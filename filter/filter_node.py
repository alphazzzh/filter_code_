"""
filter_node.py  ── V6.0 LangGraph 风控节点（防腐层 + 全局单例 + 优雅降级）
====================================================================

设计原则
─────────────────────────────────────────────────────────────
1. **防腐层（ACL）隔离**：
   - 入参转换：LangGraph State（极简字典）→ 内部 ASRRecord 等复杂对象
   - 出参剥离：Stage3 巨型情报字典 → State 仅保留 final_score + redline_alert
   - 异步旁路日志：详细诊断数据后台落盘，绝不阻塞 Graph 主流程

2. **全局单例模型管理**：
   - RiskControlNode 初始化时加载全部 Pipeline（Stage1/2/3 + 引擎）
   - 绝不在 process() 中重复加载

3. **零 Event Loop 阻塞**：
   - CPU 密集任务（stage1、拓扑+LID、stage3 打分）一律 asyncio.to_thread()
   - 拓扑分析 + LID 合并为一次 to_thread 调用
   - Stage2 全链路异步（BGE/LTP HTTP → httpx.AsyncClient）

4. **优雅降级**：
   - try-except 包裹全流程
   - 超时或异常 → final_score=50 + risk_error 写入错误信息
   - 绝不阻断 LangGraph 运行
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, TypedDict

# 多语言引擎（可选依赖）
try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

# 导入核心引擎组件
from main import parse_transcript_cell
from models import ASRRecord
from models_stage2 import DialogueTurn
from stage_one_filter import StageOneFilter
from stage_two_pipeline import StageTwoPipeline
from stage_three_scorer import IntelligenceScorer, BotConfidenceEngine, AdvancedVoicemailDetector
from topology_engine import TopologyAnalyzer, TopologyEngine

logger = logging.getLogger(__name__)


# ==========================================
# 1. LangGraph State 契约 (Anti-Corruption Layer Boundary)
# ==========================================

class GlobalSessionState(TypedDict, total=False):
    """
    LangGraph 极简通信契约。

    设计考量
    ─────────────────────────────────────────────────────────
    明细数据（tags, roles, bot_confidence, score_breakdown 等）
    已由风控节点在内部异步写入 risk_audit_details.jsonl，
    绝对不塞入 State，防范大对象序列化阻塞或大模型幻觉。
    运维建议：Filebeat / Logstash 采集 .jsonl 入库 ES。

    字段说明
    ─────────────────────────────────────────────────────────
    session_id       : 会话唯一 ID（前端/网关传入）
    raw_content      : 格式如 "A说:...\nB说:..." 的纯文本
    dynamic_topic    : (可选) RAG 动态检索主题
    final_score      : [输出] 最终风险打分 (0-100)，供条件路由
    risk_error       : [输出] 节点崩溃/超时降级时的错误信息
    global_redline_alert : [输出] 全局红线熔断标记（极高危信号）
    """
    session_id: str
    raw_content: str
    dynamic_topic: Optional[str]
    final_score: Optional[int]
    risk_error: Optional[str]
    global_redline_alert: Optional[str]


# ==========================================
# 2. CPU 卸载辅助函数（顶层函数，可被 pickle）
# ==========================================

def _cpu_stage1(stage1: StageOneFilter, records: list[ASRRecord]) -> list[ASRRecord]:
    """阶段一：极速硬正则过滤（纯 CPU）"""
    return stage1.process_batch(records)


def _cpu_topo_and_lid(
    s1_records: list[ASRRecord],
    topo_engine: TopologyEngine,
    lid_model: Any,  # fasttext model 或 None
) -> dict:
    """
    拓扑分析 + LID 识别（合并为一次线程卸载，减少调度开销）。

    返回：
    {
        "turns": list[DialogueTurn],
        "topo_metrics": TopoMetrics,
        "nlp_features_extra": dict,
    }
    """
    # 拓扑分析
    topo_analyzer = TopologyAnalyzer()
    turns = topo_analyzer.merge_turns(s1_records)
    topo_metrics = topo_engine.compute_metrics(turns)

    # LID 语种检测
    nlp_features_extra: dict = {}
    if lid_model is not None:
        effective_text = " ".join(rec.effective_text for rec in s1_records if rec.effective_text)
        if effective_text.strip():
            try:
                lid_predictions, lid_confidences = lid_model.predict(
                    effective_text.replace("\n", " ")
                )
                detected_lang = lid_predictions[0].replace("__label__", "")
                confidence = lid_confidences[0]
                nlp_features_extra["detected_language"] = detected_lang
                nlp_features_extra["lid_confidence"] = round(float(confidence), 4)
            except Exception as lid_err:
                logger.warning(f"LID 预测失败: {lid_err}")
                nlp_features_extra["detected_language"] = "unknown"
        else:
            nlp_features_extra["detected_language"] = "empty"
    else:
        nlp_features_extra["detected_language"] = "unavailable"

    # 注入拓扑特征
    nlp_features_extra["filler_word_rate"] = topo_metrics.filler_word_rate
    nlp_features_extra["is_decoupled"] = topo_metrics.is_decoupled

    return {
        "turns": turns,
        "topo_metrics": topo_metrics,
        "nlp_features_extra": nlp_features_extra,
    }


def _cpu_stage3_scoring(
    stage2_result,
    scorer: IntelligenceScorer,
    bot_engine: BotConfidenceEngine,
    voicemail_engine: AdvancedVoicemailDetector,
    topo_metrics,
    nlp_features_extra: dict,
) -> dict:
    """阶段三：打分 + Bot/Voicemail 评估（纯 CPU）"""
    if nlp_features_extra:
        stage2_result.metadata["nlp_features_extra"] = nlp_features_extra

    final_result = scorer.evaluate(stage2_result)
    bot_res = bot_engine.evaluate(
        stage2_result, filler_word_rate=topo_metrics.filler_word_rate
    )
    vm_res = voicemail_engine.evaluate(
        stage2_result, is_decoupled=topo_metrics.is_decoupled
    )

    final_result.update({
        "bot_confidence": {
            "bot_score": bot_res["bot_score"],
            "bot_label": bot_res["bot_label"].value,
        },
        "voicemail_detection": {
            "voicemail_score": vm_res["voicemail_score"],
            "is_voicemail": vm_res["is_voicemail"],
        },
        "topology_metrics": {
            "is_decoupled": topo_metrics.is_decoupled,
            "filler_word_rate": topo_metrics.filler_word_rate,
        },
        "language_detection": nlp_features_extra,
    })
    return final_result


def _cpu_audit_log(
    audit_record: dict,
    audit_log_path: str,
) -> None:
    """异步旁路日志落盘（纯文件 I/O）"""
    try:
        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
    except Exception as io_err:
        logger.error(f"写入审计日志失败: {io_err}")


# ==========================================
# 3. 核心黑盒组件：风控节点（防腐层封装）
# ==========================================

class RiskControlNode:
    """
    V6.0 LangGraph 风控节点 —— 防腐层封装。

    架构要点
    ─────────────────────────────────────────────────────────
    - 全局单例：初始化时加载全部 Pipeline，process() 零模型加载
    - 防腐层（ACL）：State ↔ 内部复杂对象的转换隔离
    - CPU 卸载：asyncio.to_thread() 消除 Event Loop 阻塞
    - 异步旁路日志：详细诊断数据后台落盘，不阻塞 Graph
    - 优雅降级：异常/超时 → final_score=50 + risk_error

    使用方式
    ─────────────────────────────────────────────────────────
    # 1. 全局初始化（服务启动时执行一次）
    risk_node = RiskControlNode()

    # 2. 注册到 LangGraph
    workflow.add_node("analyze_risk", risk_node.process)

    # 3. 条件路由（基于 final_score）
    def route_by_risk_score(state):
        score = state.get("final_score", 50)
        if score >= 80:
            return "trigger_alert"
        return END
    """

    def __init__(
        self,
        bge_model_name: str = "BAAI/bge-m3",
        ltp_service_url: str | None = None,
        bge_service_url: str | None = None,
        log_dir: str = "logs",
        stage2_timeout: float = 30.0,
    ):
        """
        初始化全部 Pipeline 引擎（全局单例）。

        Parameters
        ----------
        bge_model_name : BGE-M3 向量模型路径
        ltp_service_url : LTP 微服务地址
        bge_service_url : BGE TEI 服务地址
        log_dir : 审计日志目录
        stage2_timeout : Stage2 推理 SLA 超时秒数
        """
        logger.info("[RiskControlNode] 正在加载引擎...")

        # ── 环境变量优先，参数其次，默认值兜底 ──
        actual_bge_path    = os.getenv("MODEL_BGE_PATH", bge_model_name)
        actual_bge_svc_url = os.getenv("BGE_SERVICE_URL", bge_service_url or "")
        actual_ltp_url     = os.getenv("LTP_SERVICE_URL", ltp_service_url or "http://localhost:8900")
        self._log_dir      = os.getenv("RISK_LOG_DIR", log_dir)
        self._stage2_timeout = stage2_timeout

        logger.info(
            f"模型路径配置: BGE={actual_bge_path}, "
            f"BGE-TEI={actual_bge_svc_url or '(未配置)'}, "
            f"LTP-HTTP={actual_ltp_url}"
        )

        # ── 阶段一：纯 CPU 正则引擎（无状态，线程安全）──
        self._stage1 = StageOneFilter()

        # ── 阶段二：异步模式全链路推理引擎 ──
        self._stage2 = StageTwoPipeline(
            bge_model_name  = actual_bge_path,
            bge_service_url = actual_bge_svc_url or None,
            ltp_service_url = actual_ltp_url,
            _async          = True,
        )

        # ── 阶段三：纯 CPU 打分引擎 + 独立引擎 ──
        self._scorer = IntelligenceScorer()
        self._bot_engine = BotConfidenceEngine()
        self._voicemail_engine = AdvancedVoicemailDetector()
        self._topo_engine = TopologyEngine()

        # ── LID 语种识别模型（可选依赖）──
        self._lid_model = None
        if _FASTTEXT_AVAILABLE:
            lid_path = os.getenv("MODEL_LID_PATH", "models/lid.176.bin")
            try:
                self._lid_model = fasttext.load_model(lid_path)
                logger.info(f"✅ LID 语种识别模型加载成功: {lid_path}")
            except Exception as lid_err:
                logger.warning(f"⚠️ LID 模型加载失败（不影响核心功能）: {lid_err}")
        else:
            logger.info("ℹ️ fasttext 未安装，跳过 LID 语种识别")

        # ── 确保日志目录存在 ──
        os.makedirs(self._log_dir, exist_ok=True)
        self._audit_log_path = os.path.join(self._log_dir, "risk_audit_details.jsonl")

        logger.info(f"[RiskControlNode] 引擎加载完毕。审计日志: {self._audit_log_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 防腐层入口：LangGraph Node Function
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def process(self, state: GlobalSessionState) -> dict:
        """
        风控节点主入口 —— 防腐层隔离。

        流水线编排（5 步 + 旁路日志）：
        ─────────────────────────────────────────────────────────
        Step 1: 入参转换 — raw_content → List[ASRRecord]（轻量同步）
        Step 2: Stage1 正则过滤 → asyncio.to_thread（CPU 密集）
        Step 3: 拓扑分析 + LID 识别 → asyncio.to_thread（CPU 密集）
        Step 4: Stage2 异步推理 → 原生 await（BGE/LTP HTTP 异步）
        Step 5: Stage3 打分 + Bot/Voicemail → asyncio.to_thread（CPU 密集）
        Step 6: 出参剥离 + 旁路日志 → State 仅保留极简字段
        """
        session_id = state.get("session_id", "unknown_session")
        raw_content = state.get("raw_content", "")
        dynamic_topic = state.get("dynamic_topic")

        # ── 前置校验 ──
        if not raw_content:
            logger.warning(f"[{session_id}] 收到空的对话内容")
            return {"final_score": 50, "risk_error": "No content provided"}

        try:
            # ── Step 1: 入参转换（防腐层：State → 内部对象）──
            # 轻量同步操作，耗时 <1ms，inline 执行
            records = parse_transcript_cell(raw_content, session_id)
            if not records:
                logger.warning(f"[{session_id}] 文本解析失败，无有效记录")
                return {"final_score": 50, "risk_error": "Parse failed"}

            # ── Step 2: 阶段一正则过滤（CPU 密集，卸载到后台线程）──
            s1_records = await asyncio.to_thread(
                _cpu_stage1, self._stage1, records
            )

            # ── Step 3: 拓扑分析 + LID 识别（CPU 密集，合并卸载）──
            topo_lid_result = await asyncio.to_thread(
                _cpu_topo_and_lid,
                s1_records,
                self._topo_engine,
                self._lid_model,
            )

            turns            = topo_lid_result["turns"]
            topo_metrics     = topo_lid_result["topo_metrics"]
            nlp_features_extra = topo_lid_result["nlp_features_extra"]

            # ── Step 4: 阶段二异步推理（BGE/LTP HTTP 原生 await）──
            extra_meta: dict[str, Any] = {"nlp_features_extra": nlp_features_extra}
            if dynamic_topic:
                extra_meta["dynamic_topic"] = dynamic_topic

            stage2_result = await asyncio.wait_for(
                self._stage2.process_conversation_async(
                    conversation_id=session_id,
                    records=s1_records,
                    extra_metadata=extra_meta,
                    pre_merged_turns=turns,
                ),
                timeout=self._stage2_timeout,
            )

            # ── Step 5: 阶段三打分 + Bot/Voicemail（CPU 密集，卸载）──
            final_result = await asyncio.to_thread(
                _cpu_stage3_scoring,
                stage2_result,
                self._scorer,
                self._bot_engine,
                self._voicemail_engine,
                topo_metrics,
                nlp_features_extra,
            )

            # ── Step 6: 出参剥离（防腐层：内部对象 → State）──
            return self._extract_state_output(final_result, session_id)

        except asyncio.TimeoutError:
            logger.error(f"[{session_id}] Stage2 推理超时（{self._stage2_timeout}s），触发熔断降级。")
            return {"final_score": 50, "risk_error": f"Stage2 timeout ({self._stage2_timeout}s)"}

        except Exception as e:
            logger.error(f"[{session_id}] 风控节点处理异常: {e}", exc_info=True)
            return {"final_score": 50, "risk_error": str(e)}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 防腐层核心：出参剥离 + 旁路日志
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_state_output(self, final_result: dict, session_id: str) -> dict:
        """
        出参剥离：从 Stage3 巨型情报字典中提取极简字段回 State，
        同时异步旁路写入完整诊断数据到 .jsonl。

        State 仅保留（防腐层边界）：
          - final_score : 0-100 风险分（供 LangGraph 条件路由）
          - global_redline_alert : 红线熔断标记（极高危信号）
          - risk_error : None（正常流程无错误）

        详细数据（tags, roles, breakdown, bot_confidence 等）
        全部写入 risk_audit_details.jsonl，不塞入 State。
        """
        final_score = final_result.get("final_score", 50)
        redline_alert = final_result.get("global_redline_alert")

        # ── 异步旁路日志：详细诊断数据后台落盘 ──
        # 使用 asyncio.create_task 调度，不阻塞当前 process() 返回
        audit_record = self._build_audit_record(final_result, session_id)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                asyncio.to_thread(
                    _cpu_audit_log,
                    audit_record,
                    self._audit_log_path,
                )
            )
        except RuntimeError:
            # 无运行中的 loop（理论上不会出现，process 本身是 async）
            _cpu_audit_log(audit_record, self._audit_log_path)

        # ── 构建极简 State 输出 ──
        state_update: dict[str, Any] = {
            "final_score": final_score,
            "risk_error": None,
        }

        if redline_alert:
            state_update["global_redline_alert"] = redline_alert

        return state_update

    @staticmethod
    def _build_audit_record(final_result: dict, session_id: str) -> dict:
        """
        构建审计日志记录（完整诊断数据，供 ES 入库分析）。

        包含但不限于：
          - final_score, tags, roles, score_breakdown
          - bot_confidence, voicemail_detection
          - topology_metrics, language_detection
          - interaction_summary, nlp_features_summary, dynamic_search
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "final_score": final_result.get("final_score", 50),
            "tags": final_result.get("tags", []),
            "roles": final_result.get("roles", {}),
            "global_redline_alert": final_result.get("global_redline_alert"),
            "bot_confidence": final_result.get("bot_confidence", {}),
            "voicemail_detection": final_result.get("voicemail_detection", {}),
            "topology_metrics": final_result.get("topology_metrics", {}),
            "language_detection": final_result.get("language_detection", {}),
            "score_breakdown": final_result.get("score_breakdown", []),
            "interaction_summary": final_result.get("interaction_summary", {}),
            "nlp_features_summary": final_result.get("nlp_features_summary", {}),
            "dynamic_search": final_result.get("dynamic_search", {}),
        }


# ==========================================
# 4. 向后兼容：FilterNode 别名
# ==========================================

# 旧代码中 from filter_node import FilterNode 仍可工作
FilterNode = RiskControlNode

# 旧版 ConversationState 别名（向后兼容）
ConversationState = GlobalSessionState
