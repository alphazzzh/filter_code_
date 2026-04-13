"""
filter_node.py  ── V5.5 LangGraph 节点封装（全链路异步化 + CPU 并行卸载）
============================================================

设计原则
─────────────────────────────────────────────────────────────
1. **零 Event Loop 阻塞**：所有 CPU 密集任务（stage1 正则、拓扑分析、
   LID 预测、stage3 打分、日志落盘）一律通过 asyncio.to_thread() 卸载
   到后台线程，绝不阻塞 FastAPI 主循环。
2. **CPU 任务并行化**：拓扑分析和 LID 识别无数据依赖，使用
   asyncio.gather() 并行执行，减少串行等待。
3. **无手动线程池**：废弃 ThreadPoolExecutor，改用 asyncio.to_thread()
   的默认 executor，减少手动配置和资源管理负担。
4. **全链路异步 I/O**：stage2 的 BGE/LTP HTTP 调用走 httpx.AsyncClient，
   原生 async/await，无需线程池包装。
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import TypedDict, Optional

# 多语言引擎（可选依赖）
try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

# 导入核心引擎组件
from main import parse_transcript_cell
from stage_one_filter import StageOneFilter
from stage_two_pipeline import StageTwoPipeline
from stage_three_scorer import IntelligenceScorer, AdvancedVoicemailDetector
from topology_engine import TopologyAnalyzer, TopologyEngine

logger = logging.getLogger(__name__)

# ==========================================
# 1. 极简通信契约 (State Schema)
# ==========================================
class ConversationState(TypedDict):
    session_id: str
    raw_content: str
    dynamic_topic: Optional[str]
    final_score: Optional[int]
    error: Optional[str]


# ==========================================
# 2. CPU 卸载辅助函数（顶层函数，可被 pickle）
# ==========================================

def _cpu_stage1(stage1: StageOneFilter, records: list) -> list:
    """阶段一：极速硬正则过滤（纯 CPU）"""
    return stage1.process_batch(records)


def _cpu_topo_and_lid(
    s1_records: list,
    topo_engine: TopologyEngine,
    lid_model,  # fasttext model 或 None
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


def _cpu_stage3_and_log(
    stage2_result,
    scorer: IntelligenceScorer,
    voicemail_engine: AdvancedVoicemailDetector,
    topo_metrics,
    nlp_features_extra: dict,
    session_id: str,
    audit_log_path: str,
) -> dict:
    """阶段三：打分 + 审计日志落盘（纯 CPU + 文件 I/O）"""
    if nlp_features_extra:
        stage2_result.metadata["nlp_features_extra"] = nlp_features_extra

    final_res = scorer.evaluate(stage2_result)
    voicemail_res = voicemail_engine.evaluate(
        stage2_result,
        is_decoupled=topo_metrics.is_decoupled,
    )

    audit_record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "final_score": final_res.get("final_score", 50),
        "tags": final_res.get("tags", []),
        "roles": final_res.get("roles", {}),
        "voicemail_detection": voicemail_res,
        "language_detection": nlp_features_extra,
        "topology_metrics": {
            "filler_word_rate": topo_metrics.filler_word_rate,
            "is_decoupled": topo_metrics.is_decoupled,
        },
        "score_breakdown": final_res.get("score_breakdown", []),
        "interaction_summary": final_res.get("interaction_summary", {}),
        "nlp_features_summary": final_res.get("nlp_features_summary", {}),
        "dynamic_search": final_res.get("dynamic_search", {}),
    }

    try:
        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
    except Exception as io_err:
        logger.error(f"[{session_id}] 写入审计日志失败: {io_err}")

    return final_res


# ==========================================
# 3. 核心黑盒组件：风控节点
# ==========================================
class FilterNode:
    """
    V5.5 LangGraph 节点封装。

    架构要点
    ─────────────────────────────────────────────────────────
    - CPU 密集任务一律 asyncio.to_thread() 卸载，零 Event Loop 阻塞
    - 拓扑分析 + LID 识别在同一次 to_thread 调用中完成（减少调度开销）
    - Stage2 全链路异步（BGE/LTP HTTP 走 httpx.AsyncClient）
    - 无手动 ThreadPoolExecutor，使用 asyncio 默认 executor
    """

    def __init__(
        self,
        bge_model_name: str = "BAAI/bge-m3",
        ltp_service_url: str | None = None,
        bge_service_url: str | None = None,
        log_dir: str = "logs",
    ):
        logger.info("[FilterNode] 正在加载引擎...")

        # 环境变量优先，参数其次，默认值兜底
        actual_bge_path    = os.getenv("MODEL_BGE_PATH", bge_model_name)
        actual_bge_svc_url = os.getenv("BGE_SERVICE_URL", bge_service_url or "")
        actual_ltp_url     = os.getenv("LTP_SERVICE_URL", ltp_service_url or "http://localhost:8900")
        self.log_dir = os.getenv("RISK_LOG_DIR", log_dir)

        logger.info(
            f"模型路径配置: BGE={actual_bge_path}, "
            f"BGE-TEI={actual_bge_svc_url or '(未配置)'}, "
            f"LTP-HTTP={actual_ltp_url}"
        )

        # 阶段一：纯 CPU 正则引擎（无状态，线程安全）
        self.stage1 = StageOneFilter()

        # 阶段二：异步模式全链路推理引擎
        self.stage2 = StageTwoPipeline(
            bge_model_name  = actual_bge_path,
            bge_service_url = actual_bge_svc_url or None,
            ltp_service_url = actual_ltp_url,
            _async          = True,
        )

        # 阶段三：纯 CPU 打分引擎
        self.scorer = IntelligenceScorer()
        self.voicemail_engine = AdvancedVoicemailDetector()
        self.topo_engine = TopologyEngine()

        # LID 语种识别模型（可选依赖）
        self.lid_model = None
        if _FASTTEXT_AVAILABLE:
            lid_path = os.getenv("MODEL_LID_PATH", "models/lid.176.bin")
            try:
                self.lid_model = fasttext.load_model(lid_path)
                logger.info(f"✅ LID 语种识别模型加载成功: {lid_path}")
            except Exception as lid_err:
                logger.warning(f"⚠️ LID 模型加载失败（不影响核心功能）: {lid_err}")
        else:
            logger.info("ℹ️ fasttext 未安装，跳过 LID 语种识别")

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_log_path = os.path.join(self.log_dir, "risk_audit_details.jsonl")

        logger.info(f"[FilterNode] 引擎加载完毕。诊断日志: {self.audit_log_path}")

    async def process(self, state: ConversationState) -> dict:
        """
        全链路风控处理（零 Event Loop 阻塞）。

        流水线编排：
        ─────────────────────────────────────────────────────────
        Step 1: 解析文本（同步轻量，inline 执行）
        Step 2: Stage1 正则过滤 → asyncio.to_thread（CPU 密集）
        Step 3: 拓扑分析 + LID 识别 → asyncio.to_thread（CPU 密集）
        Step 4: Stage2 异步推理 → 原生 await（BGE/LTP HTTP 异步）
        Step 5: Stage3 打分 + 日志 → asyncio.to_thread（CPU + 文件 I/O）
        """
        session_id = state.get("session_id", "unknown_session")
        raw_content = state.get("raw_content", "")
        dynamic_topic = state.get("dynamic_topic")

        if not raw_content:
            logger.warning(f"[{session_id}] 收到空的对话内容")
            return {"error": "No content provided", "final_score": 50}

        try:
            # ── Step 1: 解析文本（轻量同步操作，耗时 <1ms，inline 执行）──
            records = parse_transcript_cell(raw_content, session_id)
            if not records:
                return {"error": "Parse failed", "final_score": 50}

            # ── Step 2: 阶段一 ──
            # CPU 密集（正则匹配 + 特征填充），卸载到后台线程
            s1_records = await asyncio.to_thread(
                _cpu_stage1, self.stage1, records
            )

            # ── Step 3: 拓扑分析 + LID 识别 ──
            # 两者无数据依赖，合并为一次 to_thread 调用（减少线程调度开销）
            # 内部串行执行，总耗时 = topo + lid（而非 2×max(topo, lid)），
            # 但省去了一次线程创建 + GIL 切换的开销
            topo_lid_result = await asyncio.to_thread(
                _cpu_topo_and_lid,
                s1_records,
                self.topo_engine,
                self.lid_model,
            )

            turns            = topo_lid_result["turns"]
            topo_metrics     = topo_lid_result["topo_metrics"]
            nlp_features_extra = topo_lid_result["nlp_features_extra"]

            # ── Step 4: 阶段二 ──
            # 全链路异步：BGE/LTP HTTP 走 httpx.AsyncClient，原生 await
            extra_meta = {"nlp_features_extra": nlp_features_extra}
            if dynamic_topic:
                extra_meta["dynamic_topic"] = dynamic_topic

            stage2_result = await asyncio.wait_for(
                self.stage2.process_conversation_async(
                    conversation_id=session_id,
                    records=s1_records,
                    extra_metadata=extra_meta,
                    pre_merged_turns=turns,
                ),
                timeout=30.0,
            )

            # ── Step 5: 阶段三打分 + 审计日志 ──
            # 纯 CPU 打分 + 文件 I/O 落盘，卸载到后台线程
            final_result = await asyncio.to_thread(
                _cpu_stage3_and_log,
                stage2_result,
                self.scorer,
                self.voicemail_engine,
                topo_metrics,
                nlp_features_extra,
                session_id,
                self.audit_log_path,
            )

            return {
                "final_score": final_result.get("final_score", 50),
                "error": None,
            }

        except asyncio.TimeoutError:
            logger.error(f"[{session_id}] 阶段二模型推理超时！触发熔断。")
            return {"final_score": 50, "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{session_id}] 风控节点处理异常: {e}", exc_info=True)
            return {"final_score": 50, "error": str(e)}
