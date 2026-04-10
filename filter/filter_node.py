import asyncio
import logging
import json
import os
from datetime import datetime
from typing import TypedDict, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# 多语言引擎
try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

# 导入你 V5.1 引擎的各个核心组件
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
# 2. 核心黑盒组件：风控节点
# ==========================================
class FilterNode:
    """
    V5.1  LangGraph 节点封装。
    支持环境变量配置、LID 语种识别，带异步安全的本地 JSONL 落盘机制。
    """
    def __init__(
        self, 
        bge_model_name: str = "BAAI/bge-m3", 
        ltp_service_url: str | None = None,
        bge_service_url: str | None = None,
        log_dir: str = "logs"
    ):
        logger.info("[Filter Node] 正在加载...")
        
        # 👇 最佳实践：环境变量优先，参数其次，默认值兜底
        actual_bge_path    = os.getenv("MODEL_BGE_PATH", bge_model_name)
        actual_bge_svc_url = os.getenv("BGE_SERVICE_URL", bge_service_url or "")
        actual_ltp_url     = os.getenv("LTP_SERVICE_URL", ltp_service_url or "http://localhost:8900")
        self.log_dir = os.getenv("RISK_LOG_DIR", log_dir)
        
        logger.info(f"模型路径配置: BGE={actual_bge_path}, BGE-TEI={actual_bge_svc_url or '(未配置)'}, LTP-HTTP={actual_ltp_url}")

        self.stage1 = StageOneFilter()
        self.stage2 = StageTwoPipeline(
            bge_model_name  = actual_bge_path,
            bge_service_url = actual_bge_svc_url or None,
            ltp_service_url = actual_ltp_url,
        )
        self.scorer = IntelligenceScorer()
        self.voicemail_engine = AdvancedVoicemailDetector()
        self.topo_engine = TopologyEngine()
        
        # 👇 找回丢失的 LID 语种识别引擎
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
        
        self.cpu_pool = ThreadPoolExecutor(max_workers=int(os.getenv("NODE_MAX_CONCURRENT", 64)))
        self.gpu_semaphore = asyncio.Semaphore(int(os.getenv("NODE_MAX_CONCURRENT", 64)))
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_log_path = os.path.join(self.log_dir, "risk_audit_details.jsonl")
        
        logger.info(f"[Filter Node] 引擎加载完毕。诊断日志将写入: {self.audit_log_path}")

    async def process(self, state: ConversationState) -> dict:
        session_id = state.get("session_id", "unknown_session")
        raw_content = state.get("raw_content", "")
        dynamic_topic = state.get("dynamic_topic")

        if not raw_content:
            logger.warning(f"[{session_id}] 收到空的对话内容")
            return {"error": "No content provided", "final_score": 50}

        try:
            loop = asyncio.get_running_loop()
            
            # ── 1. 解析文本 ──
            records = parse_transcript_cell(raw_content, session_id)
            if not records:
                return {"error": "Parse failed", "final_score": 50}

            # ── 2. 阶段一：极速硬正则过滤 ──
            s1_func = partial(self.stage1.process_batch, records)
            s1_records = await loop.run_in_executor(self.cpu_pool, s1_func)

            # ── 3. 拓扑引擎分析与 LID 识别 ──
            topo_analyzer = TopologyAnalyzer()
            turns = topo_analyzer.merge_turns(s1_records)
            topo_metrics = self.topo_engine.compute_metrics(turns)

            # 👇 执行 LID 语种检测
            nlp_features_extra = {}
            if self.lid_model is not None:
                effective_text = " ".join(rec.effective_text for rec in s1_records if rec.effective_text)
                if effective_text.strip():
                    try:
                        lid_predictions, lid_confidences = self.lid_model.predict(effective_text.replace("\n", " "))
                        detected_lang = lid_predictions[0].replace("__label__", "")
                        confidence = lid_confidences[0]
                        nlp_features_extra["detected_language"] = detected_lang
                        nlp_features_extra["lid_confidence"] = round(float(confidence), 4)
                    except Exception as lid_err:
                        logger.warning(f"[{session_id}] LID 预测失败: {lid_err}")
                        nlp_features_extra["detected_language"] = "unknown"
                else:
                    nlp_features_extra["detected_language"] = "empty"
            else:
                nlp_features_extra["detected_language"] = "unavailable"

            # ── 4. 阶段二：推理 (受限并发) ──
            extra_meta = {"nlp_features_extra": nlp_features_extra}
            nlp_features_extra["filler_word_rate"] = topo_metrics.filler_word_rate
            nlp_features_extra["is_decoupled"] = topo_metrics.is_decoupled
            
            if dynamic_topic:
                extra_meta["dynamic_topic"] = dynamic_topic

            s2_func = partial(
                self.stage2.process_conversation,
                conversation_id=session_id,
                records=s1_records,
                extra_metadata=extra_meta
            )
            
            async with self.gpu_semaphore:
                stage2_result = await asyncio.wait_for(
                    loop.run_in_executor(self.cpu_pool, s2_func),
                    timeout=30.0 
                )

            # ── 5. 阶段三：打分与落盘 (在 CPU 线程池中执行 I/O) ──
            def _run_scoring_and_log():
                # 注入 LID 结果到元数据供打分器消费
                if nlp_features_extra:
                    stage2_result.metadata["nlp_features_extra"] = nlp_features_extra

                final_res = self.scorer.evaluate(stage2_result)
                voicemail_res = self.voicemail_engine.evaluate(
                    stage2_result, 
                    is_decoupled=topo_metrics.is_decoupled
                )
                
                # 组装全量情报字典（包含找回的 language_detection）
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
                        "is_decoupled": topo_metrics.is_decoupled
                    },
                    "score_breakdown": final_res.get("score_breakdown", []),
                    "interaction_summary": final_res.get("interaction_summary", {}),
                    "nlp_features_summary": final_res.get("nlp_features_summary", {}),
                    "dynamic_search": final_res.get("dynamic_search", {})
                }
                
                try:
                    with open(self.audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
                except Exception as io_err:
                    logger.error(f"[{session_id}] 写入审计日志失败: {io_err}")
                
                return final_res

            final_result = await loop.run_in_executor(self.cpu_pool, _run_scoring_and_log)

            return {
                "final_score": final_result.get("final_score", 50),
                "error": None
            }

        except asyncio.TimeoutError:
            logger.error(f"[{session_id}] 阶段二模型推理超时！触发熔断。")
            return {"final_score": 50, "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{session_id}] 风控节点处理异常: {e}", exc_info=True)
            return {"final_score": 50, "error": str(e)}