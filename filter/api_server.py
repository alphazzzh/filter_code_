# api_server.py
# ============================================================
# V5.1 ASR 情报风控引擎 —— 企业级高并发 API 接入层
#
# V5.0 架构
# ─────────────────────────────────────────────────────────────
# ① 三阶段流水线编排（StageOneFilter → StageTwoPipeline → IntelligenceScorer）
# ② GPU 并发控制（Semaphore）+ CPU 线程池卸载（ThreadPoolExecutor）
# ③ SLA 超时熔断（asyncio.wait_for）+ 异常降级（206 Partial Content）
# ④ BOT 数据进入全链路（不再在阶段一硬性拦截）
#
# V5.1 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 全局过载保护（MAX_GLOBAL_REQUESTS=50 + asyncio.Lock 原子计数器）
# ② Step 2.5 新增拓扑分析 + fastText LID 语种打标
# ③ 阶段二/三异常降级为 206（原 500），便于 DevOps 监控
# ④ CUDA OOM 捕获 + torch.cuda.empty_cache() 显存碎片清理
# ⑤ 依赖注入模型路径（MODEL_BGE_PATH / BGE_SERVICE_URL / LTP_SERVICE_URL / MODEL_LID_PATH）
# ============================================================

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# 导入内部引擎
from models import ASRRecord, ConnectionStatus, BotLabel
from stage_one_filter import StageOneFilter
from stage_two_pipeline import StageTwoPipeline
from stage_three_scorer import IntelligenceScorer, BotConfidenceEngine, AdvancedVoicemailDetector
from topology_engine import TopologyEngine, TopologyAnalyzer

# 多语言引擎（可选依赖，加载失败不阻塞启动）
try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

# CUDA 显存管理（可选依赖，无 torch 时不影响服务启动）
try:
    import torch
    _HAS_TORCH_CUDA = hasattr(torch, "cuda") and torch.cuda.is_available()
except ImportError:
    _HAS_TORCH_CUDA = False

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 引擎全局状态管理与并发控制
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EngineState:
    # 模型实例
    stage1: StageOneFilter
    stage2: StageTwoPipeline
    scorer: IntelligenceScorer
    bot_engine: BotConfidenceEngine
    voicemail_engine: AdvancedVoicemailDetector
    topo_engine: TopologyEngine
    lid_model: Any | None = None  # fasttext 语种识别模型（可选）
    
    # 并发控制器
    cpu_pool: ThreadPoolExecutor
    
    # 👇 过载保护计数器（配合 asyncio.Lock 保证原子性）
    active_requests: int = 0
    _request_lock: Optional[asyncio.Lock] = None

state = EngineState()

# 【配置项】生产环境建议抽取到环境变量
CPU_WORKERS = 16               # CPU 密集型任务的线程池大小 (建议设置为 CPU 核心数)
STAGE2_TIMEOUT_SECONDS = 30.0 # Stage2 处理的 SLA 超时时间

# 👇 系统最大容忍并发水位线
# 假设 8 个在算，其余在排队，最多允许 50 个请求停留在系统内
# 生产环境根据服务器内存和容忍延迟调整
MAX_GLOBAL_REQUESTS = 50

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：
    确保模型在启动时加载到内存/显存（预热），而不是在第一次请求时加载（避免冷启动超时）。
    """
    logger.info("🚀 正在预热企业级 ML 引擎与线程池...")
    state.cpu_pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)
    
    # 过载保护计数器的异步锁（必须在 event loop 内创建）
    state._request_lock = asyncio.Lock()

    # 从环境变量读取模型路径（适配微服务部署）
    bge_path    = os.getenv("MODEL_BGE_PATH", "BAAI/bge-m3")
    bge_svc_url = os.getenv("BGE_SERVICE_URL", "")
    ltp_url     = os.getenv("LTP_SERVICE_URL", "http://localhost:8900")
    logger.info(f"模型路径配置: BGE={bge_path}, BGE-TEI={bge_svc_url or '(未配置)'}, LTP-HTTP={ltp_url}")

    # 实例化所有引擎（LTP/BGE 已抽离为独立微服务，主进程不再加载大模型）
    # 使用异步模式 StageTwoPipeline
    state.stage1 = StageOneFilter()
    state.stage2 = StageTwoPipeline(
        bge_model_name  = bge_path,
        bge_service_url = bge_svc_url or None,
        ltp_service_url = ltp_url,
        _async          = True,
    )
    state.scorer = IntelligenceScorer()
    state.bot_engine = BotConfidenceEngine()
    state.voicemail_engine = AdvancedVoicemailDetector()
    state.topo_engine = TopologyEngine()

    # 加载 LID 语种识别模型（可选依赖）
    if _FASTTEXT_AVAILABLE:
        lid_path = os.getenv("MODEL_LID_PATH", "models/lid.176.bin")
        try:
            state.lid_model = fasttext.load_model(lid_path)
            logger.info(f"✅ LID 语种识别模型加载成功: {lid_path}")
        except Exception as lid_err:
            logger.warning(f"⚠️ LID 模型加载失败（不影响核心功能）: {lid_err}")
            state.lid_model = None
    else:
        logger.info("ℹ️ fasttext 未安装，跳过 LID 语种识别")

    logger.info("✅ ML 引擎加载完毕，API 准备就绪。")
    
    yield # 运行中...

    logger.info("🛑 正在优雅关闭 API 服务，释放线程池与显存...")
    state.cpu_pool.shutdown(wait=True)

app = FastAPI(title="V5.1 ASR Risk Control Engine - Enterprise", version="5.1.0", lifespan=lifespan)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Pydantic 契约定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TurnContent(BaseModel):
    id: str
    speaker: Union[int, str]
    content: str

class DataPayload(BaseModel):
    session_id: Optional[str] = None
    content: Union[str, List[TurnContent]] 
    language: Optional[str] = None

    @validator('content')
    def parse_content_if_string(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list): 
                    return parsed
            except json.JSONDecodeError:
                pass # 忽略 JSON 解析错误
            # 👇 核心改动：如果不是 JSON 数组，直接原样返回纯文本字符串，交由下游正则处理
            return v 
        return v

class AnalyzeRequest(BaseModel):
    session_id: str
    data: DataPayload
    dynamic_topic: Optional[str] = Field(default=None)

class StandardResponse(BaseModel):
    status: int
    message: str
    session_id: str
    data: Optional[Dict[str, Any]] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 核心路由处理（异步卸载与降级架构）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/v1/health")
async def health_check():
    """探针接口：因为 ML 任务被卸载到了线程池，此接口永远能瞬间响应 200"""
    return {"status": "ok"}

@app.post("/api/analyze", response_model=StandardResponse)
async def analyze_conversation(req: AnalyzeRequest, response: Response, debug: bool = False):
    """
    企业级风控分析接口：严格的并发控制与超时降级机制。
    """
    # 👇 1. 全局过载保护 (Load Shedding / 快速熔断)
    #    使用 asyncio.Lock 保证 active_requests 的检查与递增是原子操作
    async with state._request_lock:
        if state.active_requests >= MAX_GLOBAL_REQUESTS:
            logger.warning(
                f"服务器触发过载保护！当前活跃请求: {state.active_requests}，"
                f"已拒载 session_id: {req.session_id}"
            )
            response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
            return StandardResponse(
                status=429,
                message="Server is at full capacity. Please try again later.",
                session_id=req.session_id,
            )
        state.active_requests += 1
    
    try:
        logger.info(f"Received request for session_id: {req.session_id}")
        loop = asyncio.get_running_loop()
        
        # ── Step 1: 解析 ──
        raw_turns = req.data.content
        records = []
        
        # 👇 智能路由：如果是纯字符串，调用 main.py 里的强大正则解析器
        if isinstance(raw_turns, str):
            from main import parse_transcript_cell
            records = parse_transcript_cell(raw_turns, req.session_id)
            
        # 👇 否则，走原来的结构化 JSON 数组解析逻辑
        else:
            for turn in raw_turns:
                if isinstance(turn, dict):
                    r_id  = str(turn.get("id", ""))
                    r_spk = str(turn.get("speaker", ""))
                    r_txt = str(turn.get("content", ""))
                else:
                    # 兼容 Pydantic Model (V1 dict() / V2 model_dump())
                    r_id  = str(getattr(turn, "id", ""))
                    r_spk = str(getattr(turn, "speaker", ""))
                    r_txt = str(getattr(turn, "content", ""))
                    
                records.append(ASRRecord(record_id=r_id, speaker_id=r_spk, raw_text=r_txt))
        if not records:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return StandardResponse(status=400, message="Content array is empty", session_id=req.session_id)

        # ── Step 2: 阶段一 (CPU 计算密集，放入线程池避免卡死 API 网关) ──
        s1_func = partial(state.stage1.process_batch, records)
        s1_records = await loop.run_in_executor(state.cpu_pool, s1_func)
        
        # 直接放行所有阶段一产出，保留语气词供后续拓扑引擎分析
        valid_records = s1_records

        # ── Step 2.5: 拓扑分析 + LID 语种打标（纯 CPU，线程池执行）──
        topo_analyzer = TopologyAnalyzer()
        turns = topo_analyzer.merge_turns(valid_records)
        topo_metrics = state.topo_engine.compute_metrics(turns)

        # LID 语种识别（打标用，不做翻译，利用 BGE-M3 原生多语言能力）
        nlp_features_extra: dict[str, Any] = {}
        if state.lid_model is not None:
            effective_text = " ".join(
                rec.effective_text for rec in valid_records if rec.effective_text
            )
            if effective_text.strip():
                try:
                    lid_predictions, lid_confidences = state.lid_model.predict(effective_text.replace("\n", " "))
                    detected_lang = lid_predictions[0].replace("__label__", "")
                    confidence = lid_confidences[0]
                    nlp_features_extra["detected_language"] = detected_lang
                    nlp_features_extra["lid_confidence"] = round(float(confidence), 4)
                    logger.info(f"[{req.session_id}] LID 检测语种: {detected_lang} (置信度: {confidence:.4f})")
                except Exception as lid_err:
                    logger.warning(f"[{req.session_id}] LID 预测失败: {lid_err}")
                    nlp_features_extra["detected_language"] = "unknown"
            else:
                nlp_features_extra["detected_language"] = "empty"
        else:
            nlp_features_extra["detected_language"] = "unavailable"

        # ── Step 3: 阶段二 (异步全链路推理，无需线程池包装) ──
        # 将 nlp_features_extra 注入 extra_metadata，随 stage2 结果流到 scorer
        extra_meta = {"dynamic_topic": req.dynamic_topic} if req.dynamic_topic else {}
        nlp_features_extra["filler_word_rate"] = topo_metrics.filler_word_rate
        nlp_features_extra["is_decoupled"] = topo_metrics.is_decoupled
        extra_meta["nlp_features_extra"] = nlp_features_extra

        try:
            # 🔥 核心改造：直接 await 异步方法，不再 run_in_executor
            stage2_result = await asyncio.wait_for(
                state.stage2.process_conversation_async(
                    conversation_id=req.session_id,
                    records=valid_records,
                    extra_metadata=extra_meta,
                    pre_merged_turns=turns,
                ),
                timeout=STAGE2_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # 【企业级特性】超时触发平滑降级，返回 206
            logger.warning(f"[{req.session_id}] Stage2 推理超时 ({STAGE2_TIMEOUT_SECONDS}s)，触发降级。")
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            return StandardResponse(
                status=206, message="Partial Content: Stage2 Timeout", session_id=req.session_id,
                data={"final_score": 50, "tags": ["stage2_timeout_degraded"], "_error": "SLA Timeout"}
            )
        except Exception as stage_exc:
            # 推理报错（含 CUDA OOM）平滑降级
            logger.error(f"[{req.session_id}] Stage2 推理崩溃: {stage_exc}", exc_info=True)
            # 🔧 显式处理 CUDA OOM：释放显存碎片，避免后续请求连锁失败
            _exc_msg = str(stage_exc).lower()
            if "out of memory" in _exc_msg or "cuda" in _exc_msg:
                if _HAS_TORCH_CUDA:
                    try:
                        torch.cuda.empty_cache()
                        logger.warning(f"[{req.session_id}] CUDA OOM 检测到，已执行 torch.cuda.empty_cache()")
                    except Exception as cuda_cleanup_err:
                        logger.error(f"[{req.session_id}] CUDA 缓存清理失败: {cuda_cleanup_err}")
                else:
                    logger.warning(f"[{req.session_id}] 检测到 CUDA 异常但 torch 不可用，跳过显存清理")
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            return StandardResponse(
                status=206, message="Partial Content: Stage2 Error", session_id=req.session_id,
                data={"final_score": 50, "tags": ["stage2_error_degraded"], "_error": "Inference error (OOM or CUDA fault)"}
            )

        # ── Step 4: 阶段三打分与拓扑 (纯 CPU 极快计算，继续卸载) ──
        def _run_scoring_sync():
            # 将 LID 打标结果合并入 stage2_result 的 interaction_features
            if nlp_features_extra:
                stage2_result.metadata["nlp_features_extra"] = nlp_features_extra

            final_result = state.scorer.evaluate(stage2_result)
            # 复用 Step 2.5 已计算的 topo_metrics，避免重复计算
            bot_res = state.bot_engine.evaluate(stage2_result, filler_word_rate=topo_metrics.filler_word_rate)
            vm_res = state.voicemail_engine.evaluate(stage2_result, is_decoupled=topo_metrics.is_decoupled)
            
            final_result.update({
                "bot_confidence": {"bot_score": bot_res["bot_score"], "bot_label": bot_res["bot_label"].value},
                "voicemail_detection": {"voicemail_score": vm_res["voicemail_score"], "is_voicemail": vm_res["is_voicemail"]},
                "topology_metrics": {
                    "is_decoupled": topo_metrics.is_decoupled,
                    "filler_word_rate": topo_metrics.filler_word_rate,
                },
                # 透传 LID 检测结果
                "language_detection": nlp_features_extra,
            })
            return final_result

        final_result = await loop.run_in_executor(state.cpu_pool, _run_scoring_sync)

        # ── Step 5: 成功返回 ──
        if not debug:
            final_result.pop("bot_confidence", None)
            final_result.pop("voicemail_detection", None)
            final_result.pop("topology_metrics", None)
            final_result.pop("language_detection", None)
        return StandardResponse(status=200, message="OK", session_id=req.session_id, data=final_result)

    except Exception as e:
        logger.error(f"处理 session_id: {req.session_id} 发生致命错误: {str(e)}", exc_info=True)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return StandardResponse(status=500, message="Internal Server Error", session_id=req.session_id)
        
    finally:
        # 👇 2. 无论请求是成功、失败、还是降级，离开系统时必须释放计数器（加锁保证原子性）
        async with state._request_lock:
            state.active_requests -= 1