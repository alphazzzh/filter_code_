# api_server.py
# ============================================================
# V5.0 ASR 情报风控引擎 —— 企业级高并发 API 接入层
# ============================================================

import asyncio
import json
import logging
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
from topology_engine import TopologyEngine

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
    
    # 并发控制器
    cpu_pool: ThreadPoolExecutor
    gpu_semaphore: asyncio.Semaphore
    
    # 👇 过载保护计数器
    active_requests: int = 0

state = EngineState()

# 【配置项】生产环境建议抽取到环境变量
CPU_WORKERS = 16               # CPU 密集型任务的线程池大小 (建议设置为 CPU 核心数)
MAX_CONCURRENT_GPU = 8        # 允许同时进入 Stage2 (GPU BGE-M3) 的最大并发数，保护显存防 OOM
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
    
    # 严格限制进入 Stage2 推理的并发数，保护显存
    state.gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU)

    # 实例化所有引擎（加载 BGE 模型、LTP 模型等）
    state.stage1 = StageOneFilter()
    state.stage2 = StageTwoPipeline()
    state.scorer = IntelligenceScorer()
    state.bot_engine = BotConfidenceEngine()
    state.voicemail_engine = AdvancedVoicemailDetector()
    state.topo_engine = TopologyEngine()
    logger.info("✅ ML 引擎加载完毕，API 准备就绪。")
    
    yield # 运行中...

    logger.info("🛑 正在优雅关闭 API 服务，释放线程池与显存...")
    state.cpu_pool.shutdown(wait=True)

app = FastAPI(title="V5.0 ASR Risk Control Engine - Enterprise", version="5.0.0", lifespan=lifespan)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1 & 2. 契约定义 (与之前保持完全一致)
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
                if not isinstance(parsed, list): raise ValueError("Must be a JSON array.")
                return parsed
            except json.JSONDecodeError:
                raise ValueError("Content is a string but NOT valid JSON.")
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
# 3. 核心路由处理 (异步卸载与降级架构)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/v1/health")
async def health_check():
    """探针接口：因为 ML 任务被卸载到了线程池，此接口永远能瞬间响应 200"""
    return {"status": "ok", "gpu_queue": state.gpu_semaphore._value}

@app.post("/api/v1/analyze", response_model=StandardResponse)
async def analyze_conversation(req: AnalyzeRequest, response: Response):
    """
    企业级风控分析接口：严格的并发控制与超时降级机制。
    """
    # 👇 1. 全局过载保护 (Load Shedding / 快速熔断)
    #    拦截器在解析任何 JSON 之前执行，保证消耗最小
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

    # 未满载，允许进入系统，计数器 +1
    state.active_requests += 1
    
    try:
        logger.info(f"Received request for session_id: {req.session_id}")
        loop = asyncio.get_running_loop()
        
        # ── Step 1: 解析 ──
        raw_turns = req.data.content
        records = [
            ASRRecord(
                record_id=str(turn if isinstance(turn, dict) else turn.dict().get("id", "")),
                speaker_id=str(turn if isinstance(turn, dict) else turn.dict().get("speaker", "")),
                raw_text=str(turn if isinstance(turn, dict) else turn.dict().get("content", ""))
            )
            for turn in raw_turns
        ]
        if not records:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return StandardResponse(status=400, message="Content array is empty", session_id=req.session_id)

        # ── Step 2: 阶段一 (CPU 计算密集，放入线程池避免卡死 API 网关) ──
        s1_func = partial(state.stage1.process_batch, records)
        s1_records = await loop.run_in_executor(state.cpu_pool, s1_func)
        
        valid_records = [r for r in s1_records if r.connection_status != ConnectionStatus.UNCONNECTED]
        if not valid_records:
            return StandardResponse(
                status=200, message="OK", session_id=req.session_id,
                data={"final_score": 5, "tags": ["stage_one_physical_interception"], "reason": "极低价值废料拦截"}
            )

        # ── Step 3: 阶段二 (GPU 计算密集，使用 Semaphore 排队 + 超时熔断) ──
        extra_meta = {"dynamic_topic": req.dynamic_topic} if req.dynamic_topic else {}
        stage2_func = partial(
            state.stage2.process_conversation,
            conversation_id=req.session_id,
            records=valid_records,
            extra_metadata=extra_meta
        )

        try:
            # 申请 GPU 锁，控制最大并发量
            async with state.gpu_semaphore:
                # 给大模型推理加上 SLA 超时熔断机制
                stage2_result = await asyncio.wait_for(
                    loop.run_in_executor(state.cpu_pool, stage2_func),
                    timeout=STAGE2_TIMEOUT_SECONDS
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
            # 推理报错（如 OOM）平滑降级
            logger.error(f"[{req.session_id}] Stage2 推理崩溃: {stage_exc}", exc_info=True)
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            return StandardResponse(
                status=206, message="Partial Content: Stage2 Error", session_id=req.session_id,
                data={"final_score": 50, "tags": ["stage2_error_degraded"], "_error": str(stage_exc)}
            )

        # ── Step 4: 阶段三打分与拓扑 (纯 CPU 极快计算，继续卸载) ──
        def _run_scoring_sync():
            final_result = state.scorer.evaluate(stage2_result)
            topo_metrics = state.topo_engine.compute_metrics(stage2_result.dialogue_turns)
            bot_res = state.bot_engine.evaluate(stage2_result, filler_word_rate=topo_metrics.filler_word_rate)
            vm_res = state.voicemail_engine.evaluate(stage2_result, is_decoupled=topo_metrics.is_decoupled)
            
            final_result.update({
                "bot_confidence": {"bot_score": bot_res["bot_score"], "bot_label": bot_res["bot_label"].value},
                "voicemail_detection": {"voicemail_score": vm_res["voicemail_score"], "is_voicemail": vm_res["is_voicemail"]},
                "topology_metrics": {"is_decoupled": topo_metrics.is_decoupled}
            })
            return final_result

        final_result = await loop.run_in_executor(state.cpu_pool, _run_scoring_sync)

        # ── Step 5: 成功返回 ──
        return StandardResponse(status=200, message="OK", session_id=req.session_id, data=final_result)

    except Exception as e:
        logger.error(f"处理 session_id: {req.session_id} 发生致命错误: {str(e)}", exc_info=True)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return StandardResponse(status=500, message=f"Internal Server Error: {str(e)}", session_id=req.session_id)
        
    finally:
        # 👇 2. 无论请求是成功、失败、还是降级，离开系统时必须释放计数器
        state.active_requests -= 1