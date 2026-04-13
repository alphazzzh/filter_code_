# api_server.py
# ============================================================
# V5.5 ASR 情报风控引擎 —— 企业级高并发 API 接入层
#
# V5.5 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 废弃 ThreadPoolExecutor，改用 asyncio.to_thread() 卸载 CPU 任务
# ② 拓扑分析 + LID 识别合并为一次 to_thread 调用（消除裸跑阻塞）
# ③ Stage2 全链路异步（BGE/LTP HTTP → httpx.AsyncClient，原生 await）
# ④ Stage3 打分 + Bot/Voicemail 评估卸载到 to_thread（CPU + I/O）
# ⑤ 全局过载保护（MAX_GLOBAL_REQUESTS + asyncio.Lock 原子计数器）
# ⑥ SLA 超时熔断 + CUDA OOM 捕获 + 异常降级（206 Partial Content）
# ============================================================

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
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
# 0. CPU 卸载辅助函数（顶层函数，可被 pickle）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cpu_stage1(stage1: StageOneFilter, records: list) -> list:
    """阶段一：极速硬正则过滤（纯 CPU）"""
    return stage1.process_batch(records)


def _cpu_topo_and_lid(
    s1_records: list,
    topo_engine: TopologyEngine,
    lid_model,  # fasttext model 或 None
    session_id: str,
) -> dict:
    """
    拓扑分析 + LID 识别（合并为一次线程卸载，消除裸跑阻塞）。

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
        effective_text = " ".join(
            rec.effective_text for rec in s1_records if rec.effective_text
        )
        if effective_text.strip():
            try:
                lid_predictions, lid_confidences = lid_model.predict(
                    effective_text.replace("\n", " ")
                )
                detected_lang = lid_predictions[0].replace("__label__", "")
                confidence = lid_confidences[0]
                nlp_features_extra["detected_language"] = detected_lang
                nlp_features_extra["lid_confidence"] = round(float(confidence), 4)
                logger.info(
                    f"[{session_id}] LID 检测语种: {detected_lang} "
                    f"(置信度: {confidence:.4f})"
                )
            except Exception as lid_err:
                logger.warning(f"[{session_id}] LID 预测失败: {lid_err}")
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 引擎全局状态管理
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

    # 👇 过载保护计数器（配合 asyncio.Lock 保证原子性）
    active_requests: int = 0
    _request_lock: Optional[asyncio.Lock] = None

state = EngineState()

# 【配置项】生产环境建议抽取到环境变量
STAGE2_TIMEOUT_SECONDS = 30.0  # Stage2 处理的 SLA 超时时间

# 👇 系统最大容忍并发水位线
MAX_GLOBAL_REQUESTS = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：
    确保模型在启动时加载到内存/显存（预热），而不是在第一次请求时加载（避免冷启动超时）。
    V5.5：不再创建 ThreadPoolExecutor，CPU 任务走 asyncio.to_thread()。
    """
    logger.info("🚀 正在预热企业级 ML 引擎...")

    # 过载保护计数器的异步锁（必须在 event loop 内创建）
    state._request_lock = asyncio.Lock()

    # 从环境变量读取模型路径（适配微服务部署）
    bge_path    = os.getenv("MODEL_BGE_PATH", "BAAI/bge-m3")
    bge_svc_url = os.getenv("BGE_SERVICE_URL", "")
    ltp_url     = os.getenv("LTP_SERVICE_URL", "http://localhost:8900")
    logger.info(
        f"模型路径配置: BGE={bge_path}, "
        f"BGE-TEI={bge_svc_url or '(未配置)'}, LTP-HTTP={ltp_url}"
    )

    # 实例化所有引擎
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

    yield  # 运行中...

    logger.info("🛑 正在优雅关闭 API 服务...")

app = FastAPI(
    title="V5.5 ASR Risk Control Engine - Enterprise",
    version="5.5.0",
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Pydantic 契约定义
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
                pass
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
# 3. 核心路由处理（全链路异步 + CPU 卸载 + 降级架构）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/v1/health")
async def health_check():
    """探针接口：CPU 任务已卸载到 to_thread，此接口永远能瞬间响应 200"""
    return {"status": "ok"}


@app.post("/api/analyze", response_model=StandardResponse)
async def analyze_conversation(
    req: AnalyzeRequest, response: Response, debug: bool = False
):
    """
    企业级风控分析接口。

    V5.5 架构：
    ─────────────────────────────────────────────────────────
    Step 1: 解析文本（轻量同步，inline 执行，<1ms）
    Step 2: Stage1 正则过滤 → asyncio.to_thread（CPU 密集）
    Step 3: 拓扑分析 + LID 识别 → asyncio.to_thread（CPU 密集，合并卸载）
    Step 4: Stage2 异步推理 → 原生 await（BGE/LTP HTTP 异步）
    Step 5: Stage3 打分 + Bot/Voicemail → asyncio.to_thread（CPU 密集）
    """
    # ── 全局过载保护（Load Shedding / 快速熔断）──
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

        # ── Step 1: 解析（轻量同步，inline）──
        raw_turns = req.data.content
        records: list[ASRRecord] = []

        if isinstance(raw_turns, str):
            from main import parse_transcript_cell
            records = parse_transcript_cell(raw_turns, req.session_id)
        else:
            for turn in raw_turns:
                if isinstance(turn, dict):
                    r_id  = str(turn.get("id", ""))
                    r_spk = str(turn.get("speaker", ""))
                    r_txt = str(turn.get("content", ""))
                else:
                    r_id  = str(getattr(turn, "id", ""))
                    r_spk = str(getattr(turn, "speaker", ""))
                    r_txt = str(getattr(turn, "content", ""))
                records.append(
                    ASRRecord(record_id=r_id, speaker_id=r_spk, raw_text=r_txt)
                )

        if not records:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return StandardResponse(
                status=400, message="Content array is empty",
                session_id=req.session_id,
            )

        # ── Step 2: 阶段一（CPU 密集，卸载到后台线程）──
        s1_records = await asyncio.to_thread(_cpu_stage1, state.stage1, records)
        valid_records = s1_records

        # ── Step 3: 拓扑分析 + LID 识别（CPU 密集，合并卸载）──
        topo_lid_result = await asyncio.to_thread(
            _cpu_topo_and_lid,
            valid_records,
            state.topo_engine,
            state.lid_model,
            req.session_id,
        )

        turns            = topo_lid_result["turns"]
        topo_metrics     = topo_lid_result["topo_metrics"]
        nlp_features_extra = topo_lid_result["nlp_features_extra"]

        # ── Step 4: 阶段二（全链路异步，原生 await）──
        extra_meta: dict[str, Any] = {}
        if req.dynamic_topic:
            extra_meta["dynamic_topic"] = req.dynamic_topic
        extra_meta["nlp_features_extra"] = nlp_features_extra

        try:
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
            logger.warning(
                f"[{req.session_id}] Stage2 推理超时 "
                f"({STAGE2_TIMEOUT_SECONDS}s)，触发降级。"
            )
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            return StandardResponse(
                status=206, message="Partial Content: Stage2 Timeout",
                session_id=req.session_id,
                data={
                    "final_score": 50,
                    "tags": ["stage2_timeout_degraded"],
                    "_error": "SLA Timeout",
                },
            )
        except Exception as stage_exc:
            logger.error(
                f"[{req.session_id}] Stage2 推理崩溃: {stage_exc}",
                exc_info=True,
            )
            # CUDA OOM 处理
            _exc_msg = str(stage_exc).lower()
            if "out of memory" in _exc_msg or "cuda" in _exc_msg:
                if _HAS_TORCH_CUDA:
                    try:
                        torch.cuda.empty_cache()
                        logger.warning(
                            f"[{req.session_id}] CUDA OOM 检测到，"
                            "已执行 torch.cuda.empty_cache()"
                        )
                    except Exception as cuda_cleanup_err:
                        logger.error(
                            f"[{req.session_id}] CUDA 缓存清理失败: "
                            f"{cuda_cleanup_err}"
                        )
                else:
                    logger.warning(
                        f"[{req.session_id}] 检测到 CUDA 异常但 torch 不可用，"
                        "跳过显存清理"
                    )
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            return StandardResponse(
                status=206, message="Partial Content: Stage2 Error",
                session_id=req.session_id,
                data={
                    "final_score": 50,
                    "tags": ["stage2_error_degraded"],
                    "_error": "Inference error (OOM or CUDA fault)",
                },
            )

        # ── Step 5: 阶段三打分 + Bot/Voicemail（CPU 密集，卸载）──
        final_result = await asyncio.to_thread(
            _cpu_stage3_scoring,
            stage2_result,
            state.scorer,
            state.bot_engine,
            state.voicemail_engine,
            topo_metrics,
            nlp_features_extra,
        )

        # ── 成功返回 ──
        if not debug:
            final_result.pop("bot_confidence", None)
            final_result.pop("voicemail_detection", None)
            final_result.pop("topology_metrics", None)
            final_result.pop("language_detection", None)
        return StandardResponse(
            status=200, message="OK",
            session_id=req.session_id, data=final_result,
        )

    except Exception as e:
        logger.error(
            f"处理 session_id: {req.session_id} 发生致命错误: {str(e)}",
            exc_info=True,
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return StandardResponse(
            status=500, message="Internal Server Error",
            session_id=req.session_id,
        )

    finally:
        # 释放过载保护计数器（加锁保证原子性）
        async with state._request_lock:
            state.active_requests -= 1
