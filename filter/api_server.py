# api_server.py
# ============================================================
# V5.0 ASR 情报风控引擎 —— 标准化 API 接入层
# ============================================================

import json
import logging
from typing import Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse

# 导入内部引擎
from models import ASRRecord, ConnectionStatus, BotLabel
from stage_one_filter import StageOneFilter
from stage_two_pipeline import StageTwoPipeline
from stage_three_scorer import IntelligenceScorer, BotConfidenceEngine, AdvancedVoicemailDetector
from topology_engine import TopologyEngine

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化引擎组件 (以单例模式加载模型，避免重复开销)
# 注意：实际生产中可通过 lifespan event 或依赖注入处理
filter_engine = StageOneFilter()
pipeline_engine = StageTwoPipeline()
scorer_engine = IntelligenceScorer()
bot_engine = BotConfidenceEngine()
voicemail_engine = AdvancedVoicemailDetector()
topology_engine = TopologyEngine()

app = FastAPI(title="V5.0 ASR Risk Control Engine", version="5.0.0")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 严格定义入参契约 (Input Schema)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TurnContent(BaseModel):
    """单轮对话结构定义"""
    id: str
    speaker: Union[int, str]
    content: str

class DataPayload(BaseModel):
    """
    文档规范的二级数据对象 (Level 2)
    兼容文档中要求的 language, start_time 等未使用字段
    """
    session_id: Optional[str] = None
    # 核心：兼容纯 JSON 数组，或者 String 化的 JSON 数组
    content: Union[str, List[TurnContent]] 
    language: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[Union[int, float]] = None

    @validator('content')
    def parse_content_if_string(cls, v):
        """
        鲁棒性设计：如果上游老老实实传了 list[dict]，直接通过；
        如果上游像文档示例那样把整个数组转成了 String，这里自动反序列化！
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                # 校验解析后是否为列表
                if not isinstance(parsed, list):
                    raise ValueError("Stringified content must resolve to a JSON array.")
                return parsed
            except json.JSONDecodeError:
                raise ValueError("content field is a string but NOT a valid JSON.")
        return v

class AnalyzeRequest(BaseModel):
    """
    文档规范的一级请求对象 (Level 1)
    """
    session_id: str
    data: DataPayload
    
    # 我们刚刚加的动态搜索特性（放在第一层作为可选扩展字段）
    dynamic_topic: Optional[str] = Field(
        default=None, 
        max_length=50, 
        description="零样本动态检索的业务主题（如：国际地缘政治、新能源汽车）"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 严格定义出参契约 (Output Schema)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StandardResponse(BaseModel):
    """统一规范的返回体"""
    status: int                       # 状态码，成功为 200
    message: str                      # 状态描述，成功为 "OK"
    session_id: str                   # 透传的会话 ID
    data: Optional[dict[str, Any]] = None # 核心业务产出内容


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 核心路由处理 (更新返回体构造)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/api/v1/analyze", response_model=StandardResponse)
async def analyze_conversation(req: AnalyzeRequest):
    """
    主风控分析接口：接收标准 ASR 话单，执行三阶段分析引擎。
    """
    logger.info(f"Received request for session_id: {req.session_id}")
    
    try:
        # ── Step 1: 解析与映射 ASRRecord ──
        raw_turns = req.data.content
        records: list[ASRRecord] = []
        
        for turn in raw_turns:
            turn_dict = turn if isinstance(turn, dict) else turn.dict()
            record = ASRRecord(
                record_id  = str(turn_dict.get("id", "")),
                speaker_id = str(turn_dict.get("speaker", "")),
                raw_text   = str(turn_dict.get("content", ""))
            )
            records.append(record)
            
        if not records:
            return StandardResponse(
                status=400,              # 👈 修改为 status
                message="Content array is empty", # 👈 修改为 message
                session_id=req.session_id
            )

        # ── Step 2: 阶段一 (算力层物理过滤) ──
        s1_records = filter_engine.process_batch(records)
        
        valid_records = [
            r for r in s1_records 
            if r.connection_status != ConnectionStatus.UNCONNECTED
        ]
        
        if not valid_records:
            logger.info(f"[{req.session_id}] 阶段一触发物理拦截，判定为极低价值废料。")
            return StandardResponse(
                status=200,              # 👈 成功状态
                message="OK",            # 👈 成功信息
                session_id=req.session_id,
                data={
                    "final_score": 5, 
                    "tags": ["stage_one_physical_interception"],
                    "reason": "命中阶段一(信息匮乏/机器人/未接通)物理拦截逻辑"
                }
            )

        # ── Step 3: 阶段二 (软硬双轨特征提取 & 动态搜索) ──
        extra_meta = {}
        if req.dynamic_topic:
            extra_meta["dynamic_topic"] = req.dynamic_topic

        try:
            stage2_result = pipeline_engine.process_conversation(
                conversation_id=req.session_id,
                records=valid_records,
                extra_metadata=extra_meta
            )

            # ── Step 4: 阶段三 (打分器裁决) ──
            final_result = scorer_engine.evaluate(stage2_result)

            # ── Step 4.5: 多维置信度辅助判定 ──
            # 计算 TopologyMetrics 供新引擎使用
            topo_metrics = topology_engine.compute_metrics(stage2_result.dialogue_turns)
            bot_result = bot_engine.evaluate(stage2_result, filler_word_rate=topo_metrics.filler_word_rate)
            voicemail_result = voicemail_engine.evaluate(stage2_result, is_decoupled=topo_metrics.is_decoupled)

            # 将辅助判定结果注入 final_result
            final_result["bot_confidence"] = {
                "bot_score": bot_result["bot_score"],
                "bot_label": bot_result["bot_label"].value,
                "veto_reason": bot_result["veto_reason"],
                "details": bot_result["details"],
            }
            final_result["voicemail_detection"] = {
                "voicemail_score": voicemail_result["voicemail_score"],
                "is_voicemail": voicemail_result["is_voicemail"],
                "veto_reason": voicemail_result["veto_reason"],
                "details": voicemail_result["details"],
            }
            final_result["topology_metrics"] = {
                "filler_word_rate": topo_metrics.filler_word_rate,
                "max_sentence_length": topo_metrics.max_sentence_length,
                "avg_sentence_length": topo_metrics.avg_sentence_length,
                "is_decoupled": topo_metrics.is_decoupled,
            }

            # ── Step 5: 组装标准出参 ──
            return StandardResponse(
                status=200,                  # 👈 成功状态
                message="OK",                # 👈 成功信息
                session_id=req.session_id,
                data=final_result            # 👈 核心产出结果
            )

        except Exception as stage_exc:
            # 【隐患 4 修复】阶段二/三模型推理阶段崩溃，降级返回 206 Partial Content
            # 上游监控系统可通过 HTTP 206 发现 GPU 显存爆了 / BGE 模型加载失败等问题
            logger.error(
                f"[Stage2/3] session_id={req.session_id} 模型推理阶段失败，降级输出: {stage_exc}",
                exc_info=True,
            )
            # 构建 fallback 最小化结果（仅基于阶段一特征）
            fallback_result = {
                "conversation_id": req.session_id,
                "final_score": 5,
                "tags": ["stage2_3_degraded"],
                "track_type": "n/a",
                "roles": {},
                "interaction_summary": {},
                "score_breakdown": [{"delta": -45, "reason": "阶段二/三推理失败，降级为阶段一最小化输出"}],
                "_degradation_error": f"Stage2/3 Process Error: {str(stage_exc)}",
            }
            return JSONResponse(
                status_code=206,
                content={
                    "status": 206,
                    "message": "Partial Content: Stage2/3 inference failed, degraded to Stage1 output",
                    "session_id": req.session_id,
                    "data": fallback_result,
                }
            )

    except Exception as e:
        logger.error(f"处理 session_id: {req.session_id} 发生致命错误: {str(e)}", exc_info=True)
        # 全局异常兜底
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,           # 👈 异常状态
                "message": f"Internal Server Error: {str(e)}", # 👈 异常信息
                "session_id": req.session_id,
                "data": None
            }
        )