import asyncio
import logging
import re
from typing import TypedDict, List, Dict, Any, Optional

# 导入你翻译微服务下的核心组件
from models_translation import DialogueTurn
from translator_engine import LLMTranslatorService

logger = logging.getLogger(__name__)

# ==========================================
# 0. 智能解析器 (复用 api_translation.py 中的逻辑)
# ==========================================
def parse_transcript_to_turns(raw_text: str, session_id: str) -> list[DialogueTurn]:
    """
    智能翻译单元格解析器（严格依赖 \n 换行符）。
    """
    records = []
    lines = raw_text.strip().splitlines()
    pattern = re.compile(r"^(.+?)说[：:,，；;]\s*(.*)$")
    
    current_speaker = None
    current_text_blocks = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = pattern.match(line)
        if match:
            if current_speaker is not None and current_text_blocks:
                joined_text = " ".join(current_text_blocks).strip()
                if joined_text:
                    records.append(DialogueTurn(
                        id=f"{session_id}_{len(records):04d}",
                        speaker=current_speaker,
                        content=joined_text
                    ))
            current_speaker = match.group(1).strip()
            current_text_blocks = [match.group(2).strip()]
        else:
            if current_speaker is not None:
                current_text_blocks.append(line)
            else:
                current_speaker = "Unknown"
                current_text_blocks = [line]
                
    if current_speaker is not None and current_text_blocks:
        joined_text = " ".join(current_text_blocks).strip()
        if joined_text:
            records.append(DialogueTurn(
                id=f"{session_id}_{len(records):04d}",
                speaker=current_speaker,
                content=joined_text
            ))
            
    return records


# ==========================================
# 1. 定义与下游节点的通信契约 (State Schema)
# ==========================================
class TranslationState(TypedDict):
    """
    【数据契约】翻译节点的输入与输出字段。
    """
    # ── 输入字段 ──
    session_id: str
    raw_content: str               # 格式如 "A说:...\nB说:..." 的纯文本
    target_language: str           # 目标语言 (例如 "en", "zh")
    
    # ── 输出字段 (翻译节点写回) ──
    translated_turns: Optional[List[Dict[str, Any]]] # 翻译后的结构化数组
    translation_error: Optional[str]                 # 错误信息


# ==========================================
# 2. 核心黑盒组件：翻译节点
# ==========================================
class TranslationNode:
    """
    LLM 翻译引擎的 LangGraph 节点封装。
    """
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1", 
        model_name: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    ):
        logger.info("🚀 [Translation Node] 正在初始化 LLM 翻译服务客户端...")
        # 预加载翻译服务引擎 (内部封装了异步的 httpx 客户端或 OpenAI 客户端)
        self.translator = LLMTranslatorService(base_url=base_url, model_name=model_name)
        logger.info("✅ [Translation Node] 翻译引擎客户端就绪！")

    async def process(self, state: dict) -> dict:
        """
        核心 Node 调度函数：LangGraph 执行到此节点时会调用该方法。
        """
        session_id = state.get("session_id", "unknown_session")
        raw_content = state.get("raw_content", "")
        # 如果全局 State 没传 target_language，默认翻译成英文
        target_language = state.get("target_language", "en") 

        if not raw_content:
            logger.warning(f"[{session_id}] 翻译节点收到空文本")
            return {"translation_error": "No content provided"}

        try:
            # ── 1. 解析文本 ──
            dialogue_turns = parse_transcript_to_turns(raw_content, session_id)
            if not dialogue_turns:
                return {"translation_error": "Parse failed", "translated_turns": []}

            # ── 2. 调用翻译引擎 (已经是异步，直接 await) ──
            # translator.translate 内部负责拼接 prompt 并请求大模型
            translated_items = await self.translator.translate(dialogue_turns)

            # ── 3. 将 Pydantic 对象转为原生字典 ──
            # 在 LangGraph 的 State 中流转时，最好使用原生的 dict，防止下游序列化报错
            translated_dicts = [
                {
                    "id": item.id,
                    "speaker": item.speaker,
                    "content": item.content  # 这里已经是翻译后的文本了
                }
                for item in translated_items
            ]

            logger.debug(f"[{session_id}] 翻译完成，共处理 {len(translated_dicts)} 个轮次")

            # 👇 核心：返回要更新到 Graph State 中的增量字典
            return {
                "translated_turns": translated_dicts,
                "translation_error": None
            }

        except Exception as e:
            logger.error(f"[{session_id}] 翻译节点处理异常: {e}", exc_info=True)
            return {
                "translated_turns": [],
                "translation_error": str(e)
            }