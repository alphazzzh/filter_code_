import json
import logging
import re
from typing import List
from openai import AsyncOpenAI  
from models_translation import DialogueTurn, TranslatedItem

logger = logging.getLogger(__name__)

class LLMTranslatorService:
    SYSTEM_PROMPT = """\
你是一位专业的多语言翻译助手，精通将各类语言翻译成简体中文。
任务规则（严格遵守，不得违反）：
1. 你将收到一段 JSON 数组，每个元素包含 "id"（原始编号）、"speaker"（说话人）、"content"（原文）。
2. 请结合完整对话的上下文，理解代词指代与对话逻辑，再进行翻译。
3. 输出格式：仅输出一个合法的 JSON 数组，每个元素严格为 {"id": "<原始id>", "content": "<翻译后中文>"}。
4. 禁止输出任何 Markdown 代码块标记（如 ```json）、解释文字或多余空白。
5. 禁止改变 id 字段的值；id 必须与输入完全一致。
输出示例（仅供格式参考）：
[{"id":"content_001","content":"翻译结果一"},{"id":"content_002","content":"翻译结果二"}]
"""

    def __init__(self):
        # 在类初始化时全局创建一次客户端，复用底层连接池
        self.client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",  
            api_key="EMPTY"                       
        )

    def build_user_prompt(self, dialogue_turns: List[DialogueTurn]) -> str:
        payload = [{"id": turn.id, "speaker": turn.speaker, "content": turn.content} for turn in dialogue_turns]
        return json.dumps(payload, ensure_ascii=False)

    async def call_llm(self, user_prompt: str) -> str:
        """ 👈 修改为 async 函数 """
        logger.info("[LLM] 发起异步翻译请求，prompt 长度：%d", len(user_prompt))
        
        # 使用 await 等待异步网络 I/O，绝不阻塞主线程
        response = await self.client.chat.completions.create(
            model="你的本地模型名称",               
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _extract_json_array(self, raw: str) -> str:
        """
        强力正则，无视大模型的废话，直接提取第一对 [ ] 及其内部内容
        """
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            return match.group(0)
        # 如果正则没找到，原样返回，交由 json.loads 报错处理
        return raw.strip()

    async def translate(self, dialogue_turns: List[DialogueTurn]) -> List[TranslatedItem]:
        """ 👈 修改为 async 函数 """
        user_prompt = self.build_user_prompt(dialogue_turns)
        
        # 1. 异步等待 LLM 响应
        raw_output = await self.call_llm(user_prompt)
        
        # 2. 强力提取 JSON 数组
        cleaned_output = self._extract_json_array(raw_output)

        try:
            parsed: list = json.loads(cleaned_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM 输出无法解析为合法 JSON：{exc}；片段：{cleaned_output[:200]}") from exc

        if not isinstance(parsed, list):
            raise ValueError(f"LLM 输出应为 JSON 数组，实际得到：{type(parsed).__name__}")

        translation_map = {str(item.get("id", "")): str(item.get("content", "")) for item in parsed if isinstance(item, dict)}

        result: List[TranslatedItem] = []
        for turn in dialogue_turns:
            translated_text = translation_map.get(turn.id)
            if translated_text is None:
                logger.warning("缺少 id=%s 的翻译结果，使用空字符串兜底", turn.id)
                translated_text = ""
            result.append(TranslatedItem(id=turn.id, content=translated_text))

        return result