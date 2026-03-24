import json
import logging
import re
from typing import List
from models_translation import DialogueTurn, TranslatedItem

logger = logging.getLogger(__name__)


class LLMTranslatorService:
    """封装大语言模型上下文翻译逻辑的引擎层"""

    # 核心策略：系统提示词强制大模型输出 JSON 数组
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

    def build_user_prompt(self, dialogue_turns: List[DialogueTurn]) -> str:
        """将结构化的对话数据转为 JSON 字符串发给大模型"""
        payload = [
            {"id": turn.id, "speaker": turn.speaker, "content": turn.content}
            for turn in dialogue_turns
        ]
        return json.dumps(payload, ensure_ascii=False)

    def call_llm(self, user_prompt: str) -> str:
        """
        [此处为 LLM SDK 接入点]
        实际开发时，将这里的存根代码替换为 OpenAI / Claude / 通义千问 等大模型的真实调用代码。

        接入示例（二选一，取消注释即可）：
        -------------------------------------------------------
        # ① OpenAI / Azure OpenAI
        # from openai import OpenAI
        # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": self.SYSTEM_PROMPT},
        #         {"role": "user",   "content": user_prompt},
        #     ],
        #     temperature=0.2,
        # )
        # return response.choices[0].message.content

        # ② Anthropic Claude
        # import anthropic
        # client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # message = client.messages.create(
        #     model="claude-opus-4-6",
        #     max_tokens=4096,
        #     system=self.SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": user_prompt}],
        # )
        # return message.content[0].text
        -------------------------------------------------------
        """
        logger.info("[LLM] 收到翻译请求，输入 prompt 长度：%d 字符", len(user_prompt))

        # 本地占位符存根（测试联调使用，假装是大模型返回的数据）
        turns: list[dict] = json.loads(user_prompt)
        stub_result = [{"id": t["id"], "content": f"【大模型机翻】{t['content']}"} for t in turns]
        return json.dumps(stub_result, ensure_ascii=False)

    def _strip_markdown_fences(self, raw: str) -> str:
        """
        兜底清除大模型可能抽风输出的 Markdown 代码块标记。
        例如：```json\\n[...]\\n``` → [...]
        """
        # 移除形如 ```json ... ``` 或 ``` ... ``` 的包裹
        cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def translate(self, dialogue_turns: List[DialogueTurn]) -> List[TranslatedItem]:
        """
        主入口：执行全文上下文翻译，返回按 id 映射的翻译结果列表。

        执行流程：
          1. 将完整对话序列化为 user prompt（一次性提交，保留上下文）
          2. 调用 LLM，获取原始字符串输出
          3. 剥离 Markdown 代码块标记
          4. json.loads 解析为 List[dict]
          5. 校验每个元素包含 id 与 content 字段
          6. 按原始输入顺序做 id 映射，返回 TranslatedItem 列表

        核心原则：绝对不在此处使用 for 循环逐句调用 LLM，
        必须整段对话一次性提交以保证翻译上下文连贯性。
        """
        # ----------------------------------------------------------------
        # Step 1：构建整段对话的 user prompt，一次性提交给 LLM
        # ----------------------------------------------------------------
        user_prompt = self.build_user_prompt(dialogue_turns)
        logger.info("发送 %d 条对话至 LLM 进行整体上下文翻译", len(dialogue_turns))

        # ----------------------------------------------------------------
        # Step 2：调用 LLM，获取原始字符串输出
        # ----------------------------------------------------------------
        raw_output = self.call_llm(user_prompt)
        logger.debug("LLM 原始输出（前500字）：%s", raw_output[:500])

        # ----------------------------------------------------------------
        # Step 3：剥离 LLM 可能残留的 Markdown 标记
        # ----------------------------------------------------------------
        cleaned_output = self._strip_markdown_fences(raw_output)

        # ----------------------------------------------------------------
        # Step 4：将清理后的字符串解析为 Python 对象
        # ----------------------------------------------------------------
        try:
            parsed: list = json.loads(cleaned_output)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM 输出无法解析为合法 JSON：{exc}；"
                f"清理后输出片段：{cleaned_output[:200]}"
            ) from exc

        if not isinstance(parsed, list):
            raise ValueError(
                f"LLM 输出应为 JSON 数组，实际得到：{type(parsed).__name__}"
            )

        # ----------------------------------------------------------------
        # Step 5：校验每个元素结构，并构建 id → translated_content 映射
        # ----------------------------------------------------------------
        translation_map: dict[str, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(f"翻译结果元素不是 JSON 对象：{item}")
            if "id" not in item or "content" not in item:
                raise ValueError(
                    f"翻译结果元素缺少必要字段 id 或 content：{item}"
                )
            translation_map[str(item["id"])] = str(item["content"])

        # ----------------------------------------------------------------
        # Step 6：严格按原始输入顺序输出，缺失条目记录 warning 并空串兜底
        # ----------------------------------------------------------------
        result: List[TranslatedItem] = []
        for turn in dialogue_turns:
            translated_text = translation_map.get(turn.id)
            if translated_text is None:
                logger.warning(
                    "LLM 输出中缺少 id=%s 的翻译结果，使用空字符串兜底", turn.id
                )
                translated_text = ""
            result.append(TranslatedItem(id=turn.id, content=translated_text))

        logger.info("翻译完成，共返回 %d 条结果", len(result))
        return result