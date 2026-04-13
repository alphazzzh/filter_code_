"""
engines/bot_confidence.py  ── 多维置信度机器人检测引擎
============================================================

从 stage_three_scorer.py 中剥离的独立引擎。
判定逻辑、评分权重、一票否决规则均保持不变。

设计原则：
  - 零内部状态，线程安全
  - 输入 StageTwoResult，输出结构化判定字典
  - 可被 BotIntentFusionStrategy / filter_node 等多方复用
"""

from __future__ import annotations

import re
from typing import Any

from models_stage2 import StageTwoResult
from config_topics import PROFANITY_REGISTRY
from models import BotLabel


class BotConfidenceEngine:
    """
    多维置信度机器人检测引擎。

    评分规则（基础分 0，叠加计分）：
      - 命中 csr_bot_whitelist 软意图         → +40
      - filler_word_rate < 0.005（极度流畅）   → +30
      - ping_pong_rate 极度规律且无并发        → +30

    一票否决（得分归零，强制标记为 HUMAN）：
      - 触发 PROFANITY_REGISTRY（脏话/攻击性词汇）
      - 复杂的反问逻辑（受害者主动反问 + 识破意图）

    最终得分 >= 70 判定为 BOT。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _BOT_THRESHOLD: int = 70

    # ── 加分权重 ──────────────────────────────────────────
    _SCORE_CSR_WHITELIST:  int = 40
    _SCORE_FLUENCY:        int = 30
    _FLUENCY_THRESHOLD:    float = 0.005
    _SCORE_PING_PONG:      int = 30

    def evaluate(
        self,
        stage2_result: StageTwoResult,
        filler_word_rate: float = 0.0,
    ) -> dict[str, Any]:
        """
        执行多维置信度机器人判定。

        Parameters
        ----------
        stage2_result    : 阶段二输出
        filler_word_rate : 语气词占比（由 TopologyEngine 提供）

        Returns
        -------
        dict[str, Any]
            bot_score  : int           置信度得分 [0, 100]
            bot_label  : BotLabel      BOT / HUMAN
            veto_reason: str | None    一票否决原因（若触发）
            details    : list[str]     评分细节
        """
        score: int   = 0
        details: list[str] = []
        all_intents: set[str] = {label for t in stage2_result.dialogue_turns for label in t.intent_labels}
        full_text: str = " ".join(
            t.merged_text for t in stage2_result.dialogue_turns
            if not t.is_backchannel
        )

        # ── 一票否决检查（优先级最高）─────────────────────

        # 否决 1：脏话/攻击性词汇
        veto_word: str | None = self._check_profanity(full_text)
        if veto_word:
            return {
                "bot_score": 0,
                "bot_label": BotLabel.HUMAN,
                "veto_reason": f"命中脏话/攻击性词汇：「{veto_word}」，强制标记为 HUMAN",
                "details": ["PROFANITY_VETO"],
            }

        # 否决 2：复杂反问逻辑（受害者识破意图的反问）
        if self._check_complex_rhetorical(stage2_result, all_intents):
            return {
                "bot_score": 0,
                "bot_label": BotLabel.HUMAN,
                "veto_reason": "检测到复杂反问逻辑（受害者主动反问+识破意图），强制标记为 HUMAN",
                "details": ["COMPLEX_RETORT_VETO"],
            }

        # ── 加分项 ────────────────────────────────────────
        non_bc_turns = [t for t in stage2_result.dialogue_turns if not t.is_backchannel]
        valid_turns_count = len(non_bc_turns)
        total_words = sum(t.word_count for t in non_bc_turns)

        # 加分 1：命中 csr_bot_whitelist
        if "csr_bot_whitelist" in all_intents:
            score += self._SCORE_CSR_WHITELIST
            details.append(f"命中 csr_bot_whitelist → +{self._SCORE_CSR_WHITELIST}")

        # 加分 2：极度流畅（语气词极少）
        if filler_word_rate < self._FLUENCY_THRESHOLD and total_words >= 30:
            score += self._SCORE_FLUENCY
            details.append(
                f"filler_word_rate={filler_word_rate:.4f} < {self._FLUENCY_THRESHOLD} (字数={total_words}) → +{self._SCORE_FLUENCY}"
            )

        # 加分 3：ping_pong_rate 极度规律
        ppr = stage2_result.interaction_features.negotiation_ping_pong_rate
        if ppr < 0.05 and valid_turns_count >= 4:
            score += self._SCORE_PING_PONG
            details.append(
                f"ping_pong_rate={ppr:.4f} < 0.05 (轮次={valid_turns_count}) → +{self._SCORE_PING_PONG}"
            )

        # ── 最终判定 ──────────────────────────────────────
        bot_label = BotLabel.BOT if score >= self._BOT_THRESHOLD else BotLabel.HUMAN
        details.append(f"最终得分={score}，阈值={self._BOT_THRESHOLD}，判定={bot_label.value}")

        return {
            "bot_score":  score,
            "bot_label":  bot_label,
            "veto_reason": None,
            "details":    details,
        }

    @staticmethod
    def _check_profanity(text: str) -> str | None:
        """检查文本是否命中 PROFANITY_REGISTRY，返回第一个命中的词或 None。"""
        text_lower = text.lower()
        for word in PROFANITY_REGISTRY:
            if word.lower() in text_lower:
                return word
        return None

    @staticmethod
    def _check_complex_rhetorical(
        stage2_result: StageTwoResult,
        all_intents: set[str],
    ) -> bool:
        """
        检测复杂反问逻辑。

        判定条件：
        1. 对话中存在 dismissal（识破）意图
        2. 且存在 interrogation（反问）意图
        3. 且 compliance_rate 极低（< 0.10）
        三者同时满足 → 受害者在主动反击，判定为真人。
        """
        has_dismissal = "dismissal" in all_intents
        has_interrogation = "interrogation" in all_intents
        low_compliance = stage2_result.interaction_features.compliance_rate < 0.10

        return has_dismissal and has_interrogation and low_compliance
