"""
scoring_strategies/bot_intent_fusion.py  ── AI 机器人 × 伪装意图融合策略
============================================================

AI 外呼诈骗法则（规则 4）：
触发条件（同时满足）：
  1. 机器人特征明显（BotConfidenceEngine 判定 BOT）
  2. 命中高危伪装意图（e_commerce_cs / authority_entity / fraud_object / fraud_jargon）

执行动作：
  - delta = +15（可配置）融合惩罚加分
  - 追加高危融合标签
"""

from __future__ import annotations

from typing import Any

from models_stage2 import StageTwoResult
from config_topics import GLOBAL_SCORING_CONFIG
from engines.bot_confidence import BotConfidenceEngine
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


class BotIntentFusionStrategy(ScoringRuleStrategy):
    """AI 机器人 × 伪装意图融合策略。"""

    # 机器人 × 意图融合触发的高危主题集合
    _BOT_FUSION_HIGH_RISK_TOPICS: frozenset[str] = frozenset({
        "e_commerce_cs",
        "authority_entity",
        "fraud_object",
        "fraud_jargon",
    })

    # 融合标签映射
    _BOT_FUSION_TAG_MAP: dict[str, str] = {
        "e_commerce_cs":    "ai_scam_bot_ecommerce",
        "authority_entity": "ai_scam_bot_authority",
        "fraud_object":     "ai_scam_bot_fraud_object",
        "fraud_jargon":     "ai_scam_bot_fraud_jargon",
    }

    def __init__(self) -> None:
        self._bot_engine = BotConfidenceEngine()

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        all_intents: set[str] = ctx.extra.get("all_intents", set())

        # 检查是否命中高危伪装意图
        hit_topics = all_intents & self._BOT_FUSION_HIGH_RISK_TOPICS
        if not hit_topics:
            return

        # 调用高级引擎获取权威结论
        nlp_extra = result.metadata.get("nlp_features_extra", {})
        fwr = nlp_extra.get("filler_word_rate", 1.0)
        bot_eval = self._bot_engine.evaluate(result, filler_word_rate=fwr)
        is_bot_verdict = (bot_eval["bot_label"].value == "bot")

        if not is_bot_verdict:
            return

        # 触发融合
        fusion_delta = GLOBAL_SCORING_CONFIG.get("bot_fusion_penalty", 15)
        for topic_id in hit_topics:
            fusion_tag = self._BOT_FUSION_TAG_MAP.get(topic_id, "ai_scam_bot_generic")
            ctx.apply(
                delta=fusion_delta,
                tag=fusion_tag,
                reason=(
                    f"命中 AI 机器人批量外呼与高危意图 [{topic_id}] 融合，"
                    "判定为机器诈骗试探"
                ),
            )
