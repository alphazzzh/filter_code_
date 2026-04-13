"""
scoring_strategies/exemption.py  ── 豁免减分策略
============================================================

遍历 EXEMPTION 主题（如 dismissal/rejection），命中时扣除固定分值。
"""

from __future__ import annotations

from models_stage2 import StageTwoResult
from config_topics import TOPIC_REGISTRY, TopicCategory, TopicDefinition
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


class ExemptionStrategy(ScoringRuleStrategy):
    """豁免减分策略。"""

    def __init__(
        self,
        registry: dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._exemption_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.EXEMPTION
        }

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        all_intents: set[str] = ctx.extra.get("all_intents", set())

        for tid, td in self._exemption_topics.items():
            if tid in all_intents:
                ctx.apply(
                    delta=td.scoring_rules.standalone_score,
                    tag=td.scoring_rules.standalone_tag,
                    reason=f"命中豁免信号主题 [{tid}]",
                )
