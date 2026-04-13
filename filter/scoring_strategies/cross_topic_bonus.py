"""
scoring_strategies/cross_topic_bonus.py  ── 跨族复合加分策略
============================================================

V5.2 跨族复合加分：当命中 >= 2 个不同的主题族时额外加分，同族不叠加。
排除默认的 general 族，要求真实的跨领域作案才触发。
"""

from __future__ import annotations

from models_stage2 import StageTwoResult
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext

# 跨类别矩阵复合加分
_CROSS_TOPIC_BONUS:       int = 20
_CROSS_TOPIC_BONUS_TAG:   str = "multi_topic_compound"
_CROSS_TOPIC_MIN_HITS:    int = 2


class CrossTopicBonusStrategy(ScoringRuleStrategy):
    """跨族复合加分策略。"""

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        hit_families: set[str] = ctx.extra.get("hit_families", set())
        valid_families = hit_families - {"general"}

        if len(valid_families) >= _CROSS_TOPIC_MIN_HITS:
            ctx.apply(
                delta=_CROSS_TOPIC_BONUS,
                tag=_CROSS_TOPIC_BONUS_TAG,
                reason=(
                    f"跨族群复合命中 {len(valid_families)} 个高危作案领域 "
                    f"({','.join(sorted(valid_families))})，升级综合危险等级"
                ),
            )
