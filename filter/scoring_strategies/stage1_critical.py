"""
scoring_strategies/stage1_critical.py  ── 阶段一硬正则极高危拦截兜底
============================================================

缺陷 1 修复：阶段一硬正则极高危拦截事件（stage_one_critical_hit）
在阶段三入口处直接加分 +40，确保不被后续逻辑稀释。
"""

from __future__ import annotations

from models_stage2 import StageTwoResult
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


class Stage1CriticalStrategy(ScoringRuleStrategy):
    """阶段一硬正则极高危拦截兜底策略。"""

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        s1_meta = result.metadata.get("stage_one", {})
        if s1_meta.get("stage_one_critical_hit"):
            ctx.apply(40, "STAGE1_CRITICAL_FORCE_RECALL", "阶段一硬正则极高危拦截")
