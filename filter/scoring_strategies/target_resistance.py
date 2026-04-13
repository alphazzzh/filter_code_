"""
scoring_strategies/target_resistance.py  ── 受害者阶梯式抵抗降权策略
============================================================

V5.2 阶梯式受害者抵抗降权机制。
V5.3 阶梯惩罚值收拢至 GLOBAL_SCORING_CONFIG，支持运行时配置。

三级阶梯：
  Tier 1 (轻度): 抵抗率 > 0 但 < tier2_rate，仅轻微口头抗拒
  Tier 2 (中度): 抵抗率 >= tier2_rate，明确质疑
  Tier 3 (重度): 抵抗率 >= tier3_rate 或明确识破(dismissal)，诈骗未遂

免疫条件：顺从率 >= compliance_immunity_rate → 抵抗全部无效
"""

from __future__ import annotations

from models_stage2 import RoleLabel, StageTwoResult
from config_topics import GLOBAL_SCORING_CONFIG
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


_SCAM_ATTEMPT_REJECTED_TAG: str = "scam_attempt_rejected"


class TargetResistanceDiscountStrategy(ScoringRuleStrategy):
    """受害者阶梯式抵抗降权策略。"""

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        cfg = GLOBAL_SCORING_CONFIG

        # 全局开关短路
        if not cfg.get("enable_resistance_discount", True):
            return

        # 只有命中高危主题时才执行
        hit_families: set[str] = ctx.extra.get("hit_families", set())
        if not hit_families:
            return

        potential_resisters = [
            sr.speaker_id for sr in result.speaker_roles
            if sr.role != RoleLabel.AGENT
        ]
        if not potential_resisters:
            return

        tier1_penalty = cfg.get("resistance_tier1_penalty", -10)
        tier2_penalty = cfg.get("resistance_tier2_penalty", -20)
        tier3_penalty = cfg.get("resistance_tier3_penalty", -35)
        tier2_rate    = cfg.get("resistance_tier2_rate", 0.15)
        tier3_rate    = cfg.get("resistance_tier3_rate", 0.25)
        immunity_rate = cfg.get("resistance_compliance_immunity_rate", 0.20)

        for target_sid in potential_resisters:
            target_turns = [
                t for t in result.dialogue_turns
                if t.speaker_id == target_sid and not t.is_backchannel
            ]
            if not target_turns:
                continue

            target_intents = {label for t in target_turns for label in t.intent_labels}

            resistance_intents = {"dismissal", "rejection"}
            resistance_count = sum(
                1 for t in target_turns if (set(t.intent_labels) & resistance_intents)
            )
            resistance_rate = resistance_count / len(target_turns)

            compliance_count = sum(
                1 for t in target_turns if "compliance" in t.intent_labels
            )
            compliance_rate = compliance_count / len(target_turns)

            # 免疫：高度顺从时，抵抗全部无效
            if compliance_rate >= immunity_rate:
                continue

            if "dismissal" in target_intents or resistance_rate >= tier3_rate:
                ctx.apply(
                    delta=tier3_penalty,
                    tag="fraud_failed_target_resisted",
                    reason=(
                        f"检测到高危风险，但疑似受害者({target_sid})明确识破并脱战"
                        f"（抵抗率={resistance_rate:.2f}或触发绝对识破），案件降级为未遂线索"
                    ),
                )
                ctx.tags.add(_SCAM_ATTEMPT_REJECTED_TAG)
                return
            elif resistance_rate >= tier2_rate:
                ctx.apply(
                    delta=tier2_penalty,
                    tag="target_strong_resistance",
                    reason=f"疑似受害者({target_sid})表现出中度抵触与质疑（抵抗率={resistance_rate:.2f}）",
                )
                return
            elif resistance_rate > 0.0:
                ctx.apply(
                    delta=tier1_penalty,
                    tag="target_mild_resistance",
                    reason=f"疑似受害者({target_sid})表现出轻微抗拒（抵抗率={resistance_rate:.2f}）",
                )
                return
