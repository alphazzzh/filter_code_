"""
scoring_strategies/whitelist.py  ── 超级白名单 & 机器人豁免策略
============================================================

运行白名单豁免主题，并附加"多维洗脑撤销（Revocation）"机制。
"""

from __future__ import annotations

from typing import Any

from models_stage2 import RoleLabel, SpeakerRoleResult, StageTwoResult
from config_topics import TOPIC_REGISTRY, TopicCategory, TopicDefinition, get_topics_by_category
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


def _check_whitelist(
    result: StageTwoResult,
    all_intents: set[str],
    ctx: _ScoringContext,
) -> tuple[bool, float]:
    whitelist_topics = get_topics_by_category(TopicCategory.WHITELIST)
    whitelist_hit_topics = {tid for tid in whitelist_topics if tid in all_intents}

    if not whitelist_hit_topics:
        return False, 1.0

    # 1. 案底感知
    hard_probe_hit = any(
        e["delta"] > 0 and not e["tag"].startswith("official") for e in ctx.events
    )

    # 2. 顺从度感知
    ifeats = result.interaction_features
    compliance_rate = ifeats.compliance_rate if ifeats else 0.0

    # 3. 锚点感知
    potential_victims = [
        sr.speaker_id for sr in result.speaker_roles if sr.role != RoleLabel.AGENT
    ]
    victim_intents = {
        label
        for t in result.dialogue_turns
        if t.speaker_id in potential_victims and not t.is_backchannel
        for label in t.intent_labels
    }
    fatal_compliance_anchors = {"compliance", "action_confirmation", "providing_info", "agreement"}
    has_fatal_anchor = bool(victim_intents & fatal_compliance_anchors)

    # 核心撤销机制
    if hard_probe_hit and compliance_rate >= 0.2 and has_fatal_anchor:
        return False, 1.0

    # 句法拦截
    nlp_feats = result.metadata.get("nlp_features", {})
    if nlp_feats.get("has_imperative_syntax", False):
        return False, 1.0

    min_discount = min(
        whitelist_topics[tid].scoring_rules.whitelist_discount
        for tid in whitelist_hit_topics
    )
    return True, min_discount


class WhitelistStrategy(ScoringRuleStrategy):
    """超级白名单 & 机器人豁免策略。"""

    def __init__(
        self,
        registry: dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._whitelist_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.WHITELIST
        }

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        all_intents: set[str] = ctx.extra.get("all_intents", set())
        is_wl, discount = _check_whitelist(result, all_intents, ctx)

        # 提取案底、顺从率及致命锚点
        hard_probe_hit = any(
            e["delta"] > 0 and not e["tag"].startswith("official") for e in ctx.events
        )
        ifeats = result.interaction_features
        compliance_rate = ifeats.compliance_rate if ifeats else 0.0

        potential_victims = [
            sr.speaker_id for sr in result.speaker_roles if sr.role != RoleLabel.AGENT
        ]
        victim_intents = {
            label
            for t in result.dialogue_turns
            if t.speaker_id in potential_victims and not t.is_backchannel
            for label in t.intent_labels
        }
        fatal_compliance_anchors = {"compliance", "action_confirmation", "providing_info", "agreement"}
        has_fatal_anchor = bool(victim_intents & fatal_compliance_anchors)

        for tid, td in self._whitelist_topics.items():
            if tid in all_intents:
                if not is_wl and hard_probe_hit and compliance_rate >= 0.2 and has_fatal_anchor:
                    hit_anchors = list(victim_intents & fatal_compliance_anchors)
                    ctx.apply(
                        delta=0,
                        tag=f"{td.scoring_rules.standalone_tag}_revoked",
                        reason=(
                            f"曾触发白名单[{tid}]，但检测到高危风险且受害者最终做出实质性妥协"
                            f"(compliance={compliance_rate:.2f}, 锚点={hit_anchors})，豁免被强制撤销！"
                        ),
                    )
                    continue

                if is_wl:
                    ctx.apply(
                        delta=td.scoring_rules.standalone_score,
                        tag=td.scoring_rules.standalone_tag,
                        reason=f"命中安全白名单 [{tid}]，强制降权防误杀",
                    )
