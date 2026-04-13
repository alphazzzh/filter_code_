"""
scoring_strategies/noise_suppression.py  ── 废料降权策略（含 OOD 兜底）
============================================================

核心逻辑：高危意图绝对优先。未命中风险时，才进行废料降权。
  - 语义层降权（已知类别的废话：外卖/闲聊/打错电话）
  - 物理层兜底（OOD 未知领域的废话：基于拓扑结构和实体密度）
"""

from __future__ import annotations

from typing import Any

from models_stage2 import StageTwoResult
from config_topics import (
    TOPIC_REGISTRY,
    TopicCategory,
    TopicDefinition,
    OOD_FALLBACK_REGISTRY,
)
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


class NoiseSuppressionStrategy(ScoringRuleStrategy):
    """废料降权策略：语义层降权 + OOD 物理兜底。"""

    def __init__(
        self,
        registry: dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._noise_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.LOW_VALUE_NOISE
        }

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        # 高危意图绝对优先：命中风险时不执行废料降权
        hit_families: set[str] = ctx.extra.get("hit_families", set())
        if hit_families:
            return

        all_intents: set[str] = ctx.extra.get("all_intents", set())
        nlp_feats: dict[str, Any] = ctx.extra.get("nlp_feats", {})

        # ── 语义层降权 ──
        self._run_noise_topics(ctx, all_intents, nlp_feats)

        # ── OOD 物理兜底 ──
        self._run_ood_fallback(ctx, result, nlp_feats)

    def _run_noise_topics(
        self,
        ctx: _ScoringContext,
        all_intents: set[str],
        nlp_feats: dict[str, Any],
    ) -> None:
        for topic_id, topic_def in self._noise_topics.items():
            if topic_id in all_intents:
                # 抑制物流对毒品的误报
                if topic_id == "corporate_logistics" and not nlp_feats.get("has_drug_quantity"):
                    surviving_events = []
                    for e in ctx.events:
                        if e["tag"].startswith("has_drug_") or e["tag"].startswith("drug_quantity_"):
                            ctx.score -= e["delta"]  # 回退分数
                        else:
                            surviving_events.append(e)
                    ctx.events = surviving_events

                # Step A：单项基础扣分
                if topic_def.scoring_rules.standalone_score != 0:
                    ctx.apply(
                        delta=topic_def.scoring_rules.standalone_score,
                        tag=topic_def.scoring_rules.standalone_tag,
                        reason=f"命中低价值噪声主题 [{topic_id}]，降权处理",
                    )

                # Step B：矩阵附加扣分
                for combo in topic_def.scoring_rules.matrix_combinations:
                    hard_feat_value = bool(nlp_feats.get(combo.syntax_feature, False))
                    triggered = (
                        (combo.requires_absence and not hard_feat_value)
                        or (not combo.requires_absence and hard_feat_value)
                    )
                    if triggered:
                        ctx.apply(
                            delta=combo.bonus_score,
                            tag=combo.bonus_tag,
                            reason=(
                                f"负向矩阵压制 [{'!' if combo.requires_absence else ''}"
                                f"{combo.syntax_feature}] × [{topic_id}] → {combo.bonus_score:+d}"
                            ),
                        )

    @staticmethod
    def _run_ood_fallback(
        ctx: _ScoringContext,
        stage2_result: StageTwoResult,
        nlp_feats: dict[str, Any],
    ) -> None:
        """OOD 物理废料兜底执行器。"""
        ifeats = stage2_result.interaction_features
        valid_turns = sum(
            1 for t in stage2_result.dialogue_turns if not t.is_backchannel
        )
        total_words = sum(
            t.word_count for t in stage2_result.dialogue_turns if not t.is_backchannel
        )

        eval_context: dict[str, Any] = {
            "valid_turn_count": valid_turns,
            "compliance_rate":  ifeats.compliance_rate,
            "total_words":      total_words,
            "ping_pong_rate":   ifeats.negotiation_ping_pong_rate,
            **nlp_feats,
        }

        for rule in OOD_FALLBACK_REGISTRY:
            if rule.condition(eval_context):
                ctx.apply(
                    delta=rule.delta,
                    tag=rule.tag,
                    reason=rule.reason,
                )
