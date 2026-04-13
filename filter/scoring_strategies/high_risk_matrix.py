"""
scoring_strategies/high_risk_matrix.py  ── 高危主题动态矩阵加分策略
============================================================

遍历所有 HIGH_RISK 主题，对每个命中的主题执行：
  Step A：单项基础分（无硬特征强化，低置信度）
  Step B：遍历 matrix_combinations，查找共现矩阵命中
  Step C：记录命中的 topic_family（供跨族复合加分使用）

V5.3 变更：
  - 消费 confidence_discount：rule_based 后端时非独立探针矩阵加分衰减
  - 矩阵命中计数梯度：多矩阵命中时首矩阵满分，后续递减

上下文通信：
  - 写入 ctx.extra["hit_families"] : set[str]  → 供下游策略使用
"""

from __future__ import annotations

from typing import Any

from models_stage2 import (
    DialogueTurn,
    InteractionFeatures,
    RoleLabel,
    SpeakerRoleResult,
    StageTwoResult,
)
from config_topics import TOPIC_REGISTRY, TopicCategory, TopicDefinition
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


def _collect_all_intents(turns: list[DialogueTurn]) -> set[str]:
    return {label for t in turns for label in t.intent_labels}


def _find_role(roles: list[SpeakerRoleResult], target: RoleLabel) -> str | None:
    return next((r.speaker_id for r in roles if r.role == target), None)


class HighRiskMatrixStrategy(ScoringRuleStrategy):
    """
    高危主题动态矩阵加分策略。

    初始化时从 registry 中过滤出 HIGH_RISK 主题，避免每次 apply 重复过滤。
    """

    def __init__(
        self,
        registry: dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._high_risk_topics = {
            tid: td
            for tid, td in registry.items()
            if td.category == TopicCategory.HIGH_RISK
        }

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        all_intents: set[str] = _collect_all_intents(result.dialogue_turns)

        # ── 提取骗子（Agent）的句法特征 ──
        nlp_feats: dict[str, Any] = result.metadata.get("nlp_features", {})
        speaker_nlp_feats: dict[str, dict[str, Any]] = result.metadata.get(
            "speaker_nlp_features", {}
        )

        agent_sid = _find_role(result.speaker_roles, RoleLabel.AGENT)
        agent_nlp_feats = dict(nlp_feats)

        if agent_sid and agent_sid in speaker_nlp_feats:
            # 智能合并：专属字典里的 False 不覆盖全局的 True
            for k, v in speaker_nlp_feats[agent_sid].items():
                if v:
                    agent_nlp_feats[k] = v
        elif not agent_sid and speaker_nlp_feats:
            # 平权聊天：动态嫌疑人推举
            suspect_sid = None
            max_risk_flags = -1
            risk_keys = {"has_imperative_syntax", "has_coercive_threat", "has_guide_behavior"}

            for sid, feats in speaker_nlp_feats.items():
                risk_count = sum(1 for k in risk_keys if feats.get(k))
                if risk_count > max_risk_flags:
                    max_risk_flags = risk_count
                    suspect_sid = sid

            if suspect_sid and max_risk_flags > 0:
                for k, v in speaker_nlp_feats[suspect_sid].items():
                    if v:
                        agent_nlp_feats[k] = v

        # ── 执行矩阵扫描 ──
        hit_families: set[str] = set()
        nlp_backend: str = agent_nlp_feats.get("nlp_backend", "rule_based")

        for topic_id, topic_def in self._high_risk_topics.items():
            has_soft_intent = topic_id in all_intents
            topic_hit = False
            matrix_hit = False
            combo_hit_count = 0

            # Step B：矩阵组合扫描
            for combo in topic_def.scoring_rules.matrix_combinations:
                if not combo.is_independent and not has_soft_intent:
                    continue

                hard_feat_value = bool(agent_nlp_feats.get(combo.syntax_feature, False))

                triggered = False
                if combo.requires_absence and not hard_feat_value:
                    triggered = True
                elif not combo.requires_absence and hard_feat_value:
                    triggered = True

                if triggered:
                    actual_delta = combo.bonus_score
                    combo_hit_count += 1
                    if combo_hit_count > 1:
                        actual_delta = int(
                            actual_delta * max(0.4, 1.0 - 0.2 * (combo_hit_count - 1))
                        )
                    if not combo.is_independent and nlp_backend == "rule_based":
                        actual_delta = int(
                            actual_delta * topic_def.scoring_rules.confidence_discount
                        )

                    ctx.apply(
                        delta=actual_delta,
                        tag=combo.bonus_tag,
                        reason=(
                            f"矩阵命中 [{'!' if combo.requires_absence else ''}{combo.syntax_feature}] "
                            f"{'(独立触发)' if combo.is_independent else '× [' + topic_id + ']'} "
                            f"→ {actual_delta:+d}"
                            f"{' [discount=' + str(topic_def.scoring_rules.confidence_discount) + ']' if not combo.is_independent and nlp_backend == 'rule_based' else ''}"
                            f"{' [梯度×' + str(max(0.4, 1.0 - 0.2 * (combo_hit_count - 1)))[:4] + ']' if combo_hit_count > 1 else ''}"
                        ),
                    )
                    matrix_hit = True
                    topic_hit = True

            # Step A：单项基础分
            if not matrix_hit and has_soft_intent and topic_def.scoring_rules.standalone_score != 0:
                ctx.apply(
                    delta=topic_def.scoring_rules.standalone_score,
                    tag=topic_def.scoring_rules.standalone_tag,
                    reason=f"单项意图命中 [{topic_id}]（无硬特征强化）",
                )
                topic_hit = True

            if topic_hit:
                hit_families.add(topic_def.topic_family)

        # 写入上下文，供下游策略读取
        ctx.extra["hit_families"] = hit_families
        ctx.extra["all_intents"] = all_intents
        ctx.extra["nlp_feats"] = nlp_feats
