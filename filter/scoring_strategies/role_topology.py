"""
scoring_strategies/role_topology.py  ── 角色拓扑附加分策略
============================================================

角色拓扑结构性风险检测，与主题配置解耦：
  规则 1：Driver 情绪经营 + 收割意图复合模型（杀猪盘信号）
  规则 2：Target 防线崩溃
  规则 3：纯聊拓扑结构性惩罚
"""

from __future__ import annotations

from models_stage2 import RoleLabel, SpeakerRoleResult, StageTwoResult, TrackType
from config_topics import GLOBAL_SCORING_CONFIG
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


def _find_role(roles: list[SpeakerRoleResult], target: RoleLabel) -> str | None:
    return next((r.speaker_id for r in roles if r.role == target), None)


# 角色拓扑附加分常量
_GROOMING_BONUS:                int   = 25
_GROOMING_TAG:                  str   = "emotional_grooming_risk"
_GROOMING_EGI_THRESHOLD:        float = 0.30
_GROOMING_COMPLIANCE_THRESHOLD: float = 0.50
_RESISTANCE_BONUS:              int   = 12
_RESISTANCE_TAG:                str   = "target_defenseless"
_RESISTANCE_DECAY_THRESHOLD:    float = 1.5


class RoleTopologyStrategy(ScoringRuleStrategy):
    """角色拓扑附加分策略。"""

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        all_intents: set[str] = ctx.extra.get("all_intents", set())
        ifeats     = result.interaction_features
        driver_sid = _find_role(result.speaker_roles, RoleLabel.DRIVER)

        # 规则 1：Driver 情绪经营 + 收割意图
        if driver_sid:
            egi = ifeats.emotional_grooming_index.get(driver_sid, 0.0)
            harvest_intents = {"fraud_object", "authority_entity", "fraud_jargon"}
            if (
                egi > _GROOMING_EGI_THRESHOLD
                and ifeats.compliance_rate > _GROOMING_COMPLIANCE_THRESHOLD
                and bool(all_intents & harvest_intents)
            ):
                ctx.apply(
                    delta=_GROOMING_BONUS,
                    tag=_GROOMING_TAG,
                    reason=(
                        f"Driver({driver_sid}) EGI={egi:.2f}，"
                        f"compliance={ifeats.compliance_rate:.2f}，"
                        "情绪经营+收割意图，构成杀猪盘复合信号"
                    ),
                )

        # 规则 2：Target 防线崩溃
        if ifeats.resistance_decay > _RESISTANCE_DECAY_THRESHOLD:
            ctx.apply(
                delta=_RESISTANCE_BONUS,
                tag=_RESISTANCE_TAG,
                reason=(
                    f"resistance_decay={ifeats.resistance_decay:.2f} "
                    f"> {_RESISTANCE_DECAY_THRESHOLD}，Target 防线显著收缩"
                ),
            )

        # 规则 3：纯聊拓扑结构性惩罚
        if result.track_type == TrackType.SYMMETRIC and ifeats.compliance_rate < 0.10:
            if not driver_sid:
                ctx.apply(
                    delta=GLOBAL_SCORING_CONFIG.get("structural_chitchat_penalty", -20),
                    tag="structural_chitchat_penalty",
                    reason="对称平权聊天且无业务顺从，物理结构判定为废话，执行结构性降权",
                )
