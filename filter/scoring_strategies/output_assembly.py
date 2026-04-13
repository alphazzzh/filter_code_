"""
scoring_strategies/output_assembly.py  ── 输出组装策略
============================================================

边界钳位 [0,100] + 标签降维压制 + Floor Clamp + 结构化情报字典组装。

此策略在责任链末尾执行，负责将 _ScoringContext 中的累积状态
转换为最终的输出字典。
"""

from __future__ import annotations

from typing import Any

from models_stage2 import StageTwoResult
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext, _SCORE_MIN, _SCORE_MAX


# 高危兜底锁底线分
_HIGH_RISK_FLOOR: int = 60

# OOD 噪音标签集合
OOD_NOISE_TAGS: frozenset[str] = frozenset({
    "global_business_sparse", "global_too_short", "global_monologue_noise",
    "low_value_casual_chat", "low_value_wrong_number",
    "casual_chat_extremely_sparse", "casual_chat",
    "corporate_logistics_noise", "corporate_bidding_noise",
    "low_value_industrial_noise", "noise_brush_off_telemarketing",
    "noise_short_greeting_hangup", "noise_delivery_short",
    "structural_chitchat_penalty",
    "unconnected_voicemail_ivr",
})

# 高危信号标签前缀
_HIGH_RISK_TAG_PREFIXES: tuple[str, ...] = (
    "has_fraud_", "has_drug_", "has_coercive_", "has_extremist_",
    "suspicious_fake_cs", "has_authority_", "coercive_org",
    "emotional_grooming", "ai_scam_bot_", "multi_topic_compound",
    "STAGE1_CRITICAL",
    "fraud_imperative_", "fraud_jargon_", "fraud_object_",
    "fake_cs_", "critical_", "fraud_failed_target_resisted",
)

# 受害者抗性状态标签
_SCAM_ATTEMPT_REJECTED_TAG: str = "scam_attempt_rejected"


class OutputAssemblyStrategy(ScoringRuleStrategy):
    """
    输出组装策略。

    此策略不修改 ctx，而是将 ctx 中的累积状态转换为最终输出字典，
    存入 ctx.extra["output"] 供 IntelligenceScorer 读取。
    """

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        nlp_feats: dict[str, Any] = ctx.extra.get("nlp_feats", {})

        # 高危兜底锁（Floor Clamp）
        has_high_risk = any(
            e["delta"] > 0 and not e["tag"].startswith("official")
            for e in ctx.events
        )
        is_whitelisted = any(
            e["delta"] <= -40 and ("whitelist" in e["tag"] or "safe" in e["tag"])
            for e in ctx.events
        )
        floor_score = _HIGH_RISK_FLOOR if (has_high_risk and not is_whitelisted) else _SCORE_MIN

        final_score = max(floor_score, min(_SCORE_MAX, ctx.score))

        # LangGraph 动态主题融合
        dynamic_search = result.metadata.get("dynamic_search", {})
        if dynamic_search.get("matched"):
            topic_queried = dynamic_search.get("topic_queried", "unknown")
            final_score = max(final_score, 70)
            ctx.tags.add(f"dynamic_topic_matched_{topic_queried}")
            ctx.events.append({
                "delta": 0,
                "tag": "dynamic_topic_matched",
                "reason": f"LangGraph融合：动态主题 [{topic_queried}] 检索命中，基础分数保底提升至 70",
            })

        # 标签降维压制
        suppressed_tags = self._apply_tag_suppression(ctx.tags)

        roles: dict[str, str] = {
            r.speaker_id: r.role.value for r in result.speaker_roles
        }

        nlp_bool_summary: dict[str, Any] = {
            k: v for k, v in nlp_feats.items()
            if isinstance(v, bool)
        }
        nlp_bool_summary["nlp_backend"] = nlp_feats.get("nlp_backend", "unknown")

        ifeats = result.interaction_features

        output_dict = {
            "conversation_id":      result.conversation_id,
            "final_score":          final_score,
            "tags":                 sorted(suppressed_tags),
            "tags_suppressed":      sorted(ctx.tags - suppressed_tags),
            "track_type":           result.track_type.value,
            "roles":                roles,
            "nlp_features_summary": nlp_bool_summary,
            "interaction_summary": {
                "ping_pong_rate":   ifeats.negotiation_ping_pong_rate,
                "compliance_rate":  ifeats.compliance_rate,
                "resistance_decay": ifeats.resistance_decay,
                "word_distribution":ifeats.speaker_word_ratio,
            },
            "score_breakdown": [
                {"delta": e["delta"], "tag": e["tag"], "reason": e["reason"]}
                for e in ctx.events
            ],
        }

        if "dynamic_search" in result.metadata:
            output_dict["dynamic_search"] = result.metadata["dynamic_search"]

        # 将输出存入 extra，供 IntelligenceScorer 读取
        ctx.extra["output"] = output_dict

    @staticmethod
    def _apply_tag_suppression(tags: set[str]) -> set[str]:
        """标签降维压制：高危意图存在时，清除底层噪音标签。"""
        has_risk_signal = False

        for tag in tags:
            if tag.startswith(_HIGH_RISK_TAG_PREFIXES):
                has_risk_signal = True
                break
            if tag == _SCAM_ATTEMPT_REJECTED_TAG:
                has_risk_signal = True
                break
            if tag.startswith("ai_scam_bot_"):
                has_risk_signal = True
                break

        if not has_risk_signal:
            return tags

        return tags - OOD_NOISE_TAGS
