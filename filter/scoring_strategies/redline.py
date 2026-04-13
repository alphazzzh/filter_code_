"""
scoring_strategies/redline.py  ── 全局红线前置熔断策略
============================================================

零号优先级：在任何复杂矩阵运算之前，极速扫描底线安全词汇。
触发时直接判死（100分），绕过大模型全链路。
"""

from __future__ import annotations

from typing import Any

from models_stage2 import StageTwoResult
from config_topics import GLOBAL_REDLINE_REGISTRY
from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext


class RedlineCircuitBreakerStrategy(ScoringRuleStrategy):
    """
    全局红线前置熔断策略。

    触发条件：对话文本命中 GLOBAL_REDLINE_REGISTRY 中的任一正则。
    触发动作：设置 ctx.extra["redline_triggered"] = True，
             IntelligenceScorer 据此短路后续策略，直接返回 100 分。
    """

    def apply(self, ctx: _ScoringContext, result: StageTwoResult) -> None:
        full_text = " ".join(
            t.effective_text if hasattr(t, 'effective_text') and t.effective_text
            else (t.merged_text if hasattr(t, 'merged_text') else "")
            for t in result.dialogue_turns
            if not t.is_backchannel
        )
        if not full_text:
            return

        for redline_pattern in GLOBAL_REDLINE_REGISTRY:
            if redline_pattern.search(full_text):
                ctx.apply(
                    delta=50,
                    tag="GLOBAL_REDLINE_ALERT",
                    reason=(
                        "触发全局红线探针前置熔断，通话被立即阻断。"
                        "系统侦测到绝对违规词汇，绕过大模型直接判死。"
                    ),
                )
                # 标记熔断，IntelligenceScorer 据此短路
                ctx.extra["redline_triggered"] = True
                return

    @staticmethod
    def build_redline_output(ctx: _ScoringContext, result: StageTwoResult) -> dict[str, Any]:
        """当红线触发时，直接构造输出字典（短路后续策略）。"""
        return {
            "conversation_id":      result.conversation_id,
            "final_score":          100,
            "tags":                 ["GLOBAL_REDLINE_ALERT"],
            "tags_suppressed":      [],
            "track_type":           result.track_type.value,
            "roles":                {r.speaker_id: r.role.value for r in result.speaker_roles},
            "nlp_features_summary": {},
            "interaction_summary": {
                "ping_pong_rate":   result.interaction_features.negotiation_ping_pong_rate,
                "compliance_rate":  result.interaction_features.compliance_rate,
                "resistance_decay": result.interaction_features.resistance_decay,
                "word_distribution":result.interaction_features.speaker_word_ratio,
            },
            "score_breakdown": [
                {
                    "delta": 50,
                    "tag": "GLOBAL_REDLINE_ALERT",
                    "reason": (
                        "触发全局红线探针前置熔断，通话被立即阻断。"
                        "系统侦测到绝对违规词汇，绕过大模型直接判死。"
                    ),
                }
            ],
            "redline_triggered": True,
        }
