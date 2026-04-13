"""
scoring_strategies/  ── 打分策略包
============================================================

责任链/策略模式：每个 ScoringRuleStrategy 封装一个独立的打分逻辑块。
IntelligenceScorer.evaluate() 按序遍历策略列表，逐个 apply。

扩展方法：新增策略只需实现 ScoringRuleStrategy 并注册到策略列表，
IntelligenceScorer 零修改。
"""

from scoring_strategies.base import ScoringRuleStrategy, _ScoringContext
from scoring_strategies.redline import RedlineCircuitBreakerStrategy
from scoring_strategies.stage1_critical import Stage1CriticalStrategy
from scoring_strategies.high_risk_matrix import HighRiskMatrixStrategy
from scoring_strategies.target_resistance import TargetResistanceDiscountStrategy
from scoring_strategies.noise_suppression import NoiseSuppressionStrategy
from scoring_strategies.whitelist import WhitelistStrategy
from scoring_strategies.exemption import ExemptionStrategy
from scoring_strategies.cross_topic_bonus import CrossTopicBonusStrategy
from scoring_strategies.role_topology import RoleTopologyStrategy
from scoring_strategies.bot_intent_fusion import BotIntentFusionStrategy
from scoring_strategies.output_assembly import OutputAssemblyStrategy

__all__ = [
    "ScoringRuleStrategy",
    "_ScoringContext",
    "RedlineCircuitBreakerStrategy",
    "Stage1CriticalStrategy",
    "HighRiskMatrixStrategy",
    "TargetResistanceDiscountStrategy",
    "NoiseSuppressionStrategy",
    "WhitelistStrategy",
    "ExemptionStrategy",
    "CrossTopicBonusStrategy",
    "RoleTopologyStrategy",
    "BotIntentFusionStrategy",
    "OutputAssemblyStrategy",
]
