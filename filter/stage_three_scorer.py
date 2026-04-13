# stage_three_scorer.py  ── V6.0 策略驱动架构（责任链重构）
# ============================================================
# 从 V5.2 上帝类重构为策略驱动编排器。
#
# 变更摘要
# ─────────────────────────────────────────────────────────────
# ① IntelligenceScorer 不再包含任何打分逻辑
#    → 仅负责初始化 _ScoringContext + 遍历策略列表
# ② 所有打分逻辑拆分为独立的 ScoringRuleStrategy 子类
#    → 存放在 scoring_strategies/ 包中
# ③ BotConfidenceEngine / AdvancedVoicemailDetector 剥离
#    → 移至 engines/ 包，可被多方复用
# ④ 新增策略只需实现 ScoringRuleStrategy 并注册到策略列表
#    → IntelligenceScorer 零修改
# ============================================================

from __future__ import annotations

from typing import Any

from models_stage2 import StageTwoResult
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IntelligenceScorer —— V6.0 策略驱动编排器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntelligenceScorer:
    """
    策略驱动的共现矩阵情报打分引擎。

    核心流程
    ─────────────────────────────────────────────────────────
    evaluate()
     ├─ 初始化 _ScoringContext（工作台）
     ├─ 遍历 self._strategies，逐个调用 strategy.apply(ctx, result)
     │    ├─ RedlineCircuitBreakerStrategy   → 红线前置熔断（短路）
     │    ├─ Stage1CriticalStrategy          → 阶段一极高危兜底
     │    ├─ HighRiskMatrixStrategy          → 高危主题动态矩阵
     │    ├─ TargetResistanceDiscountStrategy → 受害者阶梯式抵抗降权
     │    ├─ NoiseSuppressionStrategy        → 废料降权 + OOD 兜底
     │    ├─ WhitelistStrategy               → 超级白名单豁免
     │    ├─ ExemptionStrategy               → 豁免减分
     │    ├─ CrossTopicBonusStrategy         → 跨族复合加分
     │    ├─ RoleTopologyStrategy            → 角色拓扑附加分
     │    ├─ BotIntentFusionStrategy         → AI 机器人意图融合
     │    └─ OutputAssemblyStrategy          → 输出组装
     └─ 从 ctx.extra["output"] 读取最终结果

    扩展方法
    ─────────────────────────────────────────────────────────
    1. 新增策略：实现 ScoringRuleStrategy，加入策略列表
    2. 替换策略：修改 _build_default_strategies() 或传入自定义列表
    3. 调整顺序：策略按列表顺序执行，前置策略可通过 ctx.extra 影响后续
    """

    def __init__(
        self,
        strategies: list[ScoringRuleStrategy] | None = None,
    ) -> None:
        self._strategies = strategies or self._build_default_strategies()

    @staticmethod
    def _build_default_strategies() -> list[ScoringRuleStrategy]:
        """构建默认策略列表（按优先级排序）。"""
        return [
            # 零号优先级：红线前置熔断（短路）
            RedlineCircuitBreakerStrategy(),
            # 一号优先级：阶段一极高危兜底
            Stage1CriticalStrategy(),
            # 二号优先级：高危主题动态矩阵（写入 hit_families / all_intents / nlp_feats）
            HighRiskMatrixStrategy(),
            # 三号优先级：受害者阶梯式抵抗降权
            TargetResistanceDiscountStrategy(),
            # 四号优先级：废料降权 + OOD 兜底（读 hit_families 决定是否执行）
            NoiseSuppressionStrategy(),
            # 五号优先级：白名单豁免
            WhitelistStrategy(),
            # 六号优先级：豁免减分
            ExemptionStrategy(),
            # 七号优先级：跨族复合加分
            CrossTopicBonusStrategy(),
            # 八号优先级：角色拓扑附加分
            RoleTopologyStrategy(),
            # 九号优先级：AI 机器人意图融合
            BotIntentFusionStrategy(),
            # 末尾：输出组装（不修改分数，只组装最终字典）
            OutputAssemblyStrategy(),
        ]

    def evaluate(self, stage2_result: StageTwoResult) -> dict[str, Any]:
        """
        对一条阶段二输出执行完整共现矩阵打分。

        流程：
        1. 初始化 _ScoringContext
        2. 按序遍历策略列表，逐个 apply
        3. 检查红线熔断短路
        4. 从 ctx.extra["output"] 读取最终结果
        """
        ctx = _ScoringContext()

        for strategy in self._strategies:
            strategy.apply(ctx, stage2_result)

            # 红线熔断短路：跳过后续策略，直接构造输出
            if ctx.extra.get("redline_triggered"):
                return RedlineCircuitBreakerStrategy.build_redline_output(
                    ctx, stage2_result
                )

        # 从 OutputAssemblyStrategy 写入的 extra 中读取最终输出
        output = ctx.extra.get("output")
        if output is not None:
            return output

        # 兜底（理论上不会走到这里，除非策略列表不完整）
        return {
            "conversation_id": stage2_result.conversation_id,
            "final_score": max(0, min(100, ctx.score)),
            "tags": sorted(ctx.tags),
            "score_breakdown": ctx.events,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 向后兼容：保留原有类名，从 engines 包导入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from engines.bot_confidence import BotConfidenceEngine  # noqa: E402, F401
from engines.voicemail_detector import AdvancedVoicemailDetector  # noqa: E402, F401
