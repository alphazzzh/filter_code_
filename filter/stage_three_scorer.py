# stage_three_scorer.py  ── V5.0 配置驱动版
# ============================================================
# 共现矩阵打分引擎（完全配置驱动）
#
# V5.0 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 废弃所有硬编码矩阵常量（_MATRIX_FRAUD / _MATRIX_DRUG 等）
# ② evaluate() 流程遍历 TOPIC_REGISTRY，按 TopicCategory 分流：
#      HIGH_RISK      → 动态矩阵加分
#      LOW_VALUE_NOISE→ 动态降权扣分
#      WHITELIST      → 豁免折扣
#      EXEMPTION      → 触发豁免减分
# ③ 每个主题的 scoring_rules.matrix_combinations 是其矩阵行，
#   列（软意图）= topic_id 本身，行（硬特征）= syntax_feature
# ④ 新增主题只需在 config_topics.py 追加，本文件零修改
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models_stage2 import (
    DialogueTurn,
    InteractionFeatures,
    RoleLabel,
    SpeakerRoleResult,
    StageTwoResult,
    TrackType,
)
from config_topics import (
    TOPIC_REGISTRY,
    TopicCategory,
    TopicDefinition,
    get_topics_by_category,
    OOD_FALLBACK_REGISTRY,
    PROFANITY_REGISTRY,
)
from models import BotLabel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局打分边界常量（这几项确实不需要配置化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BASE_SCORE: int = 50
_SCORE_MIN:  int = 0
_SCORE_MAX:  int = 100

# 跨类别矩阵复合加分（同时命中 2 个及以上 HIGH_RISK 主题族）
_CROSS_TOPIC_BONUS:       int = 20
_CROSS_TOPIC_BONUS_TAG:   str = "multi_topic_compound"
_CROSS_TOPIC_MIN_HITS:    int = 2

# 角色拓扑附加分（独立于主题，属于结构性风险信号）
_GROOMING_BONUS:               int   = 25
_GROOMING_TAG:                 str   = "emotional_grooming_risk"
_GROOMING_EGI_THRESHOLD:       float = 0.30
_GROOMING_COMPLIANCE_THRESHOLD:float = 0.50
_RESISTANCE_BONUS:             int   = 12
_RESISTANCE_TAG:               str   = "target_defenseless"
_RESISTANCE_DECAY_THRESHOLD:   float = 1.5

# ── 标签层级结构（Hierarchical Tagging）────────────────────
# 底层噪音/OOD 标签集合：当存在任何高危意图时，这些标签必须被压制清除
# 防止「电商诈骗 + 受害者骂人 → 闲聊 sparse」这种语义冲突
OOD_NOISE_TAGS: frozenset[str] = frozenset({
    # OOD 物理兜底
    "global_business_sparse", "global_too_short", "global_monologue_noise",
    # LOW_VALUE_NOISE 语义层
    "low_value_casual_chat", "low_value_wrong_number",
    "casual_chat_extremely_sparse", "casual_chat",
    "corporate_logistics_noise", "corporate_bidding_noise",
    "low_value_industrial_noise", "noise_brush_off_telemarketing",
    "noise_short_greeting_hangup", "noise_delivery_short",
    # 拓扑结构性惩罚
    "structural_chitchat_penalty",
    # 语音信箱/未接通
    "unconnected_voicemail_ivr",
})

# 高危信号标签前缀（用于快速识别是否存在业务/诈骗意图，无需列举所有 topic_id）
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部打分工作台
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class _ScoringContext:
    """贯穿整个打分流程的可变工作台。每条对话独占一个实例。"""
    score:  int                  = _BASE_SCORE
    tags:   set[str]             = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, delta: int, tag: str | None, reason: str) -> None:
        self.score += delta
        if tag:
            self.tags.add(tag)
        self.events.append({"delta": delta, "tag": tag, "reason": reason})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _collect_all_intents(turns: list[DialogueTurn]) -> set[str]:
    return {label for t in turns for label in t.intent_labels}


def _find_role(roles: list[SpeakerRoleResult], target: RoleLabel) -> str | None:
    return next((r.speaker_id for r in roles if r.role == target), None)


def _check_whitelist(
    result: StageTwoResult,
    all_intents: set[str],
) -> tuple[bool, float]:
    """
    检查是否命中白名单主题。

    逻辑：
    1. 阶段一 bot_label == "bot"（有机器人信号）
    2. 对话中含有白名单主题的软意图（csr_bot_whitelist 被命中）
    3. 对话中无 HIGH_RISK 主题的意图
    4. 硬特征 has_imperative_syntax == False

    返回 (is_whitelisted, discount_ratio)
    """
    whitelist_topics = get_topics_by_category(TopicCategory.WHITELIST)
    risk_topics      = get_topics_by_category(TopicCategory.HIGH_RISK)

    # 检查是否有白名单意图被命中
    whitelist_hit_topics = {
        tid for tid in whitelist_topics if tid in all_intents
    }
    if not whitelist_hit_topics:
        return False, 1.0

    # 检查是否有风险意图
    risk_intent_hit = bool(all_intents & set(risk_topics.keys()))
    if risk_intent_hit:
        return False, 1.0

    # 检查硬特征
    nlp_feats = result.metadata.get("nlp_features", {})
    if nlp_feats.get("has_imperative_syntax", False):
        return False, 1.0

    # 取所有命中白名单主题中最低的折扣率（最保守）
    min_discount = min(
        whitelist_topics[tid].scoring_rules.whitelist_discount
        for tid in whitelist_hit_topics
    )
    return True, min_discount


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IntelligenceScorer —— V5.0 全动态共现矩阵引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntelligenceScorer:
    """
    V5.0 配置驱动的共现矩阵情报打分引擎。

    核心流程
    ─────────────────────────────────────────────────────────
    evaluate()
     ├─ _run_whitelist_check()         白名单豁免（SHORT CIRCUIT）
     ├─ _run_high_risk_topics()        HIGH_RISK 主题动态矩阵
     │    └── 每个主题：单项分 + 矩阵组合分
     ├─ _run_noise_topics()            LOW_VALUE_NOISE 降权扣分
     ├─ _run_exemption_topics()        EXEMPTION 豁免减分
     ├─ _run_cross_topic_bonus()       跨主题复合加分
     ├─ _run_role_topology()           角色拓扑附加分
     └─ _build_output()               边界钳位 + 输出组装

    扩展方法（零代码修改）
    ─────────────────────────────────────────────────────────
    在 config_topics.TOPIC_REGISTRY 中追加新主题：
      - HIGH_RISK      → 自动纳入 _run_high_risk_topics()
      - LOW_VALUE_NOISE→ 自动纳入 _run_noise_topics()
      - WHITELIST      → 自动纳入 _run_whitelist_check()
      - EXEMPTION      → 自动纳入 _run_exemption_topics()
    """

    def __init__(
        self,
        registry: dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._registry = registry
        # 按类别预分组，避免每次 evaluate() 重复过滤
        self._high_risk_topics = {
            tid: td
            for tid, td in registry.items()
            if td.category == TopicCategory.HIGH_RISK
            and td.scoring_rules.standalone_score != 0  # 过滤纯辅助主题（emotion等）
        }
        self._noise_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.LOW_VALUE_NOISE
        }
        self._exemption_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.EXEMPTION
        }
        self._whitelist_topics = {
            tid: td for tid, td in registry.items()
            if td.category == TopicCategory.WHITELIST
        }

    def evaluate(self, stage2_result: StageTwoResult) -> dict[str, Any]:
        """
        对一条阶段二输出执行完整共现矩阵打分。
        """
        ctx = _ScoringContext()

        # 🚨 缺陷 1 修复：阶段一硬正则极高危拦截兜底
        s1_meta = stage2_result.metadata.get("stage_one", {})
        if s1_meta.get("stage_one_critical_hit"):
            ctx.apply(40, "STAGE1_CRITICAL_FORCE_RECALL", "阶段一硬正则极高危拦截")

        all_intents: set[str]     = _collect_all_intents(stage2_result.dialogue_turns)
        nlp_feats:   dict[str, Any] = stage2_result.metadata.get("nlp_features", {})
        speaker_nlp_feats: dict[str, dict[str, Any]] = stage2_result.metadata.get("speaker_nlp_features", {})

        # 🚨 严格锁定骗子（Agent）的句法特征
        agent_sid = _find_role(stage2_result.speaker_roles, RoleLabel.AGENT)
        agent_nlp_feats = speaker_nlp_feats.get(agent_sid) if agent_sid else nlp_feats

        # ── 1. HIGH_RISK：动态矩阵加分 ──
        high_risk_hit_count = self._run_high_risk_topics(ctx, all_intents, agent_nlp_feats)

        # ── 2. 全局受害者抵抗降权 (Task 3) ──
        self._run_target_resistance_discount(ctx, stage2_result, high_risk_hit_count > 0)

        # ── 3. LOW_VALUE_NOISE & OOD 物理兜底 ──
        # 🚨 核心逻辑：高危意图绝对优先。未命中风险时，才进行废料降权。
        if high_risk_hit_count == 0:
            # 3.1 语义层降权（已知类别的废话：外卖/闲聊/打错电话）
            self._run_noise_topics(ctx, all_intents, nlp_feats)
            
            # 3.2 物理层兜底（OOD 未知领域的废话：基于拓扑结构和实体密度）
            self._run_ood_fallback(ctx, stage2_result, nlp_feats)

        # ── 4. WHITELIST：超级白名单 & 机器人豁免 (Task 2) ──
        # 无视任何高危标签，强制执行扣分
        self._run_whitelist_topics(ctx, stage2_result, all_intents)

        # ── 5. 其他辅助/结构性评分 ──
        self._run_exemption_topics(ctx, stage2_result, all_intents)
        self._run_cross_topic_bonus(ctx, high_risk_hit_count)
        self._run_role_topology(ctx, stage2_result, all_intents)
        self._run_bot_intent_fusion(ctx, stage2_result, all_intents)

        return self._build_output(ctx, stage2_result, nlp_feats)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HIGH_RISK 动态矩阵
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_high_risk_topics(
        self,
        ctx:         _ScoringContext,
        all_intents: set[str],
        agent_nlp_feats: dict[str, Any],
    ) -> int:
        """
        遍历所有 HIGH_RISK 主题，对每个命中的主题执行：
          Step A：单项基础分（无硬特征强化，低置信度）
          Step B：遍历 matrix_combinations，查找共现矩阵命中
                  条件：该主题意图被命中 AND 对应硬特征为 True
          Step C：记录该主题是否有任何命中（供跨主题复合加分计数）

        返回：命中（触发过分值变化）的主题数量
        """
        hit_count = 0

        for topic_id, topic_def in self._high_risk_topics.items():
            # 检查软意图是否命中
            if topic_id not in all_intents:
                continue

            topic_hit = False
            matrix_hit = False

            # Step B：矩阵组合扫描（优先级高于单项分）
            for combo in topic_def.scoring_rules.matrix_combinations:
                hard_feat_value = bool(agent_nlp_feats.get(combo.syntax_feature, False))
                
                # 👇 新增：识别正向/负向触发逻辑
                triggered = False
                if combo.requires_absence and not hard_feat_value:
                    triggered = True  # 要求不存在且确实不存在 -> 触发
                elif not combo.requires_absence and hard_feat_value:
                    triggered = True  # 要求存在且确实存在 -> 触发

                if triggered:
                    ctx.apply(
                        delta  = combo.bonus_score,
                        tag    = combo.bonus_tag,
                        reason = (
                            f"矩阵命中 [{'!' if combo.requires_absence else ''}{combo.syntax_feature}] × [{topic_id}]"
                            f" → {combo.bonus_score:+d}"
                        ),
                    )
                    matrix_hit = True
                    topic_hit  = True

            # Step A：单项基础分（矩阵未触发时才给，避免双重计分）
            if not matrix_hit and topic_def.scoring_rules.standalone_score != 0:
                ctx.apply(
                    delta  = topic_def.scoring_rules.standalone_score,
                    tag    = topic_def.scoring_rules.standalone_tag,
                    reason = f"单项意图命中 [{topic_id}]（无硬特征强化）",
                )
                topic_hit = True

            if topic_hit:
                hit_count += 1

        return hit_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Target Resistance Discount (Task 3)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _run_target_resistance_discount(
            self,
            ctx:         _ScoringContext,
            result:      StageTwoResult,
            has_high_risk: bool,
        ) -> None:
            """
            全局受害者抵抗降权机制：
            如果高危对话中 Target 明确拒绝/抵抗且顺从度极低，则视为失败的犯罪尝试，大幅扣分。
            同时追加「诈骗未遂」状态标签（scam_attempt_rejected），具有极高情报价值。
            """
            if not has_high_risk:
                return

            target_sid = _find_role(result.speaker_roles, RoleLabel.TARGET)
            if not target_sid:
                return

            target_turns = [
                t for t in result.dialogue_turns
                if t.speaker_id == target_sid and not t.is_backchannel
            ]
            if not target_turns:
                return

            # 收集受害者的所有意图，方便快速判断是否包含绝对的质变信号
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

            # 🚨 终极微调：抵抗率 > 0.25 (量变) 或者 明确识破(质变)
            if (resistance_rate > 0.25 or "dismissal" in target_intents) and compliance_rate < 0.10:
                ctx.apply(
                    delta  = -35,
                    tag    = "fraud_failed_target_resisted",
                    reason = (
                        f"检测到高危风险，但受害者明确拒绝/脱战"
                        f"（抵抗率={resistance_rate:.2f}或触发绝对识破），案件降级为未遂线索"
                    ),
                )
                # 追加「诈骗未遂」状态标签——情报分析的核心信号
                ctx.tags.add(_SCAM_ATTEMPT_REJECTED_TAG)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOW_VALUE_NOISE 降权扣分
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_noise_topics(self, ctx: _ScoringContext, all_intents: set[str], nlp_feats: dict[str, Any]) -> None:
            for topic_id, topic_def in self._noise_topics.items():
                if topic_id in all_intents:
                    # [原有逻辑保留]：抑制物流对毒品的误报
                    if topic_id == "corporate_logistics" and not nlp_feats.get("has_drug_quantity"):
                        # 【隐患 3 修复】使用列表推导重建，避免 list.remove(e) 在重复 dict 时删错项
                        surviving_events = []
                        for e in ctx.events:
                            if e["tag"].startswith("has_drug_") or e["tag"].startswith("drug_quantity_"):
                                ctx.score -= e["delta"]  # 回退分数
                            else:
                                surviving_events.append(e)
                        ctx.events = surviving_events

                    # Step A：永远先给予低价值单项基础扣分
                    if topic_def.scoring_rules.standalone_score != 0:
                        ctx.apply(
                            delta  = topic_def.scoring_rules.standalone_score,
                            tag    = topic_def.scoring_rules.standalone_tag,
                            reason = f"命中低价值噪声主题 [{topic_id}]，降权处理"
                        )

                    # Step B：扫描矩阵，如果满足条件，执行【额外附加】扣分！
                    for combo in topic_def.scoring_rules.matrix_combinations:
                        hard_feat_value = bool(nlp_feats.get(combo.syntax_feature, False))
                        triggered = (combo.requires_absence and not hard_feat_value) or (not combo.requires_absence and hard_feat_value)
                        
                        if triggered:
                            ctx.apply(
                                delta  = combo.bonus_score,
                                tag    = combo.bonus_tag,
                                reason = f"负向矩阵压制 [{'!' if combo.requires_absence else ''}{combo.syntax_feature}] × [{topic_id}] → {combo.bonus_score:+d}"
                            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # WHITELIST 超级白名单 & 机器人豁免 (Task 2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_whitelist_topics(
        self,
        ctx:         _ScoringContext,
        result:      StageTwoResult,
        all_intents: set[str],
    ) -> None:
        """
        执行白名单处理逻辑。
        inbound_official_ivr: 超级白名单，强制扣分（-60）。
        csr_bot_whitelist:    传统机器人白名单，百分比折扣。
        """
        for tid, td in self._whitelist_topics.items():
            if tid == "inbound_official_ivr":
                if tid in all_intents:
                    ctx.apply(
                        delta  = td.scoring_rules.standalone_score,
                        tag    = td.scoring_rules.standalone_tag,
                        reason = f"命中超级白名单 [{tid}]（官方客服呼入），强制降权防误杀",
                    )

            elif tid == "csr_bot_whitelist":
                # 使用原有的复杂启发式判定机器人
                is_wl, discount = _check_whitelist(result, all_intents)
                if is_wl:
                    discounted = round(ctx.score * discount)
                    delta = discounted - ctx.score
                    ctx.apply(
                        delta  = delta,
                        tag    = td.scoring_rules.standalone_tag,
                        reason = f"命中机器人白名单（启发式），折扣率={discount:.2f}",
                    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXEMPTION 豁免减分
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_exemption_topics(
        self,
        ctx:         _ScoringContext,
        result:      StageTwoResult,
        all_intents: set[str],
    ) -> None:
        """
        遍历 EXEMPTION 主题（如 dismissal/rejection），命中时扣除固定分值。
        注意：此处为单项意图扣分，与全局降权逻辑并存。
        """
        for tid, td in self._exemption_topics.items():
            if tid in all_intents:
                ctx.apply(
                    delta  = td.scoring_rules.standalone_score,
                    tag    = td.scoring_rules.standalone_tag,
                    reason = f"命中豁免信号主题 [{tid}]",
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 跨主题复合加分
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _run_cross_topic_bonus(
        ctx: _ScoringContext, high_risk_hit_count: int
    ) -> None:
        """当同时命中 >= _CROSS_TOPIC_MIN_HITS 个独立风险主题时额外加分。"""
        if high_risk_hit_count >= _CROSS_TOPIC_MIN_HITS:
            ctx.apply(
                delta  = _CROSS_TOPIC_BONUS,
                tag    = _CROSS_TOPIC_BONUS_TAG,
                reason = (
                    f"跨主题复合命中 {high_risk_hit_count} 个高风险主题，"
                    "升级综合危险等级"
                ),
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 角色拓扑附加分（结构性风险，独立于主题配置）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _run_role_topology(
        ctx:         _ScoringContext,
        result:      StageTwoResult,
        all_intents: set[str],
    ) -> None:
        """
        角色拓扑结构性风险检测，与主题配置解耦：

        规则 1：Driver 情绪经营 + 收割意图复合模型（杀猪盘信号）
          条件：Driver EGI > 0.30 AND compliance_rate > 0.50
                AND 存在金融/客体收割意图

        规则 2：Target 防线崩溃
          条件：resistance_decay > 阈值
        """
        ifeats     = result.interaction_features
        driver_sid = _find_role(result.speaker_roles, RoleLabel.DRIVER)

        # 规则 1
        if driver_sid:
            egi = ifeats.emotional_grooming_index.get(driver_sid, 0.0)
            harvest_intents = {"fraud_object", "authority_entity", "fraud_jargon"}
            if (
                egi > _GROOMING_EGI_THRESHOLD
                and ifeats.compliance_rate > _GROOMING_COMPLIANCE_THRESHOLD
                and bool(all_intents & harvest_intents)
            ):
                ctx.apply(
                    delta  = _GROOMING_BONUS,
                    tag    = _GROOMING_TAG,
                    reason = (
                        f"Driver({driver_sid}) EGI={egi:.2f}，"
                        f"compliance={ifeats.compliance_rate:.2f}，"
                        "情绪经营+收割意图，构成杀猪盘复合信号"
                    ),
                )

        # 规则 2
        if ifeats.resistance_decay > _RESISTANCE_DECAY_THRESHOLD:
            ctx.apply(
                delta  = _RESISTANCE_BONUS,
                tag    = _RESISTANCE_TAG,
                reason = (
                    f"resistance_decay={ifeats.resistance_decay:.2f} "
                    f"> {_RESISTANCE_DECAY_THRESHOLD}，Target 防线显著收缩"
                ),
            )

        # 规则 3：纯聊拓扑结构性惩罚 (Structural Noise Deduction)
        if result.track_type == TrackType.SYMMETRIC and ifeats.compliance_rate < 0.10:
            if not driver_sid:  # 如果双方都没能被评为 DRIVER（即双方都是 PEER_A / PEER_B 平权聊天）
                ctx.apply(
                    delta  = -20,
                    tag    = "structural_chitchat_penalty",
                    reason = "对称平权聊天且无业务顺从，物理结构判定为废话，执行结构性降权"
                )


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AI 机器人 × 伪装意图 核爆融合逻辑（规则 4）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 机器人 × 意图融合触发的高危主题集合
    _BOT_FUSION_HIGH_RISK_TOPICS: frozenset[str] = frozenset({
        "e_commerce_cs",       # 电商客服伪装
        "authority_entity",    # 权威机构伪装
        "fraud_object",        # 诈骗客体
        "fraud_jargon",        # 诈骗黑话
    })

    # 融合标签映射（高危主题 → 融合标签）
    _BOT_FUSION_TAG_MAP: dict[str, str] = {
        "e_commerce_cs":    "ai_scam_bot_ecommerce",
        "authority_entity": "ai_scam_bot_authority",
        "fraud_object":     "ai_scam_bot_fraud_object",
        "fraud_jargon":     "ai_scam_bot_fraud_jargon",
    }

    _BOT_FUSION_DELTA: int = 15  # 融合惩罚加分

    @classmethod
    def _run_bot_intent_fusion(
        cls,
        ctx:         _ScoringContext,
        result:      StageTwoResult,
        all_intents: set[str],
    ) -> None:
        """
        AI 外呼诈骗法则（规则 4）：

        触发条件（同时满足）：
        1. 机器人特征明显（以下任一）：
           - 阶段一 bot_label == BOT
           - ping_pong_rate < 0.1（几乎无交互）
           - filler_word_rate < 0.005（极度流畅无口癖）
        2. 命中高危伪装意图（e_commerce_cs / authority_entity / fraud_object / fraud_jargon）

        执行动作：
        - delta = +15 融合惩罚加分
        - 追加高危融合标签
        """
        # 检查是否命中高危伪装意图
        hit_topics = all_intents & cls._BOT_FUSION_HIGH_RISK_TOPICS
        if not hit_topics:
            return

        # 检查机器人特征
        is_bot_signal = False
        s1_meta = result.metadata.get("stage_one", {})

        # 信号 1：阶段一 bot_label == BOT
        if s1_meta.get("bot_label") == BotLabel.BOT.value:
            is_bot_signal = True

        # 信号 2：ping_pong_rate < 0.1（几乎无真正交互）
        ppr = result.interaction_features.negotiation_ping_pong_rate
        if ppr < 0.1:
            is_bot_signal = True

        # 信号 3：filler_word_rate < 0.005（极度流畅）
        nlp_extra = result.metadata.get("nlp_features_extra", {})
        fwr = nlp_extra.get("filler_word_rate", 1.0)  # 默认 1.0（真人水平）
        if fwr < 0.005:
            is_bot_signal = True

        if not is_bot_signal:
            return

        # 触发融合：对每个命中的高危主题执行惩罚加分
        for topic_id in hit_topics:
            fusion_tag = cls._BOT_FUSION_TAG_MAP.get(topic_id, "ai_scam_bot_generic")
            ctx.apply(
                delta  = cls._BOT_FUSION_DELTA,
                tag    = fusion_tag,
                reason = (
                    f"命中 AI 机器人批量外呼与高危意图 [{topic_id}] 融合，"
                    f"判定为机器诈骗试探 (ping_pong={ppr:.4f}, filler_rate={fwr:.4f})"
                ),
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 输出组装
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 高危兜底锁底线分：有高危加分事件时，绝不能跌破此分数
    _HIGH_RISK_FLOOR: int = 60

    @staticmethod
    def _apply_tag_suppression(tags: set[str]) -> set[str]:
        """
        标签降维压制（Tag Suppression）：
        
        层级规则：
        - 高危意图标签（诈骗/涉毒/暴力/极端等）> OOD 噪音标签（闲聊/废料/碎片）
        - 当 tags 中存在任何高危信号时，强制移除所有 OOD_NOISE_TAGS 中的底层标签
        
        判定高危存在的依据：
        1. tags 中是否包含 _HIGH_RISK_TAG_PREFIXES 前缀的标签
        2. tags 中是否包含「诈骗未遂」标签（说明曾经命中高危意图）
        3. tags 中是否包含 ai_scam_bot_* 系列融合标签
        
        示例：
        输入: {"suspicious_fake_cs", "casual_chat_extremely_sparse", "global_business_sparse", "scam_attempt_rejected"}
        输出: {"suspicious_fake_cs", "scam_attempt_rejected"}  # 噪音标签被压制
        """
        # 快速检查：是否存在高危信号
        has_risk_signal = False
        
        for tag in tags:
            # 检查前缀匹配
            if tag.startswith(_HIGH_RISK_TAG_PREFIXES):
                has_risk_signal = True
                break
            # 检查诈骗未遂标签
            if tag == _SCAM_ATTEMPT_REJECTED_TAG:
                has_risk_signal = True
                break
            # 检查 bot 融合标签
            if tag.startswith("ai_scam_bot_"):
                has_risk_signal = True
                break

        if not has_risk_signal:
            return tags  # 无高危信号，保留全部标签

        # 存在高危信号 → 压制清除所有 OOD 噪音标签
        suppressed = tags - OOD_NOISE_TAGS
        return suppressed

    @staticmethod
    def _build_output(
        ctx:       _ScoringContext,
        result:    StageTwoResult,
        nlp_feats: dict[str, Any],
    ) -> dict[str, Any]:
        """边界钳位 [0,100] + 标签降维压制 + 结构化情报字典组装。"""
        # 【Bug 1 修复】高危兜底锁（Floor Clamp）
        # 如果曾经命中过任何正向加分的高危标签（delta > 0 且非白名单豁免），
        # 即使被受害者抵抗/豁免扣分，底线也是 60 分，保证不会落入垃圾填埋场
        has_high_risk = any(
            e["delta"] > 0 and not e["tag"].startswith("official")
            for e in ctx.events
        )
        floor_score = IntelligenceScorer._HIGH_RISK_FLOOR if has_high_risk else _SCORE_MIN

        final_score = max(floor_score, min(_SCORE_MAX, ctx.score))

        # ── 标签降维压制（Hierarchical Tag Suppression）──
        # 在输出前执行：高危意图存在时，清除底层噪音标签，防止语义冲突
        suppressed_tags = IntelligenceScorer._apply_tag_suppression(ctx.tags)

        roles: dict[str, str] = {
            r.speaker_id: r.role.value for r in result.speaker_roles
        }

        # 仅保留布尔型特征摘要（过滤 nlp_backend 等元信息）
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
            "tags_suppressed":      sorted(ctx.tags - suppressed_tags),  # 被压制掉的标签（审计用）
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

        # 透传阶段二动态检索结果（dynamic_topic → BGE 矩阵相似度 → dynamic_search）
        if "dynamic_search" in result.metadata:
            output_dict["dynamic_search"] = result.metadata["dynamic_search"]

        return output_dict
    
    def _run_ood_fallback(
        self, 
        ctx: _ScoringContext, 
        stage2_result: StageTwoResult, 
        nlp_feats: dict[str, Any]
    ) -> None:
        """
        OOD 物理废料兜底执行器。
        构建当前的运行时动态快照，并遍历执行 config_topics 中定义的规则。
        """
        ifeats = stage2_result.interaction_features
        valid_turns = sum(1 for t in stage2_result.dialogue_turns if not t.is_backchannel)
        
        # 构建统一的上下文快照 (Runtime Evaluation Context)
        eval_context: dict[str, Any] = {
            "valid_turn_count": valid_turns,
            "compliance_rate":  ifeats.compliance_rate,
            "ping_pong_rate":   ifeats.negotiation_ping_pong_rate,
            **nlp_feats  # 将所有的 NLP 布尔特征混入上下文
        }

        # 遍历配置驱动的物理兜底规则
        for rule in OOD_FALLBACK_REGISTRY:
            if rule.condition(eval_context):
                ctx.apply(
                    delta  = rule.delta, 
                    tag    = rule.tag, 
                    reason = rule.reason
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BotConfidenceEngine —— 多维置信度机器人检测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Voicemail 正则词库（用于 AdvancedVoicemailDetector）—— 多语言：中/英/日/粤
_VOICEMAIL_REGEX_WORDS: list[str] = [
    # ── 中文语音信箱/未接通提示 ──
    "无法接通", "暂时无法接通", "正在通话中", "电话正忙",
    "不在服务区", "已启用来电提醒", "无人接听", "呼叫超时",
    "已挂断", "请在提示音后留言", "录音完成后挂断",
    "留言最长", "按井号键", "语音信箱已满", "呼叫转移",
    "转接语音信箱", "正在为您转接", "号码是空号",
    "已停机", "已过期", "请勿挂机", "余额不足",
    "通话已被录音", "会议通话中", "等待音乐",
    # ── 英文 Voicemail/IVR ──
    "The number you dialed", "leave a message", "after the beep",
    "mailbox is full", "call is being transferred", "line is busy",
    "not available", "voicemail", "press 1",
    "please hold", "all agents are busy", "your call is important",
    "no one is available", "record your message", "at the tone",
    # ── 日文留守番電話/転送 ──
    "留守番電話", "発信音の後に", "ピーという音が鳴りましたら",
    "メッセージを残して", "おかけになった電話番号",
    "電源が切れています", "通話中です", "応答なし",
    # ── 粤語未接通 ──
    "嗶一聲之後", "唔得閒", "留低訊息", "話機冇訊號",
]


class BotConfidenceEngine:
    """
    多维置信度机器人检测引擎。

    评分规则（基础分 0，叠加计分）：
      - 命中 csr_bot_whitelist 软意图         → +40
      - filler_word_rate < 0.005（极度流畅）   → +30
      - ping_pong_rate 极度规律且无并发        → +30

    一票否决（得分归零，强制标记为 HUMAN）：
      - 触发 PROFANITY_REGISTRY（脏话/攻击性词汇）
      - 复杂的反问逻辑（受害者主动反问 + 识破意图）

    最终得分 > 80 判定为 BOT。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _BOT_THRESHOLD: int = 80

    # ── 加分权重 ──────────────────────────────────────────
    _SCORE_CSR_WHITELIST:  int = 40
    _SCORE_FLUENCY:        int = 30
    _FLUENCY_THRESHOLD:    float = 0.005
    _SCORE_PING_PONG:      int = 30

    def evaluate(
        self,
        stage2_result: StageTwoResult,
        filler_word_rate: float = 0.0,
    ) -> dict[str, Any]:
        """
        执行多维置信度机器人判定。

        Parameters
        ----------
        stage2_result    : 阶段二输出
        filler_word_rate : 语气词占比（由 TopologyEngine 提供）

        Returns
        -------
        dict[str, Any]
            bot_score  : int           置信度得分 [0, 100]
            bot_label  : BotLabel      BOT / HUMAN
            veto_reason: str | None    一票否决原因（若触发）
            details    : list[str]     评分细节
        """
        score: int   = 0
        details: list[str] = []
        all_intents: set[str] = {label for t in stage2_result.dialogue_turns for label in t.intent_labels}
        full_text: str = " ".join(
            t.merged_text for t in stage2_result.dialogue_turns
            if not t.is_backchannel
        )

        # ── 一票否决检查（优先级最高）─────────────────────

        # 否决 1：脏话/攻击性词汇
        veto_word: str | None = self._check_profanity(full_text)
        if veto_word:
            return {
                "bot_score": 0,
                "bot_label": BotLabel.HUMAN,
                "veto_reason": f"命中脏话/攻击性词汇：「{veto_word}」，强制标记为 HUMAN",
                "details": ["PROFANITY_VETO"],
            }

        # 否决 2：复杂反问逻辑（受害者识破意图的反问）
        if self._check_complex_rhetorical(stage2_result, all_intents):
            return {
                "bot_score": 0,
                "bot_label": BotLabel.HUMAN,
                "veto_reason": "检测到复杂反问逻辑（受害者主动反问+识破意图），强制标记为 HUMAN",
                "details": ["COMPLEX_RETORT_VETO"],
            }

        # ── 加分项 ────────────────────────────────────────

        # 加分 1：命中 csr_bot_whitelist
        if "csr_bot_whitelist" in all_intents:
            score += self._SCORE_CSR_WHITELIST
            details.append(f"命中 csr_bot_whitelist → +{self._SCORE_CSR_WHITELIST}")

        # 加分 2：极度流畅（语气词极少）
        if filler_word_rate < self._FLUENCY_THRESHOLD:
            score += self._SCORE_FLUENCY
            details.append(
                f"filler_word_rate={filler_word_rate:.4f} < {self._FLUENCY_THRESHOLD} → +{self._SCORE_FLUENCY}"
            )

        # 加分 3：ping_pong_rate 极度规律
        ppr = stage2_result.interaction_features.negotiation_ping_pong_rate
        # 机器人外呼的 ping_pong 通常为 0（独白）或极低
        if ppr < 0.05:
            score += self._SCORE_PING_PONG
            details.append(
                f"ping_pong_rate={ppr:.4f} < 0.05 → +{self._SCORE_PING_PONG}"
            )

        # ── 最终判定 ──────────────────────────────────────
        bot_label = BotLabel.BOT if score > self._BOT_THRESHOLD else BotLabel.HUMAN
        details.append(f"最终得分={score}，阈值={self._BOT_THRESHOLD}，判定={bot_label.value}")

        return {
            "bot_score":  score,
            "bot_label":  bot_label,
            "veto_reason": None,
            "details":    details,
        }

    @staticmethod
    def _check_profanity(text: str) -> str | None:
        """检查文本是否命中 PROFANITY_REGISTRY，返回第一个命中的词或 None。"""
        text_lower = text.lower()
        for word in PROFANITY_REGISTRY:
            if word.lower() in text_lower:
                return word
        return None

    @staticmethod
    def _check_complex_rhetorical(
        stage2_result: StageTwoResult,
        all_intents: set[str],
    ) -> bool:
        """
        检测复杂反问逻辑。

        判定条件：
        1. 对话中存在 dismissal（识破）意图
        2. 且存在 interrogation（反问）意图
        3. 且 compliance_rate 极低（< 0.10）
        三者同时满足 → 受害者在主动反击，判定为真人。
        """
        has_dismissal = "dismissal" in all_intents
        has_interrogation = "interrogation" in all_intents
        low_compliance = stage2_result.interaction_features.compliance_rate < 0.10

        return has_dismissal and has_interrogation and low_compliance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdvancedVoicemailDetector —— 高级无效通话检测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdvancedVoicemailDetector:
    """
    高级无效通话（Voicemail/语音信箱/未接通）检测引擎。

    评分规则（基础分 0，叠加计分）：
      - 命中 Voicemail 正则词                → +60
      - 角色 B 字数极短且符合模板             → +30
      - 交互判定为 is_decoupled（解耦盲说）   → +20

    一票否决（得分归零）：
      - ping_pong_rate > 0.1 且不是解耦状态 → 判定为有效通话

    最终得分 > 80 判定为无效通话。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _VOICEMAIL_THRESHOLD: int = 80

    # ── 加分权重 ──────────────────────────────────────────
    _SCORE_VOICEMAIL_WORD:   int = 60
    _SCORE_SHORT_TEMPLATE_B: int = 30
    _SCORE_DECOUPLED:        int = 20

    # ── 角色 B 短模板阈值 ─────────────────────────────────
    _SHORT_B_WORD_THRESHOLD: int = 15  # 角色 B 总字数 < 15 视为极短

    def evaluate(
        self,
        stage2_result: StageTwoResult,
        is_decoupled: bool = False,
    ) -> dict[str, Any]:
        """
        执行高级无效通话判定。

        Parameters
        ----------
        stage2_result : 阶段二输出
        is_decoupled  : 解耦盲说判定（由 TopologyEngine 提供）

        Returns
        -------
        dict[str, Any]
            voicemail_score : int           置信度得分 [0, 100]
            is_voicemail    : bool          是否判定为无效通话
            veto_reason     : str | None    一票否决原因
            details         : list[str]     评分细节
        """
        score: int   = 0
        details: list[str] = []
        full_text: str = " ".join(
            t.merged_text for t in stage2_result.dialogue_turns
            if not t.is_backchannel
        )

        # ── 一票否决检查 ──────────────────────────────────
        ppr = stage2_result.interaction_features.negotiation_ping_pong_rate
        if ppr > 0.1 and not is_decoupled:
            return {
                "voicemail_score": 0,
                "is_voicemail": False,
                "veto_reason": (
                    f"ping_pong_rate={ppr:.4f} > 0.1 且非解耦状态，"
                    "判定为有效双向通话"
                ),
                "details": ["PING_PONG_VETO"],
            }

        # ── 加分项 ────────────────────────────────────────

        # 加分 1：命中 Voicemail 正则词
        voicemail_word = self._check_voicemail_words(full_text)
        if voicemail_word:
            score += self._SCORE_VOICEMAIL_WORD
            details.append(f"命中 Voicemail 词「{voicemail_word}」→ +{self._SCORE_VOICEMAIL_WORD}")

        # 加分 2：角色 B 字数极短且符合模板
        if self._check_short_template_b(stage2_result):
            score += self._SCORE_SHORT_TEMPLATE_B
            details.append(
                f"角色 B 字数极短（< {self._SHORT_B_WORD_THRESHOLD} 字）→ +{self._SCORE_SHORT_TEMPLATE_B}"
            )

        # 加分 3：解耦盲说状态
        if is_decoupled:
            score += self._SCORE_DECOUPLED
            details.append(f"解耦盲说状态 (is_decoupled=True) → +{self._SCORE_DECOUPLED}")

        # ── 最终判定 ──────────────────────────────────────
        is_voicemail = score > self._VOICEMAIL_THRESHOLD
        details.append(
            f"最终得分={score}，阈值={self._VOICEMAIL_THRESHOLD}，"
            f"判定={'无效通话' if is_voicemail else '有效通话'}"
        )

        return {
            "voicemail_score": score,
            "is_voicemail":  is_voicemail,
            "veto_reason":    None,
            "details":        details,
        }

    @staticmethod
    def _check_voicemail_words(text: str) -> str | None:
        """检查文本是否命中 Voicemail 正则词库，返回第一个命中的词或 None。"""
        for word in _VOICEMAIL_REGEX_WORDS:
            if word in text:
                return word
        return None

    @staticmethod
    def _check_short_template_b(stage2_result: StageTwoResult) -> bool:
        """
        检查是否存在「字数极短的角色 B」。

        判定逻辑：
        1. 找出字数占比最小的 speaker（角色 B）
        2. 其总有效字数 < _SHORT_B_WORD_THRESHOLD
        """
        ifeats = stage2_result.interaction_features
        word_dist = ifeats.speaker_word_ratio
        if not word_dist:
            return False

        # 找字数占比最小的 speaker
        min_sid = min(word_dist, key=lambda s: word_dist.get(s, 1.0))

        # 计算该 speaker 的有效字数
        total_words = sum(
            t.word_count
            for t in stage2_result.dialogue_turns
            if t.speaker_id == min_sid and not t.is_backchannel
        )

        return total_words < 15 and total_words > 0
