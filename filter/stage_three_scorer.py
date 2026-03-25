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
)


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
                hard_feat_value = bool(nlp_feats.get(combo.syntax_feature, False))
                
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOW_VALUE_NOISE 降权扣分
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_noise_topics(self, ctx: _ScoringContext, all_intents: set[str], nlp_feats: dict[str, Any]) -> None:
            for topic_id, topic_def in self._noise_topics.items():
                if topic_id in all_intents:
                    # [原有逻辑保留]：抑制物流对毒品的误报
                    if topic_id == "corporate_logistics" and not nlp_feats.get("has_drug_quantity"):
                        _events_to_remove = []
                        for e in ctx.events:
                            if e["tag"].startswith("has_drug_") or e["tag"].startswith("drug_quantity_"):
                                ctx.score -= e["delta"]
                                _events_to_remove.append(e)
                        for e in _events_to_remove:
                            ctx.events.remove(e)

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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 输出组装
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_output(
        ctx:       _ScoringContext,
        result:    StageTwoResult,
        nlp_feats: dict[str, Any],
    ) -> dict[str, Any]:
        """边界钳位 [0,100] + 结构化情报字典组装。"""
        final_score = max(_SCORE_MIN, min(_SCORE_MAX, ctx.score))

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

        return {
            "conversation_id":      result.conversation_id,
            "final_score":          final_score,
            "tags":                 sorted(ctx.tags),
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
