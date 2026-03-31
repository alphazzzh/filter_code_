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

import math
import re
from collections import Counter
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
    GLOBAL_REDLINE_REGISTRY,
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
    ctx: _ScoringContext, 
) -> tuple[bool, float]:
    
    from config_topics import get_topics_by_category, TopicCategory
    
    whitelist_topics = get_topics_by_category(TopicCategory.WHITELIST)
    whitelist_hit_topics = {tid for tid in whitelist_topics if tid in all_intents}
    
    if not whitelist_hit_topics:
        return False, 1.0

    # 1. 案底感知：是否触发了高危诈骗探针
    hard_probe_hit = any(e["delta"] > 0 and not e["tag"].startswith("official") for e in ctx.events)
    
    # 2. 顺从度感知：获取受害者的最终顺从率
    ifeats = result.interaction_features
    compliance_rate = ifeats.compliance_rate if ifeats else 0.0

    # 👇 3. 锚点感知：提取所有受害者在【非敷衍轮次】中表现出的实质性意图
    potential_victims = [sr.speaker_id for sr in result.speaker_roles if sr.role != RoleLabel.AGENT]
    victim_intents = {
        label 
        for t in result.dialogue_turns
        if t.speaker_id in potential_victims and not t.is_backchannel
        for label in t.intent_labels
    }
    
    # 定义“致命顺从锚点”：只有受害者明确发出了遵从指令、确认动作或提供信息的意图时才算数
    fatal_compliance_anchors = {"compliance", "action_confirmation", "providing_info", "agreement"}
    has_fatal_anchor = bool(victim_intents & fatal_compliance_anchors)

    # 👇 核心撤销机制（多维联合判定）：
    # 必须同时满足：有诈骗起手式 + 顺从度较高 + 受害者确确实实做出了实质性的遵从动作
    if hard_probe_hit and compliance_rate >= 0.2 and has_fatal_anchor:
        return False, 1.0

    # 原有的句法拦截保持不变
    nlp_feats = result.metadata.get("nlp_features", {})
    if nlp_feats.get("has_imperative_syntax", False):
        return False, 1.0

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
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 零号优先级：全局红线前置熔断（Redline Pre-Circuit Breaker）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 在任何复杂矩阵运算之前，极速扫描底线安全词汇。
        # 触发时直接判死，绕过大模型全链路。
        full_text = " ".join(
            t.effective_text if hasattr(t, 'effective_text') and t.effective_text
            else (t.merged_text if hasattr(t, 'merged_text') else "")
            for t in stage2_result.dialogue_turns
            if not t.is_backchannel
        )
        if full_text:
            for redline_pattern in GLOBAL_REDLINE_REGISTRY:
                if redline_pattern.search(full_text):
                    return {
                        "conversation_id":      stage2_result.conversation_id,
                        "final_score":          100,
                        "tags":                 ["GLOBAL_REDLINE_ALERT"],
                        "tags_suppressed":      [],
                        "track_type":           stage2_result.track_type.value,
                        "roles":                {r.speaker_id: r.role.value for r in stage2_result.speaker_roles},
                        "nlp_features_summary": {},
                        "interaction_summary": {
                            "ping_pong_rate":   stage2_result.interaction_features.negotiation_ping_pong_rate,
                            "compliance_rate":  stage2_result.interaction_features.compliance_rate,
                            "resistance_decay": stage2_result.interaction_features.resistance_decay,
                            "word_distribution":stage2_result.interaction_features.speaker_word_ratio,
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

        ctx = _ScoringContext()

        # 🚨 缺陷 1 修复：阶段一硬正则极高危拦截兜底
        s1_meta = stage2_result.metadata.get("stage_one", {})
        if s1_meta.get("stage_one_critical_hit"):
            ctx.apply(40, "STAGE1_CRITICAL_FORCE_RECALL", "阶段一硬正则极高危拦截")

        all_intents: set[str]     = _collect_all_intents(stage2_result.dialogue_turns)
        nlp_feats:   dict[str, Any] = stage2_result.metadata.get("nlp_features", {})
        speaker_nlp_feats: dict[str, dict[str, Any]] = stage2_result.metadata.get("speaker_nlp_features", {})
        # 👇 1. 从 metadata 中安全提取 nlp_extra 和 filler_word_rate
        nlp_extra = stage2_result.metadata.get("nlp_features_extra", {})
        fwr = nlp_extra.get("filler_word_rate", 1.0)

        # 👇 2. 调用高级 Bot 引擎获取权威结论 (注意参数对应)
        bot_engine = BotConfidenceEngine()
        bot_eval = bot_engine.evaluate(stage2_result, filler_word_rate=fwr)
        is_bot_verdict = (bot_eval["bot_label"] == BotLabel.BOT)

        # 🚨 严格锁定骗子（Agent）的句法特征
        agent_sid = _find_role(stage2_result.speaker_roles, RoleLabel.AGENT)
        agent_nlp_feats = dict(nlp_feats) 
        if agent_sid and agent_sid in speaker_nlp_feats:
            # 👇 修复：智能合并。绝对不允许专属字典里的 False 覆盖掉全局的 True！
            for k, v in speaker_nlp_feats[agent_sid].items():
                if v:  # 只有当专属特征为 True 时，才合并进去
                    agent_nlp_feats[k] = v
        elif not agent_sid and speaker_nlp_feats:
            # 路径 2：平权聊天（没有明确 AGENT） -> 启动“动态嫌疑人推举”
            # 寻找个人特征中，命中高危动作最多的那个人，把他当做潜在嫌疑人，避免 AB 交叉污染！
            suspect_sid = None
            max_risk_flags = -1
            
            # 定义哪些句法属于强迫性动作
            risk_keys = {"has_imperative_syntax", "has_coercive_threat", "has_guide_behavior"}
            
            for sid, feats in speaker_nlp_feats.items():
                risk_count = sum(1 for k in risk_keys if feats.get(k))
                if risk_count > max_risk_flags:
                    max_risk_flags = risk_count
                    suspect_sid = sid
                    
            # 如果选出了嫌疑人，且他确实发出了高危动作，合并他的特征
            if suspect_sid and max_risk_flags > 0:
                for k, v in speaker_nlp_feats[suspect_sid].items():
                    if v: agent_nlp_feats[k] = v

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
        self._run_bot_intent_fusion(ctx, stage2_result, all_intents, is_bot_verdict)

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
            has_soft_intent = topic_id in all_intents
            topic_hit = False
            matrix_hit = False

            # Step B：矩阵组合扫描 (优先判断矩阵，不再提前 continue)
            for combo in topic_def.scoring_rules.matrix_combinations:
                # 核心逻辑：如果不是独立探针，且软意图也没命中，才跳过
                if not combo.is_independent and not has_soft_intent:
                    continue

                hard_feat_value = bool(agent_nlp_feats.get(combo.syntax_feature, False))
                
                triggered = False
                if combo.requires_absence and not hard_feat_value:
                    triggered = True  
                elif not combo.requires_absence and hard_feat_value:
                    triggered = True  

                if triggered:
                    ctx.apply(
                        delta  = combo.bonus_score,
                        tag    = combo.bonus_tag,
                        reason = (
                            f"矩阵命中 [{'!' if combo.requires_absence else ''}{combo.syntax_feature}] "
                            f"{'(独立触发)' if combo.is_independent else '× [' + topic_id + ']'} "
                            f"→ {combo.bonus_score:+d}"
                        ),
                    )
                    matrix_hit = True
                    topic_hit  = True

            # Step A：单项基础分（矩阵未触发，且大模型命中时才给）
            if not matrix_hit and has_soft_intent and topic_def.scoring_rules.standalone_score != 0:
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
            全局受害者抵抗降权机制（已修复平权聊天盲区与类型错误）：
            如果高危对话中潜在受害者明确拒绝/抵抗且顺从度极低，则视为失败的犯罪尝试，大幅扣分。
            同时追加「诈骗未遂」状态标签（scam_attempt_rejected），具有极高情报价值。
            """
            if not has_high_risk:
                return

            # 👇 核心修复：result.speaker_roles 是 list[SpeakerRoleResult]
            # 遍历对象列表，只要角色的 role 属性不是 AGENT，就提取他的 speaker_id
            potential_resisters = [
                sr.speaker_id for sr in result.speaker_roles
                if sr.role != RoleLabel.AGENT
            ]

            if not potential_resisters:
                return

            # 遍历所有潜在受害者，只要有一人成功抵抗，即可触发降权
            for target_sid in potential_resisters:
                target_turns = [
                    t for t in result.dialogue_turns
                    if t.speaker_id == target_sid and not t.is_backchannel
                ]
                if not target_turns:
                    continue

                # 收集该受害者的所有意图，方便快速判断是否包含绝对的质变信号
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
                            f"检测到高危风险，但疑似受害者({target_sid})明确拒绝/脱战"
                            f"（抵抗率={resistance_rate:.2f}或触发绝对识破），案件降级为未遂线索"
                        ),
                    )
                    # 追加「诈骗未遂」状态标签——情报分析的核心信号
                    ctx.tags.add(_SCAM_ATTEMPT_REJECTED_TAG)
                    
                    # 只要有一方成功抵抗，即刻生效并退出
                    return

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
        运行白名单豁免主题，并附加“多维洗脑撤销（Revocation）”机制。
        """
        # 调用多维校验函数获取是否允许豁免及折扣率
        is_wl, discount = _check_whitelist(result, all_intents, ctx)

        # 提取案底、顺从率及致命锚点特征，用于判断是否是被撤销（以便记录日志）
        hard_probe_hit = any(e["delta"] > 0 and not e["tag"].startswith("official") for e in ctx.events)
        ifeats = result.interaction_features
        compliance_rate = ifeats.compliance_rate if ifeats else 0.0

        potential_victims = [sr.speaker_id for sr in result.speaker_roles if sr.role != RoleLabel.AGENT]
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
                # 如果校验函数返回 False 且满足撤销条件，说明触发了多维联合撤销
                if not is_wl and hard_probe_hit and compliance_rate >= 0.2 and has_fatal_anchor:
                    hit_anchors = list(victim_intents & fatal_compliance_anchors)
                    ctx.apply(
                        delta  = 0,  
                        tag    = f"{td.scoring_rules.standalone_tag}_revoked",
                        reason = f"曾触发白名单[{tid}]，但检测到高危风险且受害者最终做出实质性妥协(compliance={compliance_rate:.2f}, 锚点={hit_anchors})，豁免被强制撤销！"
                    )
                    continue 
                
                # 如果校验通过且不存在句法拦截，正常应用白名单减分及折扣
                if is_wl:
                    # 👇 修复：绝不能把白名单自己的负分打折！直接使用原汁原味的负分进行核减
                    ctx.apply(
                        delta  = td.scoring_rules.standalone_score,
                        tag    = td.scoring_rules.standalone_tag,
                        reason = f"命中安全白名单 [{tid}]，强制降权防误杀"
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
        is_bot:      bool,
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


        # 👇 核心修复：直接听从高级引擎的结论，如果不是 Bot，直接退出！
        if not is_bot:
            return
        # # 检查机器人特征
        # is_bot_signal = False
        # s1_meta = result.metadata.get("stage_one", {})

        # # ── 短文本保护：计算非 backchannel 轮次总数和总字数 ──
        # non_bc_turns = [t for t in result.dialogue_turns if not t.is_backchannel]
        # total_words = sum(t.word_count for t in non_bc_turns)

        # # 信号 1：阶段一 bot_label == BOT
        # if s1_meta.get("bot_label") == BotLabel.BOT.value and total_words >= 30:
        #     is_bot_signal = True

        # # 信号 2：ping_pong_rate < 0.1（几乎无真正交互）
        # # 修复：只有当存在至少 4 个轮次时，ping_pong 为 0 才是不正常的机器单向输出
        # ppr = result.interaction_features.negotiation_ping_pong_rate
        # if ppr < 0.1 and len(non_bc_turns) >= 4:
        #     is_bot_signal = True

        # # 信号 3：filler_word_rate < 0.005（极度流畅）
        # # 修复：只有当对话总字数 >= 30 时，0 结巴率才有判定意义
        # nlp_extra = result.metadata.get("nlp_features_extra", {})
        # fwr = nlp_extra.get("filler_word_rate", 1.0)  # 默认 1.0（真人水平）
        # if fwr < 0.005 and total_words >= 30:
        #     is_bot_signal = True

        # if not is_bot_signal:
        #     return

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
        has_high_risk = any(
            e["delta"] > 0 and not e["tag"].startswith("official")
            for e in ctx.events
        )

        # 👇 新增：检查是否命中了强效安全白名单（分数 <= -40 且带有 safe/whitelist 标识）
        is_whitelisted = any(
            e["delta"] <= -40 and ("whitelist" in e["tag"] or "safe" in e["tag"])
            for e in ctx.events
        )

        # 👇 修改：如果存在超级白名单豁免，直接撤销高危底线锁，允许分数下探到安全区
        floor_score = IntelligenceScorer._HIGH_RISK_FLOOR if (has_high_risk and not is_whitelisted) else _SCORE_MIN

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
        total_words = sum(t.word_count for t in stage2_result.dialogue_turns if not t.is_backchannel)
        
        # 构建统一的上下文快照 (Runtime Evaluation Context)
        eval_context: dict[str, Any] = {
            "valid_turn_count": valid_turns,
            "compliance_rate":  ifeats.compliance_rate,
            "total_words":      total_words,
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

    最终得分 >= 70 判定为 BOT。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _BOT_THRESHOLD: int = 70

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
        
        # 👇 1. 新增：计算有效轮次和总字数，为后面的防误判做准备
        non_bc_turns = [t for t in stage2_result.dialogue_turns if not t.is_backchannel]
        valid_turns_count = len(non_bc_turns)
        total_words = sum(t.word_count for t in non_bc_turns)

        # 加分 1：命中 csr_bot_whitelist
        if "csr_bot_whitelist" in all_intents:
            score += self._SCORE_CSR_WHITELIST
            details.append(f"命中 csr_bot_whitelist → +{self._SCORE_CSR_WHITELIST}")

        # 加分 2：极度流畅（语气词极少）
        # 👇 2. 增加字数限制条件：and total_words >= 30
        if filler_word_rate < self._FLUENCY_THRESHOLD and total_words >= 30:
            score += self._SCORE_FLUENCY
            details.append(
                f"filler_word_rate={filler_word_rate:.4f} < {self._FLUENCY_THRESHOLD} (字数={total_words}) → +{self._SCORE_FLUENCY}"
            )

        # 加分 3：ping_pong_rate 极度规律
        ppr = stage2_result.interaction_features.negotiation_ping_pong_rate
        # 👇 3. 增加轮次限制条件：and valid_turns_count >= 4
        if ppr < 0.05 and valid_turns_count >= 4:
            score += self._SCORE_PING_PONG
            details.append(
                f"ping_pong_rate={ppr:.4f} < 0.05 (轮次={valid_turns_count}) → +{self._SCORE_PING_PONG}"
            )

        # ── 最终判定 ──────────────────────────────────────
        bot_label = BotLabel.BOT if score >= self._BOT_THRESHOLD else BotLabel.HUMAN
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

# ── 轻量级 TF-IDF 余弦相似度（无需 GPU / BGE 模型）────────────

# 预编译中文分词正则（字符级 bigram，适用于中文/日文等多语言场景）
_RE_TOKENIZER: re.Pattern = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[a-zA-Z]+",
    re.UNICODE,
)


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    轻量级 TF-IDF 余弦相似度计算。

    使用字符级 bigram + IDF 近似（基于文档频率的简单折扣），
    无需外部依赖，O(n) 时间复杂度。

    适用场景：跨轮次连贯度判定（两句话是否在语义上相关）
    不适用：需要深度语义理解的场景（请使用 BGE-M3）
    """
    def _tokenize(text: str) -> list[str]:
        return _RE_TOKENIZER.findall(text.lower())

    def _ngrams(tokens: list[str]) -> Counter[str]:
        grams: list[str] = []
        for i in range(len(tokens) - 1):
            grams.append(f"{tokens[i]}_{tokens[i + 1]}")
        return Counter(grams)

    ngrams_a = _ngrams(_tokenize(text_a))
    ngrams_b = _ngrams(_tokenize(text_b))

    if not ngrams_a or not ngrams_b:
        return 0.0

    # 简单 IDF 折扣：仅出现在一个文档中的 bigram 权重更高
    all_keys = set(ngrams_a.keys()) | set(ngrams_b.keys())
    idf: dict[str, float] = {}
    for k in all_keys:
        df = (1 if k in ngrams_a else 0) + (1 if k in ngrams_b else 0)
        idf[k] = math.log(2.0 / df) + 1.0  # 平滑 IDF

    # TF-IDF 加权向量
    def _to_tfidf_vec(ngrams: Counter[str]) -> dict[str, float]:
        total = sum(ngrams.values()) or 1.0
        return {k: (v / total) * idf[k] for k, v in ngrams.items()}

    vec_a = _to_tfidf_vec(ngrams_a)
    vec_b = _to_tfidf_vec(ngrams_b)

    # 余弦相似度
    dot_product = sum(vec_a[k] * vec_b.get(k, 0.0) for k in vec_a)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return dot_product / (norm_a * norm_b)


class AdvancedVoicemailDetector:
    """
    高级无效通话（Voicemail/语音信箱/未接通）检测引擎。

    V5.1 重构：废除字数统计防线，引入跨轮次语义连贯度测算。

    评分规则（基础分 0，叠加计分）：
      - 命中 Voicemail 正则词                → +60
      - 角色 B 字数极短且符合模板             → +30
      - 交互判定为 is_decoupled（解耦盲说）   → +20

    一票否决（得分归零）：
      - ping_pong_rate > 0.1 且不是解耦状态 → 判定为有效通话
      - 跨轮次语义连贯度 > 0.35 → 检测到真交互，推翻无效通话判定

    最终得分 > 80 判定为无效通话。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _VOICEMAIL_THRESHOLD: int = 80

    # ── 加分权重 ──────────────────────────────────────────
    _SCORE_VOICEMAIL_WORD:   int = 60
    _SCORE_SHORT_TEMPLATE_B: int = 30
    _SCORE_DECOUPLED:        int = 20

    # ── 语义连贯度阈值 ──────────────────────────────────
    _CROSS_TURN_SIM_THRESHOLD: float = 0.35  # 跨轮次语义耦合阈值

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

        # ── 一票否决检查 1：ping_pong_rate ──────────────────
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

        # ── 一票否决检查 2：跨轮次语义连贯度防线 ──────────
        # 废除旧的字数比例防线，改用 TF-IDF 语义耦合检测
        max_cross_turn_sim: float = 0.0
        valid_turns = [
            t for t in stage2_result.dialogue_turns
            if not t.is_backchannel and t.merged_text.strip()
        ]

        for i in range(len(valid_turns) - 1):
            turn_a = valid_turns[i]
            turn_b = valid_turns[i + 1]
            # 只计算相邻且角色不同的轮次
            if turn_a.speaker_id == turn_b.speaker_id:
                continue
            sim = _tfidf_cosine_similarity(
                turn_a.merged_text, turn_b.merged_text
            )
            if sim > max_cross_turn_sim:
                max_cross_turn_sim = sim

        is_voicemail = score > self._VOICEMAIL_THRESHOLD

        if is_voicemail and max_cross_turn_sim > self._CROSS_TURN_SIM_THRESHOLD:
            # 语义防线一票否决：检测到跨角色语义高度耦合，推翻无效通话判定
            veto_reason = (
                f"触发连贯度防线：检测到跨角色语义高度耦合 "
                f"(sim={max_cross_turn_sim:.4f} > {self._CROSS_TURN_SIM_THRESHOLD})，"
                f"推翻单向盲播判定"
            )
            details.append(veto_reason)
            return {
                "voicemail_score": score,
                "is_voicemail":    False,
                "veto_reason":     veto_reason,
                "details":         details,
            }

        details.append(
            f"跨轮次语义连贯度={max_cross_turn_sim:.4f}，"
            f"阈值={self._CROSS_TURN_SIM_THRESHOLD}，"
            f"最终得分={score}，判定={'无效通话' if is_voicemail else '有效通话'}"
        )

        return {
            "voicemail_score": score,
            "is_voicemail":  is_voicemail,
            "veto_reason":    None,
            "details":        details,
            "cross_turn_sim": round(max_cross_turn_sim, 4),
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
