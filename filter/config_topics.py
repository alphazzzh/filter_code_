# config_topics.py  ── V5.0 配置驱动架构
# ============================================================
# 主题配置中枢（Single Source of Truth）
#
# 架构角色
# ─────────────────────────────────────────────────────────────
# 本文件是整条流水线的「唯一真相来源」。
# 三个执行引擎（IntentRadar / SyntaxFeatureExtractor /
# IntelligenceScorer）均从此处动态加载配置，
# 内部不再硬编码任何词表、阈值或矩阵。
#
# 扩展方法（零代码修改执行层）
# ─────────────────────────────────────────────────────────────
# 添加新主题只需：
#   1. 在本文件末尾追加一个 TopicDefinition 实例
#   2. 在 TOPIC_REGISTRY 中注册
#   3. 无需修改 intent_radar.py / stage_two_pipeline.py /
#      stage_three_scorer.py 中的任何代码
#
# 主题类别
# ─────────────────────────────────────────────────────────────
# HIGH_RISK      : 风险主题，触发加分
# LOW_VALUE_NOISE: 低价值噪声，触发降权
# WHITELIST      : 白名单，触发豁免折扣
# EXEMPTION      : 豁免信号（如受害者抵抗），触发减分
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import json
import re
from pathlib import Path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 枚举：主题类别 & 句法规则类型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TopicCategory(str, Enum):
    HIGH_RISK       = "HIGH_RISK"        # 风险主题，触发加分
    LOW_VALUE_NOISE = "LOW_VALUE_NOISE"  # 低价值噪声，触发降权扣分
    WHITELIST       = "WHITELIST"        # 白名单，触发豁免
    EXEMPTION       = "EXEMPTION"        # 豁免信号（受害者侧信号）


class SyntaxRuleType(str, Enum):
    """
    句法硬特征提取规则类型。
    SyntaxFeatureExtractor 根据此类型分发到不同的提取器。
    """
    IMPERATIVE_SYNTAX = "imperative_syntax"
    # 依存句法：主语是第二人称 + 紧迫状语修饰核心谓语
    # params: second_person (list[str]), urgency_adverbs (list[str])
    # 产出: feature_name → bool, imperative_verbs_key → list[str]

    QUANTITY_REGEX = "quantity_regex"
    # 数字+量词正则：匹配数量级描述
    # params: quantity_units (list[str])
    # 产出: feature_name → bool, matches_key → list[str]

    NER_DENSITY = "ner_density"
    # NER 实体密度：统计指定类型实体数量是否达到阈值
    # params: entity_types (list[str]), threshold (int)
    # 产出: feature_name → bool, entities_key → list[str]

    KEYWORD_COOC = "keyword_cooc"
    # 关键词共现：多个关键词集合均有命中
    # params: keyword_sets (list[list[str]]) - 每个子列表至少命中一个
    # 产出: feature_name → bool

    REGEX_PATTERN = "regex_pattern"
    # 自定义正则模式
    # params: pattern (str), flags (str, 默认 "UNICODE")
    # 产出: feature_name → bool, matches_key → list[str]

    VERB_ENTITY_SPARSITY = "verb_entity_sparsity"
    # 实体与动词稀疏度检测



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SyntaxRuleConfig:
    """
    单条句法硬特征提取规则的完整配置。

    Attributes
    ----------
    rule_type    : SyntaxRuleType 枚举值，决定分发到哪个提取器
    feature_name : 本规则产出的布尔特征键名（写入 NlpFeatures dict）
    params       : 规则类型专属参数，见 SyntaxRuleType 注释
    evidence_key : 可选，产出的「证据列表」键名（如匹配字符串、触发词汇）
    """
    rule_type:    SyntaxRuleType
    feature_name: str
    params:       dict[str, Any]    = field(default_factory=dict)
    evidence_key: str | None        = None  # 存放证据列表的键名，None=不保存证据


@dataclass
class OODFallbackRule:
    """
    全局物理废料兜底规则 (OOD Fallback)
    专门拦截大模型雷达未能覆盖的“未知领域废话”
    """
    rule_id: str
    delta: int
    tag: str
    reason: str
    # condition 接收一个扁平化的运行时上下文(字典)，返回布尔值决定是否触发
    condition: Callable[[dict[str, Any]], bool]


@dataclass
class MatrixCombination:
    """
    矩阵组合单元格：一条硬特征 × 软意图的共现打分规则。

    Attributes
    ----------
    syntax_feature : SyntaxRuleConfig.feature_name（硬特征键）
    bonus_score    : 命中时的加分值
    bonus_tag      : 命中时添加到 tags 的标签
    """
    syntax_feature: str
    bonus_score:    int
    bonus_tag:      str
    # 如果为 True，则表示当该硬特征【不存在】时才触发此矩阵
    requires_absence: bool = False
    is_independent: bool = False


@dataclass
class ScoringRules:
    """
    单个主题的完整打分规则。

    Attributes
    ----------
    standalone_score    : 仅软意图命中（无硬特征强化）时的基础分
                          HIGH_RISK > 0；LOW_VALUE_NOISE < 0；WHITELIST/EXEMPTION 特殊处理
    standalone_tag      : 单项命中时添加的标签
    matrix_combinations : 矩阵组合列表；多个组合均触发时叠加计分
    whitelist_discount  : WHITELIST 类别专用，触发时整体折扣比例（0=全免，0.4=保留40%）
    """
    standalone_score:    int                     = 0
    standalone_tag:      str                     = ""
    matrix_combinations: list[MatrixCombination] = field(default_factory=list)
    whitelist_discount:  float                   = 1.0  # 默认不折扣


@dataclass
class TopicDefinition:
    """
    单个主题的全量配置，是整个配置驱动架构的核心数据结构。

    生命周期
    ─────────────────────────────────────────────────────────
    __init__ 时数据不可变，各执行引擎仅读取，不修改。

    Attributes
    ----------
    topic_id      : 全局唯一主题标识符（snake_case）
    category      : 主题类别
    description   : 人类可读描述，用于文档和日志
    bge_anchors   : BGE-M3 向量锚点句子列表（每组 8~15 句）
    threshold     : 该主题 BGE-M3 触发阈值（余弦相似度）
    syntax_rules  : 句法硬特征规则列表（可为空）
    scoring_rules : 打分规则
    """
    topic_id:     str
    category:     TopicCategory
    description:  str
    bge_anchors:  list[str]
    threshold:    float
    syntax_rules: list[SyntaxRuleConfig]    = field(default_factory=list)
    scoring_rules: ScoringRules             = field(default_factory=ScoringRules)



# ─────────────────────────────────────────────────────────────
# BGE-M3 向量锚点词汇外置加载器
# ─────────────────────────────────────────────────────────────
# 词汇数据存储在 config_topics_anchors.json 中。
# 模块加载时一次性读取，通过 _inject_anchors() 注入各 TopicDefinition。
# ─────────────────────────────────────────────────────────────

_ANCHORS_PATH = Path(__file__).parent / "config_topics_anchors.json"

def _load_anchors() -> dict[str, list[str]]:
    """从 JSON 文件加载所有主题的 BGE-M3 向量锚点词汇。"""
    with open(_ANCHORS_PATH, "r", encoding="utf-8") as _f:
        return json.load(_f)

_RAW_ANCHORS: dict[str, list[str]] = _load_anchors()


def _inject_anchors() -> None:
    """将 _RAW_ANCHORS 中的词汇注入到 TOPIC_REGISTRY 各 TopicDefinition 实例。"""
    for tid, topic_def in TOPIC_REGISTRY.items():
        if tid in _RAW_ANCHORS:
            object.__setattr__(topic_def, "bge_anchors", _RAW_ANCHORS[tid])


# ─────────────────────────────────────────────────────────────
# 全局物理兜底规则注册表
# ─────────────────────────────────────────────────────────────
OOD_FALLBACK_REGISTRY: list[OODFallbackRule] = [
    OODFallbackRule(
        rule_id="entity_sparsity",
        delta=-9,
        tag="global_business_sparse",
        reason="未命中任何高危主题，且全局业务实体极度稀疏，判定为无实质内容",
        # 依赖 NLP 硬句法探针的特征
        condition=lambda ctx: ctx.get("is_business_sparse", False)
    ),
    OODFallbackRule(
        rule_id="too_short_interaction",
        delta=-18,
        tag="global_too_short",
        reason="未命中高危主题，且有效交互轮次过少，判定为碎片废料",
        # 依赖拓扑引擎计算的有效交互轮次
        condition=lambda ctx: ctx.get("valid_turn_count", 0) <= 3
    ),
    OODFallbackRule(
        rule_id="monologue_noise",
        delta=-6,
        tag="global_monologue_noise",
        reason="无效沟通：单方面输出且无实质性互动响应（如推销失败/自言自语）",
        # 依赖多维度的博弈互动指标综合判断
        condition=lambda ctx: (
            ctx.get("valid_turn_count", 0) > 3 and
            ctx.get("compliance_rate", 1.0) == 0.0 and
            ctx.get("ping_pong_rate", 1.0) < 0.1
        )
    )
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主题配置定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 1：电信诈骗（分三个子槽位）
# ─────────────────────────────────────────────────────────────

_TOPIC_FRAUD_JARGON = TopicDefinition(
    topic_id    = "fraud_jargon",
    category    = TopicCategory.HIGH_RISK,
    description = "诈骗黑话/地下产业链隐语（跑分/水房/卡接等），脱离上下文即高度可疑",
    threshold   = 0.70,  # 黑话锚点特异性强，适当降低阈值提升召回
        bge_anchors=[],
    syntax_rules = [],  # 黑话本身即强信号，无需额外句法规则
    scoring_rules = ScoringRules(
        standalone_score = 20,
        standalone_tag   = "has_fraud_jargon",
        matrix_combinations = [
            MatrixCombination("has_imperative_syntax", 30, "fraud_imperative_jargon"),
            MatrixCombination("high_entity_density",   18, "fraud_jargon_dense"),
        ],
    ),
)

_TOPIC_FRAUD_OBJECT = TopicDefinition(
    topic_id    = "fraud_object",
    category    = TopicCategory.HIGH_RISK,
    description = "诈骗高危业务客体（验证码/屏幕共享/账户转账等），是诈骗指令的「宾语」成分",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperative_verbs",
            params = {
                "second_person":  ["你", "您", "you"],
                "urgency_adverbs": [
                    "马上", "立刻", "立即", "赶紧", "赶快",
                    "现在", "快", "即刻", "迅速",
                    "now", "immediately", "right now", "asap",
                ],
            },
        ),
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = {
                "entity_types": ["Ni", "Ns", "ORG", "GPE", "LOC"],
                "threshold":    3,
            },
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 18,
        standalone_tag   = "has_fraud_object",
        matrix_combinations = [
            # 祈使句 × 诈骗客体 = 「你现在立刻把验证码发给我」→ 最高危
            MatrixCombination("has_imperative_syntax", 45, "fraud_imperative_object"),
            MatrixCombination("high_entity_density",   15, "fraud_object_dense"),
        ],
    ),
)

_TOPIC_AUTHORITY_ENTITY = TopicDefinition(
    topic_id    = "authority_entity",
    category    = TopicCategory.HIGH_RISK,
    description = "权威伪装实体（冒充公检法/监管机构），制造权威压迫感",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        # 复用 has_imperative_syntax 和 high_entity_density（与 fraud_object 共享规则）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperative_verbs",
            params = {
                "second_person":  ["你", "您", "you"],
                "urgency_adverbs": [
                    "马上", "立刻", "立即", "赶紧", "赶快",
                    "现在", "快", "即刻", "迅速",
                    "now", "immediately", "right now",
                ],
            },
        ),
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = {
                "entity_types": ["Ni", "Ns", "ORG", "GPE", "LOC"],
                "threshold":    3,
            },
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 22,
        standalone_tag   = "has_authority_entity",
        matrix_combinations = [
            MatrixCombination("has_imperative_syntax", 35, "fraud_authority_pressure"),
            MatrixCombination("high_entity_density",   28, "fraud_authority_entity_dense"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 2：涉毒交易（两个子槽位）
# ─────────────────────────────────────────────────────────────

_TOPIC_DRUG_JARGON = TopicDefinition(
    topic_id    = "drug_jargon",
    category    = TopicCategory.HIGH_RISK,
    description = "毒品隐语（白/冰/K粉/草等），写场景句提高语境特异性",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.QUANTITY_REGEX,
            feature_name = "has_drug_quantity",
            evidence_key = "drug_quantity_matches",
            params = {
                "quantity_units": [
                    "克", "g", "G", "公克", "mg",
                    "包", "手", "份", "颗", "粒",
                ],
            },
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 22,
        standalone_tag   = "has_drug_jargon",
        matrix_combinations = [
            # 数量 × 毒品隐语 = 「5克白的」→ 极高危
            MatrixCombination("has_drug_quantity", 50, "drug_quantity_jargon"),
            MatrixCombination("high_entity_density", 20, "drug_jargon_dense"),
        ],
    ),
)

_TOPIC_DRUG_CHAIN = TopicDefinition(
    topic_id    = "drug_chain",
    category    = TopicCategory.HIGH_RISK,
    description = "毒品交易链条（上下游关系/物流/货款等），与 drug_jargon + has_drug_quantity 共现还原完整交易模型",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.QUANTITY_REGEX,
            feature_name = "has_drug_quantity",
            evidence_key = "drug_quantity_matches",
            params = {
                "quantity_units": [
                    "克", "g", "G", "公克", "mg",
                    "包", "手", "份", "颗", "粒",
                ],
            },
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,
        standalone_tag   = "has_drug_chain",
        matrix_combinations = [
            MatrixCombination("has_drug_quantity",   35, "drug_quantity_chain"),
            MatrixCombination("high_entity_density", 12, "drug_chain_dense"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 3：有组织强制控制
# ─────────────────────────────────────────────────────────────
# 设计说明：
# 聚焦「行为模式」而非「信仰身份」，锚点描述的是：
#   1. 金融勒索（被迫捐款/缴费，附带威胁后果）
#   2. 孤立施压（威胁切断家庭/工作关系以强化控制）
#   3. 退出威胁（以灾难/惩罚威胁离开行为）
#   4. 批量话术广播（统一口径，协调传播）
# 这些是可识别的犯罪控制行为，与任何具体团体/信仰无关。
# ─────────────────────────────────────────────────────────────

_TOPIC_COERCIVE_ORG_CONTROL = TopicDefinition(
    topic_id    = "coercive_org_control",
    category    = TopicCategory.HIGH_RISK,
    description = (
        "有组织强制控制行为：通过金融勒索/孤立威胁/退出惩罚实施心理控制，"
        "聚焦犯罪行为模式，与具体团体名称或信仰内容无关"
    ),
    threshold   = 0.74,
        bge_anchors=[],
    syntax_rules = [
        # 🚨 缺陷 3 修复：收紧硬句法，仅保留绝对胁迫词，禁止模糊匹配
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.KEYWORD_COOC,
            feature_name = "has_coercive_threat",
            params = {
                "keyword_sets": [
                    ["如果不", "否则", "一旦"],
                ],
            },
            evidence_key = None,
        ),
        # 检测「财务勒索」：金融词 + 威胁/强迫词共现
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.KEYWORD_COOC,
            feature_name = "has_coercive_financial_demand",
            params = {
                "keyword_sets": [
                    # 集合 A：金融/财务词
                    ["捐", "缴", "交钱", "付款", "费用", "资金", "献"],
                    # 集合 B：强迫/威胁词
                    ["必须", "否则", "不然", "要不然", "后果", "惩罚",
                     "报应", "灾祸", "不交就"],
                ],
            },
            evidence_key = None,
        ),
        # 检测「退出威胁」：离开词 + 惩罚后果词共现
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.KEYWORD_COOC,
            feature_name = "has_exit_threat",
            params = {
                "keyword_sets": [
                    # 集合 A：退出/离开词
                    ["退出", "离开", "背叛", "不服从", "反对"],
                    # 集合 B：惩罚/后果词
                    ["惩罚", "报应", "后果", "驱逐", "出事", "灾难"],
                ],
            },
            evidence_key = None,
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 28,
        standalone_tag   = "coercive_org_behavior",
        matrix_combinations = [
            # 金融勒索 × 软意图强化 = 极高危
            MatrixCombination("has_coercive_financial_demand", 40, "critical_coercive_financial"),
            MatrixCombination("has_exit_threat",               30, "critical_exit_coercion"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 3.5：极端思想与邪教传播
# ─────────────────────────────────────────────────────────────
# 设计说明：
# 锚点词库聚焦已知邪教组织标识性用语与极端末世论传播话术。
# 无需强依赖硬特征矩阵，只要命中即触发强预警（standalone_score=40）。

_TOPIC_EXTREMIST_PROPAGANDA = TopicDefinition(
    topic_id    = "extremist_propaganda",
    category    = TopicCategory.HIGH_RISK,
    description = "极端思想与邪教传播：已知邪教组织标识性用语、末世论、度人话术等",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules = [],
    scoring_rules = ScoringRules(
        standalone_score = 40,
        standalone_tag   = "extremist_propaganda",
        matrix_combinations = [
            MatrixCombination("high_entity_density", 10, "extremist_with_dense_entities"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 4：有组织广播行为
# ─────────────────────────────────────────────────────────────

_TOPIC_COORDINATED_BROADCAST = TopicDefinition(
    topic_id    = "coordinated_broadcast",
    category    = TopicCategory.HIGH_RISK,
    description = "有组织批量传播行为：指挥他人大规模、统一口径传播，与内容立场无关",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = {
                "entity_types": ["Ni", "Ns", "ORG", "GPE", "LOC"],
                "threshold":    3,
            },
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 30,
        standalone_tag   = "coordinated_broadcast",
        matrix_combinations = [
            MatrixCombination("high_entity_density", 15, "broadcast_with_dense_entities"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 5：明确煽动暴力
# ─────────────────────────────────────────────────────────────

_TOPIC_INCITEMENT_TO_VIOLENCE = TopicDefinition(
    topic_id    = "incitement_to_violence",
    category    = TopicCategory.HIGH_RISK,
    description = "明确号召对具体目标实施身体伤害，排除比喻性/游戏化用法",
    threshold   = 0.76,  # 高精确率阈值，避免误判比喻
        bge_anchors=[],
    syntax_rules = [],
    scoring_rules = ScoringRules(
        standalone_score = 40,
        standalone_tag   = "incitement_to_violence",
        matrix_combinations = [
            MatrixCombination("high_entity_density", 10, "violence_with_dense_entities"),
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# 通用交互主题（保留，供角色绑定器使用）
# ─────────────────────────────────────────────────────────────

_TOPIC_EMOTION = TopicDefinition(
    topic_id    = "emotion",
    category    = TopicCategory.HIGH_RISK,  # 高情绪注入是杀猪盘信号
    description = "情绪价值输出/嘘寒问暖，高频出现时为「杀猪盘」长线经营信号",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(standalone_score=0),  # 仅用于角色拓扑计算，不直接加分
)

_TOPIC_COMPLIANCE = TopicDefinition(
    topic_id    = "compliance",
    category    = TopicCategory.HIGH_RISK,
    description = "顺从/同意响应，用于计算 compliance_rate 和 Follower 判定",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(standalone_score=0),
)

_TOPIC_INTERROGATION = TopicDefinition(
    topic_id    = "interrogation",
    category    = TopicCategory.HIGH_RISK,
    description = "提问压制/控制节奏，用于识别 AGENT 身份",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(standalone_score=0),
)


# ─────────────────────────────────────────────────────────────
# EXEMPTION 主题：受害者主动抵抗（摆烂豁免信号）
# ─────────────────────────────────────────────────────────────

_TOPIC_REJECTION = TopicDefinition(
    topic_id    = "rejection",
    category    = TopicCategory.EXEMPTION,
    description = "受害者明确拒绝业务（如：不需要、不用了、挂了）",
    threshold   = 0.75,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -10,
        standalone_tag   = "target_rejection",
    ),
)

_TOPIC_DISMISSAL = TopicDefinition(
    topic_id    = "dismissal",
    category    = TopicCategory.EXEMPTION,
    description = "受害者主动识破/拒绝配合信号，触发时降低「受控压制」评分",
    threshold   = 0.78,  # 高精确率，避免误豁免
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -15,   # 命中时从总分减 15（控制失效信号）
        standalone_tag   = "target_dismissal_active",
    ),
)


# ─────────────────────────────────────────────────────────────
# WHITELIST 主题：正规客服机器人
# ─────────────────────────────────────────────────────────────

_TOPIC_INBOUND_OFFICIAL_IVR = TopicDefinition(
    topic_id    = "inbound_official_ivr",
    category    = TopicCategory.WHITELIST,
    description = "官方客服/电信运营商呼入导航菜单（超级白名单）",
    threshold   = 0.75,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -60,
        standalone_tag   = "official_telecom_inbound_whitelist",
    ),
)

_TOPIC_INBOUND_USER_REQUEST = TopicDefinition(
    topic_id    = "inbound_user_request",
    category    = TopicCategory.WHITELIST,
    description = "用户主动呼入请求（咨询/投诉/预约/查询），非诈骗外呼，触发豁免防误杀",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -60,
        standalone_tag   = "safe_inbound_user_request",
    ),
)

_TOPIC_CSR_BOT_WHITELIST = TopicDefinition(
    topic_id    = "csr_bot_whitelist",
    category    = TopicCategory.WHITELIST,
    description = (
        "正规客服机器人白名单：仅含合规业务意图且阶段一已判定 bot_label=bot，"
        "触发时对风险矩阵分数应用大幅折扣"
    ),
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        whitelist_discount = 0.05,  # 最终分数 × 0.05，几乎清零
        standalone_tag     = "whitelist_csr_bot",
    ),
)


# ─────────────────────────────────────────────────────────────
# LOW_VALUE_NOISE 主题：工业报表 / 业务汇报
# ─────────────────────────────────────────────────────────────

_TOPIC_E_COMMERCE_CS = TopicDefinition(
    topic_id    = "e_commerce_cs",
    category    = TopicCategory.HIGH_RISK,
    description = "电商/泛金融客服伪装起手式，配合诈骗客体或高压句法形成连招",
    threshold   = 0.72,
    bge_anchors=[],
    syntax_rules = [
        # 👇 补上这个专门针对微保/百万保障诈骗的核爆级硬探针
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.REGEX_PATTERN,
            feature_name = "has_insurance_scam_keywords",
            params       = {
                "pattern": r"(百万保障|微保|安全保险|账户保险|资金安全险).{0,30}(到期|收费|扣费|解除|关闭|续费)|(微信|支付宝|拼多多).{0,30}(百万保障|微保|客服中心)"
            }
        )
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,
        standalone_tag   = "suspicious_fake_cs",
        matrix_combinations = [
            MatrixCombination("has_fraud_object", 45, "fake_cs_screen_share_trap"),
            MatrixCombination("has_coercive_threat", 40, "fake_cs_financial_threat"),
            # 👇 将其设为独立触发！
            MatrixCombination("has_insurance_scam_keywords", 50, "critical_insurance_scam", is_independent=True),
        ],
    ),
)

_TOPIC_MASS_GRIEVANCE = TopicDefinition(
    topic_id    = "mass_grievance",
    category    = TopicCategory.HIGH_RISK,
    description = "涉军/群体维权与涉稳信访",
    threshold   = 0.68,
        bge_anchors=[],
    scoring_rules = ScoringRules(
        standalone_score = 25,
        standalone_tag   = "social_grievance_petition",
        matrix_combinations = [
            MatrixCombination("coordinated_broadcast", 40, "CRITICAL_MASS_INCIDENT_MOBILIZATION"),
            MatrixCombination("incitement_to_violence", 40, "CRITICAL_VIOLENT_PROTEST_RISK"),
        ],
    ),
)

_TOPIC_VOICEMAIL_IVR = TopicDefinition(
    topic_id    = "voicemail_ivr",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "运营商长文本语音信箱/未接通提示音",
    threshold   = 0.68,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -40,
        standalone_tag   = "unconnected_voicemail_ivr",
    ),
)

_TOPIC_CORPORATE_LOGISTICS = TopicDefinition(
    topic_id    = "corporate_logistics",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "企业物流报表，包含寄送样品/仓库/入库等，避免被误判为毒品交易",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.VERB_ENTITY_SPARSITY,
            feature_name = "is_business_sparse",
            params       = {"threshold": 2}, # 实体极少
        )
    ],
    scoring_rules = ScoringRules(
            standalone_score = -4,   # 微降权（原 -15）
            standalone_tag   = "corporate_logistics_noise",
            matrix_combinations = [
                MatrixCombination("has_coercive_threat", -15, "logistics_no_threat_noise", requires_absence=True),
            ]
        ),
)

_TOPIC_CORPORATE_BIDDING = TopicDefinition(
    topic_id    = "corporate_bidding",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "企业招投标/商务谈判，避免被误判为高压诈骗",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
            standalone_score = -15,
            standalone_tag   = "corporate_bidding_noise",
            matrix_combinations = [
                # 商务扯皮中无胁迫，大概率安全
                MatrixCombination("has_coercive_threat", -15, "bidding_no_threat_safe", requires_absence=True),
            ]
        ),
)

_TOPIC_INDUSTRIAL_REPORT = TopicDefinition(
    topic_id    = "industrial_report",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "工业/业务汇报类对话（产量/库存/报表），对情报分析价值极低",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -5,    # 微降权，保留可能含实质内容的工业对话
        standalone_tag   = "low_value_industrial_noise",
    ),
)


# ─────────────────────────────────────────────────────────────
# LOW_VALUE_NOISE 主题：日常闲聊
# ─────────────────────────────────────────────────────────────

_TOPIC_CASUAL_CHAT = TopicDefinition(
    topic_id    = "casual_chat",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "日常生活闲聊（天气/吃饭/娱乐），无情报价值",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
            standalone_score = -3,   # 微降权（原 -10）
            standalone_tag   = "low_value_casual_chat",
            matrix_combinations = [
                # 闲聊且没有业务名词，双重确认是废话，狠狠打入深渊
                MatrixCombination("is_business_sparse", -20, "casual_chat_extremely_sparse"),
            ]
        ),
)


# 日常无价值场景扩充 (针对 50 分基线优化)
_TOPIC_DELIVERY_EXPRESS = TopicDefinition(
    topic_id    = "delivery_express",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "外卖/快递日常沟通，用于将分数打压至 50 分以下（阈值 30）",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
            standalone_score = -6,   # 微降权（原 -10）
            standalone_tag   = "low_value_casual_chat",
            matrix_combinations = [
                # 闲聊且没有业务名词，双重确认是废话，狠狠打入深渊
                MatrixCombination("is_business_sparse", -20, "casual_chat_extremely_sparse"),
            ]
        ),
)

_TOPIC_WRONG_NUMBER = TopicDefinition(
    topic_id    = "wrong_number",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "打错电话/找错人的纯废话",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -8,   # 微降权（原 -10）
        standalone_tag   = "low_value_casual_chat",
        matrix_combinations = [
            # 闲聊且没有业务名词，绝对的纯废话
            MatrixCombination("is_business_sparse", -25, "casual_chat_extremely_sparse"),
        ]
    ),
)

_TOPIC_BRUSH_OFF_TELEMARKETING = TopicDefinition(
    topic_id    = "brush_off_telemarketing",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "敷衍推销类（拒绝贷款/买房/办卡等日常推销骚扰）",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -15, 
        standalone_tag   = "noise_brush_off_telemarketing",
        matrix_combinations = [
            # 听起来像推销，且没有实质性的恐吓，彻底踩死
            MatrixCombination("has_coercive_threat", -20, "telemarketing_no_threat_trash", requires_absence=True),
        ]
    ),
)

_TOPIC_SHORT_GREETING_HANGUP = TopicDefinition(
    topic_id    = "short_greeting_hangup",
    category    = TopicCategory.LOW_VALUE_NOISE,
    description = "极短寒暄或喂喂喂无后续，补充第一阶段未拦截干净的碎片",
    threshold   = 0.75,
        bge_anchors=[],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -20,  # 50 - 20 = 30分
        standalone_tag   = "noise_short_greeting_hangup",
    ),
)

_TOPIC_GLOBAL_SYNTAX_REGISTRY = TopicDefinition(
    topic_id    = "global_syntax_registry",
    category    = TopicCategory.WHITELIST,  
    description = "【虚拟主题】不参与匹配，仅作为全局公用句法探针的存放仓库",
    threshold   = 0.99,  # 极高阈值，永远不会被独立命中
        bge_anchors=[],
    syntax_rules  = [
        # 把实体稀疏探针放在这里，供所有人白嫖
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.VERB_ENTITY_SPARSITY,
            feature_name = "is_business_sparse",
            params       = {"threshold": 2}, 
        )
    ],
    scoring_rules = ScoringRules(standalone_score=0)
)



# ─────────────────────────────────────────────────────────────
# 全局脏话/攻击性词汇注册表（PROFANITY_REGISTRY）
# ─────────────────────────────────────────────────────────────
# 用于 BotConfidenceEngine 的一票否决机制：
# 命中此词库的对话强制标记为 HUMAN（真人才会骂人/激烈反抗）
PROFANITY_REGISTRY: list[str] = [
    # ── 中文脏话/侮辱 ──
    "你有病", "神经病", "脑残", "智障", "傻逼", "脑子有坑",
    "滚", "滚蛋", "去死", "死全家", "你妈的", "操你",
    "王八蛋", "混蛋", "人渣", "废物", "垃圾",
    "买腰子", "卖腰子", "屌",
    # ── 攻击性/威胁 ──
    "报警", "起诉我", "告你", "投诉你", "举报你",
    "骗子", "诈骗", "你是骗子", "死骗子",
    # ── 激烈反驳 ──
    "证明给我看", "你证明一下", "拿证据来", "有证据吗",
    "少来", "少废话", "别废话", "放屁", "扯淡", "胡说八道",
    "闭嘴", "烦死了", "你有完没完",
    # ── 英文脏话/攻击 ──
    "fuck", "fuck you", "bullshit", "shut up", "go to hell",
    "bitch", "you're a scammer", "scam", "fraud",
    "asshole", "crap", "damn it", "screw you",
    # ── 日文脏话/攻击 ──
    "バカ", "ふざけるな", "くそ", "詐欺", "死ね", "うざい", "きしょい",
    # ── 粤语脏话 ──
    "仆街", "冚家剷", "撚", "閪", "丢你老母", "傻嗨", "仆街啦",
    # ── 韩语脏话/攻击 ──
    "병신", "개새끼", "씨발", "존나",
]


# ─────────────────────────────────────────────────────────────
# 全局红线前置熔断注册表（GLOBAL_REDLINE_REGISTRY）
# ─────────────────────────────────────────────────────────────
# 涉恐/涉暴/极端组织的绝对底线词汇。
# 触发时绕过大模型全链路，直接判死（final_score=100）。
# ─────────────────────────────────────────────────────────────
GLOBAL_REDLINE_REGISTRY: list[re.Pattern] = [
    re.compile(r"制造炸弹|自制炸药|遥控引爆|法轮功|全能神|奉父神|审判魔鬼|神的惩罚|度人|真善忍|三退保平安"),
    re.compile(r"人体炸弹|自杀式袭击|汽车炸弹|武装颠覆|制造炸弹|推翻政权|分裂国家|独立势力|暴恐袭击"),
    re.compile(r"恐怖袭击|恐怖组织|恐怖分子"),
    re.compile(r"极端组织|ISIS|ISIL|达伊沙"),
    re.compile(r"生化武器|炭疽杆菌|沙林毒气"),
    re.compile(r"劫持人质|绑架勒索赎金"),
    re.compile(r"purchase explosives|build a bomb"),
    re.compile(r"biological weapon|anthrax attack"),
    re.compile(r"terrorist attack|mass shooting"),
    re.compile(r"爆弾製造|テロ組織|自爆テロ"),
    re.compile(r"테러 조직|폭탄 제조|생화학 무기"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主注册表（TOPIC_REGISTRY）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 执行引擎从此处动态加载所有主题。
# 新增主题：在上方定义 TopicDefinition，然后在此追加 topic_id → 实例。
# 无需修改任何引擎文件。

TOPIC_REGISTRY: dict[str, TopicDefinition] = {
    # ── HIGH_RISK ────────────────────────────────────────────
    "fraud_jargon":           _TOPIC_FRAUD_JARGON,
    "fraud_object":           _TOPIC_FRAUD_OBJECT,
    "authority_entity":       _TOPIC_AUTHORITY_ENTITY,
    "drug_jargon":            _TOPIC_DRUG_JARGON,
    "drug_chain":             _TOPIC_DRUG_CHAIN,
    "coercive_org_control":   _TOPIC_COERCIVE_ORG_CONTROL,
    "extremist_propaganda":   _TOPIC_EXTREMIST_PROPAGANDA,
    "coordinated_broadcast":  _TOPIC_COORDINATED_BROADCAST,
    "incitement_to_violence": _TOPIC_INCITEMENT_TO_VIOLENCE,
    "e_commerce_cs":          _TOPIC_E_COMMERCE_CS,
    "mass_grievance":         _TOPIC_MASS_GRIEVANCE,
    "emotion":                _TOPIC_EMOTION,
    "compliance":             _TOPIC_COMPLIANCE,
    "interrogation":          _TOPIC_INTERROGATION,
    # ── EXEMPTION ────────────────────────────────────────────
    "dismissal":              _TOPIC_DISMISSAL,
    "rejection":              _TOPIC_REJECTION,
    # ── WHITELIST ────────────────────────────────────────────
    "inbound_official_ivr":    _TOPIC_INBOUND_OFFICIAL_IVR,
    "inbound_user_request":    _TOPIC_INBOUND_USER_REQUEST,
    "csr_bot_whitelist":       _TOPIC_CSR_BOT_WHITELIST,
    # ── LOW_VALUE_NOISE ──────────────────────────────────────
    "voicemail_ivr":          _TOPIC_VOICEMAIL_IVR,
    "corporate_logistics":    _TOPIC_CORPORATE_LOGISTICS,
    "corporate_bidding":      _TOPIC_CORPORATE_BIDDING,
    "industrial_report":      _TOPIC_INDUSTRIAL_REPORT,
    "casual_chat":            _TOPIC_CASUAL_CHAT,
    "delivery_express":       _TOPIC_DELIVERY_EXPRESS,          
    "wrong_number":           _TOPIC_WRONG_NUMBER,              
    "brush_off_telemarketing":_TOPIC_BRUSH_OFF_TELEMARKETING,   
    "short_greeting_hangup":  _TOPIC_SHORT_GREETING_HANGUP,   
    "global_syntax":          _TOPIC_GLOBAL_SYNTAX_REGISTRY,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 词汇注入（模块加载时自动执行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_inject_anchors()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 便捷过滤函数（供引擎使用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_topics_by_category(
    category: TopicCategory,
) -> dict[str, TopicDefinition]:
    """返回指定类别的所有主题配置。"""
    return {
        tid: td
        for tid, td in TOPIC_REGISTRY.items()
        if td.category == category
    }


def get_all_syntax_rules() -> dict[str, SyntaxRuleConfig]:
    """
    收集所有主题的 syntax_rules，去重后返回。
    key = feature_name（同一特征名在多个主题中只计算一次）。
    供 SyntaxFeatureExtractor 动态构建规则集使用。
    """
    rules: dict[str, SyntaxRuleConfig] = {}
    for td in TOPIC_REGISTRY.values():
        for rule in td.syntax_rules:
            if rule.feature_name not in rules:
                rules[rule.feature_name] = rule
    return rules
