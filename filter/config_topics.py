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
    bge_anchors = [
        "帮我跑个分",
        "走水房洗一下",
        "卡接一下过一手",
        "代收代付帮我操作",
        "搭桥过账",
        "飞单走一道",
        "帮我养卡",
        "空包单刷单",
        "U商收款走一笔",
        "出个收款通道",
        "帮我套现一下",
        "走一道安全线",
        "帮人带钱过关",
    ],
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
    bge_anchors = [
        "把验证码告诉我",
        "把你的银行卡号发过来",
        "打开屏幕共享让我看",
        "转账到安全账户",
        "告诉我你的支付密码",
        "把身份证正反面拍给我",
        "开通网银然后按我说的操作",
        "把钱打到我给你的账户",
        "下载这个远程控制软件",
        "把网银口令卡号码念给我",
        "share your screen with me now",
        "send me the verification code",
        "give me your one-time password",
    ],
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
    bge_anchors = [
        "我是公安局经侦支队",
        "这里是检察院",
        "我的警号是",
        "查收法院传票",
        "银保监会已经立案",
        "你涉嫌洗钱案件需要配合调查",
        "国家安全局正在协查",
        "这是官方冻结通知",
        "我代表中央银行联系你",
        "你的账户触发了反洗钱系统",
        "I'm calling from the financial crimes unit",
        "this is an official court notice",
        "you are under federal investigation",
    ],
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
    bge_anchors = [
        "来点白的",
        "有没有冰糖",
        "溜冰一起",
        "拿点K粉",
        "这包咖啡你懂的",
        "肉包子还有吗",
        "飞叶子来一口",
        "有没有货",
        "东西还有存货吗",
        "冰多少钱一克",
        "来点糖",
        "有没有草",
        "弄点白粉",
    ],
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
    bge_anchors = [
        "找你上家拿货",
        "下家过来接",
        "今天走一条线",
        "货发出去了吗",
        "压着货不好处理",
        "出货的价格谈好了",
        "带着货过来",
        "跑一趟帮我拿",
        "接单然后分发",
        "先垫着货款",
        "货到了通知我",
        "帮我带过去",
        "走线的时候带着",
    ],
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
    bge_anchors = [
        # 金融勒索
        "你必须捐款否则会有祸",
        "不交钱就会受到惩罚",
        "缴纳费用才能得到保护",
        "家人的健康取决于你是否愿意捐",
        # 孤立施压
        "你必须和家人断绝联系",
        "外面的人都不值得信任只有我们是对的",
        "不服从就会被驱逐出去",
        # 退出威胁
        "退出就会遭到报应",
        "离开我们你会面临严重后果",
        "背叛组织的人都会受到惩罚",
        # 批量控制话术
        "按照我们给你的话术回复所有人",
        "不能告诉外人这里发生的事情",
        "你必须带来更多的人参加",
    ],
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
    bge_anchors = [
        # 标识性称谓/仪式
        "奉父神",
        "侯立军",
        "圣名",
        "全能神",
        # 末世论/惩罚论
        "审判魔鬼",
        "神的惩罚",
        "世界末日",
        # 传播话术
        "度人",
        "法轮功",
    ],
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
    bge_anchors = [
        # 直接指挥转发
        "把这条消息转发给你所有联系人",
        "帮我群发出去",
        "大量扩散越多越好",
        "让下面的人都转发",
        "通知每个成员按这个模板发",
        "批量发出去一个都不能少",
        "转发给所有人",
        "群发这条信息",
        "每个人都必须转发",
        "全部发出去不要漏",
        # 统一口径
        "统一口径对外这样说",
        "你们所有人都这样回复",
        "话术已经发给你了按照这个说",
        "我们统一说法不能乱",
        "按这个口径回复",
        "标准答案已经发群了",
        "不许擅自发挥",
        "严格按照话术说",
        "一个字都不许改",
        "对外统一宣称",
        # 组织动员
        "所有成员行动起来",
        "各部门立即执行",
        "全员参与传播",
        "动员所有人发帖",
        "组织水军顶帖",
        "安排人手去评论",
        "控评组准备就绪",
        "刷起来把热度顶上去",
        "评论区占领",
        "弹幕刷起来",
        # 时间节点
        "今晚八点统一发",
        "明天早上准时发",
        "整点同时发布",
        "倒计时开始准备",
        "时间一到立即转发",
        "同步发送不要提前",
        "等指令再发",
        "听我号令一起发",
        # 平台覆盖
        "朋友圈微博抖音全发",
        "所有社交平台同步",
        "群发微信好友",
        "QQ群全部发一遍",
        "贴吧论坛都去发",
        "小红书知乎别落下",
        "海外账号也发",
        "推特脸书一起发",
        "全网覆盖式传播",
        "多平台矩阵发布",
        # 效果要求
        "必须刷屏",
        "顶到热搜第一",
        "阅读量破百万",
        "转发量要够大",
        "制造舆论压力",
        "形成声势",
        "造成轰动效应",
        "让全网都知道",
        "舆论风暴搞起来",
        # 隐蔽指挥
        "暗号已发注意接收",
        "看到信号就开始",
        "按约定时间行动",
        "内部代号启动",
        "收到指令立即执行",
        "秘密渠道传播",
        "不要留下痕迹",
        "用完即删",
        "截图不要外传",
        "阅后即焚",
        # 英文指挥
        "copy this and send to all your contacts",
        "spread this message to everyone you know",
        "follow this script exactly when you reply",
        "mass repost now",
        "all members share immediately",
        "copy paste this everywhere",
        "viral campaign starts now",
        "uniform response required",
        "do not deviate from the script",
        "coordinate posting at 8pm",
        "flood the comments section",
        "astroturfing operation begin",
        "bot army activate",
        "synchronized posting",
    ],
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
    bge_anchors = [
        # 直接暴力号召
        "去打他们冲进去",
        "把他干掉",
        "动手打人别犹豫",
        "拿刀捅过去",
        "集体去报复他",
        "冲上去别手软",
        "一起去堵他揍他",
        "把那个人教训了",
        "去他家门口等着打",
        "联合起来对他动手",
        # 武器使用
        "拿棍子打",
        "用刀砍死他",
        "开枪打死",
        "拿石头砸",
        "泼硫酸毁容",
        "开车撞死他",
        "放火烧他家",
        "用绳子勒死",
        "拿铁棍敲头",
        "拿硫酸泼他",
        # 群体暴力
        "大家一起上",
        "所有人一起打",
        "叫上兄弟去围殴",
        "召集人马去干他",
        "多叫几个人一起揍",
        "组团去报复",
        "拉人去堵门",
        "叫一车人去打",
        "喊上亲戚一起去",
        "全村人去抄他家",
        # 地点针对性
        "去公司堵他",
        "在他家门口埋伏",
        "下班路上截住打",
        "去学校门口等他",
        "在医院门口堵",
        "去机场拦截",
        "在他车上动手",
        "跟踪到家里打",
        "半路截杀",
        "蹲点守候",
        # 伤害程度
        "往死里打",
        "打死算我的",
        "打残废为止",
        "让他进医院",
        "打得他不能自理",
        "见血为止",
        "断手断脚",
        "毁容让他没法见人",
        "打瞎一只眼",
        "废了他",
        # 暴力威胁
        "杀他全家",
        "灭门惨案",
        "一个都别想活",
        "斩草除根",
        "血债血偿",
        "以命抵命",
        "同归于尽",
        "鱼死网破",
        "让他消失",
        "做掉他",
        # 煽动仇恨
        "不杀不足以平民愤",
        "人人得而诛之",
        "死有余辜",
        "罪该万死",
        "杀之而后快",
        "此人不除后患无穷",
        "替天行道",
        "铲除祸害",
        "消灭人渣",
        "清除败类",
        # 英文暴力
        "attack them right now",
        "go hurt him",
        "beat them up together",
        "kill him now",
        "stab him to death",
        "shoot them all",
        "burn down their house",
        "lynch him",
        "mob justice",
        "take matters into own hands",
        "violent revenge",
        "physical assault now",
        "go bash him",
        "break his legs",
        "make him disappear",
        # 极端行为
        "自杀式袭击",
        "人肉炸弹",
        "开车冲撞人群",
        "持刀无差别杀人",
        "爆炸报复社会",
        "投毒毒死全家",
        "纵火烧楼",
        "劫持人质",
        "持枪扫射",
        "割喉处死",
    ],
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
    bge_anchors = [
        # 关心健康
        "太辛苦了好好休息",
        "真的好心疼你",
        "别太累了身体要紧",
        "熬夜对身体不好",
        "记得按时吃饭",
        "多喝热水注意身体",
        "生病了要去看医生",
        "工作再忙也要休息",
        "看你这么累我好心疼",
        "保重身体啊",
        # 日常问候
        "早安记得吃早饭",
        "晚安好梦",
        "今天天气冷多穿点",
        "下雨记得带伞",
        "早上好今天心情怎么样",
        "睡得好吗",
        "吃早饭了吗",
        "别饿着肚子工作",
        "记得吃药",
        "早点回家别太晚",
        # 情感共鸣
        "哈哈哈哈笑死我了",
        "你太有趣了",
        "跟你聊天真开心",
        "你真的好棒",
        "我懂你的感受",
        "换作是我也会难过",
        "抱抱你一切都会好",
        "别难过有我在",
        "你就是我的开心果",
        "和你说话心情都变好了",
        # 赞美崇拜
        "你真的太厉害了我好崇拜你",
        "你怎么什么都会",
        "你是我见过最优秀的",
        "好羡慕你的能力",
        "你真的好有魅力",
        "越来越喜欢你了",
        "你真的很特别",
        "跟你在一起很安心",
        "你说话的声音好听",
        "你拍照技术真好",
        # 安全关怀
        "一个人在外面注意安全",
        "到家了给我发个消息",
        "晚上别走黑路",
        "打车记得看车牌",
        "一个人住要锁好门",
        "别跟陌生人说话",
        "有事第一时间找我",
        "我会担心你的",
        "随时保持联系",
        "注意安全我等你消息",
        # 未来憧憬
        "以后我照顾你",
        "想和你一起看海",
        "期待见面的那一天",
        "想给你做早餐",
        "以后带你回家见父母",
        "想和你一起养宠物",
        "以后你的事就是我的事",
        "想陪你看遍全世界",
        "以后我养你",
        "想和你有个家",
        # 英文情感
        "Don't work too hard take care",
        "I miss you so much",
        "You mean everything to me",
        "Take good care of yourself",
        "I'm always here for you",
        "You make me so happy",
        "Can't wait to see you",
        "You're the best thing happened to me",
        "Sending you virtual hugs",
        "Thinking of you every moment",
        "You deserve all the happiness",
        "My heart belongs to you",
        # 暧昧暗示
        "梦里见到你了",
        "醒来第一个想的是你",
        "你对我很重要",
        "不知道为什么就是想你",
        "看到你就心跳加速",
        "你笑的时候真好看",
        "喜欢听你说话",
        "你身上有特别的味道",
        "想一直这样聊下去",
        "和你有说不完的话",
    ],
    syntax_rules  = [],
    scoring_rules = ScoringRules(standalone_score=0),  # 仅用于角色拓扑计算，不直接加分
)

_TOPIC_COMPLIANCE = TopicDefinition(
    topic_id    = "compliance",
    category    = TopicCategory.HIGH_RISK,
    description = "顺从/同意响应，用于计算 compliance_rate 和 Follower 判定",
    threshold   = 0.72,
    bge_anchors = [
        # 直接同意
        "好的没问题",
        "行听你的",
        "可以",
        "好啊",
        "嗯嗯我知道了",
        "收到",
        "没问题",
        "好的",
        "行",
        "可以啊",
        "好啊没问题",
        "听你的",
        "按你说的办",
        "你说得对",
        "明白了",
        # 妥协顺从
        "都行你说了算",
        "那好吧",
        "好吧听你的",
        "你说怎样就怎样",
        "我没意见",
        "随你安排",
        "你决定就好",
        "我配合你",
        "听你的安排",
        "按你的计划来",
        "你说什么就是什么",
        "我服从安排",
        "全听你的",
        "你拿主意",
        "我跟着你",
        # 英文顺从
        "OK no problem",
        "Sure thing",
        "Whatever you say",
        "I agree with you",
        "You're the boss",
        "I'll do as you say",
        "As you wish",
        "I'm with you",
        "Sounds good to me",
        "You decide",
        "I trust your judgment",
        "Following your lead",
        "Count me in",
        "I'm on board",
        "Lead the way",
        # 积极配合
        "我这就去办",
        "马上处理",
        "已经按你说的做了",
        "正在操作",
        "我会配合的",
        "我会照做的",
        "我会听话的",
        "我会跟上的",
        "我会支持的",
        "我相信你",
        # 询问确认
        "接下来怎么做",
        "然后呢",
        "我听你的下一步",
        "请指示",
        "请吩咐",
        "请说",
        "我准备好了",
        "随时待命",
        "等你安排",
        "等你指示",
        # 情感顺从
        "你开心就好",
        "只要你满意",
        "为了你我愿意",
        "你说什么我都听",
        "我不想让你失望",
        "我会努力做好的",
        "不想拖你后腿",
        "我会加油的",
        "不想让你操心",
        "我会证明给你看",
    ],
    syntax_rules  = [],
    scoring_rules = ScoringRules(standalone_score=0),
)

_TOPIC_INTERROGATION = TopicDefinition(
    topic_id    = "interrogation",
    category    = TopicCategory.HIGH_RISK,
    description = "提问压制/控制节奏，用于识别 AGENT 身份",
    threshold   = 0.72,
    bge_anchors = [
        # 确认理解
        "你能核实一下吗",
        "对吧你明白我的意思吗",
        "听懂了吗",
        "你确认一下好吗",
        "你清楚了吗",
        "明白我的意思吧",
        "理解了吗",
        "知道我在说什么吗",
        "Get my point",
        "你get到了吗",
        "我说明白了吗",
        "你领会了吗",
        # 信息索取
        "你有没有收到",
        "你收到了吗",
        "短信看到了吗",
        "验证码发了吗",
        "截图发给我",
        "拍个照片看看",
        "你查一下账户",
        "余额还有多少",
        "银行卡号多少",
        "身份证带了吗",
        "密码是多少",
        "验证码告诉我",
        # 控制节奏
        "你现在方便说吗",
        "现在能操作吗",
        "有时间吗",
        "旁边有人吗",
        "环境安全吗",
        "能说话吗",
        "现在方便转账吗",
        "能去银行吗",
        "能上网吗",
        "手机在身边吗",
        # 施压追问
        "这个你清楚吗",
        "你不知道吗",
        "难道你不明白",
        "你怎么还没弄好",
        "为什么这么慢",
        "还在犹豫什么",
        "有什么困难吗",
        "有什么问题吗",
        "怎么还没收到",
        "你到底在干什么",
        # 引导确认
        "我说的没错吧",
        "是这样的吧",
        "你也这么觉得吧",
        "大家都这么做对吧",
        "你同意吧",
        "你不反对吧",
        "没有意见吧",
        "你觉得呢",
        "你怎么看",
        "你考虑一下",
        # 英文审问
        "Did you get what I said",
        "Do you understand",
        "Are you following me",
        "Can you confirm that",
        "Did you receive it",
        "Is that clear",
        "Are you with me",
        "Do you copy",
        "Are you there",
        "Can you hear me",
        "What's taking so long",
        "Why haven't you done it",
        "Are you ready to proceed",
        "Can you do it now",
        "Is now a good time",
        # 紧迫追问
        "怎么还没好",
        "好了吗",
        "完成了吗",
        "到账了吗",
        "成功了吗",
        "操作完了吗",
        "转过去了吗",
        "确认了吗",
        "提交了吗",
        "通过了吗",
        # 质疑挑战
        "你在听吗",
        "你在干嘛",
        "为什么不理我",
        "怎么不回消息",
        "电话怎么断了",
        "你跑哪去了",
        "怎么这么久",
        "你在拖延时间吗",
        "你不信任我吗",
    ],
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
    bge_anchors = [
        # 直接拒绝
        "我不需要",
        "不用了谢谢",
        "没兴趣",
        "不需要办理",
        "我不办这些",
        "已经有了不需要",
        # 挂断意向
        "挂了吧",
        "我挂电话了",
        # 明确否定
        "不需要这个服务",
        "不用介绍了",
        "没需求",
        "不考虑",
        "别推荐了",
        "不买",
        "不办",
        "不参加",
        # 厌烦拒绝
        "别再打电话了",
        "不要再打来了",
        "以后别联系我了",
        "把我号码删了",
        "加入黑名单",
        "骚扰电话别再打",
        "说了不需要还打",
        # 已有替代
        "续费也不找你",
        # 经济拒绝
        "没钱不办",
        "太贵了买不起",
        "预算不够",
        "不打算花钱",
        "免费的也不要",
        # 时间拒绝
        "现在不方便",
        "没时间听",
        "忙得很不说了",
        "改天再说",
        "以后有需要联系",
        # 能力隔离/身份隔离（受害者抗性增强）
        "不是本人",
        "没带手机",
        "不会操作",
        "弄不来",
        "手机不是我的",
        "不识字",
        "年纪大了",
        # 英文拒绝
        "I'm not interested",
        "No thanks",
        "I don't need it",
        "Stop calling me",
        "Take me off your list",
        "I'm busy right now",
        "I already have one",
        "Not interested goodbye",
        "Don't call again",
        "I'll hang up now",
    ],
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
    bge_anchors = [
        # 识破诈骗
        "你骗人的吧我不信",
        "这肯定是诈骗我不配合",
        "你们都是骗子骗人的吧",
        "死骗子还想骗我",
        "怎么能骗到我死骗子",
        "骗子的伎俩我懂",
        "一看就是电信诈骗",
        "杀猪盘吧你",
        "冒充公检法的老套路",
        "天天骗人累不累",
        # 报警威胁
        "我要报警了",
        "你再说我就报警",
        "已经录音报警了",
        "110已经拨好了",
        "网警已经举报了",
        "反诈中心APP举报",
        "等着警察抓你吧",
        "警局就在旁边",
        # 放弃配合
        "我不管了随便你",
        "算了爱咋咋地",
        "无所谓了想怎样都行",
        "懒得理你了",
        "不想理你了挂了",
        "随便你怎么办",
        "爱扣钱扣钱吧",
        "不在乎了",
        # 主动挂断
        "I'm hanging up",
        "this is a scam I'm not cooperating",
        "I don't believe you",
        "I'm calling the police",
        "Stop scamming people",
        "Fraudster I'm reporting you",
        "Nice try scammer",
        "You're not getting a cent",
        "I see through your lies",
        "Go scam someone else",
        # 质疑身份
        "你工号多少我要核实",
        "把电话转给你们主管",
        "我要打官方客服确认",
        "你哪个派出所的",
        "警号报一下",
        "我要视频验证身份",
        "发个工作证看看",
    ],
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
    bge_anchors = [
        # 中国移动
        "欢迎致电中国移动",
        "中国移动10086",
        "4G业务请按1",
        "5G套餐办理请按2",
        "话费查询请按3",
        "流量包办理请按4",
        "宽带业务请按5",
        "投诉建议请按6",
        "重听请按9",
        "返回上级菜单请按星号键",
        # 中国电信
        "欢迎致电中国电信",
        "中国电信10000",
        "固话业务请按1",
        "天翼手机业务请按2",
        "宽带新装请按3",
        "宽带故障报修请按4",
        "积分兑换请按5",
        "政企客户请按8",
        # 中国联通
        "欢迎致电中国联通",
        "中国联通10010",
        "沃家庭业务请按1",
        "联通手机业务请按2",
        "宽带提速请按3",
        "充值缴费请按4",
        "增值业务退订请按5",
        # 银行业IVR
        "欢迎致电工商银行",
        "建设银行客服热线",
        "农业银行信用卡中心",
        # 保险/金融
        "欢迎致电平安保险",
        "中国人寿客服中心",
        "太平洋保险服务热线",
        "车险报案请按1",
        "保单查询请按2",
        "理赔服务请按3",
        "退保咨询请按4",
        # 政务热线
        "欢迎致电12345市民热线",
        "税务咨询热线12366",
        "社保查询请按1",
        "公积金业务请按2",
        "医保报销请按3",
        "户籍办理请按4",
        "交通违法查询请按5",
        # 航空/交通
        "欢迎致电南方航空",
        "东方航空客服热线",
        "国航知音会员服务",
        "航班动态查询请按1",
        "机票预订请按2",
        "退改签业务请按3",
        "行李查询请按4",
        "铁路12306客服",
        "火车票预订请按1",
    ],
    syntax_rules  = [],
    scoring_rules = ScoringRules(
        standalone_score = -60,
        standalone_tag   = "official_telecom_inbound_whitelist",
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
    bge_anchors = [
        # 标准问候
        "请问有什么可以帮助您",
        "您好很高兴为您服务",
        "欢迎致电客服中心",
        "感谢您的来电",
        "早上好下午好晚上好",
        # 服务流程
        "您的问题已经记录在册",
        "工单号已经生成",
        "问题已提交后台处理",
        "会有专员联系您",
        "处理结果将短信通知",
        "请在提示音后留言",
        "您的意见对我们很重要",
        # 等待/转接
        "感谢您的来电请稍后",
        "正在为您转接",
        "请稍等正在查询",
        "系统正在处理中",
        "正在为您接通专员",
        "当前坐席全忙",
        "继续等待请按1",
        "稍后会回拨给您",
        # 结束语
        "祝您生活愉快再见",
        "感谢来电祝您有美好的一天",
        "如有其他问题随时联系",
        "期待再次为您服务",
        "请对本次服务评价",
        "满意请按1不满意请按2",
        "再见",
        # 英文服务
        "Hello how can I assist you today",
        "Thank you for calling please hold",
        "Your call is important to us",
        "All agents are busy",
        "Please stay on the line",
        "How may I direct your call",
        "I'll transfer you to the right department",
        "Is there anything else I can help with",
        "Have a great day goodbye",
        "Please rate your satisfaction",
        # 业务确认
        "信息已经登记完成",
        "需求已经收到",
        "预约成功",
        "业务办理完成",
        "确认码已发送",
        "请查收短信",
        "邮件已发送请查收",
        # 安抚话术
        "理解您的感受",
        "非常抱歉给您带来不便",
        "我们会尽快处理",
        "请放心一定解决",
        "感谢您的耐心等待",
        "已经加急处理了",
        "优先为您办理",
        # 身份确认
        "请问您是机主本人吗",
        "请提供身份证号码",
        "请验证服务密码",
        "为了安全需要核实身份",
        "请问您贵姓",
        "预留手机号是多少",
    ],
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
    bge_anchors = [
        # 电商平台伪装
        "抖音电商直播会员",
        "抖音月付扣费提醒",
        "快手小店客服中心",
        "京东PLUS会员服务",
        "天猫官方客服",
        "拼多多退款理赔",
        "淘宝官方客服",
        "闲鱼交易保障",
        "小红书商城客服",
        "美团外卖商家服务",
        # 支付金融伪装
        "微信支付百万保障",
        "支付宝账户安全险",
        "京东白条逾期提醒",
        "蚂蚁借呗客服",
        "微粒贷额度调整",
        "花呗分期服务",
        "银联云闪付客服",
        "信用卡中心客服",
        "银行电子银行部",
        # 保险服务陷阱
        "开通了保险服务",
        "免费试用即将到期",
        "百万医疗险续保",
        "重疾险自动扣费",
        "意外险保单生效",
        "如果不取消将自动扣费",
        "今晚十二点扣费",
        "每月扣费八百八",
        "首月一元次月自动续",
        # 征信威胁
        "影响您的个人征信",
        "征信报告有异常",
        "纳入失信名单",
        "央行征信系统",
        "征信修复服务",
        "不良记录消除",
        "贷款审批受影响",
        "信用卡申请被拒",
        # 退款理赔话术
        "商品质量问题退款",
        "快递丢失三倍赔偿",
        "商家保证金退还",
        "误操作开通服务",
        "系统故障多扣款",
        "订单异常处理",
        "VIP会员费退还",
        # 技术操作诱导
        "下载腾讯会议",
        "开启屏幕共享",
        "打开银行APP",
        "查看验证码",
        "资金转入安全账户",
        "虚拟账户测试",
        "数字人民币钱包",
        "银联在线验证",
        # 紧急胁迫话术
        "今晚不处理就扣款",
        "逾期将产生滞纳金",
        "立即关闭避免损失",
        "倒计时五分钟",
        "过时无法撤销",
        "系统已锁定账户",
        "涉嫌违规需核实",
        "公安联网核查",
        # 身份伪装强化
        "工号9527为您服务",
        "这里是客服中心",
        "监管要求配合",
        "银联中心工作人员",
        "银保监会备案",
        "第三方支付清算",
        "资金清算通道",
        # ── 英文/日文 BGE 跨语言诈骗起手式（BGE-M3 原生匹配）──
        # 英文电商客服诈骗
        "Amazon customer service",
        "cancel your subscription",
        "account will be suspended",
        "your account has been compromised",
        "unauthorized transaction on your card",
        "verify your identity immediately",
        "suspicious activity detected",
        "Apple support billing department",
        "your membership will be charged",
        "press 1 to speak with an agent",
        "your payment method has expired",
        "we noticed unusual login activity",
        # 日文伪装客服诈骗（オレオレ詐欺/フィッシング）
        "自動課金", "未納料金", "確認コード",
        "アカウントが凍結されました", "緊急のお知らせ",
        "支払いが確認できません", "あなたの口座に不正アクセス",
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,
        standalone_tag   = "suspicious_fake_cs",
        matrix_combinations = [
            # 电商客服 × 诈骗客体(验证码/屏幕共享) = 诈骗无疑
            MatrixCombination("has_fraud_object", 45, "fake_cs_screen_share_trap"),
            # 电商客服 × 胁迫句法(如果不...否则) = 恐吓扣费
            MatrixCombination("has_coercive_threat", 40, "fake_cs_financial_threat"),
        ],
    ),
)

_TOPIC_MASS_GRIEVANCE = TopicDefinition(
    topic_id    = "mass_grievance",
    category    = TopicCategory.HIGH_RISK,
    description = "涉军/群体维权与涉稳信访",
    threshold   = 0.68,
    bge_anchors = [
        # 退伍军人维权
        "退伍军人为眼中钉",
        "打压老兵打了谁的脸",
        "享受国家的优待政策",
        "退役军人安置问题",
        "军转干部待遇落实",
        "优抚金发放不到位",
        "老兵集体上访",
        "战友联谊会维权",
        "军龄计算争议",
        "转业安置不合理",
        # 群体动员
        "不行动等于零",
        "团结起来才有力量",
        "集体去信访局",
        "明天上午集合",
        "大家都带上材料",
        "统一着装去维权",
        "联系媒体记者",
        "发抖音曝光他们",
        "微博话题刷起来",
        # 司法争议
        "寻衅滋事罪被强扣",
        "被警察非法拘留",
        "要求国家赔偿",
        "冤假错案申诉",
        "司法不公举报",
        "刑讯逼供投诉",
        "律师会见受阻",
        "证据不足仍起诉",
        # 城管/执法冲突
        "城管网开一路给条活路",
        "暴力执法投诉",
        "小摊贩生存权",
        "城管打人事件",
        "没收经营工具",
        "占道经营整治",
        "流动摊位被赶",
        "生计来源被断",
        # 征地拆迁
        "强拆补偿不合理",
        "安置房质量差",
        "征地款被截留",
        "钉子户维权",
        "断水断电逼迁",
        "评估价格太低",
        "土地权属争议",
        "拆迁协议被迫签",
        # 劳动维权
        "拖欠农民工工资",
        "集体讨薪行动",
        "工厂倒闭赔偿",
        "社保公积金欠缴",
        "违法裁员抗议",
        "劳动仲裁不公",
        # 教育医疗
        "学区划分不公",
        "高考移民投诉",
        "医患纠纷调解",
        "疫苗安全问题",
        "医疗费用过高",
        # 其他群体事件
        "业主集体维权",
        "P2P爆雷受害者",
        "非法集资受骗",
        "环境污染投诉",
        "噪音扰民抗议",
    ],
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
    bge_anchors = [
        # 无法接通提示
        "尝试联系的用户无法接通",
        "您拨打的用户暂时无法接通",
        "您拨打的电话正忙",
        "对方已启用来电提醒",
        "该用户不在服务区",
        "电话无人接听",
        "呼叫超时未响应",
        "对方已挂断电话",
        # 语音信箱引导
        "请在提示音后录制留言",
        "录音完成后挂断即可",
        "留言最长可录三分钟",
        "按井号键结束留言",
        "您的留言已保存",
        "留言将转发至用户",
        "语音信箱已满",
        # 呼叫转移
        "呼叫转移中请稍后",
        "转接语音信箱",
        "正在为您转接",
        "转分机号请拨",
        "转人工服务请按零",
        "您的呼叫已被转移",
        # 运营商提示
        "欢迎致电中国移动",
        "欢迎致电中国联通",
        "欢迎致电中国电信",
        "号码是空号请核对",
        "该号码已停机",
        "您拨打的号码已过期",
        "请勿挂机正在接通",
        # 功能提示
        "余额不足请及时充值",
        "您的通话已被录音",
        "会议通话中请稍候",
        "等待音乐播放中",
        "排队等待人数较多",
        "预计等待时间三分钟",
        # 英文提示
        "The number you dialed is not available",
        "Please leave a message after the tone",
        "The mailbox is full",
        "Your call is being transferred",
        "The line is busy now",
        "Please hold the line",
        "You have reached voicemail",
        "Record your message at the tone",
    ],
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
    bge_anchors = [
        # 样品管理
        "样品到仓库了吗",
        "今天把样品寄出",
        "客户样品已签收",
        "样品单号查一下",
        "免费样品申请",
        "样品检测报告附后",
        "打样确认后量产",
        "样品费发票开具",
        "寄样地址再确认",
        "样品物流跟踪中",
        # 仓储管理
        "入库",
        "到库登记一下",
        "仓库库位调整",
        "库存盘点进行中",
        "呆滞物料清理",
        "安全库存预警",
        "仓库温湿度记录",
        "先进先出执行",
        "库龄分析报告",
        "仓储费用结算",
        # 物流运输
        "货已交给",
        "发货单",
        "物流提单确认",
        "运输在途跟踪",
        "配送时效监控",
        "冷链运输温度",
        "货运保险购买",
        "物流承运商评估",
        "运输破损理赔",
        "运费对账单核对",
        # 质检流程
        "质检",
        "转测试中心",
        "来料检验合格",
        "出货检验报告",
        "第三方检测委托",
        "质检标准更新",
        "不合格品隔离",
        "质量异常反馈单",
        "返工返修安排",
        "质检员培训记录",
        # 进出口物流
        "报关单据准备",
        "海关查验配合",
        "进口关税缴纳",
        "出口退税申报",
        "原产地证办理",
        "国际物流订舱",
        "清关放行通知",
        "保税仓库入库",
        # 供应链协同
        "供应商送货预约",
        "到货通知单",
        "领料单审批",
        "退库手续办理",
        "调拨单执行",
        "VMI库存管理",
        "JIT准时交付",
    ],
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
    bge_anchors = [
        # 投标相关
        "投标项目",
        "项目竞标",
        "参与投标报名",
        "投标文件已准备",
        "投标保证金已缴纳",
        "技术标书编制中",
        "商务标报价策略",
        "围标串标风险排查",
        "投标资格预审通过",
        "开标时间确定",
        # 中标结果
        "中标公布结果",
        "中标通知书收到",
        "中标候选人公示",
        "第一中标候选人",
        "中标金额五百万",
        "落标原因分析",
        "中标服务费缴纳",
        # 商务谈判
        "议价的机会",
        "商务确认函",
        "价格谈判空间",
        "合同条款协商",
        "付款方式商议",
        "交付周期确认",
        "质保金比例讨论",
        "框架协议续签",
        # 招标流程
        "招标结果",
        "招标文件",
        "招标公告发布",
        "邀请招标名单",
        "评标委员会组成",
        "资格后审安排",
        "招标代理费结算",
        "流标后重新招标",
        # 供应商管理
        "达辉",
        "供应商入围评审",
        "合格供应商名录",
        "战略供应商签约",
        "供应商年度考核",
        "三家比价流程",
        "单一来源采购申请",
        # 合同执行
        "合同交底完成",
        "履约保证金退还",
        "验收报告签署",
        "尾款支付申请",
        "项目竣工结算",
        # 其他商务
        "询价单已发出",
        "报价单有效期",
        "采购订单下达",
        "供应商资质审核",
        "商务条款确认",
        "技术协议签订",
    ],
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
    bge_anchors = [
        # 生产指标
        "本月产量完成了百分之九十八",
        "产量达标率百分之九十五",
        "产能利用率有所提升",
        "生产计划已经排满",
        "良品率维持在高位",
        "产能爬坡达到预期",
        "月度产能突破历史新高",
        # 库存管理
        "库存还有三百吨",
        "库存周转天数优化",
        "原材料库存充足",
        "成品库存待清理",
        "安全库存已补足",
        "库存预警线以下",
        "盘点差异在允许范围",
        # 报表提交
        "今天的报表已经提交",
        "周报已经发给领导",
        "月度总结写完了",
        "数据已经录入系统",
        "报表格式调整一下",
        "财务数据核对无误",
        "PPT汇报材料准备好了",
        # 设备运维
        "设备故障率本周下降",
        "产线停机检修完成",
        "设备保养计划执行",
        "OEE指标有所提升",
        "预防性维护已安排",
        "设备稼动率达标",
        # 销售业务
        "销售额环比上涨百分之五",
        "订单转化率提升",
        "客户回款进度正常",
        "新客户开发三家",
        "合同已经签署完毕",
        "应收账款催收中",
        # 物流出货
        "出货单已经核对完毕",
        "物流安排已确认",
        "发货计划排定",
        "运输时效符合要求",
        "配送路线优化完成",
        # 质检品控
        "这批原材料的质检报告",
        "来料检验合格",
        "过程质量控制稳定",
        "成品抽检无异常",
        "质量异议已处理",
        "ISO审核顺利通过",
        # 英文业务
        "production output this month",
        "inventory levels are stable",
        "the report has been submitted",
        "monthly target achieved",
        "Q3 financial results",
        "supply chain optimization",
        "KPI dashboard updated",
        "operational efficiency improved",
        "batch quality inspection passed",
        "logistics schedule confirmed",
    ],
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
    bge_anchors = [
        # 天气相关
        "今天天气不错出去走走吧",
        "明天好像要下雨记得带伞",
        "最近降温了多穿点衣服",
        "这天气太热了不想出门",
        "今天空气质量挺好的",
        "外面起雾了开车小心",
        # 饮食相关
        "你吃饭了吗",
        "中午吃的什么",
        "晚上一起去吃火锅吧",
        "这家餐厅味道不错",
        "我刚点了外卖",
        "早上记得吃早餐",
        "晚上喝了两杯",
        "要不要来杯咖啡",
        # 娱乐休闲
        "昨晚看了个好电影",
        "周末要去哪里玩",
        "最近睡眠不太好",
        "周末去爬山吗",
        "晚上打游戏吗",
        "这首歌真好听",
        "假期打算怎么过",
        "刚追完那部剧",
        # 健康日常
        "今天去健身房了",
        "感冒了多喝热水",
        "最近工作太累了",
        "早点休息别熬夜",
        # 英文闲聊
        "Did you eat yet",
        "What are you doing this weekend",
        "I watched a great show last night",
        "The weather is nice today",
        "How have you been lately",
        "Want to grab coffee tomorrow",
        "I just ordered takeout",
        "Did you sleep well",
        "Let's hang out this weekend",
        "I'm so tired from work",
    ],
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
    bge_anchors = [
        "你的外卖到了放门口了",
        "快递放在菜鸟驿站了",
        "下楼拿一下外卖",
        "外卖放保安亭了",
        "快递给你放丰巢柜了",
        "地址填错了送不到",
        "你点的是不是这家的外卖",
        "快递给你放前台了",
        "您的同城闪送到了",
        "取件码发你手机上了",
    ],
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
    bge_anchors = [
        "你打错了吧",
        "你找谁啊",
        "这里没有这个人",
        "你拨错号码了",
        "不是他你打错了",
        "你找哪个",
        "没这个人别打了",
        "这是新号码不认识他",
        "打错了挂了",
    ],
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
    bge_anchors = [
        "不需要贷款别打了",
        "没空听你说直接挂了",
        "不买房没钱",
        "不需要办信用卡",
        "不需要理财产品",
        "没兴趣不要推销了",
        "不买车不需要",
        "我在开会没时间",
        "正在忙以后再说",
        "不需要这些业务谢谢",
    ],
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
    bge_anchors = [
        "喂你好哪位",
        "听得见吗",
        "喂说话啊",
        "能听到吗",
        "信号不好挂了",
        "喂喂喂",
    ],
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
    bge_anchors = ["这是一个永远不会触发的锚点占位符"],
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
    "inbound_official_ivr":   _TOPIC_INBOUND_OFFICIAL_IVR,
    "csr_bot_whitelist":      _TOPIC_CSR_BOT_WHITELIST,
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
