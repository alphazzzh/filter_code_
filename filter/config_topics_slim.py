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
    bge_anchors=_ANCHOR_PLACEHOLDER_FRAUD_JARGON,,
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
        # ── 日文诈骗客体 ──
        "画面共有を開始してください",
        "認証番号を教えてください",
        "パスワードを入力してください",
        "口座番号を教えて",
        "身分証の写真を送って",
        "カード情報を入力して",
        "安全な口座に移してください",
        "リモート操作アプリをダウンロードして",
        "銀行アプリを開いてください",
        "暗証番号を教えてください",
        "クレジットカードの裏面を撮って",
        # ── 韩语诈骗客体 ──
        "화면 공유를 켜주세요",
        "인증번호 알려주세요",
        "비밀번호 입력해 주세요",
        "계좌번호 알려 주세요",
        "신분증 사진 보내 주세요",
        "안전 계좌로 이체해 주세요",
        "원격 제어 앱 설치해 주세요",
        "카드 정보 입력해 주세요",
        "OTP 번호 알려 주세요",
    ],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperative_verbs",
            params = {
                "second_persbge_anchors=_ANCHOR_PLACEHOLDER_FRAUD_OBJECT,,
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
        # ── 日文权威伪装 ──
        "警察署の金融犯罪課です",
        "検察庁から連絡しています",
        "裁判所の呼び出しです",
        "あなたは事件に関与しています",
        "捜査に協力してください",
        "警視庁サイバー犯罪対策課",
        "金融庁の調査が進んでいます",
        "あなたの口座が凍結されました",
        "警察手帳の番号をお伝えします",
        "弁護士会から依頼を受けています",
        # ── 韩语权威伪装 ──
        "경찰청 금융범죄수사대입니다",
        "검찰에서 연락했습니다",
        "법원 소환장입니다",
        "당신은 사건에 연루되었습니다",
        "수사에 협조해 주세요",
        "경찰 사건번호 알려드립니다",
        "금융감독원 조사 중입니다",
        "당신 계좌가 동결되었습니다",
    ],
    syntax_rules = [
        # 复用 has_imperative_syntax 和 high_entity_density（与 fraud_object 共享规则）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperbge_anchors=_ANCHOR_PLACEHOLDER_AUTHORITY_ENTITY,,
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
        # ── 英文毒品隐语 ──
        "I need some blow",
        "got any crystal",
        "looking for ice",
        "can you hook me up with some snow",
        "need some Molly",
        "got any weed",
        "where can I find heroin",
        "looking for some powder",
        "need a gram of coke",
        "any acid tabs available",
        "got some speed",
        "can you score me some meth",
        # ── 日文毒品隐语 ──
        "白いのがない",
        "シャブやりたい",
        "大麻ない",
        "覚せい剤ある",
        "MDMAないかな",
        "コカイン手に入る",
        "ヘロイン欲しい",
        "草ある",
        "タバコみたいなのある",
        "やつ持ってるか",
        # ── 韩语毒品隐语 ──
        "하얀 거 좀",
        "필로폰 있어",
        "대초 없어",
        "마리화나 구해줘",
        "모르핀 있나요",
        "코카인 구할 수 있어",
        "엑스터시 필요해",
        "얼음 있어",
        "초 좀 구해",
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
            MatrixCbge_anchors=_ANCHOR_PLACEHOLDER_DRUG_JARGON,,
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
        # ── 英文毒品交易链条 ──
        "contact your supplier",
        "the shipment is on the way",
        "pick up the package",
        "the drop is ready",
        "hold the product for me",
        "money is ready pick up",
        "need a runner for delivery",
        "front me the goods",
        "the connect wants to meet",
        "stash the product",
        "move the package across",
        "count the cash",
        "the mule will carry it",
        "delivery confirmed send payment",
        "next shipment coming Friday",
        # ── 日文毒品交易链条 ──
        "上の人に聞いて",
        "荷物届いた",
        "受け渡し場所",
        "代金は後で",
        "運んでくれ",
        "在庫あるか確認",
        "次の出荷予定",
        "受け取り手配置",
        "現金で払う",
        "隠して運んで",
        # ── 韩语毒品隐语链条 ──
        "윗사람한테 물어봐",
        "물건 도착했어",
        "인수 장소 알려줘",
        "돈은 나중에 줄게",
        "배송해 줘",
        "재고 확인해 줘",
        "다음 배송 언제야",
        "현물 받을 준비 해",
        "현금 결제할게",
        "숨겨서 가져와 줘",
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
        matrix_cobge_anchors=_ANCHOR_PLACEHOLDER_DRUG_CHAIN,,
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
        # ── 英文强制控制 ──
        "you must donate or face consequences",
        "pay us or your family will suffer",
        "if you leave you will be punished",
        "cut off all contact with your family",
        "only we can protect you now",
        "betray us and you will regret it",
        "follow the script when talking to outsiders",
        "do not tell anyone what happens here",
        "recruit more members or else",
        "your salvation depends on your donations",
        "we are the only ones who understand you",
        "everyone outside is trying to deceive you",
        # ── 日文强制控制 ──
        "寄付しないと災いが起きる",
        "家族との縁を切らなければならない",
        "出ていくなら罰が下る",
        "外の人は全員信じてはいけない",
        "組織を裏切る者は罰される",
        "お金を納めなければならない",
        "外部には話してはいけない",
        "指示通りに話してください",
        "他の人を連れてきて",
        "ここで起きていることは秘密だ",
        # ── 韩语强制控制 ──
        "기부하지 않으면 벌을 받을 겁니다",
        "가족과 연락을 끊어야 합니다",
        "떠나면 벌이 내릴 것입니다",
        "바깥 사람들은 다 적입니다",
        "조직을 배신하면 후폭풍이 올 겁니다",
        "돈을 내야만 보호받을 수 있습니다",
        "밖에서 일어나는 일을 알려주면 안 돼",
        "우리가 준 대본대로 말해",
        "더 많은 사람을 데려와",
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
     bge_anchors=_ANCHOR_PLACEHOLDER_COERCIVE_ORG_CONTROL,,
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
        # ── 英文极端主义传播 ──
        "the end of the world is coming",
        "follow the true path to salvation",
        "our leader is the chosen one",
        "the final judgment is upon us",
        "you must spread our message to save souls",
        "the world will be purified",
        "only believers will be spared",
        "join us to escape the coming disaster",
        "our teachings are the ultimate truth",
        "non-believers will face divine punishment",
        # ── 日文极端主义传播 ──
        "世界の終わりが来る",
        "救済されるのは信者だけ",
        "私たちの教えこそ真実",
        "裁きの日が近い",
        "指導者は選ばれた存在",
        "末法の世だからこそ",
        "信じる者だけが助かる",
        "人々を導かなければならない",
        "来るべき災厄から逃れられる",
        # ── 韩语极端主义传播 ──
        "세상 끝이 온다",
        "구원받는 것은 신자뿐이다",
        "우리의 가르침이 진리다",
        "심판의 날이 다가온다",
        "지도자는 선택받은 자다",
        "불신자는 벌을 받을 것이다",
        "우리의 메시지를 전파해야 한다",
        "재앙이 오기 전에 우리에게 합류하라",
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
    bge_anchors=_ANCHOR_PLACEHOLDER_EXTREMIST_PROPAGANDA,,
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
        "在他车上动手"bge_anchors=_ANCHOR_PLACEHOLDER_COORDINATED_BROADCAST,,
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
        # ── 日文情感（殺豬盤信号） ──
        "お疲れ様ゆっくり休んで",
        "体が心配だよ",
        "夜更かしは肌に悪いよ",
        "ご飯ちゃんと食べた",
        "おはよう朝ごはん食べた",
        "今日寒いから暖かくして",
        "一緒にいると楽しい",
        "あなたは本当に素敵だよ",
        "あなたのことがもっと好きになった",
        "いつも考えてるよ",
        "帰ったら連絡してね",
        "一人で気をつけて",
        "何かあったらすぐ連絡して",
        "将来は私が面倒見る",
        "あなたの声が好き",
        "いつか会いたいな",
        "ずっとこうして話していたい",
        "夢で会ったよ",
        "起きて一番にあなたのこと考えた",
        "あなたが世界で一番大切",
        # ── 韩语情感（殺豬盤信号） ──
        "너무 힘들겠다 쉬어",
        "건강이 걱정돼",
        "밥 잘 챙겨 먹어",
        "안녕 아침 먹었어",
        "오늘 추워 따뜻하게 입어",
        "너랑 대화하면 행복해",
        "너 정말 대단하다",
        "너 더 좋아졌어",
        "항상 생각하고 있어",
        "도착하면 연락해 줘",
        "혼자 조심해",
        "무슨 일 있으면 바로 연락해",
        "앞으로 내가 챙겨줄게",
        "너 목소리 좋다",
        "언젠가 만나고 싶어",
        "꿈에서 봤어",
        "일어나자마자 네 생각났어",
        "넌 내 세상에서 제일 소중해",
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
    cabge_anchors=_ANCHOR_PLACEHOLDER_INCITEMENT_TO_VIOLENCE,bge_anchors=_ANCHOR_PLACEHOLDER_EMOTION,bge_anchors=_ANCHOR_PLACEHOLDER_COMPLIANCE,bge_anchors=_ANCHOR_PLACEHOLDER_INTERROGATION,bge_anchors=_ANCHOR_PLACEHOLDER_REJECTION,bge_anchors=_ANCHOR_PLACEHOLDER_DISMISSAL,bge_anchors=_ANCHOR_PLACEHOLDER_INBOUND_OFFICIAL_IVR,bge_anchors=_ANCHOR_PLACEHOLDER_CSR_BOT_WHITELIST,bge_anchors=_ANCHOR_PLACEHOLDER_INBOUND_USER_REQUEST,bge_anchors=_ANCHOR_PLACEHOLDER_E_COMMERCE_CS,bge_anchors=_ANCHOR_PLACEHOLDER_MASS_GRIEVANCE,bge_anchors=_ANCHOR_PLACEHOLDER_VOICEMAIL_IVR,bge_anchors=_ANCHOR_PLACEHOLDER_CORPORATE_LOGISTICS,bge_anchors=_ANCHOR_PLACEHOLDER_CORPORATE_BIDDING,bge_anchors=_ANCHOR_PLACEHOLDER_INDUSTRIAL_REPORT,bge_anchors=_ANCHOR_PLACEHOLDER_CASUAL_CHAT,bge_anchors=_ANCHOR_PLACEHOLDER_DELIVERY_EXPRESS,bge_anchors=_ANCHOR_PLACEHOLDER_WRONG_NUMBER,bge_anchors=_ANCHOR_PLACEHOLDER_BRUSH_OFF_TELEMARKETING,bge_anchors=_ANCHOR_PLACEHOLDER_SHORT_GREETING_HANGUP,bge_anchors=_ANCHOR_PLACEHOLDER_GLOBAL_SYNTAX_REGISTRY,