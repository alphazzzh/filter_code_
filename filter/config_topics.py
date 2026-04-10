# config_topics.py  ── V5.1 配置驱动架构（行为学特征增强）
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
# V5.1 变更摘要
# ─────────────────────────────────────────────────────────────
# ① SyntaxRuleType 新增 4 种行为学/心理学特征类型：
#      ISOLATION_REQUEST / MICRO_ACTION_COMMAND /
#      CONDITIONAL_THREAT / ACTION_TARGET_TRIPLET
# ② ScoringRules 新增 confidence_discount 字段（规则后端衰减系数）
# ③ standalone_score 全面压低，防止单项意图直达危险区
# ④ 多语言支持：锚点词汇中文全覆盖，英文部分覆盖（~15 主题），
#    日/韩仅 inbound_user_request 等少量主题覆盖，粤语锚点待补充
# ⑤ 新增 PROFANITY_REGISTRY（脏话/攻击性词汇库）
# ⑥ 新增 GLOBAL_REDLINE_REGISTRY（红线前置熔断正则）
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
from typing import Any, Callable, Union
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 枚举：主题类别 & 句法规则类型 & 匹配模式
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

    # ── V5.1 新增：行为学/心理学特征（对抗话术软化）───────────

    ISOLATION_REQUEST = "isolation_request"
    # 物理隔离与信息阻断检测
    # 探测维度：空间隔离（找个没人的房间/把门反锁）、通讯阻断（不要挂电话/
    #   开启飞行模式/拦截短信）、社交阻断（不能告诉家人/国家机密）
    # params: isolation_keywords (list[str]) — 隔离/阻断指令词库
    # 产出: feature_name → bool

    MICRO_ACTION_COMMAND = "micro_action_command"
    # 服从性测试与微动作指令检测
    # 探测维度：手机设备控制（打开免提/点右上角/跟着我读）、屏幕操作引导
    # params: device_action_keywords (list[str]) — 设备/操作微指令词库
    # 产出: feature_name → bool

    CONDITIONAL_THREAT = "conditional_threat"
    # 条件胁迫与逻辑陷阱检测（升级 KEYWORD_COOC）
    # 利用「条件从句 + 负面主句」结构：[如果不...] → [征信/违法/拘留]
    # params: condition_clauses (list[str]), threat_clauses (list[str])
    # 产出: feature_name → bool

    ACTION_TARGET_TRIPLET = "action_target_triplet"
    # 语义角色三元组检测（替代粗暴 NER_DENSITY）
    # 提取 [施事者] + [动作: 转移/归集/下载] + [受事者: 资金/屏幕/验证码]
    # params: action_verbs (list[str]), target_entities (list[str])
    # 产出: feature_name → bool

    # ── V5.2 新增：更丰富的行为与心理特征 ───────────────

    TEMPORAL_URGENCY = "temporal_urgency"
    # 时间压力表达：制造紧迫感催促受害者立即行动
    # params: keywords (list[str]) — 时间压力词库
    # 产出: feature_name → bool

    PRIVACY_INTRUSION = "privacy_intrusion"
    # 隐私信息索取：要求受害者提供敏感个人信息
    # params: keywords (list[str]) — 隐私信息索取词库
    # 产出: feature_name → bool

    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    # 情绪操控：利用情感绑定（心疼/担心/为你好）建立信任
    # params: keywords (list[str]) — 情绪操控词库
    # 产出: feature_name → bool

    FINANCIAL_FLOW = "financial_flow"
    # 资金流向描述：涉及资金转移/归集/缴纳（复用三元组提取器，LTP 增强）
    # params: verbs (list[str]), targets (list[str])
    # 产出: feature_name → bool

    IDENTITY_IMPERSONATION = "identity_impersonation"
    # 身份冒充标识：伪装公检法/银行/平台工作人员身份
    # params: keywords (list[str]) — 身份冒充词库
    # 产出: feature_name → bool

    CHANNEL_SHIFTING = "channel_shifting"
    # 渠道转移引导：引导受害者转移到非官方通信渠道
    # params: keywords (list[str]) — 渠道转移词库
    # 产出: feature_name → bool


class MatchMode(str, Enum):
    """
    V5.3 关键词匹配模式：控制纯字符串提取器的词边界策略。

    SUBSTR        : 子串包含（kw in text），向后兼容，默认模式
    EXACT_WORD    : 精确词匹配 — LTP/HanLP 后端使用分词结果（kw in tokens），
                    rule_based 后端因 tokens 为逐字拆分不可用，自动退回单字边界正则
    REGEX_BOUNDARY: 中文词边界正则 — 单字关键词要求前后不是中文字符/\\w，
                    多字关键词退回 SUBSTR（多字本身误报率低）
    """
    SUBSTR         = "substr"
    EXACT_WORD     = "exact_word"
    REGEX_BOUNDARY = "regex_boundary"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局评分参数 — 执行引擎的硬编码值收拢到此
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 设计原则：所有结构性/拓扑级评分参数不写死在 stage_three_scorer.py 中，
# 由配置中枢统一管控，运行时只需修改此处即可生效。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GLOBAL_SCORING_CONFIG = {
    # ── 受害者抵抗阶梯降权 ──
    "enable_resistance_discount": True,     # 全局开关，False 即屏蔽阶梯减分
    "resistance_tier1_penalty":   -10,      # Tier1 轻度抗拒
    "resistance_tier2_penalty":   -20,      # Tier2 中度质疑
    "resistance_tier3_penalty":   -35,      # Tier3 明识破/高频拒绝
    "resistance_compliance_immunity_rate": 0.20,  # 顺从率阈值（≥此值免疫）
    "resistance_tier3_rate":      0.25,     # Tier3 抵抗率阈值
    "resistance_tier2_rate":      0.15,     # Tier2 抵抗率阈值
    # ── 拓扑结构性惩罚 ──
    "structural_chitchat_penalty": -20,     # 对称平权聊天且无顺从 → 结构性降权
    # ── Bot × 意图融合 ──
    "bot_fusion_penalty":          15,      # 融合惩罚加分
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic 强类型参数契约
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 每种 SyntaxRuleType 对应一个 Pydantic BaseModel，
# 在 SyntaxRuleConfig.__post_init__ 中自动校验。
# 写错 key 拼写 → 系统启动时 ValidationError 立刻崩溃。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImperativeSyntaxParams(BaseModel):
    """IMPERATIVE_SYNTAX 参数：第二人称 + 紧迫状语"""
    second_person:  list[str]  = Field(..., min_length=1, description="第二人称代词列表")
    urgency_adverbs: list[str] = Field(..., min_length=1, description="紧迫状语列表")

class QuantityRegexParams(BaseModel):
    """QUANTITY_REGEX 参数：量词列表"""
    quantity_units: list[str] = Field(..., min_length=1, description="量词列表")

class NerDensityParams(BaseModel):
    """NER_DENSITY 参数：实体类型 + 阈值"""
    entity_types: list[str] = Field(..., min_length=1, description="NER 实体类型列表")
    threshold:    int       = Field(default=3, ge=1, description="实体数量阈值")

class KeywordCoocParams(BaseModel):
    """KEYWORD_COOC 参数：多组关键词集合"""
    keyword_sets: list[list[str]] = Field(..., min_length=1, description="关键词分组，每组至少一个列表")
    match_mode:   MatchMode       = Field(default=MatchMode.SUBSTR, description="匹配模式：substr/exact_word/regex_boundary")

    @field_validator("keyword_sets")
    @classmethod
    def _no_empty_sets(cls, v: list[list[str]]) -> list[list[str]]:
        for i, s in enumerate(v):
            if not s:
                raise ValueError(f"keyword_sets[{i}] 不能为空列表")
        return v

class RegexPatternParams(BaseModel):
    """REGEX_PATTERN 参数：正则模式字符串 + 标志"""
    pattern: str = Field(..., min_length=1, description="正则模式字符串")
    flags:   str = Field(default="UNICODE", description="re 模块标志名")

class VerbEntitySparsityParams(BaseModel):
    """VERB_ENTITY_SPARSITY 参数：实体稀疏阈值"""
    threshold: int = Field(default=3, ge=1, description="业务实体数量阈值")

class IsolationRequestParams(BaseModel):
    """ISOLATION_REQUEST 参数：隔离/阻断指令词库"""
    isolation_keywords: list[str] = Field(..., min_length=1, description="隔离/阻断指令词库")
    match_mode:         MatchMode = Field(default=MatchMode.SUBSTR, description="匹配模式")

class MicroActionCommandParams(BaseModel):
    """MICRO_ACTION_COMMAND 参数：设备/操作微指令词库"""
    device_action_keywords: list[str] = Field(..., min_length=1, description="设备/操作微指令词库")
    match_mode:             MatchMode = Field(default=MatchMode.SUBSTR, description="匹配模式")

class ConditionalThreatParams(BaseModel):
    """CONDITIONAL_THREAT 参数：条件从句 + 威胁主句"""
    condition_clauses: list[str] = Field(..., min_length=1, description="条件从句词库")
    threat_clauses:   list[str] = Field(..., min_length=1, description="威胁主句词库")
    match_mode:       MatchMode = Field(default=MatchMode.SUBSTR, description="匹配模式")

class ActionTargetTripletParams(BaseModel):
    """ACTION_TARGET_TRIPLET 参数：施事动词 + 受事实体"""
    action_verbs:    list[str] = Field(..., min_length=1, description="施事动词列表")
    target_entities: list[str] = Field(..., min_length=1, description="受事实体列表")
    match_mode:     MatchMode = Field(default=MatchMode.SUBSTR, description="匹配模式")

class SimpleKeywordsParams(BaseModel):
    """
    通用关键字匹配参数。
    用于 TEMPORAL_URGENCY / PRIVACY_INTRUSION / EMOTIONAL_MANIPULATION /
    IDENTITY_IMPERSONATION / CHANNEL_SHIFTING 等纯字符串匹配规则。
    """
    keywords:   list[str] = Field(..., min_length=1, description="关键字词库")
    match_mode: MatchMode = Field(default=MatchMode.SUBSTR, description="匹配模式")


# ── 映射表：SyntaxRuleType → Pydantic 模型类 ────────────────
# 未在此表中的类型（如未来扩展）将跳过校验，保持向后兼容。
_RULE_TYPE_PARAMS_MAP: dict[SyntaxRuleType, type[BaseModel]] = {
    SyntaxRuleType.IMPERATIVE_SYNTAX:      ImperativeSyntaxParams,
    SyntaxRuleType.QUANTITY_REGEX:         QuantityRegexParams,
    SyntaxRuleType.NER_DENSITY:            NerDensityParams,
    SyntaxRuleType.KEYWORD_COOC:           KeywordCoocParams,
    SyntaxRuleType.REGEX_PATTERN:          RegexPatternParams,
    SyntaxRuleType.VERB_ENTITY_SPARSITY:   VerbEntitySparsityParams,
    SyntaxRuleType.ISOLATION_REQUEST:      IsolationRequestParams,
    SyntaxRuleType.MICRO_ACTION_COMMAND:   MicroActionCommandParams,
    SyntaxRuleType.CONDITIONAL_THREAT:     ConditionalThreatParams,
    SyntaxRuleType.ACTION_TARGET_TRIPLET:  ActionTargetTripletParams,
    SyntaxRuleType.FINANCIAL_FLOW:         ActionTargetTripletParams,  # 复用三元组参数结构
    SyntaxRuleType.TEMPORAL_URGENCY:       SimpleKeywordsParams,
    SyntaxRuleType.PRIVACY_INTRUSION:      SimpleKeywordsParams,
    SyntaxRuleType.EMOTIONAL_MANIPULATION: SimpleKeywordsParams,
    SyntaxRuleType.IDENTITY_IMPERSONATION: SimpleKeywordsParams,
    SyntaxRuleType.CHANNEL_SHIFTING:       SimpleKeywordsParams,
}

# 反向映射：Pydantic 模型类 → 列表字段名（用于 evidence 收集）
_PARAMS_KEYWORDS_FIELD: dict[type[BaseModel], str] = {
    IsolationRequestParams:      "isolation_keywords",
    MicroActionCommandParams:    "device_action_keywords",
    SimpleKeywordsParams:        "keywords",
    ActionTargetTripletParams:   "action_verbs",      # FINANCIAL_FLOW 也用这个
}


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
    params       : Pydantic 强类型参数模型（自动校验 key 拼写和类型）
    evidence_key : 可选，产出的「证据列表」键名（如匹配字符串、触发词汇）
    """
    rule_type:    SyntaxRuleType
    feature_name: str
    params:       Union[BaseModel, dict[str, Any]] = field(default_factory=dict)
    evidence_key: str | None = None  # 存放证据列表的键名，None=不保存证据

    def __post_init__(self) -> None:
        """
        构造后自动校验：如果 params 是原始 dict，尝试转换为对应的 Pydantic 模型。
        如果 rule_type 已注册到 _RULE_TYPE_PARAMS_MAP 中，强制校验。
        校验失败时抛出 pydantic.ValidationError，系统无法启动。
        """
        if isinstance(self.params, dict) and not self.params:
            return  # 空字典跳过校验（兼容 VERB_ENTITY_SPARSITY 等仅设 threshold 的情况）
        model_cls = _RULE_TYPE_PARAMS_MAP.get(self.rule_type)
        if model_cls is not None:
            if isinstance(self.params, dict):
                # dict → Pydantic 模型，触发校验
                self.params = model_cls.model_validate(self.params)
            elif isinstance(self.params, BaseModel) and not isinstance(self.params, model_cls):
                # 模型类型不匹配
                raise TypeError(
                    f"SyntaxRuleConfig({self.rule_type.value!r}): "
                    f"期望 params 为 {model_cls.__name__}，"
                    f"实际为 {type(self.params).__name__}"
                )
            # isinstance(self.params, model_cls) → 校验通过，无需操作


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
    # 如果为 True，则该组合无需软意图（topic_id）命中即可独立触发
    # 常用于正则硬探针（如 insurance_scam_keywords），confidence_discount 不打折
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
    confidence_discount : 当矩阵触发的硬特征由 rule_based 后端产出时的衰减系数
                          1.0 = 不衰减（有 NLP 支撑），0.5 = 折半（仅正则兜底）
    """
    standalone_score:    int                     = 0
    standalone_tag:      str                     = ""
    matrix_combinations: list[MatrixCombination] = field(default_factory=list)
    whitelist_discount:  float                   = 1.0  # 默认不折扣
    confidence_discount: float                   = 0.5  # 默认折半（rule_based 无依存句法）


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
    topic_family: str = "general"               # V5.2: 主题族，用于跨族复合加分
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
        # 👇 只有轮次少 且 字数少，才算是真废料
        condition=lambda ctx: ctx.get("valid_turn_count", 0) <= 3 and ctx.get("total_words", 0) < 30
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
    topic_family= "fraud",
    description = "诈骗黑话/地下产业链隐语（跑分/水房/卡接等），脱离上下文即高度可疑",
    threshold   = 0.70,  # 黑话锚点特异性强，适当降低阈值提升召回
        bge_anchors=[],
    syntax_rules = [],  # 黑话本身即强信号，无需额外句法规则
    scoring_rules = ScoringRules(
        standalone_score = 10,    # V5.1: 压低（原20），仅黑话最多爬到 60 分
        standalone_tag   = "has_fraud_jargon",
        matrix_combinations = [
            MatrixCombination("has_imperative_syntax",          15, "fraud_imperative_jargon"),
            MatrixCombination("high_entity_density",            10, "fraud_jargon_dense"),
            MatrixCombination("has_isolation_request",          18, "fraud_jargon_isolation"),     # V5.1: 黑话+隔离=高危
            MatrixCombination("has_micro_action_command",       12, "fraud_jargon_micro_cmd"),     # V5.1: 黑话+微动作
            MatrixCombination("has_conditional_threat",         15, "fraud_jargon_threat"),        # V5.1: 黑话+胁迫
            MatrixCombination("has_action_target_triplet",      18, "fraud_jargon_triplet"),       # V5.1: 黑话+三元组=高危
        ],
    ),
)

_TOPIC_FRAUD_OBJECT = TopicDefinition(
    topic_id    = "fraud_object",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "fraud",
    description = "诈骗高危业务客体（验证码/屏幕共享/账户转账等），是诈骗指令的「宾语」成分",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperative_verbs",
            params = ImperativeSyntaxParams(
                second_person=["你", "您", "you"],
                urgency_adverbs=[
                    "马上", "立刻", "立即", "赶紧", "赶快",
                    "现在", "快", "即刻", "迅速",
                    "now", "immediately", "right now", "asap",
                ],
            ),
        ),
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = NerDensityParams(
                entity_types=["Ni", "Ns", "ORG", "GPE", "LOC"],
                threshold=3,
            ),
        ),
        # V5.1 新增：物理隔离与信息阻断（公检法诈骗绝对高危前置动作）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.ISOLATION_REQUEST,
            feature_name = "has_isolation_request",
            evidence_key = "isolation_evidence",
            params = IsolationRequestParams(
                isolation_keywords=[
                    # 空间隔离
                    "找个没人的地方", "找个没人的房间", "到外面去", "把门反锁",
                    "反锁门", "关上门", "独自一人", "没有人的地方",
                    # 通讯阻断
                    "不要挂电话", "千万别挂", "不能挂断", "别挂",
                    "开启飞行模式", "打开飞行模式", "关掉WiFi", "关掉 wifi",
                    "拦截短信", "不要接电话", "不要回短信", "把手机静音",
                    # 社交阻断
                    "不能告诉家人", "不能告诉任何人", "这是国家机密",
                    "保密", "不要跟别人说", "不要告诉朋友", "对谁都不能说",
                    "你自己知道就好", "这件事不能让第三个人知道",
                    # 英文隔离
                    "don't tell anyone", "keep it secret", "go to a private room",
                    "don't hang up", "turn off your phone",
                ],
            ),
        ),
        # V5.1 新增：服从性测试微动作指令
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.MICRO_ACTION_COMMAND,
            feature_name = "has_micro_action_command",
            evidence_key = "micro_action_evidence",
            params = MicroActionCommandParams(
                device_action_keywords=[
                    # 手机设备操作
                    "打开免提", "开免提", "点右上角", "点一下右上角",
                    "点设置", "点击设置", "打开设置", "进入设置",
                    "往下滑", "滑到底部", "点击链接", "点开链接",
                    "扫码", "扫二维码", "截图", "截个屏", "录屏",
                    # 行为控制测试
                    "跟着我读", "你现在打开", "你点一下", "你按一下",
                    "不要动手机", "手机放一边", "按照我说的做",
                    "一步一步来", "你先", "然后", "再点",
                    # 英文微指令
                    "open your settings", "tap the icon", "click the link",
                    "follow my instructions", "do exactly as I say",
                ],
            ),
        ),
        # V5.1 新增：语义角色三元组（谁对实体做了什么）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.ACTION_TARGET_TRIPLET,
            feature_name = "has_action_target_triplet",
            evidence_key = "triplet_evidence",
            params = ActionTargetTripletParams(
                action_verbs=[
                    "转账", "汇款", "打钱", "输入", "提供", "告知",
                    "发送", "下载", "安装", "共享", "截图", "录屏",
                    "归集", "转移", "提取", "验证", "确认", "授权",
                    "tell me", "send me", "transfer", "share your",
                    "download", "install", "verify",
                ],
                target_entities=[
                    "验证码", "密码", "短信", "屏幕", "账户", "银行卡",
                    "身份证", "余额", "资金", "收款码", "付款码",
                    "信用卡", "cvv", "otp", "one-time password",
                    "screen", "password", "bank account", "credit card",
                ],
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 8,    # V5.1: 压低单项（原18），仅提敏感词最多爬到 58 分（关注区）
        standalone_tag   = "has_fraud_object",
        matrix_combinations = [
            # 祈使句 × 诈骗客体 = 「你现在立刻把验证码发给我」→ 高危
            MatrixCombination("has_imperative_syntax",       20, "fraud_imperative_object"),
            MatrixCombination("high_entity_density",          8, "fraud_object_dense"),
            # V5.1 新增矩阵：行为学特征 × 诈骗客体
            MatrixCombination("has_isolation_request",       22, "fraud_isolation_object"),      # 隔离+客体=极高危
            MatrixCombination("has_micro_action_command",    18, "fraud_micro_cmd_object"),       # 微动作+客体=高危
            MatrixCombination("has_action_target_triplet",   20, "fraud_triplet_object"),         # 三元组+客体=高危
            MatrixCombination("has_conditional_threat",      18, "fraud_conditional_threat"),     # 条件胁迫+客体=高危
        ],
    ),
)

_TOPIC_AUTHORITY_ENTITY = TopicDefinition(
    topic_id    = "authority_entity",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "fraud",
    description = "权威伪装实体（冒充公检法/监管机构），制造权威压迫感",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_imperative_syntax",
            evidence_key = "imperative_verbs",
            params = ImperativeSyntaxParams(
                second_person=["你", "您", "you"],
                urgency_adverbs=[
                    "马上", "立刻", "立即", "赶紧", "赶快",
                    "现在", "快", "即刻", "迅速",
                    "now", "immediately", "right now",
                ],
            ),
        ),
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = NerDensityParams(
                entity_types=["Ni", "Ns", "ORG", "GPE", "LOC"],
                threshold=3,
            ),
        ),
        # V5.1: 条件胁迫（公检法诈骗核心：不配合→拘留/征信）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CONDITIONAL_THREAT,
            feature_name = "has_conditional_threat",
            evidence_key = "conditional_threat_evidence",
            params = ConditionalThreatParams(
                condition_clauses=[
                    "如果不", "如果不配合", "如果不处理", "如果不转账",
                    "如果你不", "要是你不", "不配合的话", "不处理的话",
                    "否则", "一旦", "要是", "假如不",
                    "if you don't", "otherwise", "if not",
                ],
                threat_clauses=[
                    "征信", "征信受损", "信用记录", "影响征信",
                    "涉嫌", "涉嫌违法", "涉嫌犯罪", "涉嫌洗钱",
                    "拘留", "逮捕", "判刑", "坐牢", "通缉",
                    "冻结", "冻结账户", "冻结资产", "封号",
                    "后果", "法律责任", "承担后果",
                    "影响子女", "影响家人", "牵连",
                    "criminal", "arrest", "frozen", "credit score",
                ],
            ),
        ),
        # V5.2 新增：身份冒充标识（公检法诈骗核心起手式）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IDENTITY_IMPERSONATION,
            feature_name = "has_identity_impersonation",
            evidence_key = "impersonation_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    # 公检法冒充
                    "我是警察", "我是警官", "我是公安局", "我是刑侦",
                    "我是检察官", "我是法院", "我是监管局", "我是银保监",
                    "XX分局", "XX派出所", "刑侦大队",
                    # 银行/平台冒充
                    "我是银行工作人员", "客服中心", "风控部门",
                    "银联", "反诈中心", "征信中心",
                    # 英文冒充
                    "I'm calling from", "this is the police", "IRS calling",
                    "FBI", "fraud department",
                ],
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 10,    # V5.1: 压低单项（原22）
        standalone_tag   = "has_authority_entity",
        matrix_combinations = [
            MatrixCombination("has_imperative_syntax",      18, "fraud_authority_pressure"),
            MatrixCombination("high_entity_density",        12, "fraud_authority_entity_dense"),
            MatrixCombination("has_isolation_request",      22, "authority_isolation_extreme"),    # V5.1: 隔离+权威=极危
            MatrixCombination("has_conditional_threat",     25, "authority_conditional_coercion"),  # V5.1: 条件胁迫+权威=极高危
            MatrixCombination("has_identity_impersonation", 20, "authority_impersonation_extreme"), # V5.2: 身份冒充+权威=极高危
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 2：涉毒交易（两个子槽位）
# ─────────────────────────────────────────────────────────────

_TOPIC_DRUG_JARGON = TopicDefinition(
    topic_id    = "drug_jargon",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "narcotics",
    description = "毒品隐语（白/冰/K粉/草等），写场景句提高语境特异性",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.QUANTITY_REGEX,
            feature_name = "has_drug_quantity",
            evidence_key = "drug_quantity_matches",
            params = QuantityRegexParams(
                quantity_units=[
                    "克", "g", "G", "公克", "mg",
                    "包", "手", "份", "颗", "粒",
                ],
            ),
        ),
        # V5.2 新增：渠道转移（暗语化线上交易特征）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CHANNEL_SHIFTING,
            feature_name = "has_channel_shifting",
            evidence_key = "channel_shifting_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    # 加密通讯
                    "加微信", "加V", "加WX", "微信联系", "加TG",
                    "下载电报", "Telegram", "蝙蝠", "Signal",
                    "阅后即焚", "加密软件", "私密聊天", "加密通讯",
                    # 线上交易渠道
                    "转账到这里", "打到这个", "扫码付款", "收款码",
                    # 英文暗语渠道
                    "WhatsApp", "Snapchat", "disappearing messages",
                ],
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,    # V5.2: 适度抬高（有矩阵支撑）
        standalone_tag   = "has_drug_jargon",
        matrix_combinations = [
            MatrixCombination("has_drug_quantity", 25, "drug_quantity_jargon"),
            MatrixCombination("high_entity_density", 10, "drug_jargon_dense"),
            MatrixCombination("has_channel_shifting", 20, "drug_jargon_covert_channel"),  # V5.2: 暗语+渠道转移=高危
        ],
    ),
)

_TOPIC_DRUG_CHAIN = TopicDefinition(
    topic_id    = "drug_chain",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "narcotics",
    description = "毒品交易链条（上下游关系/物流/货款等），与 drug_jargon + has_drug_quantity 共现还原完整交易模型",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.QUANTITY_REGEX,
            feature_name = "has_drug_quantity",
            evidence_key = "drug_quantity_matches",
            params = QuantityRegexParams(
                quantity_units=[
                    "克", "g", "G", "公克", "mg",
                    "包", "手", "份", "颗", "粒",
                ],
            ),
        ),
        # V5.2 新增：资金流向（毒品交易核心：钱和货的流向）
        # NOTE: FINANCIAL_FLOW 复用 ActionTargetTripletParams，key 必须为
        #       action_verbs / target_entities（与三元组提取器一致）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.FINANCIAL_FLOW,
            feature_name = "has_financial_flow",
            evidence_key = "financial_flow_evidence",
            params = ActionTargetTripletParams(
                action_verbs=[
                    "打到", "转到", "汇入", "汇到", "打钱", "转账",
                    "付钱", "给钱", "发红包", "扫码", "充值",
                    "withdraw", "wire transfer", "send money",
                ],
                target_entities=[
                    "账户", "卡号", "银行卡", "支付宝", "微信",
                    "收款码", "地址", "到付", "货到付款",
                    "account", "card", "bitcoin", "crypto",
                ],
            ),
        ),
        # V5.3 新增：毒品交易动宾三元组（LTP 增强检测"出货→克"等暗语结构）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.ACTION_TARGET_TRIPLET,
            feature_name = "has_drug_triplet",
            evidence_key = "drug_triplet_evidence",
            params = ActionTargetTripletParams(
                action_verbs=["拿", "发", "走", "带", "出货", "拿货", "接货", "送货", "走货"],
                target_entities=["肉", "冰", "货", "手", "克", "包", "份", "K粉", "草", "丸子"],
                match_mode=MatchMode.EXACT_WORD,  # V5.3: "走"不能匹配"走路"
            ),
        ),
        # V5.3 新增：交易链胁迫（"不付→断货/涨价"）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CONDITIONAL_THREAT,
            feature_name = "has_drug_chain_threat",
            evidence_key = "drug_chain_threat_evidence",
            params = ConditionalThreatParams(
                condition_clauses=["不付", "不发货", "不打款", "不转账", "没钱"],
                threat_clauses=["断货", "涨价", "不送了", "没货", "终止合作"],
                match_mode=MatchMode.SUBSTR,  # 短语级匹配，子串包含即可
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 12,     # V5.2: 适度抬高（有矩阵支撑）
        standalone_tag   = "has_drug_chain",
        matrix_combinations = [
            MatrixCombination("has_drug_quantity",       18, "drug_quantity_chain"),
            MatrixCombination("high_entity_density",      8, "drug_chain_dense"),
            MatrixCombination("has_financial_flow",      22, "drug_chain_financial_flow"),  # V5.2: 交易链+资金流向=高危
            MatrixCombination("has_channel_shifting",    18, "drug_chain_covert_channel"),  # V5.2: 交易链+渠道转移
            MatrixCombination("has_drug_triplet",        20, "drug_chain_action_target"),   # V5.3: 交易动宾结构
            MatrixCombination("has_drug_chain_threat",   18, "drug_chain_coercion"),        # V5.3: 交易链胁迫
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
    topic_family= "coercion",
    description = (
        "有组织强制控制行为：通过金融勒索/孤立威胁/退出惩罚实施心理控制，"
        "聚焦犯罪行为模式，与具体团体名称或信仰内容无关"
    ),
    threshold   = 0.74,
        bge_anchors=[],
    syntax_rules = [
        # V5.3: 替换 KEYWORD_COOC 为 CONDITIONAL_THREAT（句法确认条件胁迫结构）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CONDITIONAL_THREAT,
            feature_name = "has_coercive_threat",
            evidence_key = "coercive_threat_evidence",
            params = ConditionalThreatParams(
                condition_clauses=["如果不", "否则", "一旦", "除非"],
                threat_clauses=["惩罚", "报应", "后果", "灾难", "灾祸", "驱逐", "出事"],
                match_mode=MatchMode.REGEX_BOUNDARY,  # V5.3: 词边界保护
            ),
        ),
        # 检测「财务勒索」：金融词 + 威胁/强迫词共现
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.KEYWORD_COOC,
            feature_name = "has_coercive_financial_demand",
            params = KeywordCoocParams(
                keyword_sets=[
                    # 集合 A：金融/财务词
                    ["捐", "缴", "交钱", "付款", "费用", "资金", "献"],
                    # 集合 B：强迫/威胁词
                    ["必须", "否则", "不然", "要不然", "后果", "惩罚",
                     "报应", "灾祸", "不交就"],
                ],
                match_mode=MatchMode.REGEX_BOUNDARY,  # V5.3: "交"→"交警" 误报保护
            ),
            evidence_key = None,
        ),
        # 检测「退出威胁」：离开词 + 惩罚后果词共现
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.KEYWORD_COOC,
            feature_name = "has_exit_threat",
            params = KeywordCoocParams(
                keyword_sets=[
                    # 集合 A：退出/离开词
                    ["退出", "离开", "背叛", "不服从", "反对"],
                    # 集合 B：惩罚/后果词
                    ["惩罚", "报应", "后果", "驱逐", "出事", "灾难"],
                ],
                match_mode=MatchMode.REGEX_BOUNDARY,  # V5.3: 词边界保护
            ),
            evidence_key = None,
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 12,    # V5.1: 压低（原28）
        standalone_tag   = "coercive_org_behavior",
        matrix_combinations = [
            MatrixCombination("has_coercive_financial_demand", 20, "critical_coercive_financial"),   # V5.1: 压低（原40）
            MatrixCombination("has_exit_threat",               15, "critical_exit_coercion"),        # V5.1: 压低（原30）
            MatrixCombination("has_isolation_request",         18, "coercive_isolation_extreme"),    # V5.1: 隔离+强制控制=极危
            MatrixCombination("has_conditional_threat",        18, "coercive_conditional_threat"),   # V5.1: 条件胁迫+强制控制
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
    topic_family= "extremism",
    description = "极端思想与邪教传播：已知邪教组织标识性用语、末世论、度人话术等",
    threshold   = 0.70,
        bge_anchors=[],
    syntax_rules = [
        # V5.2 新增：情绪操控（邪教核心：感情绑定+末日恐慌）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.EMOTIONAL_MANIPULATION,
            feature_name = "has_emotional_manipulation",
            evidence_key = "emotional_manipulation_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "福报", "拯救", "赎罪", "神明", "唯一出路",
                    "末日", "灾难", "审判", "度人", "修行",
                    "为你好", "心疼你", "我是为你着想", "只有我能帮你",
                    "家人不理解", "世人不懂", "开悟", "觉醒",
                    "blessing", "salvation", "the only way", "enlightenment",
                ],
            ),
        ),
        # V5.2 新增：资金流向（邪教敛财核心）
        # NOTE: FINANCIAL_FLOW 复用 ActionTargetTripletParams，key 必须为
        #       action_verbs / target_entities（与三元组提取器一致）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.FINANCIAL_FLOW,
            feature_name = "has_extremist_financial_flow",
            evidence_key = "extremist_financial_evidence",
            params = ActionTargetTripletParams(
                action_verbs=["捐", "献", "交", "奉献", "供养", "布施", "奉献给", "上交"],
                target_entities=["会费", "善款", "诚意金", "奉献金", "功德", "香火钱", "组织"],
                match_mode=MatchMode.REGEX_BOUNDARY,  # V5.3: "交"→"交警" 误报保护
            ),
        ),
        # V5.2 新增：渠道转移（邪教传播渠道控制）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CHANNEL_SHIFTING,
            feature_name = "has_extremist_channel_shift",
            evidence_key = "extremist_channel_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "加我们", "加群", "内部群", "核心群", "学习群",
                    "下载APP", "安装软件", "使用这个软件",
                    "不要用微信", "不要在网上说", "私下联系",
                ],
            ),
        ),
        # V5.3 新增：邪教条件胁迫（"不信→天谴/报应"，精神控制核心句法）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CONDITIONAL_THREAT,
            feature_name = "has_extremist_conditional_threat",
            evidence_key = "extremist_conditional_threat_evidence",
            params = ConditionalThreatParams(
                condition_clauses=["不信", "违背", "退出", "反对", "不服从", "质疑"],
                threat_clauses=["天谴", "报应", "神明惩罚", "灾难降临", "审判", "毁灭", "下地狱"],
                match_mode=MatchMode.SUBSTR,  # 短语级匹配
            ),
        ),
        # V5.3 新增：隔离请求（邪教信息阻断：切断外部信息源）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.ISOLATION_REQUEST,
            feature_name = "has_extremist_isolation",
            evidence_key = "extremist_isolation_evidence",
            params = IsolationRequestParams(
                isolation_keywords=[
                    "不要告诉家人", "不要跟别人说", "家人不理解",
                    "世俗", "外面的人不懂", "不要上网查",
                ],
                match_mode=MatchMode.SUBSTR,
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,
        standalone_tag   = "extremist_propaganda",
        matrix_combinations = [
            MatrixCombination("high_entity_density",                    8, "extremist_with_dense_entities"),
            MatrixCombination("has_emotional_manipulation",           20, "extremist_mind_control"),         # V5.2
            MatrixCombination("has_extremist_financial_flow",         25, "extremist_financial_harvest"),     # V5.2
            MatrixCombination("has_extremist_channel_shift",          18, "extremist_channel_control"),       # V5.2
            MatrixCombination("has_extremist_conditional_threat",     20, "extremist_coercion"),              # V5.3
            MatrixCombination("has_extremist_isolation",              18, "extremist_information_blockade"),  # V5.3
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 4：有组织广播行为
# ─────────────────────────────────────────────────────────────

_TOPIC_COORDINATED_BROADCAST = TopicDefinition(
    topic_id    = "coordinated_broadcast",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "subversion",
    description = "有组织批量传播行为：指挥他人大规模、统一口径传播，与内容立场无关",
    threshold   = 0.72,
        bge_anchors=[],
    syntax_rules = [
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.NER_DENSITY,
            feature_name = "high_entity_density",
            evidence_key = "entity_list",
            params = NerDensityParams(
                entity_types=["Ni", "Ns", "ORG", "GPE", "LOC"],
                threshold=3,
            ),
        ),
        # V5.2 新增：渠道转移引导
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CHANNEL_SHIFTING,
            feature_name = "has_broadcast_channel_shift",
            evidence_key = "broadcast_channel_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "转发到", "发到群里", "转发群", "扩散", "传播出去",
                    "统一口径", "按这个发", "复制粘贴", "截图转发",
                    "share this", "forward to", "spread the word",
                ],
            ),
        ),
        # V5.2 新增：时间压力
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.TEMPORAL_URGENCY,
            feature_name = "has_broadcast_urgency",
            evidence_key = "broadcast_urgency_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "马上转发", "立刻扩散", "紧急通知", "十万火急",
                    "赶紧通知", "时间不多了", "快去告诉", "尽快转发",
                    "urgent", "immediately forward", "act now",
                ],
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 12,
        standalone_tag   = "coordinated_broadcast",
        matrix_combinations = [
            MatrixCombination("high_entity_density",          15, "broadcast_with_dense_entities"),
            MatrixCombination("has_broadcast_channel_shift",  20, "broadcast_channel_amplification"),  # V5.2
            MatrixCombination("has_broadcast_urgency",        18, "broadcast_urgent_amplification"),    # V5.2
        ],
    ),
)


# ─────────────────────────────────────────────────────────────
# HIGH_RISK 主题 5：明确煽动暴力
# ─────────────────────────────────────────────────────────────

_TOPIC_INCITEMENT_TO_VIOLENCE = TopicDefinition(
    topic_id    = "incitement_to_violence",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "violence",
    description = "明确号召对具体目标实施身体伤害，排除比喻性/游戏化用法",
    threshold   = 0.76,  # 高精确率阈值，避免误判比喻
        bge_anchors=[],
    syntax_rules = [
        # V5.2 新增：时间压力（暴力煽动的紧迫感）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.TEMPORAL_URGENCY,
            feature_name = "has_violence_urgency",
            evidence_key = "violence_urgency_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "趁现在", "就在今天", "马上行动",
                    "时机已到", "不能再等", "时间紧迫", "抓紧时间",
                    "tonight", "right now", "before it's too late",
                ],
            ),
        ),
        # V5.3 新增：祈使句检测（暴力煽动中的指令性句式："给我打！""冲！"）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.IMPERATIVE_SYNTAX,
            feature_name = "has_violence_imperative",
            evidence_key = "violence_imperative_evidence",
            params = ImperativeSyntaxParams(
                second_person=["你", "你们", "您", "大家", "兄弟们"],
                urgency_adverbs=["给我", "马上", "立刻", "冲", "上", "打", "杀"],
            ),
        ),
        # V5.3 新增：暴力动宾三元组（"打→他们/人"，LTP 增强确认暴力指令）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.ACTION_TARGET_TRIPLET,
            feature_name = "has_violence_triplet",
            evidence_key = "violence_triplet_evidence",
            params = ActionTargetTripletParams(
                action_verbs=["打", "砸", "烧", "杀", "冲", "砍", "围"],
                target_entities=["他们", "他", "她", "人", "店", "车", "门", "房子"],
                match_mode=MatchMode.EXACT_WORD,  # V5.3: "打"不能匹配"打电话"
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 15,
        standalone_tag   = "incitement_to_violence",
        matrix_combinations = [
            MatrixCombination("high_entity_density",       8, "violence_with_dense_entities"),
            MatrixCombination("has_violence_urgency",     20, "violence_urgent_action"),       # V5.2
            MatrixCombination("has_violence_imperative",  22, "violence_imperative_command"),   # V5.3
            MatrixCombination("has_violence_triplet",     20, "violence_action_target"),        # V5.3
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
        standalone_score=-50,
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
    topic_family= "fraud",
    description = "电商/泛金融客服伪装起手式，配合诈骗客体或高压句法形成连招",
    threshold   = 0.72,
    bge_anchors=[],
    syntax_rules = [
        # 微保/百万保障诈骗核爆级硬探针
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.REGEX_PATTERN,
            feature_name = "has_insurance_scam_keywords",
            params = RegexPatternParams(
                pattern=r"(百万保障|微保|安全保险|账户保险|资金安全险).{0,30}(到期|收费|扣费|解除|关闭|续费)|(微信|支付宝|拼多多).{0,30}(百万保障|微保|客服中心)"
            )
        ),
        # 备用金/白条/客服引导链接等泛金融高危探针
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.REGEX_PATTERN,
            feature_name = "has_financial_scam_keywords",
            params = RegexPatternParams(
                pattern=r"(备用金|白条|金条|借呗|微粒贷|百万保障).{0,30}(额度|关闭|激活|服务费|违约金|征信)|(京东|金融|淘宝|客服|银联).{0,20}(注销|额度|回执|屏幕共享|共享屏幕)"
            )
        ),
        # V5.1 新增：服从性测试微动作指令（电商客服特供）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.MICRO_ACTION_COMMAND,
            feature_name = "has_ecommerce_micro_cmd",
            evidence_key = "ecommerce_micro_cmd_evidence",
            params = MicroActionCommandParams(
                device_action_keywords=[
                    # 屏幕操作
                    "打开微信", "打开支付宝", "打开手机银行",
                    "点右上角", "点我的", "点设置", "点击服务",
                    "进入钱包", "进入支付", "往下滑",
                    "点击链接", "扫码", "截个屏", "录屏",
                    "共享屏幕", "屏幕共享",
                    # 引导下载
                    "下载APP", "安装软件", "打开这个APP",
                    # 引导输入
                    "输入验证码", "告诉我验证码", "念一下验证码",
                    "把密码输入", "输入支付密码",
                    # 英文版电商引导
                    "open the app", "go to settings", "share your screen",
                ],
            ),
        ),
        # V5.1 新增：条件胁迫（「不关闭百万保障会扣费」）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CONDITIONAL_THREAT,
            feature_name = "has_ecommerce_conditional_threat",
            evidence_key = "ecommerce_threat_evidence",
            params = ConditionalThreatParams(
                condition_clauses=[
                    "如果不", "如果不关闭", "不处理的话", "不取消",
                    "不及时", "逾期", "今天不",
                    "if you don't", "unless you",
                ],
                threat_clauses=[
                    "扣费", "收费", "自动续费", "每月扣",
                    "影响征信", "征信", "信用",
                    "冻结", "封号", "限制", "无法使用",
                    "charged", "fee", "freeze",
                ],
            ),
        ),
        # V5.2 新增：隐私信息索取（电商诈骗核心收割动作）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.PRIVACY_INTRUSION,
            feature_name = "has_ecommerce_privacy_intrusion",
            evidence_key = "ecommerce_privacy_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "验证码", "短信验证码", "手机验证码", "动态验证码",
                    "身份证号", "身份证正反面", "身份证照片",
                    "银行卡号", "卡号", "信用卡号", "有效期",
                    "CVV", "cvv", "安全码", "背面三位数",
                    "登录密码", "支付密码", "交易密码",
                    "one-time password", "OTP", "social security number",
                ],
            ),
        ),
        # V5.2 新增：渠道转移（离开平台到非官方渠道）
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.CHANNEL_SHIFTING,
            feature_name = "has_ecommerce_channel_shift",
            evidence_key = "ecommerce_channel_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "下载APP", "安装软件", "打开这个APP",
                    "点击链接", "点开链接", "打开网址",
                    "加微信", "加QQ", "加客服微信",
                    "扫码", "扫二维码", "扫这个码",
                    "屏幕共享", "共享屏幕", "开启屏幕共享",
                    "download the app", "click the link", "scan the QR code",
                ],
            ),
        ),
        # V5.2 新增：时间压力
        SyntaxRuleConfig(
            rule_type    = SyntaxRuleType.TEMPORAL_URGENCY,
            feature_name = "has_ecommerce_temporal_urgency",
            evidence_key = "ecommerce_urgency_evidence",
            params = SimpleKeywordsParams(
                keywords=[
                    "今天之内", "逾期不处理", "最后期限", "马上到期",
                    "今天不处理就", "不操作就扣费", "超过24小时",
                    "抓紧时间", "马上处理", "立即取消",
                    "within 24 hours", "before it expires", "act immediately",
                ],
            ),
        ),
    ],
    scoring_rules = ScoringRules(
        standalone_score = 8,     # V5.1: 压低（原15）
        standalone_tag   = "suspicious_fake_cs",
        matrix_combinations = [
            MatrixCombination("has_fraud_object",                 20, "fake_cs_screen_share_trap"),
            MatrixCombination("has_coercive_threat",              18, "fake_cs_financial_threat"),
            MatrixCombination("has_ecommerce_micro_cmd",          20, "fake_cs_micro_cmd_trap"),       # V5.1
            MatrixCombination("has_ecommerce_conditional_threat", 20, "fake_cs_conditional_trap"),      # V5.1
            MatrixCombination("has_isolation_request",            22, "fake_cs_isolation_trap"),         # V5.1
            # 独立触发（正则硬探针保留高权重）
            MatrixCombination("has_insurance_scam_keywords",      30, "critical_insurance_scam", is_independent=True),
            MatrixCombination("has_financial_scam_keywords",      30, "critical_financial_scam", is_independent=True),
            # V5.2 新增矩阵
            MatrixCombination("has_ecommerce_privacy_intrusion",  22, "fake_cs_privacy_harvest"),       # V5.2
            MatrixCombination("has_ecommerce_channel_shift",      20, "fake_cs_channel_shift"),         # V5.2
            MatrixCombination("has_ecommerce_temporal_urgency",   18, "fake_cs_urgency_trap"),          # V5.2
        ],
    ),
)

_TOPIC_MASS_GRIEVANCE = TopicDefinition(
    topic_id    = "mass_grievance",
    category    = TopicCategory.HIGH_RISK,
    topic_family= "grievance",
    description = "涉军/群体维权与涉稳信访",
    threshold   = 0.72,
        bge_anchors=[],
    scoring_rules = ScoringRules(
        standalone_score = 12,    # V5.1: 压低（原25）
        standalone_tag   = "social_grievance_petition",
        matrix_combinations = [
            MatrixCombination("coordinated_broadcast",  20, "CRITICAL_MASS_INCIDENT_MOBILIZATION"),   # V5.1: 压低（原40）
            MatrixCombination("incitement_to_violence", 20, "CRITICAL_VIOLENT_PROTEST_RISK"), 
            MatrixCombination("is_business_sparse",    -20, "false_positive_grievance_noise"),         # V5.1: 压低（原40）
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
            params = VerbEntitySparsityParams(threshold=2),  # 实体极少
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
        standalone_score = -15,   # V5.1: 加大扣分（原-5），50-15=35
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
            standalone_score = -20,  # V5.1: 大幅扣分（原-3），50-20=30（安全/废料区）
            standalone_tag   = "low_value_casual_chat",
            matrix_combinations = [
                # 闲聊且没有业务名词，双重确认是废话，打入深渊
                MatrixCombination("is_business_sparse", -15, "casual_chat_extremely_sparse"),  # V5.1: 总计-35
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
            params = VerbEntitySparsityParams(threshold=2), 
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
