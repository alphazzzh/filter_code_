# stage_two_pipeline.py  ── V5.2 配置驱动架构（多维特征增强）
# ============================================================
# Y 型双轨流水线编排（软语义 + 硬句法）
#
# V5.0 变更摘要
# ─────────────────────────────────────────────────────────────
# ① SyntaxFeatureExtractor 废弃硬编码的 _URGENCY_ADVERBS /
#   _SECOND_PERSON / _RE_DRUG_QUANTITY 等常量
# ② 改为从 config_topics.get_all_syntax_rules() 动态加载所有规则，
#   按 SyntaxRuleType 分发到对应的提取器方法
# ③ 新增句法规则只需在 config_topics.py 追加 SyntaxRuleConfig，
#   本文件零修改
# ④ NLP 后端层（LTP/HanLP/规则降级）保持不变
#
# V5.1 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 新增 4 个行为学/心理学特征提取器：
#      _extract_isolation_request / _extract_micro_action_command /
#      _extract_conditional_threat / _extract_action_target_triplet
# ② 正则缓存从模块级全局变量迁移到 SyntaxFeatureExtractor 实例级，
#   新增 _compile_patterns() + reload()，支持热更新
# ③ CONDITIONAL_THREAT 和 ACTION_TARGET_TRIPLET 有依存句法增强
#   + 正则降级双路径
# ④ needs_nlp 条件补入 CONDITIONAL_THREAT / ACTION_TARGET_TRIPLET
#
# V5.2 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 新增 6 种特征类型提取器：
#      TEMPORAL_URGENCY / PRIVACY_INTRUSION / EMOTIONAL_MANIPULATION
#      FINANCIAL_FLOW / IDENTITY_IMPERSONATION / CHANNEL_SHIFTING
# ② 新增通用轻量级关键字提取器 _extract_simple_keywords
# ③ FINANCIAL_FLOW 复用 _extract_action_target_triplet（LTP 增强）
# ④ needs_nlp 条件补入 FINANCIAL_FLOW
# ============================================================

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from models_stage2 import ASRRecord, StageTwoResult, TrackType
from topology_engine import TopologyAnalyzer
from intent_radar import IntentRadar
from role_binder import RoleBinder
from config_topics import (
    SyntaxRuleConfig,
    SyntaxRuleType,
    MatchMode,
    TOPIC_REGISTRY,
    get_all_syntax_rules,
    # V5.2+ Pydantic 强类型参数模型（消除 config↔pipeline 字符串 Key 隐形耦合）
    ImperativeSyntaxParams,
    QuantityRegexParams,
    NerDensityParams,
    KeywordCoocParams,
    RegexPatternParams,
    VerbEntitySparsityParams,
    IsolationRequestParams,
    MicroActionCommandParams,
    ConditionalThreatParams,
    ActionTargetTripletParams,
    SimpleKeywordsParams,
    _RULE_TYPE_PARAMS_MAP,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NlpFeatures —— 动态 key-value 快照
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NlpFeatures:
    """
    硬句法轨道产出的结构化特征。

    特征 key 由 config_topics 中的 SyntaxRuleConfig.feature_name
    动态决定，不再固定为三个字段。
    _bool_features  : {feature_name: bool}     产出的布尔判断
    _evidence       : {evidence_key: list[str]} 证据字符串列表（审计用）
    nlp_backend     : 实际使用的 NLP 后端名称
    """
    _bool_features: dict[str, bool]        = field(default_factory=dict)
    _evidence:      dict[str, list[str]]   = field(default_factory=dict)
    nlp_backend:    str                    = "rule_based"

    def set_feature(self, name: str, value: bool) -> None:
        self._bool_features[name] = value

    def get_feature(self, name: str) -> bool:
        return self._bool_features.get(name, False)

    def add_evidence(self, key: str, items: list[str]) -> None:
        self._evidence.setdefault(key, [])
        self._evidence[key].extend(items)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._bool_features,
            **{k: v for k, v in self._evidence.items()},
            "nlp_backend": self.nlp_backend,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实例级预编译正则缓存（V5.1 迁移）
# V5.0 时期为模块级全局变量，V5.1 迁移到 SyntaxFeatureExtractor
# 实例内部，配合 reload() 支持配置热更新。
# 下方仅保留无状态的辅助构建函数。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_quantity_pattern(units: list[str]) -> re.Pattern:
    """
    根据量词列表构建数字+量词正则。
    模式：(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*(unit1|unit2|...)
    """
    unit_group = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    pattern = (
        r"(?<!\d)"                        # 避免匹配电话号码中间的数字
        r"(\d+(?:\.\d+)?)"                # 整数或小数
        r"\s*"                            # 可选空白
        r"(" + unit_group + r")"          # 量词
    )
    return re.compile(pattern, re.UNICODE)


def _build_org_suffix_pattern() -> re.Pattern:
    """NER 降级兜底：机构名尾缀正则（当无 LTP/HanLP 时使用）。"""
    return re.compile(
        r"[\u4e00-\u9fff]{2,8}"
        r"(?:局|院|委|部|厅|处|署|站|所|办|公司|银行|中心|机构)",
        re.UNICODE,
    )


# 模块级常量：NER 降级兜底的机构名尾缀正则（无状态，不随配置变化）
_RE_ORG_SUFFIX: re.Pattern = _build_org_suffix_pattern()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP 后端抽象层（LTP / HanLP / 规则降级，三级自动降级）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _NlpBackend:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def analyze(self, text: str) -> dict[str, Any]:
        """
        返回标准化解析结果：
        {
          "tokens": list[str],
          "dep":    list[tuple[int, str]],  # (head_0based, rel)
          "ner":    list[tuple[str, str]],  # (entity, type)
        }
        dep_rel 内部标准：HED / SBV / ADV / VOB
        """
        raise NotImplementedError


class _LtpBackend(_NlpBackend):
    """LTP 4.x HTTP 微服务后端，标签：HED/SBV/ADV/VOB。

    通过 HTTP 调用独立的 LTP 微服务（Dynamic Batching + 横向扩容），
    替代进程内 LTP 实例，避免内存/显存峰值拖慢主事件循环。

    环境变量：
      LTP_SERVICE_URL  微服务地址（默认 http://localhost:8900）
    降级策略：
      微服务不可用时，自动降级到 HanLP → 规则后端。
    """

    def __init__(self, service_url: str | None = None) -> None:
        from ltp_service.client import LtpHttpClient
        url = service_url or os.getenv("LTP_SERVICE_URL", "http://localhost:8900")
        self._client = LtpHttpClient(base_url=url)
        # 启动时做健康检查，确认微服务可用
        health = self._client.health()
        if health.get("status") != "ok" or not health.get("model_loaded", False):
            raise ConnectionError(
                f"[LTP] 微服务不可用 (url={url}, health={health})"
            )

    @property
    def name(self) -> str:
        return "ltp"

    def analyze(self, text: str) -> dict[str, Any]:
        """调用微服务 /analyze 接口，返回与旧版 _LtpBackend.analyze 兼容格式。"""
        return self._client.analyze(text)


class _HanLpBackend(_NlpBackend):
    """HanLP 2.x 后端，UD 标签映射为内部标准。"""

    _UD_MAP: dict[str, str] = {
        "root": "HED", "nsubj": "SBV", "advmod": "ADV",
        "obj":  "VOB",  "dobj":  "VOB", "nsubj:pass": "SBV",
    }

    def __init__(self) -> None:
        import hanlp  # type: ignore
        self._hanlp = hanlp.load(
            hanlp.pretrained.mtl
            .CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )

    @property
    def name(self) -> str:
        return "hanlp"

    def analyze(self, text: str) -> dict[str, Any]:
        result  = self._hanlp([text])
        tokens  = result.get("tok/fine", [[]])[0]
        dep_raw = result.get("dep", [[]])[0]
        dep: list[tuple[int, str]] = [
            (h - 1, self._UD_MAP.get(r.lower(), r.upper()))
            for h, r in dep_raw
        ]
        ner_raw = result.get("ner/ontonotes", [[]])[0]
        ner: list[tuple[str, str]] = [
            (e, t) for e, t, *_ in ner_raw
        ]
        return {"tokens": tokens, "dep": dep, "ner": ner}


class _RuleBasedFallback(_NlpBackend):
    """规则降级后端，无外部依赖，准确率低但保证流水线不中断。"""

    @property
    def name(self) -> str:
        return "rule_based"

    def analyze(self, text: str) -> dict[str, Any]:
        orgs = _RE_ORG_SUFFIX.findall(text)
        ner: list[tuple[str, str]] = [(o, "ORG") for o in orgs]
        return {"tokens": list(text), "dep": [], "ner": ner}


def _load_nlp_backend(ltp_service_url: str | None = None) -> _NlpBackend:
    """按 LTP 微服务 → HanLP → 规则 顺序自动降级。

    ltp_service_url: LTP 微服务地址（默认从 LTP_SERVICE_URL 环境变量读取）。
    微服务不可用时自动降级到 HanLP，最终兜底规则后端。
    """
    for BackendCls, label, kwargs in [
        (_LtpBackend, "LTP-HTTP", {"service_url": ltp_service_url}),
        (_HanLpBackend, "HanLP", {}),
    ]:
        try:
            return BackendCls(**kwargs)
        except Exception as e:
            warnings.warn(f"[NLP] {label} 加载失败（{e}），尝试下一个。", RuntimeWarning)
    return _RuleBasedFallback()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SyntaxFeatureExtractor —— 配置驱动版
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SyntaxFeatureExtractor:
    """
    配置驱动的硬句法特征提取器。

    初始化时从 config_topics.get_all_syntax_rules() 加载所有规则，
    extract() 遍历规则列表，按 SyntaxRuleType 动态分发到对应提取器。

    V5.1 变更：
    - 正则缓存从模块级迁移到实例级（_compiled_quantity_patterns /
      _compiled_regex_patterns），支持 reload() 热更新
    - 新增 4 个行为学/心理学特征提取器
    - CONDITIONAL_THREAT / ACTION_TARGET_TRIPLET 支持依存句法增强 + 正则降级

    扩展方法
    ─────────────────────────────────────────────────────────
    在 config_topics.py 为某个主题添加 SyntaxRuleConfig：
      → 本文件自动识别并执行，零代码修改。
    """

    def __init__(self, backend: Optional[_NlpBackend] = None) -> None:
        self._backend: _NlpBackend             = backend or _load_nlp_backend()
        # 从配置加载所有去重后的句法规则（feature_name → SyntaxRuleConfig）
        self._rules:  dict[str, SyntaxRuleConfig] = get_all_syntax_rules()
        # 实例级预编译正则缓存（支持热更新）
        self._compiled_quantity_patterns: dict[str, re.Pattern] = {}
        self._compiled_regex_patterns:    dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """扫描当前 self._rules，预编译所有 QUANTITY_REGEX / REGEX_PATTERN 正则。

        V5.3: 清理 .get() 死代码，直接访问 Pydantic 模型属性。
        """
        self._compiled_quantity_patterns.clear()
        self._compiled_regex_patterns.clear()
        for _rule in self._rules.values():
            if _rule.rule_type == SyntaxRuleType.QUANTITY_REGEX:
                units = _rule.params.quantity_units
                if units:
                    self._compiled_quantity_patterns[_rule.feature_name] = (
                        _build_quantity_pattern(units)
                    )
            elif _rule.rule_type == SyntaxRuleType.REGEX_PATTERN:
                pattern_str = _rule.params.pattern
                if pattern_str:
                    flags_str = _rule.params.flags
                    flags = getattr(re, flags_str, re.UNICODE)
                    self._compiled_regex_patterns[_rule.feature_name] = re.compile(
                        pattern_str, flags
                    )

    def reload(self) -> None:
        """
        热更新：重新从 config_topics 加载规则并重建正则缓存。
        调用此方法后，后续的 extract() 将使用最新配置。
        """
        self._rules = get_all_syntax_rules()
        self._compile_patterns()

    def extract(self, text: str) -> NlpFeatures:
        """
        对输入文本执行所有配置中的句法规则，返回 NlpFeatures 快照。

        分发逻辑：
          QUANTITY_REGEX          → _extract_quantity_regex()
          REGEX_PATTERN           → _extract_regex_pattern()
          NER_DENSITY             → _extract_ner_density()         (需要 NLP 解析)
          IMPERATIVE_SYNTAX       → _extract_imperative()          (需要 NLP 解析)
          KEYWORD_COOC            → _extract_keyword_cooc()
          VERB_ENTITY_SPARSITY    → _extract_verb_entity_sparsity()
          ISOLATION_REQUEST       → _extract_isolation_request()    (V5.1)
          MICRO_ACTION_COMMAND    → _extract_micro_action_command() (V5.1)
          CONDITIONAL_THREAT      → _extract_conditional_threat()   (需要 NLP 解析, V5.1)
          ACTION_TARGET_TRIPLET   → _extract_action_target_triplet() (需要 NLP 解析, V5.1)
        """
        feats = NlpFeatures(nlp_backend=self._backend.name)

        # 判断是否有需要 NLP 解析的规则（避免对仅正则的短文本调用重量级模型）
        needs_nlp = any(
            r.rule_type in (
                SyntaxRuleType.NER_DENSITY,
                SyntaxRuleType.IMPERATIVE_SYNTAX,
                SyntaxRuleType.VERB_ENTITY_SPARSITY,
                SyntaxRuleType.CONDITIONAL_THREAT,      # V5.1: 条件胁迫依赖依存句法
                SyntaxRuleType.ACTION_TARGET_TRIPLET,   # V5.1: 三元组提取依赖 NER
                SyntaxRuleType.FINANCIAL_FLOW,          # V5.2: 资金流向复用三元组，依赖 NER
            )
            for r in self._rules.values()
        )
        # V5.3: EXACT_WORD 模式需要分词结果，即使其他规则不需要 NLP
        needs_nlp = needs_nlp or self._needs_nlp_for_match_mode()

        parsed: dict[str, Any] = self._backend.analyze(text) if needs_nlp else {}

        for rule in self._rules.values():
            if rule.rule_type == SyntaxRuleType.QUANTITY_REGEX:
                self._extract_quantity_regex(text, rule, feats)

            elif rule.rule_type == SyntaxRuleType.REGEX_PATTERN:
                self._extract_regex_pattern(text, rule, feats)

            elif rule.rule_type == SyntaxRuleType.NER_DENSITY:
                self._extract_ner_density(parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.IMPERATIVE_SYNTAX:
                self._extract_imperative(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.KEYWORD_COOC:
                self._extract_keyword_cooc(text, parsed, feats)


            elif rule.rule_type == SyntaxRuleType.VERB_ENTITY_SPARSITY:
                self._extract_verb_entity_sparsity(parsed, rule, feats)

            # ── V5.1 新增：行为学/心理学特征提取器 ────────────
            elif rule.rule_type == SyntaxRuleType.ISOLATION_REQUEST:
                self._extract_isolation_request(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.MICRO_ACTION_COMMAND:
                self._extract_micro_action_command(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.CONDITIONAL_THREAT:
                self._extract_conditional_threat(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.ACTION_TARGET_TRIPLET:
                self._extract_action_target_triplet(text, parsed, rule, feats)

            # ── V5.2 新增：多维行为/心理/交易特征提取器 ─────────
            elif rule.rule_type == SyntaxRuleType.TEMPORAL_URGENCY:
                self._extract_simple_keywords(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.PRIVACY_INTRUSION:
                self._extract_simple_keywords(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.EMOTIONAL_MANIPULATION:
                self._extract_simple_keywords(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.IDENTITY_IMPERSONATION:
                self._extract_simple_keywords(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.CHANNEL_SHIFTING:
                self._extract_simple_keywords(text, parsed, feats)

            elif rule.rule_type == SyntaxRuleType.FINANCIAL_FLOW:
                # 复用三元组提取器（LTP 增强 + 正则降级双路径）
                self._extract_action_target_triplet(text, parsed, rule, feats)

        return feats

    # ── V5.3 词边界匹配核心方法 ──────────────────────────────────

    def _needs_nlp_for_match_mode(self) -> bool:
        """检查是否有规则使用了 EXACT_WORD 模式（需要分词结果）。"""
        return any(
            getattr(r.params, "match_mode", MatchMode.SUBSTR) == MatchMode.EXACT_WORD
            for r in self._rules.values()
        )

    @staticmethod
    def _match_keyword(
        kw:          str,
        text:        str,
        tokens:      list[str],
        nlp_backend: str,
        mode:        MatchMode,
    ) -> bool:
        """
        V5.3 NLP 后端感知的词边界匹配。

        策略：
          SUBSTR         → kw in text（向后兼容）
          EXACT_WORD     → LTP/HanLP: kw in tokens（真分词精确匹配）
                           rule_based: 退回 _regex_chinese_boundary（tokens 是逐字拆分不可用）
          REGEX_BOUNDARY → 中文字边界正则：前后不能有中文字符或下划线
                           使用 (?<![\\u4e00-\\u9fff\\w]) 和 (?![\\u4e00-\\u9fff\\w])
                           覆盖"字符串边界 + 标点/空格 + 非中文词边界"场景

        注意：\\b 对中文无效（中文没有空格分词），故 REGEX_BOUNDARY 使用
              Unicode 范围检查代替标准 \\b。
        """
        if mode == MatchMode.SUBSTR:
            return kw in text

        if mode == MatchMode.EXACT_WORD:
            if nlp_backend in ("LTP", "hanlp"):
                return kw in tokens
            # rule_based 后端 tokens = list(text) 逐字拆分，不可用于精确词匹配
            # 退回中文边界正则
            return SyntaxFeatureExtractor._regex_chinese_boundary(kw, text)

        if mode == MatchMode.REGEX_BOUNDARY:
            return SyntaxFeatureExtractor._regex_chinese_boundary(kw, text)

        return kw in text  # 兜底

    @staticmethod
    def _regex_chinese_boundary(kw: str, text: str) -> bool:
        """
        中文词边界正则匹配（rule_based 后端降级方案）。

        核心问题：中文没有空格分词，\\b 对中文完全无效。
        正则无法区分"交"在"交警"（子串）和"请交费"（独立词）中的角色。
        因此本方法的策略是**保守保护**：

        1. 单字关键词（如"交"、"捐"）：
           检查是否出现在标点/空格/字符串边界旁，且后面不紧贴中文字符。
           这只能拦截最明显的误报（"交"出现在字符串末尾或标点后），
           对于"请交费"这种正常语境中的独立用法也会漏判。
           → 单字误报防护建议使用 EXACT_WORD 模式（依赖 LTP 分词）。

        2. 多字关键词（如"交钱"、"退出"）：
           直接使用 SUBSTR（多字本身的误报率远低于单字）。
           例："交钱"不太可能成为某个更长中文词的子串。

        结论：如果 NLP 后端可用，应优先使用 EXACT_WORD；
              REGEX_BOUNDARY 只是 rule_based 后端退而求其次的保底方案。
        """
        # 单字关键词：保守边界保护
        if len(kw) == 1:
            # 只检查后方边界：关键词后面不紧贴中文字符或字母数字
            # 前方不做限制（中文句子中词前面几乎总是中文字符）
            pattern = rf"{re.escape(kw)}(?![\u4e00-\u9fff\w])"
            return bool(re.search(pattern, text))
        # 多字关键词：误报率低，直接子串匹配
        return kw in text

    # ── 各类型提取器 ──────────────────────────────────────────

    def _extract_quantity_regex(
        self, text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        QUANTITY_REGEX：数字 + 量词正则匹配。
        使用实例级预编译缓存正则，支持热更新。
        """
        pattern = self._compiled_quantity_patterns.get(rule.feature_name)
        if pattern is None:
            return
        matches = pattern.findall(text)
        if matches:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(
                    rule.evidence_key,
                    [f"{num}{unit}" for num, unit in matches],
                )

    def _extract_regex_pattern(
        self, text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """REGEX_PATTERN：自定义正则，使用实例级预编译缓存。"""
        pattern = self._compiled_regex_patterns.get(rule.feature_name)
        if pattern is None:
            return
        found = pattern.findall(text)
        if found:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(
                    rule.evidence_key,
                    [str(m) for m in found],
                )

    @staticmethod
    def _extract_ner_density(
        parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        NER_DENSITY：统计指定类型实体数量是否达到阈值。

        entity_types 参数支持 LTP 标签（Ni/Ns）和 HanLP 标签（ORG/GPE/LOC）。

        V5.3: 清理 .get() 死代码。
        """
        p = rule.params
        entity_types: frozenset[str] = frozenset(p.entity_types)
        threshold: int = p.threshold

        entities: list[str] = [
            ent_text
            for ent_text, ent_type in parsed.get("ner", [])
            if ent_type in entity_types
        ]

        if len(entities) >= threshold:
            feats.set_feature(rule.feature_name, True)
        if rule.evidence_key:
            feats.add_evidence(rule.evidence_key, entities)

    @staticmethod
    def _extract_imperative(
        text: str,
        parsed: dict[str, Any],
        rule: SyntaxRuleConfig,
        feats: NlpFeatures,
    ) -> None:
        """
        IMPERATIVE_SYNTAX：依存句法检测「高压祈使句」。

        算法逻辑（详见 V4.0 注释）：
        Step 1: 找到所有 HED（核心谓语）节点
        Step 2: 在每个 HED 的 SBV 子节点中寻找第二人称主语
        Step 3: 在每个 HED 的 ADV 子节点中寻找紧迫状语
        Step 4: 两项同时满足 → 命中，记录核心动词

        降级：dep 为空时使用正则近似检测。

        V5.3: 清理 .get() 死代码。
        """
        p = rule.params
        second_person:   frozenset[str] = frozenset(p.second_person)
        urgency_adverbs: frozenset[str] = frozenset(p.urgency_adverbs)

        tokens: list[str]            = parsed.get("tokens", [])
        dep:    list[tuple[int, str]] = parsed.get("dep", [])

        # 规则降级路径
        if not dep:
            re_pattern = (
                r"^(?:" + "|".join(re.escape(w) for w in second_person) + r")"
                r"[\u4e00-\u9fff\s]{0,5}"
                r"(?:" + "|".join(re.escape(a) for a in urgency_adverbs) + r")"
            )
            if re.search(re_pattern, text):
                feats.set_feature(rule.feature_name, True)
                if rule.evidence_key:
                    feats.add_evidence(rule.evidence_key, ["[rule_based_match]"])
            return

        root_indices: list[int] = [
            i for i, (_, rel) in enumerate(dep) if rel == "HED"
        ]
        imperative_verbs: list[str] = []

        for root_idx in root_indices:
            has_2nd_subj = any(
                head == root_idx and rel == "SBV"
                and tokens[i].lower() in second_person
                for i, (head, rel) in enumerate(dep)
            )
            has_urgency_adv = any(
                head == root_idx and rel == "ADV"
                and tokens[i].lower() in urgency_adverbs
                for i, (head, rel) in enumerate(dep)
            )
            if has_2nd_subj and has_urgency_adv:
                imperative_verbs.append(
                    tokens[root_idx] if root_idx < len(tokens) else "?"
                )

        if imperative_verbs:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, imperative_verbs)

    def _extract_keyword_cooc(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        KEYWORD_COOC：多关键词集合共现检测。

        逻辑：params.keyword_sets 是一个列表的列表。
        每个子列表至少有一个词在 text 中出现，
        且所有子列表均满足 → 判定命中。

        例（金融勒索检测）：
          keyword_sets = [
              ["捐", "缴", "交钱"],   # 子列表 A：金融词
              ["必须", "否则", "后果"],# 子列表 B：威胁词
          ]
          → 文本中同时含有 A 中至少一个词 AND B 中至少一个词 → 命中

        V5.3: 支持 match_mode 参数，默认 SUBSTR 向后兼容。
        """
        p = rule.params
        keyword_sets: list[list[str]] = p.keyword_sets
        match_mode:   MatchMode       = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not keyword_sets:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        all_sets_matched = all(
            any(self._match_keyword(kw, text, tokens, nlp_backend, match_mode) for kw in kw_set)
            for kw_set in keyword_sets
        )
        if all_sets_matched:
            feats.set_feature(rule.feature_name, True)

    @staticmethod
    def _extract_verb_entity_sparsity(
        parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """VERB_ENTITY_SPARSITY：实体与业务词极度稀疏检测

        V5.3: 清理 .get() 死代码。
        """
        p = rule.params
        threshold: int = p.threshold
        
        # 统计核心业务名词（机构、地名、人名等）的数量
        entities = [ent_type for _, ent_type in parsed.get("ner", [])]
        business_entity_count = sum(
            1 for e in entities if e in ("Ni", "Ns", "ORG", "GPE", "LOC", "nh", "PER")
        )
        
        # 如果整通电话的实体数连 threshold 都不够，判定为缺乏有效主宾语的闲聊废料
        if business_entity_count < threshold:
            feats.set_feature(rule.feature_name, True)

    # ── V5.1 新增：行为学/心理学特征提取器 ──────────────────

    def _extract_isolation_request(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        ISOLATION_REQUEST：物理隔离与信息阻断检测。

        诈骗核心前提：切断受害者外部信息源。
        探测维度：
          - 空间隔离：「找个没人的房间」「把门反锁」「到外面去」
          - 通讯阻断：「不要挂电话」「开启飞行模式」「拦截短信」「关掉WiFi」
          - 社交阻断：「不能告诉家人」「国家机密」「不要跟别人说」

        V5.3: 支持 match_mode 参数。
        """
        p = rule.params
        isolation_keywords: list[str] = p.isolation_keywords
        match_mode:         MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not isolation_keywords:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        hit_words = [kw for kw in isolation_keywords
                     if self._match_keyword(kw, text, tokens, nlp_backend, match_mode)]
        if hit_words:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, hit_words)

    def _extract_micro_action_command(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        MICRO_ACTION_COMMAND：服从性测试与微动作指令检测。

        高阶诈骗在切入核心客体前，通过微小指令测试受害者被控程度。
        探测维度：
          - 设备操作：「打开免提」「点右上角」「点一下设置」
          - 屏幕引导：「往下滑」「点击链接」「扫码」
          - 行为测试：「跟着我读一遍」「你现在打开」「不要动手机」

        V5.3: 支持 match_mode 参数。
        """
        p = rule.params
        device_keywords: list[str] = p.device_action_keywords
        match_mode:      MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not device_keywords:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        hit_words = [kw for kw in device_keywords
                     if self._match_keyword(kw, text, tokens, nlp_backend, match_mode)]
        if hit_words:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, hit_words)

    def _extract_conditional_threat(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        CONDITIONAL_THREAT：条件胁迫与逻辑陷阱检测。

        升级版 KEYWORD_COOC，利用「条件从句 + 负面主句」结构：
          [如果不...] → [征信受损/涉嫌违法/拘留/影响子女]

        降级策略：无依存句法时使用正则匹配条件从句模式。

        V5.3: 支持 match_mode 参数；清理 .get() 死代码。
        """
        p = rule.params
        condition_clauses: list[str] = p.condition_clauses
        threat_clauses:   list[str] = p.threat_clauses
        match_mode:       MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not condition_clauses or not threat_clauses:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        # 检测条件从句是否出现
        has_condition = any(
            self._match_keyword(cond, text, tokens, nlp_backend, match_mode)
            for cond in condition_clauses
        )
        # 检测威胁主句是否出现
        has_threat = any(
            self._match_keyword(threat, text, tokens, nlp_backend, match_mode)
            for threat in threat_clauses
        )

        # ── 依存句法增强：确认条件-威胁在同一小句范围内 ──
        dep: list[tuple[int, str]] = parsed.get("dep", [])

        syntactic_confirm = False
        if dep and tokens and has_condition and has_threat:
            # 在依存树中寻找「条件词 → 核心谓语」和「威胁词 → 核心谓语」
            # 共享同一个 HED 时判定为同一小句
            root_indices = [i for i, (_, rel) in enumerate(dep) if rel == "HED"]
            for root_idx in root_indices:
                children = [
                    (i, tokens[i].lower(), rel)
                    for i, (head, rel) in enumerate(dep)
                    if head == root_idx
                ]
                child_texts = [t for _, t, _ in children]
                # 检查这个谓语子树是否同时包含条件和威胁成分
                has_cond_in_tree = any(c in " ".join(child_texts) for c in condition_clauses)
                has_thrt_in_tree = any(t in " ".join(child_texts) for t in threat_clauses)
                if has_cond_in_tree and has_thrt_in_tree:
                    syntactic_confirm = True
                    break

        # 最终判定：关键词共现 OR 依存句法确认
        if (has_condition and has_threat) or syntactic_confirm:
            feats.set_feature(rule.feature_name, True)
            hit_list = []
            if has_condition:
                hit_list.extend(c for c in condition_clauses
                                if self._match_keyword(c, text, tokens, nlp_backend, match_mode))
            if has_threat:
                hit_list.extend(t for t in threat_clauses
                                if self._match_keyword(t, text, tokens, nlp_backend, match_mode))
            if rule.evidence_key and hit_list:
                feats.add_evidence(rule.evidence_key, hit_list[:6])  # 最多保留 6 条证据

    def _extract_action_target_triplet(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        ACTION_TARGET_TRIPLET：语义角色三元组检测。

        V5.3: 支持 match_mode 参数；补被动语态（被字句）；清理 .get() 死代码。
        """
        p = rule.params
        action_verbs:    list[str] = p.action_verbs
        target_entities: list[str] = p.target_entities
        match_mode:     MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not action_verbs or not target_entities:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        hit_verbs   = [v for v in action_verbs
                       if self._match_keyword(v, text, tokens, nlp_backend, match_mode)]
        hit_targets = [t for t in target_entities
                       if self._match_keyword(t, text, tokens, nlp_backend, match_mode)]

        # ── 依存句法增强：兼容 VOB, FOB, "把"字句(POB) 及"被"字句(SBV) ──
        dep: list[tuple[int, str]] = parsed.get("dep", [])

        syntactic_confirm = False
        if dep and tokens and hit_verbs and hit_targets:
            
            # 遍历所有节点 (i为子节点索引, head为父节点索引, rel为关系)
            for i, (head, rel) in enumerate(dep):
                if head >= len(tokens): 
                    continue
                    
                child_text = tokens[i]
                head_text  = tokens[head]

                # 🎯 场景 1 & 2：直接宾语 (VOB) 或 前置宾语 (FOB)
                # 例如："输入(head) 验证码(child)" 或 "验证码(child) 输入(head)了吗"
                if rel in ("VOB", "FOB"):
                    if head_text in hit_verbs and child_text in hit_targets:
                        syntactic_confirm = True
                        break

                # 🎯 场景 3："把"字句 / 介词宾语结构 (POB 向上追溯 ADV)
                # 例如："把(head) 验证码(child) 发(prep_head) 给我"
                if rel == "POB":
                    # 如果当前词是目标实体（如验证码），且它的父节点是"把"或"将"
                    if child_text in hit_targets and head_text in ("把", "将", "给"):
                        # 向上追溯第二跳：找到"把"依附的核心动词
                        prep_head_idx = dep[head][0] # "把"的父节点索引
                        prep_rel      = dep[head][1] # "把"与父节点的关系
                        
                        if prep_head_idx < len(tokens):
                            prep_head_text = tokens[prep_head_idx]
                            # 如果"把"是作为状语(ADV)依附在我们的高危动作词上(如"发")
                            if prep_rel == "ADV" and prep_head_text in hit_verbs:
                                syntactic_confirm = True
                                break

                # 🎯 场景 4（V5.3 新增）："被"字句 / 被动语态
                # 例如："验证码(SBV) 被(ADV) 输入了(HED)"
                # LTP 标注：验证码 -SBV→ 输入(HED)，被 -ADV→ 输入(HED)
                # 也覆盖"让/叫/给"被动标记
                if rel == "SBV" and child_text in hit_targets:
                    # 检查同一个 head（动词）下是否有"被/让/叫/给"作 ADV
                    has_passive_marker = any(
                        head2 == head and rel2 == "ADV" and tokens[j] in ("被", "让", "叫", "给")
                        for j, (head2, rel2) in enumerate(dep)
                    )
                    if has_passive_marker and head_text in hit_verbs:
                        syntactic_confirm = True
                        break

        # ── 降级判定与证据组装（保持不变） ──
        if hit_verbs and hit_targets:
            if syntactic_confirm:
                feats.set_feature(rule.feature_name, True)
            else:
                for verb in hit_verbs:
                    for target in hit_targets:
                        v_pos = text.find(verb)
                        t_pos = text.find(target)
                        # 降级：如果在 15 个字符内共现，也算命中
                        if v_pos >= 0 and t_pos >= 0 and abs(v_pos - t_pos) < 15:
                            feats.set_feature(rule.feature_name, True)
                            break
                    if feats.get_feature(rule.feature_name):
                        break

        if feats.get_feature(rule.feature_name) and rule.evidence_key:
            triplets = [f"{v}→{t}" for v in hit_verbs for t in hit_targets]
            feats.add_evidence(rule.evidence_key, triplets[:6])


    # ── V5.2 新增：通用轻量级关键字提取器 ──────────────────

    def _extract_simple_keywords(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        通用字符串集合匹配提取器。

        适用于纯字符串匹配型规则（无 NLP 依赖），避免为每种新特征
        写一个结构完全相同的提取方法。支持的规则类型：
          TEMPORAL_URGENCY / PRIVACY_INTRUSION / EMOTIONAL_MANIPULATION
          IDENTITY_IMPERSONATION / CHANNEL_SHIFTING

        统一使用 SimpleKeywordsParams.keywords 作为匹配词库。

        V5.3: 支持 match_mode 参数，默认 SUBSTR 向后兼容。
        """
        p = rule.params
        keywords:   list[str] = p.keywords
        match_mode: MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not keywords:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        hit_words = [kw for kw in keywords
                     if self._match_keyword(kw, text, tokens, nlp_backend, match_mode)]
        if hit_words:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, hit_words[:3])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StageTwoPipeline —— V5.0 配置驱动编排器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StageTwoPipeline:
    """
    Y 型双轨流水线编排器，配置驱动，无硬编码。

    两轨并行：
      软语义轨道 → dialogue_turns[*].intent_labels（BGE-M3 向量雷达）
      硬句法轨道 → metadata["nlp_features"]（SyntaxFeatureExtractor）
    
    额外轨道：
      动态搜索轨道 → metadata["dynamic_search"]（BGE-M3 RAG 检索）
      角色绑定     → speaker_roles（RoleBinder）
      拓扑分析     → track_type / interaction_features
    """

    def __init__(
        self,
        bge_model_name:   str   = "BAAI/bge-m3",
        use_fp16:         bool  = True,
        intent_threshold: float = 0.72,
        ltp_service_url:  str   | None = None,
        bge_service_url:  str   | None = None,
        nlp_backend:      Optional[_NlpBackend] = None,
    ) -> None:
        self._topology = TopologyAnalyzer()
        self._radar    = IntentRadar.get_instance(
            model_name      = bge_model_name,
            use_fp16        = use_fp16,
            bge_service_url = bge_service_url,
        )
        self._binder = RoleBinder(
            radar    = self._radar,
            topology = self._topology,
        )
        # 若未传入外部 nlp_backend，则使用 _load_nlp_backend 并透传 ltp_service_url
        if nlp_backend is not None:
            self._syntax = SyntaxFeatureExtractor(backend=nlp_backend)
        else:
            self._syntax = SyntaxFeatureExtractor(
                backend=_load_nlp_backend(ltp_service_url=ltp_service_url)
            )

    def process_conversation(
        self,
        conversation_id: str,
        records:         list[ASRRecord],
        extra_metadata:  dict[str, Any] | None = None,
    ) -> StageTwoResult:
        """
        处理一段完整对话，输出双轨融合的 StageTwoResult。
        metadata["nlp_features"] 中存放 NlpFeatures.to_dict() 的产出。
        """
        from typing import Any as _Any  # 避免顶层循环引用

        # ── 软语义轨道 ────────────────────────────────────────
        merged_turns  = self._topology.merge_turns(records)
        track_type    = self._topology.classify_track(merged_turns)
        labeled_turns, role_results, ifeats = self._binder.bind(
            turns=merged_turns, track_type=track_type
        )

        # ── 硬句法轨道 ────────────────────────────────────────
        # 提取全局特征
        full_text: str = " ".join(t.merged_text for t in labeled_turns if not t.is_backchannel and t.merged_text.strip())
        global_nlp_feats: NlpFeatures = self._syntax.extract(full_text) if full_text.strip() else NlpFeatures(nlp_backend=self._syntax._backend.name)

        # 提取按角色特征 (防角色污染)
        speaker_nlp_feats: dict[str, dict[str, Any]] = {}
        for role_res in role_results:
            sid = role_res.speaker_id
            speaker_text = " ".join(t.merged_text for t in labeled_turns if t.speaker_id == sid and not t.is_backchannel and t.merged_text.strip())
            speaker_nlp_feats[sid] = self._syntax.extract(speaker_text).to_dict() if speaker_text.strip() else NlpFeatures(nlp_backend=self._syntax._backend.name).to_dict()

        
        # ── 2. 动态搜索轨道（鲁棒整合 + 滑动窗口切块） ──
        dynamic_search_result = None

        raw_topic = extra_metadata.get("dynamic_topic") if extra_metadata else None
        
        # 🛡️ 类型归一化：无论传入的是单字符串，还是列表，统统转成干净的 list[str]
        topics_to_search = []
        if isinstance(raw_topic, str) and raw_topic.strip():
            topics_to_search = [raw_topic.strip()]
        elif isinstance(raw_topic, list):
            topics_to_search = [str(t).strip() for t in raw_topic if str(t).strip()]

        if topics_to_search:
            # 清洗对话文本，剔除无意义的语气词
            valid_turns = [
                t for t in labeled_turns 
                if not t.is_backchannel and len(t.merged_text.strip()) > 1
            ]
            
            if valid_turns:
                # 【RAG 优化方向二：滑动窗口上下文重组】(此处性能开销极小，只做切片)
                window_size = 5
                stride = 2
                search_chunks = []
                
                for i in range(0, len(valid_turns), stride):
                    window = valid_turns[i : i + window_size]
                    if not window:
                        break
                    chunk_text = " ".join([f"[{t.speaker_id}] {t.merged_text.strip()}" for t in window])
                    search_chunks.append(chunk_text)

                # 🚀 核心升级：多主题独立搜索，拒绝语义稀释！
                best_result = None
                for single_topic in topics_to_search:
                    # 每个主题拥有独立的纯净向量，互不干扰
                    res = self._radar.dynamic_search(
                        search_chunks=search_chunks, 
                        dynamic_topic=single_topic,
                        default_threshold=0.40,
                        top_k=1  
                    )
                    
                    # 内部竞价：保留得分最高的那个主题的情报
                    current_score = res.get("max_score", 0.0)
                    if not best_result or current_score > best_result.get("max_score", 0.0):
                        best_result = res
                
                dynamic_search_result = best_result
            else:
                dynamic_search_result = {
                    "topic_queried": str(topics_to_search),
                    "matched": False,
                    "max_score": 0.0,
                    "top_matches": [],
                    "status": "skipped_due_to_pure_backchannel"
                }

        # ── 3. 汇总元信息 ──
        metadata: dict[str, Any] = {
            "raw_record_count":  len(records),
            "merged_turn_count": len(labeled_turns),
            "track_type":        track_type.value,
            "nlp_features":      global_nlp_feats.to_dict(),
            "speaker_nlp_features": speaker_nlp_feats, 
        }
        
        # 鲁棒性设计 3：将检索结果安全注入元数据
        if dynamic_search_result:
            metadata["dynamic_search"] = dynamic_search_result
            
        if extra_metadata:
            # 剔除传入的 dynamic_topic，避免返回结果里冗余
            clean_extra = {k: v for k, v in extra_metadata.items() if k != "dynamic_topic"}
            metadata.update(clean_extra)

        return StageTwoResult(
            conversation_id      = conversation_id,
            track_type           = track_type,
            dialogue_turns       = labeled_turns,
            speaker_roles        = role_results,
            interaction_features = ifeats,
            stage_two_done       = True,
            metadata             = metadata,
        )
