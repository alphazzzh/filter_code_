# stage_two_pipeline.py  ── V6.0 依赖注入 + 异步核心架构
# ============================================================
# Y 型双轨流水线编排（软语义 + 硬句法）
#
# V6.0 变更摘要（本版本核心改动）
# ─────────────────────────────────────────────────────────────
# ① 依赖注入（DI）：TopologyAnalyzer / IntentRadar / RoleBinder /
#   SyntaxFeatureExtractor 全部通过 __init__ 参数注入，
#   未传入时由 _create_default_xxx 工厂方法创建默认实例。
#   支持外部传入 Mock 对象进行单元测试。
# ② 消除异步/同步重复代码：process_conversation_async 为唯一
#   业务核心，process_conversation 改为纯同步包装器（try asyncio
#   event loop → fallback new loop），实现绝对 DRY。
# ③ NlpBackend Protocol：用 typing.Protocol 替代 _NlpBackend
#   类继承，定义清晰的接口契约，支持结构化子类型。
#
# V5.2 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 新增 6 种特征类型提取器：
#      TEMPORAL_URGENCY / PRIVACY_INTRUSION / EMOTIONAL_MANIPULATION
#      FINANCIAL_FLOW / IDENTITY_IMPERSONATION / CHANNEL_SHIFTING
# ② 新增通用轻量级关键字提取器 _extract_simple_keywords
# ③ FINANCIAL_FLOW 复用 _extract_action_target_triplet（LTP 增强）
# ④ needs_nlp 条件补入 FINANCIAL_FLOW
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
# V5.0 变更摘要
# ─────────────────────────────────────────────────────────────
# ① SyntaxFeatureExtractor 废弃硬编码的 _URGENCY_ADVERBS /
#   _SECOND_PERSON / _RE_DRUG_QUANTITY 等常量
# ② 改为从 config_topics.get_all_syntax_rules() 动态加载所有规则，
#   按 SyntaxRuleType 分发到对应的提取器方法
# ③ 新增句法规则只需在 config_topics.py 追加 SyntaxRuleConfig，
#   本文件零修改
# ④ NLP 后端层（LTP/HanLP/规则降级）保持不变
# ============================================================

from __future__ import annotations

import asyncio
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import logging

logger = logging.getLogger(__name__)

from models_stage2 import ASRRecord, StageTwoResult, TrackType, DialogueTurn
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
# NlpBackend 协议接口（V6.0：Protocol 替代类继承）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@runtime_checkable
class NlpBackend(Protocol):
    """
    NLP 后端协议接口。

    任何实现了 name / analyze / analyze_batch / analyze_async /
    analyze_batch_async 的对象均可作为 NLP 后端注入到
    SyntaxFeatureExtractor 中。支持结构化子类型（不需要显式继承）。
    """

    @property
    def name(self) -> str: ...

    def analyze(self, text: str) -> dict[str, Any]: ...

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]: ...

    async def analyze_async(self, text: str) -> dict[str, Any]: ...

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]: ...


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
# NLP 后端实现层（LTP / HanLP / 规则降级，三级自动降级）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _LtpBackend:
    """LTP 4.x HTTP 微服务后端，标签：HED/SBV/ADV/VOB。

    通过 HTTP 调用独立的 LTP 微服务（Dynamic Batching + 横向扩容），
    替代进程内 LTP 实例，避免内存/显存峰值拖慢主事件循环。

    环境变量：
      LTP_SERVICE_URL  微服务地址（默认 http://localhost:8900）
    降级策略：
      微服务不可用时，自动降级到规则后端（_RuleBasedFallback），
      不阻断打分流水线。

    注：实现 NlpBackend Protocol，无需显式继承。
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
        # 延迟初始化降级后端（仅在运行时降级时才创建，避免浪费资源）
        self._fallback: _RuleBasedFallback | None = None

    @property
    def name(self) -> str:
        return "ltp"

    def _get_fallback(self) -> _RuleBasedFallback:
        """懒加载规则降级后端"""
        if self._fallback is None:
            self._fallback = _RuleBasedFallback()
        return self._fallback

    def analyze(self, text: str) -> dict[str, Any]:
        """调用微服务 /analyze 接口，失败时降级到规则后端。"""
        try:
            return self._client.analyze(text)
        except Exception as e:
            logger.warning(
                f"[LtpBackend] analyze HTTP 失败，降级到 rule_based: {e}"
            )
            return self._get_fallback().analyze(text)

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """调用微服务批量接口，失败时降级到规则后端（逐条）。"""
        try:
            return self._client.analyze_batch(texts)
        except Exception as e:
            logger.warning(
                f"[LtpBackend] analyze_batch HTTP 失败，降级到 rule_based: {e}"
            )
            return [self._get_fallback().analyze(t) for t in texts]

    # ── V5.5 异步方法 ──────────────────────────────────────

    async def analyze_async(self, text: str) -> dict[str, Any]:
        """异步调用微服务，失败时降级到规则后端。"""
        try:
            return await self._client.analyze_async(text)
        except Exception as e:
            logger.warning(
                f"[LtpBackend] analyze_async HTTP 失败，降级到 rule_based: {e}"
            )
            return self._get_fallback().analyze(text)

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]:
        """异步批量调用微服务，失败时降级到规则后端。"""
        try:
            return await self._client.analyze_batch_async(texts)
        except Exception as e:
            logger.warning(
                f"[LtpBackend] analyze_batch_async HTTP 失败，降级到 rule_based: {e}"
            )
            return [self._get_fallback().analyze(t) for t in texts]


class _HanLpBackend:
    """HanLP 2.x 后端，UD 标签映射为内部标准。

    注：实现 NlpBackend Protocol，无需显式继承。
    """

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

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """HanLP 原生支持 batch 输入，1 次调用替代 N 次。"""
        results = self._hanlp(texts)
        batch_output: list[dict[str, Any]] = []
        n = len(texts)
        tok_all  = results.get("tok/fine", [])
        dep_all  = results.get("dep", [])
        ner_all  = results.get("ner/ontonotes", [])
        for i in range(n):
            tokens  = tok_all[i] if i < len(tok_all) else []
            dep_raw = dep_all[i] if i < len(dep_all) else []
            dep: list[tuple[int, str]] = [
                (h - 1, self._UD_MAP.get(r.lower(), r.upper()))
                for h, r in dep_raw
            ]
            ner_raw = ner_all[i] if i < len(ner_all) else []
            ner: list[tuple[str, str]] = [
                (e, t) for e, t, *_ in ner_raw
            ]
            batch_output.append({"tokens": tokens, "dep": dep, "ner": ner})
        return batch_output

    # ── 异步方法（委托给同步实现，HanLP 不支持原生异步）──────────

    async def analyze_async(self, text: str) -> dict[str, Any]:
        return self.analyze(text)

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]:
        return self.analyze_batch(texts)


class _RuleBasedFallback:
    """规则降级后端，无外部依赖，准确率低但保证流水线不中断。

    注：实现 NlpBackend Protocol，无需显式继承。
    """

    @property
    def name(self) -> str:
        return "rule_based"

    def analyze(self, text: str) -> dict[str, Any]:
        orgs = _RE_ORG_SUFFIX.findall(text)
        ner: list[tuple[str, str]] = [(o, "ORG") for o in orgs]
        return {"tokens": list(text), "dep": [], "ner": ner}

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.analyze(t) for t in texts]

    async def analyze_async(self, text: str) -> dict[str, Any]:
        return self.analyze(text)

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]:
        return self.analyze_batch(texts)


def _load_nlp_backend(ltp_service_url: str | None = None) -> NlpBackend:
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

    V6.0 变更：
    - backend 参数类型从 Optional[_NlpBackend] 改为 Optional[NlpBackend]
      （Protocol 接口，支持结构化子类型）

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

    def __init__(self, backend: NlpBackend | None = None) -> None:
        self._backend: NlpBackend            = backend or _load_nlp_backend()
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
        """
        feats = NlpFeatures(nlp_backend=self._backend.name)

        needs_nlp = any(
            r.rule_type in (
                SyntaxRuleType.NER_DENSITY,
                SyntaxRuleType.IMPERATIVE_SYNTAX,
                SyntaxRuleType.VERB_ENTITY_SPARSITY,
                SyntaxRuleType.CONDITIONAL_THREAT,
                SyntaxRuleType.ACTION_TARGET_TRIPLET,
                SyntaxRuleType.FINANCIAL_FLOW,
            )
            for r in self._rules.values()
        ) or self._needs_nlp_for_match_mode()

        parsed: dict[str, Any] = self._backend.analyze(text) if needs_nlp else {}

        self._apply_rules(text, parsed, feats)

        return feats

    def extract_batch(self, texts: list[str]) -> list[NlpFeatures]:
        """
        批量提取特征：将 N 次 NLP HTTP 请求压缩为 1 次。
        """
        needs_nlp = any(
            r.rule_type in (
                SyntaxRuleType.NER_DENSITY,
                SyntaxRuleType.IMPERATIVE_SYNTAX,
                SyntaxRuleType.VERB_ENTITY_SPARSITY,
                SyntaxRuleType.CONDITIONAL_THREAT,
                SyntaxRuleType.ACTION_TARGET_TRIPLET,
                SyntaxRuleType.FINANCIAL_FLOW,
            )
            for r in self._rules.values()
        ) or self._needs_nlp_for_match_mode()

        if needs_nlp:
            parsed_batch = self._backend.analyze_batch(texts)
        else:
            parsed_batch = [{}] * len(texts)

        results: list[NlpFeatures] = []
        for text, parsed in zip(texts, parsed_batch):
            if not text.strip():
                results.append(NlpFeatures(nlp_backend=self._backend.name))
                continue

            feats = NlpFeatures(nlp_backend=self._backend.name)
            self._apply_rules(text, parsed, feats)
            results.append(feats)

        return results

    # ── V5.5 异步方法 ──────────────────────────────────────

    async def extract_async(self, text: str) -> NlpFeatures:
        """异步单条提取。"""
        feats = NlpFeatures(nlp_backend=self._backend.name)

        needs_nlp = any(
            r.rule_type in (
                SyntaxRuleType.NER_DENSITY,
                SyntaxRuleType.IMPERATIVE_SYNTAX,
                SyntaxRuleType.VERB_ENTITY_SPARSITY,
                SyntaxRuleType.CONDITIONAL_THREAT,
                SyntaxRuleType.ACTION_TARGET_TRIPLET,
                SyntaxRuleType.FINANCIAL_FLOW,
            )
            for r in self._rules.values()
        ) or self._needs_nlp_for_match_mode()

        parsed: dict[str, Any] = await self._backend.analyze_async(text) if needs_nlp else {}
        self._apply_rules(text, parsed, feats)
        return feats

    async def extract_batch_async(self, texts: list[str]) -> list[NlpFeatures]:
        """异步批量提取特征。"""
        needs_nlp = any(
            r.rule_type in (
                SyntaxRuleType.NER_DENSITY,
                SyntaxRuleType.IMPERATIVE_SYNTAX,
                SyntaxRuleType.VERB_ENTITY_SPARSITY,
                SyntaxRuleType.CONDITIONAL_THREAT,
                SyntaxRuleType.ACTION_TARGET_TRIPLET,
                SyntaxRuleType.FINANCIAL_FLOW,
            )
            for r in self._rules.values()
        ) or self._needs_nlp_for_match_mode()

        if needs_nlp:
            parsed_batch = await self._backend.analyze_batch_async(texts)
        else:
            parsed_batch = [{}] * len(texts)

        results: list[NlpFeatures] = []
        for text, parsed in zip(texts, parsed_batch):
            if not text.strip():
                results.append(NlpFeatures(nlp_backend=self._backend.name))
                continue
            feats = NlpFeatures(nlp_backend=self._backend.name)
            self._apply_rules(text, parsed, feats)
            results.append(feats)
        return results

    def _apply_rules(self, text: str, parsed: dict[str, Any], feats: NlpFeatures) -> None:
        """对单条文本应用全部规则（extract 和 extract_batch 共用）。"""
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
                self._extract_keyword_cooc(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.VERB_ENTITY_SPARSITY:
                self._extract_verb_entity_sparsity(parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.ISOLATION_REQUEST:
                self._extract_isolation_request(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.MICRO_ACTION_COMMAND:
                self._extract_micro_action_command(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.CONDITIONAL_THREAT:
                self._extract_conditional_threat(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.ACTION_TARGET_TRIPLET:
                self._extract_action_target_triplet(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.TEMPORAL_URGENCY:
                self._extract_simple_keywords(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.PRIVACY_INTRUSION:
                self._extract_simple_keywords(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.EMOTIONAL_MANIPULATION:
                self._extract_simple_keywords(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.IDENTITY_IMPERSONATION:
                self._extract_simple_keywords(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.CHANNEL_SHIFTING:
                self._extract_simple_keywords(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.FINANCIAL_FLOW:
                self._extract_action_target_triplet(text, parsed, rule, feats)

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
        """
        if mode == MatchMode.SUBSTR:
            return kw in text

        if mode == MatchMode.EXACT_WORD:
            if nlp_backend in ("LTP", "hanlp"):
                return kw in tokens
            return SyntaxFeatureExtractor._regex_chinese_boundary(kw, text)

        if mode == MatchMode.REGEX_BOUNDARY:
            return SyntaxFeatureExtractor._regex_chinese_boundary(kw, text)

        return kw in text  # 兜底

    @staticmethod
    def _regex_chinese_boundary(kw: str, text: str) -> bool:
        """
        中文词边界正则匹配（rule_based 后端降级方案）。
        """
        if len(kw) == 1:
            pattern = rf"{re.escape(kw)}(?![\u4e00-\u9fff\w])"
            return bool(re.search(pattern, text))
        return kw in text

    # ── 各类型提取器 ──────────────────────────────────────────

    def _extract_quantity_regex(
        self, text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
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
        p = rule.params
        second_person:   frozenset[str] = frozenset(p.second_person)
        urgency_adverbs: frozenset[str] = frozenset(p.urgency_adverbs)

        tokens: list[str]            = parsed.get("tokens", [])
        dep:    list[tuple[int, str]] = parsed.get("dep", [])

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
        p = rule.params
        threshold: int = p.threshold
        
        entities = [ent_type for _, ent_type in parsed.get("ner", [])]
        business_entity_count = sum(
            1 for e in entities if e in ("Ni", "Ns", "ORG", "GPE", "LOC", "nh", "PER")
        )
        
        if business_entity_count < threshold:
            feats.set_feature(rule.feature_name, True)

    # ── V5.1 新增：行为学/心理学特征提取器 ──────────────────

    def _extract_isolation_request(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
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
        p = rule.params
        condition_clauses: list[str] = p.condition_clauses
        threat_clauses:   list[str] = p.threat_clauses
        match_mode:       MatchMode = getattr(p, "match_mode", MatchMode.SUBSTR)
        if not condition_clauses or not threat_clauses:
            return

        tokens      = parsed.get("tokens", [])
        nlp_backend = feats.nlp_backend

        has_condition = any(
            self._match_keyword(cond, text, tokens, nlp_backend, match_mode)
            for cond in condition_clauses
        )
        has_threat = any(
            self._match_keyword(threat, text, tokens, nlp_backend, match_mode)
            for threat in threat_clauses
        )

        dep: list[tuple[int, str]] = parsed.get("dep", [])

        syntactic_confirm = False
        if dep and tokens and has_condition and has_threat:
            root_indices = [i for i, (_, rel) in enumerate(dep) if rel == "HED"]
            for root_idx in root_indices:
                children = [
                    (i, tokens[i].lower(), rel)
                    for i, (head, rel) in enumerate(dep)
                    if head == root_idx
                ]
                child_texts = [t for _, t, _ in children]
                has_cond_in_tree = any(c in " ".join(child_texts) for c in condition_clauses)
                has_thrt_in_tree = any(t in " ".join(child_texts) for t in threat_clauses)
                if has_cond_in_tree and has_thrt_in_tree:
                    syntactic_confirm = True
                    break

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
                feats.add_evidence(rule.evidence_key, hit_list[:6])

    def _extract_action_target_triplet(
        self, text: str, parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
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

        dep: list[tuple[int, str]] = parsed.get("dep", [])

        syntactic_confirm = False
        if dep and tokens and hit_verbs and hit_targets:
            
            for i, (head, rel) in enumerate(dep):
                if head >= len(tokens): 
                    continue
                    
                child_text = tokens[i]
                head_text  = tokens[head]

                if rel in ("VOB", "FOB"):
                    if head_text in hit_verbs and child_text in hit_targets:
                        syntactic_confirm = True
                        break

                if rel == "POB":
                    if child_text in hit_targets and head_text in ("把", "将", "给"):
                        prep_head_idx = dep[head][0]
                        prep_rel      = dep[head][1]
                        
                        if prep_head_idx < len(tokens):
                            prep_head_text = tokens[prep_head_idx]
                            if prep_rel == "ADV" and prep_head_text in hit_verbs:
                                syntactic_confirm = True
                                break

                if rel == "SBV" and child_text in hit_targets:
                    has_passive_marker = any(
                        head2 == head and rel2 == "ADV" and tokens[j] in ("被", "让", "叫", "给")
                        for j, (head2, rel2) in enumerate(dep)
                    )
                    if has_passive_marker and head_text in hit_verbs:
                        syntactic_confirm = True
                        break

        if hit_verbs and hit_targets:
            if syntactic_confirm:
                feats.set_feature(rule.feature_name, True)
            else:
                for verb in hit_verbs:
                    for target in hit_targets:
                        v_pos = text.find(verb)
                        t_pos = text.find(target)
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
# StageTwoPipeline —— V6.0 依赖注入 + 异步核心架构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StageTwoPipeline:
    """
    Y 型双轨流水线编排器，V6.0 依赖注入 + 异步核心。

    V6.0 核心变更
    ─────────────────────────────────────────────────────────
    ① 依赖注入（DI）：
       - topology / radar / binder / syntax 全部通过 __init__ 参数注入
       - 未传入时由 _create_default_xxx 工厂方法创建默认实例
       - 支持外部传入 Mock 对象进行单元测试

    ② 绝对 DRY：
       - process_conversation_async 为唯一业务核心
       - process_conversation 改为纯同步包装器（try running loop → new loop）

    ③ NlpBackend Protocol：
       - 用 typing.Protocol 替代 _NlpBackend 类继承
       - 支持结构化子类型，无需显式继承

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
        # ── 依赖注入参数（全部可选，未传入时走默认工厂）──
        topology:  TopologyAnalyzer       | None = None,
        radar:     IntentRadar            | None = None,
        binder:    RoleBinder             | None = None,
        syntax:    SyntaxFeatureExtractor | None = None,
        nlp_backend: NlpBackend           | None = None,
        # ── 向后兼容参数（用于默认工厂方法）──
        bge_model_name:   str   = "BAAI/bge-m3",
        use_fp16:         bool  = True,
        intent_threshold: float = 0.72,
        ltp_service_url:  str   | None = None,
        bge_service_url:  str   | None = None,
        *,
        _async: bool = False,
    ) -> None:
        self._is_async = _async

        # ── 依赖注入：优先使用外部传入的实例 ──
        if topology is not None:
            self._topology = topology
        else:
            self._topology = self._create_default_topology()

        if radar is not None:
            self._radar = radar
        else:
            self._radar = self._create_default_radar(
                bge_model_name=bge_model_name,
                use_fp16=use_fp16,
                bge_service_url=bge_service_url,
                is_async=_async,
            )

        if binder is not None:
            self._binder = binder
        else:
            self._binder = self._create_default_binder(
                radar=self._radar,
                topology=self._topology,
            )

        if syntax is not None:
            self._syntax = syntax
        elif nlp_backend is not None:
            self._syntax = SyntaxFeatureExtractor(backend=nlp_backend)
        else:
            self._syntax = self._create_default_syntax(
                ltp_service_url=ltp_service_url,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 工厂方法（可被子类覆盖或替换）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _create_default_topology() -> TopologyAnalyzer:
        """创建默认拓扑分析器。"""
        return TopologyAnalyzer()

    @staticmethod
    def _create_default_radar(
        bge_model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        bge_service_url: str | None = None,
        is_async: bool = False,
    ) -> IntentRadar:
        """创建默认意图雷达（同步或异步单例）。"""
        if is_async:
            return IntentRadar.get_async_instance(
                model_name      = bge_model_name,
                use_fp16        = use_fp16,
                bge_service_url = bge_service_url,
            )
        return IntentRadar.get_instance(
            model_name      = bge_model_name,
            use_fp16        = use_fp16,
            bge_service_url = bge_service_url,
        )

    @staticmethod
    def _create_default_binder(
        radar: IntentRadar,
        topology: TopologyAnalyzer,
    ) -> RoleBinder:
        """创建默认角色绑定器。"""
        return RoleBinder(radar=radar, topology=topology)

    @staticmethod
    def _create_default_syntax(
        ltp_service_url: str | None = None,
    ) -> SyntaxFeatureExtractor:
        """创建默认句法特征提取器。"""
        return SyntaxFeatureExtractor(
            backend=_load_nlp_backend(ltp_service_url=ltp_service_url)
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 同步入口：纯包装器（零业务逻辑，绝对 DRY）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def process_conversation(
        self,
        conversation_id: str,
        records:         list[ASRRecord],
        extra_metadata:  dict[str, Any] | None = None,
        pre_merged_turns: list[DialogueTurn] | None = None,
    ) -> StageTwoResult:
        """
        同步处理入口 —— 纯包装器，委托给 process_conversation_async。

        策略：尝试在已有 event loop 中运行（asyncio 生态兼容），
        若已在运行中的 loop 内被调用（如 filter_node 异步上下文），
        则创建新 loop 在独立线程中执行，避免阻塞当前 loop。

        V6.0：此方法零业务逻辑，所有核心代码在 _process_core 中。
        """
        coro = self.process_conversation_async(
            conversation_id  = conversation_id,
            records          = records,
            extra_metadata   = extra_metadata,
            pre_merged_turns = pre_merged_turns,
        )
        try:
            # 尝试获取当前运行中的 event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 无运行中的 loop → 直接 asyncio.run()
            return asyncio.run(coro)

        # 已有运行中的 loop（例如在 async 函数内被同步调用）
        # → 在独立线程中运行新 loop，避免嵌套 loop 崩溃
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 异步核心：唯一业务逻辑入口（绝对 DRY）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def process_conversation_async(
        self,
        conversation_id: str,
        records:         list[ASRRecord],
        extra_metadata:  dict[str, Any] | None = None,
        pre_merged_turns: list[DialogueTurn] | None = None,
    ) -> StageTwoResult:
        """
        异步核心 —— 唯一业务逻辑入口。

        与 V5.5 的 process_conversation_async 逻辑完全一致：
          - 软语义轨道：拓扑分析 + 角色绑定 + 意图标注
          - 硬句法轨道：批量 NLP 特征提取
          - 动态搜索轨道：BGE-M3 语义检索
          - 元信息汇总

        V6.0：同步版 process_conversation 不再有独立业务逻辑，
              全部委托给此方法，实现绝对 DRY。

        Parameters
        ----------
        pre_merged_turns : 若上游已合并过 turns，直接透传以避免重复 merge_turns。
        """
        # ── 软语义轨道 ────────────────────────────────────────
        if pre_merged_turns is not None:
            merged_turns = pre_merged_turns
        else:
            merged_turns = self._topology.merge_turns(records)
        track_type    = self._topology.classify_track(merged_turns)
        labeled_turns, role_results, ifeats = self._binder.bind(
            turns=merged_turns, track_type=track_type
        )

        # ── 硬句法轨道（异步批量） ────────────────────────────
        full_text: str = " ".join(t.merged_text for t in labeled_turns if not t.is_backchannel and t.merged_text.strip())

        texts_to_analyze: list[str] = []
        speaker_ids: list[str] = []

        texts_to_analyze.append(full_text)
        for role_res in role_results:
            sid = role_res.speaker_id
            speaker_text = " ".join(t.merged_text for t in labeled_turns if t.speaker_id == sid and not t.is_backchannel and t.merged_text.strip())
            texts_to_analyze.append(speaker_text)
            speaker_ids.append(sid)

        # 异步批量提取！
        batch_feats = await self._syntax.extract_batch_async(texts_to_analyze)

        global_nlp_feats: NlpFeatures = batch_feats[0] if full_text.strip() else NlpFeatures(nlp_backend=self._syntax._backend.name)
        speaker_nlp_feats: dict[str, dict[str, Any]] = {}
        for idx, sid in enumerate(speaker_ids):
            speaker_text = texts_to_analyze[idx + 1]
            speaker_nlp_feats[sid] = batch_feats[idx + 1].to_dict() if speaker_text.strip() else NlpFeatures(nlp_backend=self._syntax._backend.name).to_dict()

        # ── 动态搜索轨道（异步） ──
        dynamic_search_result = None

        raw_topic = extra_metadata.get("dynamic_topic") if extra_metadata else None

        topics_to_search = []
        if isinstance(raw_topic, str) and raw_topic.strip():
            topics_to_search = [raw_topic.strip()]
        elif isinstance(raw_topic, list):
            topics_to_search = [str(t).strip() for t in raw_topic if str(t).strip()]

        if topics_to_search:
            valid_turns = [
                t for t in labeled_turns
                if not t.is_backchannel and len(t.merged_text.strip()) > 1
            ]

            if valid_turns:
                window_size = 5
                stride = 2
                search_chunks = []

                for i in range(0, len(valid_turns), stride):
                    window = valid_turns[i : i + window_size]
                    if not window:
                        break
                    chunk_text = " ".join([f"[{t.speaker_id}] {t.merged_text.strip()}" for t in window])
                    search_chunks.append(chunk_text)

                # 异步批量检索！
                dynamic_search_result = await self._radar.dynamic_search_batch_async(
                    search_chunks=search_chunks,
                    dynamic_topics=topics_to_search,
                    default_threshold=0.40,
                    top_k=1,
                )
            else:
                dynamic_search_result = {
                    "topic_queried": str(topics_to_search),
                    "matched": False,
                    "max_score": 0.0,
                    "top_matches": [],
                    "status": "skipped_due_to_pure_backchannel"
                }

        # ── 汇总元信息 ──
        metadata: dict[str, Any] = {
            "raw_record_count":  len(records),
            "merged_turn_count": len(labeled_turns),
            "track_type":        track_type.value,
            "nlp_features":      global_nlp_feats.to_dict(),
            "speaker_nlp_features": speaker_nlp_feats,
        }

        if dynamic_search_result:
            metadata["dynamic_search"] = dynamic_search_result

        if extra_metadata:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 向后兼容导出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# _NlpBackend 类名已废弃，但保留导出以兼容旧 import
# 新代码应使用 NlpBackend Protocol
_NlpBackend = NlpBackend
