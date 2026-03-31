# stage_two_pipeline.py  ── V5.0 配置驱动版
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
# ============================================================

from __future__ import annotations

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
    TOPIC_REGISTRY,
    get_all_syntax_rules,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NlpFeatures —— 动态 key-value 快照
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NlpFeatures:
    """
    硬句法轨道产出的结构化特征。

    V5.0 变化：特征 key 由 config_topics 中的 SyntaxRuleConfig.feature_name
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
# 全局预编译正则缓存
# 从 config_topics 中收集 QUANTITY_REGEX 和 REGEX_PATTERN 规则，
# 在模块加载时一次性编译，禁止在循环中重复编译。
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
# NLP 后端抽象层（与 V4.0 保持一致，仅接口不变）
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
    """LTP 4.x 后端，标签：HED/SBV/ADV/VOB。"""

    def __init__(self, model_path: str = "/home/zzh/923/model/ltp_small") -> None:
        from ltp import LTP  # type: ignore
        import os
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"[LTP] 本地模型文件夹不存在: {model_path}")
        self._ltp = LTP(model_path)

    @property
    def name(self) -> str:
        return "ltp"

    def analyze(self, text: str) -> dict[str, Any]:
        output = self._ltp.pipeline([text], tasks=["cws", "dep", "ner"])
        tokens: list[str] = output.cws[0] if output.cws else []
        
        # ==========================================================
        # 1. 鲁棒兼容处理依存句法 (dep)
        # ==========================================================
        dep: list[tuple[int, str]] = []
        if output.dep:
            dep_data = output.dep[0]
            if isinstance(dep_data, dict):
                # 新版 LTP 格式: {'head': [2, 0], 'label': ['SBV', 'HED']}
                heads = dep_data.get("head", [])
                labels = dep_data.get("label", [])
                for h, l in zip(heads, labels):
                    dep.append((h - 1, l.upper()))
            elif isinstance(dep_data, list):
                # 旧版 LTP 格式: [{'head': 2, 'label': 'SBV'}, ...]
                for d in dep_data:
                    if isinstance(d, dict):
                        dep.append((d.get("head", 1) - 1, d.get("label", "UNK").upper()))
        
        # ==========================================================
        # 2. 鲁棒兼容处理命名实体识别 (ner) (修复幽灵 Bug)
        # ==========================================================
        ner: list[tuple[str, str]] = []
        if output.ner:
            ner_data = output.ner[0]
            if isinstance(ner_data, dict):
                # 新版 LTP 格式: {'label': ['Nh'], 'text': ['张三'], 'offset': [...]}
                labels = ner_data.get("label", [])
                texts = ner_data.get("text", [])
                for t, l in zip(texts, labels):
                    ner.append((t, l))
            elif isinstance(ner_data, list):
                # 旧版 LTP 格式: [{'label': 'Nh', 'text': '张三'}, ...]
                for d in ner_data:
                    if isinstance(d, dict):
                        t = d.get("text", d.get("name", ""))
                        l = d.get("label", d.get("tag", ""))
                        ner.append((t, l))
                        
        return {"tokens": tokens, "dep": dep, "ner": ner}


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


def _load_nlp_backend(ltp_model_path: str = "LTP/small") -> _NlpBackend:
    """按 LTP → HanLP → 规则 顺序自动降级，ltp_model_path 透传到 LTP 后端。"""
    for BackendCls, label in [(_LtpBackend, "LTP"), (_HanLpBackend, "HanLP")]:
        try:
            if BackendCls is _LtpBackend:
                return _LtpBackend(model_path=ltp_model_path)
            return BackendCls()
        except Exception as e:
            warnings.warn(f"[NLP] {label} 加载失败（{e}），尝试下一个。", RuntimeWarning)
    return _RuleBasedFallback()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SyntaxFeatureExtractor —— 配置驱动版
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SyntaxFeatureExtractor:
    """
    V5.0 配置驱动的硬句法特征提取器。

    初始化时从 config_topics.get_all_syntax_rules() 加载所有规则，
    extract() 遍历规则列表，按 SyntaxRuleType 动态分发到对应提取器。

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
        """扫描当前 self._rules，预编译所有 QUANTITY_REGEX / REGEX_PATTERN 正则。"""
        self._compiled_quantity_patterns.clear()
        self._compiled_regex_patterns.clear()
        for _rule in self._rules.values():
            if _rule.rule_type == SyntaxRuleType.QUANTITY_REGEX:
                _units = _rule.params.get("quantity_units", [])
                if _units:
                    self._compiled_quantity_patterns[_rule.feature_name] = (
                        _build_quantity_pattern(_units)
                    )
            elif _rule.rule_type == SyntaxRuleType.REGEX_PATTERN:
                _pattern_str = _rule.params.get("pattern", "")
                if _pattern_str:
                    _flags_str = _rule.params.get("flags", "UNICODE")
                    _flags     = getattr(re, _flags_str, re.UNICODE)
                    self._compiled_regex_patterns[_rule.feature_name] = re.compile(
                        _pattern_str, _flags
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
          QUANTITY_REGEX   → _extract_quantity_regex()
          REGEX_PATTERN    → _extract_regex_pattern()
          NER_DENSITY      → _extract_ner_density()   (需要 NLP 解析)
          IMPERATIVE_SYNTAX→ _extract_imperative()     (需要 NLP 解析)
          KEYWORD_COOC     → _extract_keyword_cooc()
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
            )
            for r in self._rules.values()
        )
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
                self._extract_keyword_cooc(text, rule, feats)


            elif rule.rule_type == SyntaxRuleType.VERB_ENTITY_SPARSITY:
                self._extract_verb_entity_sparsity(parsed, rule, feats)

            # ── V5.1 新增：行为学/心理学特征提取器 ────────────
            elif rule.rule_type == SyntaxRuleType.ISOLATION_REQUEST:
                self._extract_isolation_request(text, rule, feats)

            elif rule.rule_type == SyntaxRuleType.MICRO_ACTION_COMMAND:
                self._extract_micro_action_command(text, rule, feats)

            elif rule.rule_type == SyntaxRuleType.CONDITIONAL_THREAT:
                self._extract_conditional_threat(text, parsed, rule, feats)

            elif rule.rule_type == SyntaxRuleType.ACTION_TARGET_TRIPLET:
                self._extract_action_target_triplet(text, parsed, rule, feats)

        return feats

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
        """
        entity_types: frozenset[str] = frozenset(
            rule.params.get("entity_types", ["Ni", "Ns", "ORG", "GPE", "LOC"])
        )
        threshold: int = int(rule.params.get("threshold", 3))

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
        """
        second_person:  frozenset[str] = frozenset(
            rule.params.get("second_person", ["你", "您", "you"])
        )
        urgency_adverbs: frozenset[str] = frozenset(
            rule.params.get("urgency_adverbs", [
                "马上", "立刻", "立即", "赶紧", "赶快",
                "现在", "快", "即刻", "迅速",
            ])
        )

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

    @staticmethod
    def _extract_keyword_cooc(
        text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        KEYWORD_COOC：多关键词集合共现检测。

        逻辑：params["keyword_sets"] 是一个列表的列表。
        每个子列表至少有一个词在 text 中出现，
        且所有子列表均满足 → 判定命中。

        例（金融勒索检测）：
          keyword_sets = [
              ["捐", "缴", "交钱"],   # 子列表 A：金融词
              ["必须", "否则", "后果"],# 子列表 B：威胁词
          ]
          → 文本中同时含有 A 中至少一个词 AND B 中至少一个词 → 命中
        """
        keyword_sets: list[list[str]] = rule.params.get("keyword_sets", [])
        if not keyword_sets:
            return

        all_sets_matched = all(
            any(kw in text for kw in kw_set)
            for kw_set in keyword_sets
        )
        if all_sets_matched:
            feats.set_feature(rule.feature_name, True)

    @staticmethod
    def _extract_verb_entity_sparsity(
        parsed: dict[str, Any], rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """VERB_ENTITY_SPARSITY：实体与业务词极度稀疏检测"""
        threshold: int = int(rule.params.get("threshold", 3))
        
        # 统计核心业务名词（机构、地名、人名等）的数量
        entities = [ent_type for _, ent_type in parsed.get("ner", [])]
        business_entity_count = sum(
            1 for e in entities if e in ("Ni", "Ns", "ORG", "GPE", "LOC", "nh", "PER")
        )
        
        # 如果整通电话的实体数连 threshold 都不够，判定为缺乏有效主宾语的闲聊废料
        if business_entity_count < threshold:
            feats.set_feature(rule.feature_name, True)

    # ── V5.1 新增：行为学/心理学特征提取器 ──────────────────

    @staticmethod
    def _extract_isolation_request(
        text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        ISOLATION_REQUEST：物理隔离与信息阻断检测。

        诈骗核心前提：切断受害者外部信息源。
        探测维度：
          - 空间隔离：「找个没人的房间」「把门反锁」「到外面去」
          - 通讯阻断：「不要挂电话」「开启飞行模式」「拦截短信」「关掉WiFi」
          - 社交阻断：「不能告诉家人」「国家机密」「不要跟别人说」
        """
        isolation_keywords: list[str] = rule.params.get("isolation_keywords", [])
        if not isolation_keywords:
            return

        hit_words = [kw for kw in isolation_keywords if kw in text]
        if hit_words:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, hit_words)

    @staticmethod
    def _extract_micro_action_command(
        text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        MICRO_ACTION_COMMAND：服从性测试与微动作指令检测。

        高阶诈骗在切入核心客体前，通过微小指令测试受害者被控程度。
        探测维度：
          - 设备操作：「打开免提」「点右上角」「点一下设置」
          - 屏幕引导：「往下滑」「点击链接」「扫码」
          - 行为测试：「跟着我读一遍」「你现在打开」「不要动手机」
        """
        device_keywords: list[str] = rule.params.get("device_action_keywords", [])
        if not device_keywords:
            return

        hit_words = [kw for kw in device_keywords if kw in text]
        if hit_words:
            feats.set_feature(rule.feature_name, True)
            if rule.evidence_key:
                feats.add_evidence(rule.evidence_key, hit_words)

    @staticmethod
    def _extract_conditional_threat(
        text: str,
        parsed: dict[str, Any],
        rule: SyntaxRuleConfig,
        feats: NlpFeatures,
    ) -> None:
        """
        CONDITIONAL_THREAT：条件胁迫与逻辑陷阱检测。

        升级版 KEYWORD_COOC，利用「条件从句 + 负面主句」结构：
          [如果不...] → [征信受损/涉嫌违法/拘留/影响子女]

        降级策略：无依存句法时使用正则匹配条件从句模式。
        """
        condition_clauses: list[str] = rule.params.get("condition_clauses", [])
        threat_clauses:   list[str] = rule.params.get("threat_clauses", [])
        if not condition_clauses or not threat_clauses:
            return

        # 检测条件从句是否出现
        has_condition = any(cond in text for cond in condition_clauses)
        # 检测威胁主句是否出现
        has_threat = any(threat in text for threat in threat_clauses)

        # ── 依存句法增强：确认条件-威胁在同一小句范围内 ──
        tokens: list[str]            = parsed.get("tokens", [])
        dep:    list[tuple[int, str]] = parsed.get("dep", [])

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
                hit_list.extend(c for c in condition_clauses if c in text)
            if has_threat:
                hit_list.extend(t for t in threat_clauses if t in text)
            if rule.evidence_key and hit_list:
                feats.add_evidence(rule.evidence_key, hit_list[:6])  # 最多保留 6 条证据

    @staticmethod
    def _extract_action_target_triplet(
        text: str,
        parsed: dict[str, Any],
        rule: SyntaxRuleConfig,
        feats: NlpFeatures,
    ) -> None:
        action_verbs:    list[str] = rule.params.get("action_verbs", [])
        target_entities: list[str] = rule.params.get("target_entities", [])
        if not action_verbs or not target_entities:
            return

        hit_verbs = [v for v in action_verbs if v in text]
        hit_targets = [t for t in target_entities if t in text]

        # ── 依存句法增强：兼容 VOB, FOB 以及“把”字句(POB) ──
        tokens: list[str]            = parsed.get("tokens", [])
        dep:    list[tuple[int, str]] = parsed.get("dep", [])

        syntactic_confirm = False
        if dep and tokens and hit_verbs and hit_targets:
            
            # 遍历所有节点 (i为子节点索引, head为父节点索引, rel为关系)
            for i, (head, rel) in enumerate(dep):
                if head >= len(tokens): 
                    continue
                    
                child_text = tokens[i]
                head_text  = tokens[head]

                # 🎯 场景 1 & 2：直接宾语 (VOB) 或 前置宾语 (FOB)
                # 例如：“输入(head) 验证码(child)” 或 “验证码(child) 输入(head)了吗”
                if rel in ("VOB", "FOB"):
                    if head_text in hit_verbs and child_text in hit_targets:
                        syntactic_confirm = True
                        break

                # 🎯 场景 3：“把”字句 / 介词宾语结构 (POB 向上追溯 ADV)
                # 例如：“把(head) 验证码(child) 发(prep_head) 给我”
                if rel == "POB":
                    # 如果当前词是目标实体（如验证码），且它的父节点是“把”或“将”
                    if child_text in hit_targets and head_text in ("把", "将", "给"):
                        # 向上追溯第二跳：找到“把”依附的核心动词
                        prep_head_idx = dep[head][0] # “把”的父节点索引
                        prep_rel      = dep[head][1] # “把”与父节点的关系
                        
                        if prep_head_idx < len(tokens):
                            prep_head_text = tokens[prep_head_idx]
                            # 如果“把”是作为状语(ADV)依附在我们的高危动作词上(如"发")
                            if prep_rel == "ADV" and prep_head_text in hit_verbs:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StageTwoPipeline —— V5.0 配置驱动编排器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StageTwoPipeline:
    """
    V5.0 Y 型双轨流水线编排器，配置驱动，无硬编码。

    两轨并行：
      软语义轨道 → dialogue_turns[*].intent_labels
      硬句法轨道 → metadata["nlp_features"]
    """

    def __init__(
        self,
        bge_model_name:   str   = "BAAI/bge-m3",
        use_fp16:         bool  = True,
        intent_threshold: float = 0.72,
        ltp_model_path:   str   = "LTP/small",
        nlp_backend:      Optional[_NlpBackend] = None,
    ) -> None:
        self._topology = TopologyAnalyzer()
        self._radar    = IntentRadar.get_instance(
            model_name = bge_model_name,
            use_fp16   = use_fp16,
        )
        self._binder = RoleBinder(
            radar    = self._radar,
            topology = self._topology,
        )
        # 若未传入外部 nlp_backend，则使用 _load_nlp_backend 并透传 ltp_model_path
        if nlp_backend is not None:
            self._syntax = SyntaxFeatureExtractor(backend=nlp_backend)
        else:
            self._syntax = SyntaxFeatureExtractor(
                backend=_load_nlp_backend(ltp_model_path=ltp_model_path)
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

        # 鲁棒性设计 1：安全的字典读取
        dynamic_topic = extra_metadata.get("dynamic_topic") if extra_metadata else None
        
        if dynamic_topic and isinstance(dynamic_topic, str):
            # 清洗对话文本，剔除无意义的语气词 "嗯", "啊"
            valid_turns = [
                t for t in labeled_turns 
                if not t.is_backchannel and len(t.merged_text.strip()) > 1
            ]
            
            if valid_turns:
                # =========================================================
                # 【RAG 优化方向二：滑动窗口上下文重组 (Contextual Chunking)】
                # 设定窗口大小为 5 句话，步长为 2 句话，保留上下文交叉
                # =========================================================
                window_size = 5
                stride = 2
                search_chunks = []
                
                for i in range(0, len(valid_turns), stride):
                    window = valid_turns[i : i + window_size]
                    if not window:
                        break
                    # 拼接带说话人标识的完整上下文，例如： "[0] 喂你好 [1] 你的快递到了"
                    chunk_text = " ".join([f"[{t.speaker_id}] {t.merged_text.strip()}" for t in window])
                    search_chunks.append(chunk_text)

                # 调用雷达进行搜索，传入拼接好的上下文块
                dynamic_search_result = self._radar.dynamic_search(
                    search_chunks=search_chunks, 
                    dynamic_topic=dynamic_topic,
                    default_threshold=0.65,
                    top_k=1  
                )
            else:
                # 对话全是废话/静音，直接返回未匹配
                dynamic_search_result = {
                    "topic_queried": dynamic_topic,
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
