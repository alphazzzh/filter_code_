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


# 在模块加载时，扫描 config 中所有 QUANTITY_REGEX 规则，预编译正则
_COMPILED_QUANTITY_PATTERNS: dict[str, re.Pattern] = {}
_COMPILED_REGEX_PATTERNS:    dict[str, re.Pattern] = {}
_RE_ORG_SUFFIX:              re.Pattern            = _build_org_suffix_pattern()

for _rule in get_all_syntax_rules().values():
    if _rule.rule_type == SyntaxRuleType.QUANTITY_REGEX:
        _units = _rule.params.get("quantity_units", [])
        if _units:
            _COMPILED_QUANTITY_PATTERNS[_rule.feature_name] = (
                _build_quantity_pattern(_units)
            )
    elif _rule.rule_type == SyntaxRuleType.REGEX_PATTERN:
        _pattern_str = _rule.params.get("pattern", "")
        if _pattern_str:
            _flags_str  = _rule.params.get("flags", "UNICODE")
            _flags      = getattr(re, _flags_str, re.UNICODE)
            _COMPILED_REGEX_PATTERNS[_rule.feature_name] = re.compile(
                _pattern_str, _flags
            )


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

    def __init__(self, model_path: str = "LTP/small") -> None:
        from ltp import LTP  # type: ignore
        self._ltp = LTP(model_path)

    @property
    def name(self) -> str:
        return "ltp"

    def analyze(self, text: str) -> dict[str, Any]:
        output = self._ltp.pipeline([text], tasks=["cws", "dep", "ner"])
        tokens: list[str] = output.cws[0] if output.cws else []
        dep: list[tuple[int, str]] = [
            (d["head"] - 1, d["label"].upper())
            for d in (output.dep[0] if output.dep else [])
        ]
        ner: list[tuple[str, str]] = [
            (s[0], s[1]) for s in (output.ner[0] if output.ner else [])
        ]
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


def _load_nlp_backend() -> _NlpBackend:
    """按 LTP → HanLP → 规则 顺序自动降级。"""
    for BackendCls, label in [(_LtpBackend, "LTP"), (_HanLpBackend, "HanLP")]:
        try:
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
                SyntaxRuleType.VERB_ENTITY_SPARSITY,  # 业务实体稀疏度探针也依赖 NER 解析
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

        return feats

    # ── 各类型提取器 ──────────────────────────────────────────

    @staticmethod
    def _extract_quantity_regex(
        text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """
        QUANTITY_REGEX：数字 + 量词正则匹配。
        使用模块加载时预编译的缓存正则，零运行时开销。
        """
        pattern = _COMPILED_QUANTITY_PATTERNS.get(rule.feature_name)
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

    @staticmethod
    def _extract_regex_pattern(
        text: str, rule: SyntaxRuleConfig, feats: NlpFeatures
    ) -> None:
        """REGEX_PATTERN：自定义正则，使用预编译缓存。"""
        pattern = _COMPILED_REGEX_PATTERNS.get(rule.feature_name)
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
        self._syntax = SyntaxFeatureExtractor(backend=nlp_backend)

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
                    top_k=10  # 请求返回前 10 个高分片段
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
