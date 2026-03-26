# stage_one_filter.py
# ============================================================
# ASR 流水线 —— 阶段一：无损特征提取与高危数据拦截
#
# 模块职责划分
# ─────────────────────────────────────────────────────────────
#  Normalizer          : NFKC 归一化 + 标点压缩（纯文本变换）
#  BotFeatureExtractor : 语气词占比 + 前缀锚点（只读，不修改文本）
#  LIDRouter           : fastText 语种识别 + 短文本回退
#  ASRErrorCorrector   : 多语言 ASR 容错（中文拼音 / 英文 Double Metaphone）
#  StageOneFilter      : 编排以上子模块，输出填充完整的 ASRRecord
# ============================================================

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

# ── 第三方库 ──────────────────────────────────────────────────
try:
    import fasttext  # fasttext-wheel
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

import jellyfish          # Double Metaphone, Soundex 等
from pypinyin import lazy_pinyin  # 中文拼音转写

from models import (
    ASRRecord,
    BotFeatures,
    BotLabel,
    ConnectionStatus,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Normalizer —— NFKC 归一化 + 标点压缩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Normalizer:
    """
    纯文本归一化器。
    职责：全半角统一 + 无意义连续标点压缩。
    绝对不删除原始字词，输出文本字数不会减少超过正常压缩幅度。
    """

    # 预编译正则：连续 2 次以上的同一中文标点 → 保留 1 个
    # 覆盖常见标点：，。！？、；：…
    _RE_REPEAT_PUNCT: re.Pattern = re.compile(
        r"([，。！？、；：…])\1{1,}", re.UNICODE
    )

    # 预编译正则：多个空白字符 → 单个空格（ASR 转写常见噪声）
    _RE_MULTI_SPACE: re.Pattern = re.compile(r"\s{2,}")

    def normalize(self, text: str) -> str:
        """
        执行顺序：
        1. NFKC 归一化（全角→半角、兼容字符统一）
        2. 连续同类中文标点压缩为单个
        3. 多余空白压缩
        """
        # Step 1：Unicode NFKC 归一化
        text = unicodedata.normalize("NFKC", text)

        # Step 2：连续标点压缩（`，，，` → `，`）
        text = self._RE_REPEAT_PUNCT.sub(r"\1", text)

        # Step 3：空白压缩
        text = self._RE_MULTI_SPACE.sub(" ", text).strip()

        return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BotFeatureExtractor —— 轻量级机器人特征提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 语气词词表（可扩展）
_FILLER_WORDS: frozenset[str] = frozenset([
    "嗯", "啊", "哦", "哈", "呢", "吧", "嘛", "呃",
    "那个", "就是", "然后", "对", "对对", "好的",
    "umm", "uh", "er", "like", "you know",
])


class BotFeatureExtractor:
    """
    从归一化文本中提取轻量级机器人特征。
    只读操作，不修改传入文本。
    """

    def __init__(self, filler_words: frozenset[str] = _FILLER_WORDS) -> None:
        self._filler_words = filler_words
        # 预编译：用于分词的简单中文字符切割（粒度：单字 + 英文词）
        self._RE_TOKEN: re.Pattern = re.compile(
            r"[\u4e00-\u9fff]|[a-zA-Z]+", re.UNICODE
        )

    def extract(self, text: str) -> BotFeatures:
            tokens: list[str] = self._RE_TOKEN.findall(text.lower())
            total: int = len(tokens)

            if total == 0:
                filler_ratio = 0.0
            else:
                filler_count = sum(1 for t in tokens if t in self._filler_words)
                filler_ratio = round(filler_count / total, 4)

            # 👇 激活全局指纹防线特征：截取前 30 个字计算哈希摘要
            prefix_str = text[:30]
            # 使用原生 hash (工业级系统可换为 mmh3)，生成特征快照
            simhash_val = str(hash(prefix_str) % (10 ** 8)) if prefix_str else None

            return BotFeatures(
                filler_word_ratio=filler_ratio,
                prefix_tokens=text[:20],
                simhash_value=simhash_val, # 👈 下游 Flink 可在 5 分钟滑动窗口中以此做碰撞拦截
                total_tokens=total,         # 👈 保存总 Token 数，供 infer_bot_label 短文本保护使用
            )

    # 短文本 Token 阈值：低于此值无法通过语气词可靠判定 BOT
    _BOT_MIN_TOKENS: int = 30

    @staticmethod
    def infer_bot_label(features: BotFeatures) -> BotLabel:
        """
        启发式判定：仅在长对话中，语气词占比极低才疑似机器人。
        短文本（< 30 Token）一律返回 UNCERTAIN，避免误杀口齿清晰的真人短对话。
        """
        # 【Bug 2 修复】短文本保护：Token 数不足 30，无法可靠判定，一律给 UNCERTAIN
        if features.total_tokens < BotFeatureExtractor._BOT_MIN_TOKENS:
            return BotLabel.UNCERTAIN

        if features.filler_word_ratio < 0.02:
            return BotLabel.BOT
        elif features.filler_word_ratio > 0.05:
            return BotLabel.HUMAN
        else:
            return BotLabel.UNCERTAIN



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LIDRouter —— 语种识别与路由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SHORT_TEXT_THRESHOLD: int = 10   # 字符数低于此值时跳过 LID
_DEFAULT_LANG: str = "zh"         # 短文本默认语种


class LIDRouter:
    """
    语种识别路由器，封装 fastText lid.176.bin 模型。
    当文本过短时，自动回退到默认语种（zh），避免误判。

    Parameters
    ----------
    model_path : str | Path
        fastText lid.176.bin 的路径。
        如果路径不存在或 fasttext 未安装，退化为全量回退模式（全部返回 zh）。
    """

    def __init__(
        self,
        model_path: str | Path = "lid.176.bin",
        short_threshold: int = _SHORT_TEXT_THRESHOLD,
        default_lang: str = _DEFAULT_LANG,
    ) -> None:
        self._short_threshold = short_threshold
        self._default_lang = default_lang
        self._model = None

        # 延迟加载 fastText 模型（失败时优雅降级）
        model_path = Path(model_path)
        if _FASTTEXT_AVAILABLE and model_path.exists():
            try:
                self._model = fasttext.load_model(str(model_path))
            except Exception as e:
                # 模型加载失败，记录警告并继续（降级为全量回退）
                import warnings
                warnings.warn(
                    f"[LIDRouter] fastText 模型加载失败：{e}，"
                    "将使用默认语种回退策略。",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def detect(self, text: str) -> tuple[str, Optional[float]]:
        """
        识别语种。

        Returns
        -------
        (lang_code, confidence)
            lang_code  : ISO 639-1 代码，如 'zh', 'en', 'ug'
            confidence : 0~1 之间的置信度；短文本回退时为 None
        """
        # 短文本直接回退
        if len(text) < self._short_threshold:
            return self._default_lang, None

        # fastText 不可用时全量回退
        if self._model is None:
            return self._default_lang, None

        # fastText 预测，取 top-1
        # 返回格式：(['__label__zh'], [0.9987])
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)
        lang_code: str = labels[0].replace("__label__", "")
        raw_prob = float(probs[0])
        clamped_prob = min(1.0, max(0.0, raw_prob))
        confidence: float = round(clamped_prob, 4)

        return lang_code, confidence


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ASRErrorCorrector —— 多语言 ASR 容错
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 中文结巴去重正则（预编译）──────────────────────────────────
# 匹配 2 次以上连续重复的非数字字符（防止误删电话号码中的连续数字）
_RE_ZH_STUTTER: re.Pattern = re.compile(
    r"([^\d\s])\1{2,}", re.UNICODE
)

# ── 英文词级别结巴去重（预编译）──────────────────────────────
# 匹配连续重复出现 2 次以上的同一英文词（忽略大小写）
_RE_EN_STUTTER: re.Pattern = re.compile(
    r"\b(\w+)(\s+\1){2,}\b", re.IGNORECASE | re.UNICODE
)


def _pinyin_similarity(word_a: str, word_b: str) -> float:
    """
    中文拼音容错：将两个词转为拼音序列后，
    计算字符级 Levenshtein 相似度（jellyfish）。

    Returns
    -------
    float : 0~1，越接近 1 表示越相似
    """
    py_a: str = " ".join(lazy_pinyin(word_a))
    py_b: str = " ".join(lazy_pinyin(word_b))
    dist: int = jellyfish.levenshtein_distance(py_a, py_b)
    max_len: int = max(len(py_a), len(py_b), 1)
    return 1.0 - dist / max_len


def _double_metaphone_match(word_a: str, word_b: str) -> bool:
    """
    英文 Double Metaphone 音似匹配。
    两个词的主码或副码任意一对相同，则认为发音相似。

    例：Smith / Smyth → ('SM0', 'XMT') vs ('SM0', 'XMT') → True
    """
    codes_a: tuple[str, str] = jellyfish.double_metaphone(word_a)
    codes_b: tuple[str, str] = jellyfish.double_metaphone(word_b)
    # 过滤空码后进行集合交叉
    set_a: set[str] = {c for c in codes_a if c}
    set_b: set[str] = {c for c in codes_b if c}
    return bool(set_a & set_b)


def _ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    字符 N-gram Jaccard 相似度（维吾尔语等缺乏音素字典时的兜底方案）。

    Returns
    -------
    float : Jaccard 相似度 0~1
    """
    if not s1 or not s2:
        return 0.0
    grams_a: set[str] = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    grams_b: set[str] = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    if not grams_a and not grams_b:
        return 1.0
    intersection: int = len(grams_a & grams_b)
    union: int = len(grams_a | grams_b)
    return intersection / union if union else 0.0


class ASRErrorCorrector:
    """
    多语言 ASR 容错处理器。
    核心职责：
      1. 中文：去结巴 + 拼音相似度检测（供调用方决策）
      2. 英文/印欧：去结巴 + Double Metaphone 音似规整
      3. 其他语种（维吾尔语等）：字符 N-gram Jaccard 相似度兜底
    """

    def correct(self, text: str, lang: str) -> str:
        """
        根据语种分发处理。

        Parameters
        ----------
        text : 归一化后的文本
        lang : LID 输出的语种代码

        Returns
        -------
        str : 容错处理后的文本
        """
        if lang.startswith("zh"):
            return self._correct_chinese(text)
        elif lang.startswith(("en", "es", "fr", "de", "it", "pt")):
            return self._correct_indo_european(text)
        else:
            # 其他语种：仅做字符级结巴去重，不做音似纠错
            return self._correct_generic(text)

    # ── 分支 A：中文 ──────────────────────────────────────────

    @staticmethod
    def _correct_chinese(text: str) -> str:
        """
        1. 字符级结巴去重（正则预编译）。
        2. 说明：拼音容错函数（_pinyin_similarity）暴露在模块级别，
           供调用方在候选词比对场景中按需调用，
           此处不做全文扫描（开销过大）。
        """
        return _RE_ZH_STUTTER.sub(r"\1", text)

    # ── 分支 B：英文 / 印欧语系 ──────────────────────────────

    @staticmethod
    def _correct_indo_european(text: str) -> str:
        """
        1. 词级别结巴去重（"I I I want" → "I want"）。
        2. Double Metaphone 仅在外部候选词比对时使用（_double_metaphone_match），
           此处只做结巴处理。
        """
        return _RE_EN_STUTTER.sub(r"\1", text)

    # ── 分支 C：其他语种 ──────────────────────────────────────

    @staticmethod
    def _correct_generic(text: str) -> str:
        """
        字符级结巴去重（通用正则），适用于维吾尔语等。
        N-gram 相似度（_ngram_jaccard）供外部调用，不在此处全文扫描。
        """
        return _RE_ZH_STUTTER.sub(r"\1", text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. StageOneFilter —— 阶段一流水线编排
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StageOneFilter:
    """
    阶段一流水线：无损特征提取 + 高危数据拦截。

    执行顺序（严格遵守"先提取、后清洗"原则）
    ─────────────────────────────────────────────────────────
    动作 1：Normalizer         → 填充 normalized_text
    动作 2：BotFeatureExtractor → 填充 bot_features / bot_label
    动作 2b：强制 CONNECTED 放行  → connection_status = CONNECTED
    动作 3：LIDRouter           → 填充 lang / lang_confidence
    动作 4：ASRErrorCorrector   → 填充 cleaned_text
    ─────────────────────────────────────────────────────────
    所有操作均通过 process(record) 接口，
    接收并返回同一个 ASRRecord 实例（原地修改字段）。
    raw_text 字段在整个流水线中永远不会被修改。
    """

    def __init__(
        self,
        fasttext_model_path: str | Path = "lid.176.bin",
        filler_words: Optional[frozenset[str]] = None,
    ) -> None:
        # 子模块实例化（各自在 __init__ 中完成昂贵资源的预加载）
        self._normalizer    = Normalizer()
        self._bot_extractor = BotFeatureExtractor(
            filler_words=filler_words or _FILLER_WORDS
        )
        # V1.0 UnconnectedDetector 已铲除：单句级熵值检测在多轮次 V5.0 下误判率过高
        self._lid_router    = LIDRouter(model_path=fasttext_model_path)
        self._corrector     = ASRErrorCorrector()

    # ── 公开接口 ──────────────────────────────────────────────

    def process(self, record: ASRRecord) -> ASRRecord:
        """
        执行完整的阶段一处理流程。

        Parameters
        ----------
        record : ASRRecord
            必须已填充 record_id 和 raw_text，其余字段为 None。

        Returns
        -------
        ASRRecord
            原地修改并返回同一实例，stage_one_done 置为 True。

        Notes
        -----
        若 connection_status 为 UNCONNECTED，流程**仍会继续**执行
        后续动作（LID、ASR 容错），以便完整保留所有特征供下游审计。
        如需提前短路，在业务层判断 stage_one_done 后自行过滤。
        """
        # ── 动作 1：归一化 ─────────────────────────────────────
        record.normalized_text = self._normalizer.normalize(record.raw_text)

        # 后续所有动作均基于 normalized_text，保证 raw_text 只读
        working_text: str = record.normalized_text

        # ── 动作 2a：机器人特征提取 ────────────────────────────
        record.bot_features = self._bot_extractor.extract(working_text)
        record.bot_label    = self._bot_extractor.infer_bot_label(record.bot_features)

        # ── 动作 2b：强制放行（V1.0 UnconnectedDetector 已铲除）──
        record.unconnected_features = None
        record.connection_status    = ConnectionStatus.CONNECTED

        # ── 动作 3：语种识别 ───────────────────────────────────
        lang, confidence = self._lid_router.detect(working_text)
        record.lang             = lang
        record.lang_confidence  = confidence

        # ── 动作 4：ASR 容错处理 ───────────────────────────────
        record.cleaned_text = self._corrector.correct(working_text, lang)

        # ── 标记阶段完成 ───────────────────────────────────────
        record.stage_one_done = True

        # ── 追加 metadata 调试信息 ─────────────────────────────
        record.metadata["stage_one"] = {
            "lang_fallback":  confidence is None,
        }

        return record

    def process_batch(self, records: list[ASRRecord]) -> list[ASRRecord]:
        """
        批量处理，返回已完成阶段一的记录列表。
        注：高并发场景可将此方法替换为 concurrent.futures.ThreadPoolExecutor。
        """
        return [self.process(r) for r in records]
