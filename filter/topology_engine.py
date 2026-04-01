# topology_engine.py
# ============================================================
# 阶段二：同源轮次合并 + 对话拓扑分流 + 拓扑特征度量
#
# 模块组成
# ─────────────────────────────────────────────────────────────
#  TopologyAnalyzer    : 同源轮次合并 + 轨道分类（ASYMMETRIC / SYMMETRIC）
#  TopologyEngine      : 拓扑特征计算引擎（filler_word_rate / 解耦盲说等）
#  TopologyMetrics     : 多维度量快照数据类
#
# V5.1 变更
# ─────────────────────────────────────────────────────────────
# ① _FILLER_WORDS 扩充至中/英/日/粤/韩五语种
# ② 新增 TopologyEngine + TopologyMetrics（原仅 TopologyAnalyzer）
# ③ 新增 is_decoupled 解耦盲说判定
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from models_stage2 import ASRRecord, DialogueTurn, TrackType


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 常量与预编译（模块级，禁止在循环中重复编译）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 纯附和判定词集（冻结集合，O(1) 查找）
_BACKCHANNEL_CHARS: frozenset[str] = frozenset(
    "嗯啊哦呢吧哈哟喔唔嘿呀哼"
)
_BACKCHANNEL_WORDS: frozenset[str] = frozenset([
    "嗯嗯", "哦哦", "啊啊", "好的", "好", "行", "对", "对对",
    "对对对", "是", "是是", "嗯哼", "哈哈",
    "ok", "okay", "yeah", "yep", "uh-huh", "mhm",
])

# 语气词集合（用于 filler_word_rate 计算）—— 多语言支持：中/英/日/粤/韩
_FILLER_WORDS: frozenset[str] = frozenset({
    # ── 中文语气词 ──
    "嗯", "啊", "那个", "就是说", "哎", "噢", "呀", "呢", "嘛",
    "哦", "哈", "哎哟", "唔", "嘿", "噢", "哦哦", "哈哈", "唉",
    # ── 粤语语气词 ──
    "咁", "即係", "誒", "唔", "呢", "咗", "嘅", "噃", "囉", "喇",
    # ── 英文语气词 ──
    "um", "uh", "well", "like", "you know", "hmm", "er", "ah",
    "oh", "okay", "ok", "yeah", "yep", "right", "I mean",
    "so", "basically", "actually", "literally", "sort of", "kind of",
    # ── 日文语气词 ──
    "ええと", "あの", "まぁ", "なんか", "うーん", "えっと",
    "そうですね", "あー", "えー", "いや", "うん", "その",
    # ── 韩语语气词（韩文 + 罗马音）──
    "음", "어", "그러니까", "저기요", "아", "막상막하",
})

# 指令性关键词（用于 is_decoupled 解耦盲说判定）
_DIRECTIVE_KEYWORDS: frozenset[str] = frozenset([
    "你", "您", "请", "马上", "立刻", "必须", "需要", "应该",
    "给我", "帮", "提供", "输入", "发送", "点击", "下载",
    "转账", "汇款", "验证码", "密码", "账户", "银行卡",
])

# 中文字符计数正则（Unicode CJK 统一表意文字范围）
_RE_CJK: re.Pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# 英文词计数正则
_RE_EN_WORD: re.Pattern = re.compile(r"\b[a-zA-Z']+\b")

# 拓扑判定阈值
_ASYMMETRIC_DOMINANT_RATIO: float = 0.80  # 主导方字数占比阈值
_SYMMETRIC_MAX_RATIO:       float = 0.60  # 对称轨道最大一方字数占比阈值（补集 ≥ 0.40）
_MIN_SPEAKERS_FOR_ANALYSIS: int   = 2     # 分析需要的最少说话人数


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _count_words(text: str) -> int:
    """
    多语言字数统计：
    - 中文：按字符计（每个 CJK 字符 = 1 词）
    - 英文：按空白分隔词计
    两者之和作为最终字数，适配中英混杂的 ASR 输出。
    """
    cjk_count:     int = len(_RE_CJK.findall(text))
    en_word_count: int = len(_RE_EN_WORD.findall(text))
    return cjk_count + en_word_count


def _is_backchannel(text: str) -> bool:
    """
    判断一段文本是否为纯附和/静音轮次。

    判定逻辑（优先级由高到低）：
    1. 空文本或纯空白 → True
    2. 词级精确匹配：整段文本在 _BACKCHANNEL_WORDS 中 → True
    3. 字符级：去除空白后，所有字符均在 _BACKCHANNEL_CHARS 中 → True
    4. 字数 ≤ 3 且词级无实质内容 → True（宽松兜底）
    """
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in _BACKCHANNEL_WORDS:
        return True
    if all(ch in _BACKCHANNEL_CHARS for ch in stripped if not ch.isspace()):
        return True
    if _count_words(stripped) <= 2 and lower in _BACKCHANNEL_WORDS:
        return True
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TopologyAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TopologyAnalyzer:
    """
    对话拓扑分析器。

    设计约定
    ─────────────────────────────────────────────────────────
    - 输入 records 必须已按时间顺序排列（由上游保证）。
    - 该类本身无状态，所有方法可被并发安全调用。
    - 不依赖任何向量模型或外部服务，纯本地规则运算。
    """

    # ── 公开接口：同源轮次合并 ────────────────────────────────

    def merge_turns(self, records: list[ASRRecord]) -> list[DialogueTurn]:
        """
        将 ASR 碎片流合并为按说话人分组的连续轮次。

        合并规则
        ──────────────────────────────────────────────────────
        1. 遍历 records，相邻同 speaker_id 的碎片直接拼接（空格分隔）。
        2. 合并完成后，对每个合并轮次：
           a. 若整轮仅含纯语气词，标记 is_backchannel=True，
              并将文本收敛为单个代表性词（如"嗯"），避免重复噪声。
           b. 否则 is_backchannel=False，保留完整合并文本。
        3. 对于连续出现的 backchannel 轮次（同 speaker），
           进一步合并为 1 条，raw_record_count 累加。

        Parameters
        ----------
        records : list[ASRRecord]
            已排序的 ASR 碎片列表，每条含 speaker_id + effective_text。

        Returns
        -------
        list[DialogueTurn]
            合并后的轮次序列，保留轮次顺序，turn_index 从 0 开始。
        """
        if not records:
            return []

        # ── Step 1：相邻同 speaker 合并 ───────────────────────
        raw_merged: list[tuple[str, list[str]]] = []   # (speaker_id, [texts])
        for rec in records:
            text = rec.effective_text
            if raw_merged and raw_merged[-1][0] == rec.speaker_id:
                raw_merged[-1][1].append(text)
            else:
                raw_merged.append((rec.speaker_id, [text]))

# ── Step 2：逐组构建 DialogueTurn + backchannel 判定 ──
        pre_turns: list[DialogueTurn] = []
        for speaker_id, texts in raw_merged:
            joined = " ".join(t for t in texts if t)
            is_bc  = _is_backchannel(joined)
            display_text = "嗯" if is_bc and any(
                ch in "嗯啊哦" for ch in joined
            ) else (joined if is_bc else joined)

            pre_turns.append(
                DialogueTurn(
                    speaker_id       = speaker_id,
                    merged_text      = display_text,
                    word_count       = _count_words(joined) if not is_bc else 0,
                    raw_record_count = len(texts),
                    is_backchannel   = is_bc,
                    turn_index       = 0,
                )
            )

        # ── Step 3：重新编号 turn_index ───────────────────────
        final: list[DialogueTurn] = []
        for idx, turn in enumerate(pre_turns): # 👈 修复：把 folded 改成 pre_turns
            final.append(
                DialogueTurn(
                    speaker_id       = turn.speaker_id,
                    merged_text      = turn.merged_text,
                    word_count       = turn.word_count,
                    raw_record_count = turn.raw_record_count,
                    is_backchannel   = turn.is_backchannel,
                    turn_index       = idx,
                )
            )

        return final

    # ── 公开接口：拓扑轨道分类 ────────────────────────────────

    def classify_track(self, turns: list[DialogueTurn]) -> TrackType:
        """
        根据字数分布和轮次节奏，将对话分流到拓扑轨道。

        判定规则（按优先级）
        ──────────────────────────────────────────────────────
        1. 只有一个 speaker → ASYMMETRIC（独白）
        2. 主导方字数占比 > _ASYMMETRIC_DOMINANT_RATIO → ASYMMETRIC
        3. 最大方字数占比 ≤ _SYMMETRIC_MAX_RATIO → SYMMETRIC
        4. 兜底：ASYMMETRIC（保守策略，避免漏判高危对话）

        Parameters
        ----------
        turns : list[DialogueTurn]
            merge_turns() 的输出。

        Returns
        -------
        TrackType
        """
        if not turns:
            return TrackType.ASYMMETRIC  # 空对话保守处理

        # 计算各 speaker 的有效字数（不含 backchannel）
        word_totals: dict[str, int] = {}
        for turn in turns:
            if not turn.is_backchannel:
                word_totals[turn.speaker_id] = (
                    word_totals.get(turn.speaker_id, 0) + turn.word_count
                )

        # 只有一个实质说话人
        if len(word_totals) < _MIN_SPEAKERS_FOR_ANALYSIS:
            return TrackType.ASYMMETRIC

        total_words: int = sum(word_totals.values())
        if total_words == 0:
            return TrackType.ASYMMETRIC

        # 最大占比
        max_ratio: float = max(word_totals.values()) / total_words

        if max_ratio > _ASYMMETRIC_DOMINANT_RATIO:
            return TrackType.ASYMMETRIC
        elif max_ratio <= _SYMMETRIC_MAX_RATIO:
            return TrackType.SYMMETRIC
        else:
            # 60% ~ 80% 的灰色地带：进一步用轮次节奏辅助判定
            return self._break_tie_by_rhythm(turns, word_totals, total_words)

    # ── 私有方法：轮次节奏辅助判定 ───────────────────────────

    @staticmethod
    def _break_tie_by_rhythm(
        turns:       list[DialogueTurn],
        word_totals: dict[str, int],
        total_words: int,
    ) -> TrackType:
        """
        在字数占比处于灰色地带（60%~80%）时，
        通过轮次切换速度辅助区分轨道。

        规则：
        - 计算「轮次交替密度」= 相邻轮次 speaker 发生切换的次数 / (总轮次 - 1)
        - 交替密度 > 0.6（快速来回）→ SYMMETRIC（协商节奏）
        - 交替密度 ≤ 0.6 → ASYMMETRIC（主导发言模式）
        """
        non_bc_turns = [t for t in turns if not t.is_backchannel]
        if len(non_bc_turns) < 2:
            return TrackType.ASYMMETRIC

        switch_count: int = sum(
            1 for i in range(1, len(non_bc_turns))
            if non_bc_turns[i].speaker_id != non_bc_turns[i - 1].speaker_id
        )
        alternation_density: float = switch_count / (len(non_bc_turns) - 1)

        return (
            TrackType.SYMMETRIC
            if alternation_density > 0.6
            else TrackType.ASYMMETRIC
        )

    # ── 工具方法：供外部调用的字数分布统计 ──────────────────

    @staticmethod
    def compute_word_distribution(
        turns: list[DialogueTurn],
    ) -> dict[str, float]:
        """
        计算各 speaker 的字数占比，排除 backchannel 轮次。
        返回 {speaker_id: ratio}，所有 ratio 之和 = 1.0。
        """
        word_totals: dict[str, int] = {}
        
        # 🚨 修复点：先将所有出现过的 speaker 初始化为 0（防止纯嗯啊方消失）
        for turn in turns:
            if turn.speaker_id not in word_totals:
                word_totals[turn.speaker_id] = 0

        # 再累加非 backchannel 的字数
        for turn in turns:
            if not turn.is_backchannel:
                word_totals[turn.speaker_id] += turn.word_count

        total: int = sum(word_totals.values())
        if total == 0:
            return {sid: 0.0 for sid in word_totals}
        return {sid: round(cnt / total, 4) for sid, cnt in word_totals.items()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TopologyMetrics —— 拓扑特征度量快照
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TopologyMetrics:
    """
    对话拓扑多维度量快照，由 TopologyEngine 计算产出。

    Attributes
    ----------
    filler_word_rate   : 语气词（嗯/啊/那个/呢）占总字数的比例
    max_sentence_length: 单句最大字数（基于标点切分）
    avg_sentence_length: 单句平均字数
    is_decoupled       : 双方是否处于「解耦盲说」状态
                         （一方命中指令性关键词，另一方无相关语义回应且持续高密度输出）
    """
    filler_word_rate:    float
    max_sentence_length: int
    avg_sentence_length: float
    is_decoupled:        bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TopologyEngine —— 拓扑特征计算引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 标点切分正则（中文句号/问号/叹号/分号 + 英文 . ! ? 及换行）
_RE_SENTENCE_SPLIT: re.Pattern = re.compile(
    r"[。！？；\n.!?]+"
)


class TopologyEngine:
    """
    对话拓扑特征计算引擎。

    在 TopologyAnalyzer 的基础之上，提供更细粒度的统计度量，
    供 BotConfidenceEngine 和 AdvancedVoicemailDetector 使用。

    无状态设计，所有方法可并发安全调用。
    """

    def compute_metrics(self, turns: list[DialogueTurn]) -> TopologyMetrics:
        """
        对一组合并后的轮次计算全部拓扑度量。

        Parameters
        ----------
        turns : list[DialogueTurn]
            TopologyAnalyzer.merge_turns() 的输出。

        Returns
        -------
        TopologyMetrics
        """
        return TopologyMetrics(
            filler_word_rate    = self._compute_filler_word_rate(turns),
            max_sentence_length = self._compute_max_sentence_length(turns),
            avg_sentence_length = self._compute_avg_sentence_length(turns),
            is_decoupled       = self._compute_is_decoupled(turns),
        )

    # ── 语气词占比 ──────────────────────────────────────────

    @staticmethod
    def _compute_filler_word_rate(turns: list[DialogueTurn]) -> float:
        """
        计算语气词占总词数的比例（多语言：中/英/日/粤）。

        算法：将轮次文本转为小写后，使用子串匹配检测 FILLER_WORDS 中的
        多字节语气词（如"ええと"、"you know"），同时保留单字符级别匹配。

        机器人外呼通常极度流畅（filler_word_rate < 0.005），
        真人对话通常 > 0.02。
        """
        total_words: int = 0
        filler_hits: int = 0

        for turn in turns:
            if turn.is_backchannel:
                continue
            text = turn.merged_text
            total_words += _count_words(text)

            # 先检测多字节语气词（长词优先匹配，避免子串误命中）
            text_lower = text.lower()
            matched_positions: set[int] = set()  # 已匹配的字符位置
            # 按长度降序排列，确保长词优先
            sorted_fillers = sorted(_FILLER_WORDS, key=len, reverse=True)
            for filler in sorted_fillers:
                filler_lower = filler.lower()
                start = 0
                while True:
                    idx = text_lower.find(filler_lower, start)
                    if idx == -1:
                        break
                    # 标记已匹配的位置
                    for pos in range(idx, idx + len(filler)):
                        matched_positions.add(pos)
                    filler_hits += 1
                    start = idx + len(filler)

            # 单字符级别语气词检测（仅对未被多字节匹配的字符）
            for i, ch in enumerate(text):
                if not ch.isspace() and i not in matched_positions and ch.lower() in "嗯啊哦呢呀哈哟喔唔嘿吧诶":
                    # 单字符语气词贡献 0.5（因为一个中文字通常 = 1 词但语气价值较低）
                    filler_hits += 0.5

        if total_words == 0:
            return 0.0
        return round(filler_hits / total_words, 6)

    # ── 单句最大字数 ────────────────────────────────────────

    @staticmethod
    def _compute_max_sentence_length(turns: list[DialogueTurn]) -> int:
        """基于标点切分，计算单句最大字数。"""
        max_len: int = 0
        for turn in turns:
            if turn.is_backchannel:
                continue
            sentences = _RE_SENTENCE_SPLIT.split(turn.merged_text)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    wc = _count_words(sent)
                    if wc > max_len:
                        max_len = wc
        return max_len

    # ── 单句平均字数 ────────────────────────────────────────

    @staticmethod
    def _compute_avg_sentence_length(turns: list[DialogueTurn]) -> float:
        """基于标点切分，计算单句平均字数。"""
        sentence_lengths: list[int] = []
        for turn in turns:
            if turn.is_backchannel:
                continue
            sentences = _RE_SENTENCE_SPLIT.split(turn.merged_text)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    sentence_lengths.append(_count_words(sent))

        if not sentence_lengths:
            return 0.0
        return round(sum(sentence_lengths) / len(sentence_lengths), 2)

    # ── 解耦盲说判定 ────────────────────────────────────────

    @staticmethod
    def _compute_is_decoupled(turns: list[DialogueTurn]) -> bool:
        """
        判定双方是否处于「解耦盲说」状态。

        判定条件（全部满足）：
        1. 至少存在两个 speaker
        2. 一方（dominant）的文本命中指令性关键词
        3. 另一方（passive）的高密度输出（> 30% 字数占比）
        4. dominant 的 ping_pong 节奏极低（< 0.1），即几乎无真正互动

        典型场景：AI 客服盲播 → 受害者没在听 → 客服自顾自读话术
        """
        non_bc_turns = [t for t in turns if not t.is_backchannel]
        if len(non_bc_turns) < 3:
            return False

        # 收集 speakers
        speakers: list[str] = list(dict.fromkeys(t.speaker_id for t in non_bc_turns))
        if len(speakers) < 2:
            return False

        # 计算各 speaker 字数
        word_totals: dict[str, int] = {}
        for turn in non_bc_turns:
            word_totals[turn.speaker_id] = word_totals.get(turn.speaker_id, 0) + turn.word_count

        total_words: int = sum(word_totals.values())
        if total_words == 0:
            return False

        # 找出字数占比 > 30% 的双方
        candidates = [
            sid for sid, wc in word_totals.items()
            if wc / total_words > 0.30
        ]
        if len(candidates) < 2:
            return False

        # 判定：是否存在一方命中指令性关键词而另一方没有语义回应
        # 分别检查两方
        directive_speaker: str | None = None
        passive_speaker: str | None = None

        for sid in candidates:
            sid_text = " ".join(t.merged_text for t in non_bc_turns if t.speaker_id == sid)
            has_directive = any(kw in sid_text for kw in _DIRECTIVE_KEYWORDS)
            if has_directive and directive_speaker is None:
                directive_speaker = sid
            elif not has_directive and passive_speaker is None:
                passive_speaker = sid

        if directive_speaker is None or passive_speaker is None:
            return False

        # 验证 ping_pong 率极低（一方在持续高密度输出而不关心对方反应）
        ping_pong_count: int = 0
        for i in range(1, len(non_bc_turns) - 1):
            if (
                non_bc_turns[i - 1].speaker_id != non_bc_turns[i].speaker_id
                and non_bc_turns[i].speaker_id != non_bc_turns[i + 1].speaker_id
                and non_bc_turns[i - 1].speaker_id == non_bc_turns[i + 1].speaker_id
            ):
                ping_pong_count += 1

        ping_pong_rate = ping_pong_count / max(len(non_bc_turns) - 2, 1)
        return ping_pong_rate < 0.1
