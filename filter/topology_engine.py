# topology_engine.py
# ============================================================
# 阶段二：同源轮次合并 + 对话拓扑分流
#
# TopologyAnalyzer 职责
# ─────────────────────────────────────────────────────────────
#  merge_turns()      : 合并相邻同角色碎片，收敛纯语气词为 backchannel
#  classify_track()   : 按字数分布 + 轮次节奏判定 TrackType
# ============================================================

from __future__ import annotations

import re
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
    
