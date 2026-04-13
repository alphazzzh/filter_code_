"""
engines/voicemail_detector.py  ── 高级无效通话检测引擎
============================================================

从 stage_three_scorer.py 中剥离的独立引擎。
矩阵评分 + 语义防线逻辑均保持不变。

设计原则：
  - 零内部状态，线程安全
  - 输入 StageTwoResult + is_decoupled，输出结构化判定字典
  - 可被 filter_node / LangGraph 节点等方复用
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from models_stage2 import StageTwoResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Voicemail 正则词库 —— 多语言：中/英/日/粤/韩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VOICEMAIL_REGEX_WORDS: list[str] = [
    # ── 中文语音信箱/未接通提示 ──
    "无法接通", "暂时无法接通", "正在通话中", "电话正忙",
    "不在服务区", "已启用来电提醒", "无人接听", "呼叫超时",
    "已挂断", "请在提示音后留言", "录音完成后挂断",
    "留言最长", "按井号键", "语音信箱已满", "呼叫转移",
    "转接语音信箱", "正在为您转接", "号码是空号",
    "已停机", "已过期", "请勿挂机", "余额不足",
    "通话已被录音", "会议通话中", "等待音乐",
    # ── 英文 Voicemail/IVR ──
    "The number you dialed", "leave a message", "after the beep",
    "mailbox is full", "call is being transferred", "line is busy",
    "not available", "voicemail", "press 1",
    "please hold", "all agents are busy", "your call is important",
    "no one is available", "record your message", "at the tone",
    # ── 日文留守番電話/転送 ──
    "留守番電話", "発信音の後に", "ピーという音が鳴りましたら",
    "メッセージを残して", "おかけになった電話番号",
    "電源が切れています", "通話中です", "応答なし",
    # ── 粤語未接通 ──
    "嗶一聲之後", "唔得閒", "留低訊息", "話機冇訊號",
    # ── 韓語 음성사서함/전화 ──
    "음성 사서함", "전화를 받지", "연결이 되지", "통화 중입니다",
    "메시지를 남겨", "전화번호는",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 轻量级 TF-IDF 余弦相似度（无需 GPU / BGE 模型）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 预编译中文分词正则（字符级 bigram，适用于中文/日文等多语言场景）
_RE_TOKENIZER: re.Pattern = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[a-zA-Z]+",
    re.UNICODE,
)


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    轻量级 TF-IDF 余弦相似度计算。

    使用字符级 bigram + IDF 近似（基于文档频率的简单折扣），
    无需外部依赖，O(n) 时间复杂度。

    适用场景：跨轮次连贯度判定（两句话是否在语义上相关）
    不适用：需要深度语义理解的场景（请使用 BGE-M3）
    """
    def _tokenize(text: str) -> list[str]:
        return _RE_TOKENIZER.findall(text.lower())

    def _ngrams(tokens: list[str]) -> Counter[str]:
        grams: list[str] = []
        for i in range(len(tokens) - 1):
            grams.append(f"{tokens[i]}_{tokens[i + 1]}")
        return Counter(grams)

    ngrams_a = _ngrams(_tokenize(text_a))
    ngrams_b = _ngrams(_tokenize(text_b))

    if not ngrams_a or not ngrams_b:
        return 0.0

    # 简单 IDF 折扣：仅出现在一个文档中的 bigram 权重更高
    all_keys = set(ngrams_a.keys()) | set(ngrams_b.keys())
    idf: dict[str, float] = {}
    for k in all_keys:
        df = (1 if k in ngrams_a else 0) + (1 if k in ngrams_b else 0)
        idf[k] = math.log(2.0 / df) + 1.0  # 平滑 IDF

    # TF-IDF 加权向量
    def _to_tfidf_vec(ngrams: Counter[str]) -> dict[str, float]:
        total = sum(ngrams.values()) or 1.0
        return {k: (v / total) * idf[k] for k, v in ngrams.items()}

    vec_a = _to_tfidf_vec(ngrams_a)
    vec_b = _to_tfidf_vec(ngrams_b)

    # 余弦相似度
    dot_product = sum(vec_a[k] * vec_b.get(k, 0.0) for k in vec_a)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return dot_product / (norm_a * norm_b)


class AdvancedVoicemailDetector:
    """
    高级无效通话（Voicemail/语音信箱/未接通）检测引擎。

    V5.1 重构：废除字数统计防线，引入跨轮次语义连贯度测算。

    评分规则（基础分 0，叠加计分）：
      - 命中 Voicemail 正则词                → +60
      - 角色 B 字数极短且符合模板             → +30
      - 交互判定为 is_decoupled（解耦盲说）   → +20

    一票否决（得分归零）：
      - ping_pong_rate > 0.1 且不是解耦状态 → 判定为有效通话
      - 跨轮次语义连贯度 > 0.35 → 检测到真交互，推翻无效通话判定

    最终得分 > 80 判定为无效通话。
    """

    # ── 评分阈值 ──────────────────────────────────────────
    _VOICEMAIL_THRESHOLD: int = 80

    # ── 加分权重 ──────────────────────────────────────────
    _SCORE_VOICEMAIL_WORD:   int = 60
    _SCORE_SHORT_TEMPLATE_B: int = 30
    _SCORE_DECOUPLED:        int = 20

    # ── 语义连贯度阈值 ──────────────────────────────────
    _CROSS_TURN_SIM_THRESHOLD: float = 0.35  # 跨轮次语义耦合阈值

    # ── 角色 B 短模板阈值 ─────────────────────────────────
    _SHORT_B_WORD_THRESHOLD: int = 15  # 角色 B 总字数 < 15 视为极短

    def evaluate(
        self,
        stage2_result: StageTwoResult,
        is_decoupled: bool = False,
    ) -> dict[str, Any]:
        """
        执行高级无效通话判定。

        Parameters
        ----------
        stage2_result : 阶段二输出
        is_decoupled  : 解耦盲说判定（由 TopologyEngine 提供）

        Returns
        -------
        dict[str, Any]
            voicemail_score : int           置信度得分 [0, 100]
            is_voicemail    : bool          是否判定为无效通话
            veto_reason     : str | None    一票否决原因
            details         : list[str]     评分细节
        """
        score: int   = 0
        details: list[str] = []
        full_text: str = " ".join(
            t.merged_text for t in stage2_result.dialogue_turns
            if not t.is_backchannel
        )

        # ── 一票否决检查 1：ping_pong_rate ──────────────────
        ppr = stage2_result.interaction_features.negotiation_ping_pong_rate
        if ppr > 0.1 and not is_decoupled:
            return {
                "voicemail_score": 0,
                "is_voicemail": False,
                "veto_reason": (
                    f"ping_pong_rate={ppr:.4f} > 0.1 且非解耦状态，"
                    "判定为有效双向通话"
                ),
                "details": ["PING_PONG_VETO"],
            }

        # ── 加分项 ────────────────────────────────────────

        # 加分 1：命中 Voicemail 正则词
        voicemail_word = self._check_voicemail_words(full_text)
        if voicemail_word:
            score += self._SCORE_VOICEMAIL_WORD
            details.append(f"命中 Voicemail 词「{voicemail_word}」→ +{self._SCORE_VOICEMAIL_WORD}")

        # 加分 2：角色 B 字数极短且符合模板
        if self._check_short_template_b(stage2_result):
            score += self._SCORE_SHORT_TEMPLATE_B
            details.append(
                f"角色 B 字数极短（< {self._SHORT_B_WORD_THRESHOLD} 字）→ +{self._SCORE_SHORT_TEMPLATE_B}"
            )

        # 加分 3：解耦盲说状态
        if is_decoupled:
            score += self._SCORE_DECOUPLED
            details.append(f"解耦盲说状态 (is_decoupled=True) → +{self._SCORE_DECOUPLED}")

        # ── 一票否决检查 2：跨轮次语义连贯度防线 ──────────
        max_cross_turn_sim: float = 0.0
        valid_turns = [
            t for t in stage2_result.dialogue_turns
            if not t.is_backchannel and t.merged_text.strip()
        ]

        for i in range(len(valid_turns) - 1):
            turn_a = valid_turns[i]
            turn_b = valid_turns[i + 1]
            # 只计算相邻且角色不同的轮次
            if turn_a.speaker_id == turn_b.speaker_id:
                continue
            sim = _tfidf_cosine_similarity(
                turn_a.merged_text, turn_b.merged_text
            )
            if sim > max_cross_turn_sim:
                max_cross_turn_sim = sim

        is_voicemail = score > self._VOICEMAIL_THRESHOLD

        if is_voicemail and max_cross_turn_sim > self._CROSS_TURN_SIM_THRESHOLD:
            # 语义防线一票否决：检测到跨角色语义高度耦合，推翻无效通话判定
            veto_reason = (
                f"触发连贯度防线：检测到跨角色语义高度耦合 "
                f"(sim={max_cross_turn_sim:.4f} > {self._CROSS_TURN_SIM_THRESHOLD})，"
                f"推翻单向盲播判定"
            )
            details.append(veto_reason)
            return {
                "voicemail_score": score,
                "is_voicemail":    False,
                "veto_reason":     veto_reason,
                "details":         details,
            }

        details.append(
            f"跨轮次语义连贯度={max_cross_turn_sim:.4f}，"
            f"阈值={self._CROSS_TURN_SIM_THRESHOLD}，"
            f"最终得分={score}，判定={'无效通话' if is_voicemail else '有效通话'}"
        )

        return {
            "voicemail_score": score,
            "is_voicemail":  is_voicemail,
            "veto_reason":    None,
            "details":        details,
            "cross_turn_sim": round(max_cross_turn_sim, 4),
        }

    @staticmethod
    def _check_voicemail_words(text: str) -> str | None:
        """检查文本是否命中 Voicemail 正则词库，返回第一个命中的词或 None。"""
        for word in _VOICEMAIL_REGEX_WORDS:
            if word in text:
                return word
        return None

    @staticmethod
    def _check_short_template_b(stage2_result: StageTwoResult) -> bool:
        """
        检查是否存在「字数极短的角色 B」。

        判定逻辑：
        1. 找出字数占比最小的 speaker（角色 B）
        2. 其总有效字数 < _SHORT_B_WORD_THRESHOLD
        """
        ifeats = stage2_result.interaction_features
        word_dist = ifeats.speaker_word_ratio
        if not word_dist:
            return False

        # 找字数占比最小的 speaker
        min_sid = min(word_dist, key=lambda s: word_dist.get(s, 1.0))

        # 计算该 speaker 的有效字数
        total_words = sum(
            t.word_count
            for t in stage2_result.dialogue_turns
            if t.speaker_id == min_sid and not t.is_backchannel
        )

        return total_words < 15 and total_words > 0
