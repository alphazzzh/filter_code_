"""
scoring_strategies/base.py  ── 策略基类 + 评分工作台
============================================================

_ScoringContext  : 贯穿整个打分流程的可变工作台
ScoringRuleStrategy : 抽象策略基类，所有策略必须实现 apply()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from models_stage2 import StageTwoResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局打分边界常量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BASE_SCORE: int = 50
_SCORE_MIN:  int = 0
_SCORE_MAX:  int = 100


@dataclass
class _ScoringContext:
    """
    贯穿整个打分流程的可变工作台。每条对话独占一个实例。

    策略间通过 ctx 共享状态：
      - score  : 当前累计分数
      - tags   : 已命中的标签集合
      - events : 加分/扣分明细（审计用）
      - extra  : 策略间通信字典（如 hit_families 传递给下游策略）
    """
    score:  int                  = _BASE_SCORE
    tags:   set[str]             = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)
    extra:  dict[str, Any]       = field(default_factory=dict)

    def apply(self, delta: int, tag: str | None, reason: str) -> None:
        self.score += delta
        if tag:
            self.tags.add(tag)
        self.events.append({"delta": delta, "tag": tag, "reason": reason})


class ScoringRuleStrategy(ABC):
    """
    打分策略抽象基类。

    每个策略封装一个独立的打分逻辑块，接收统一的 _ScoringContext，
    通过 ctx.apply() 修改分数/标签，通过 ctx.extra 与下游策略通信。

    约定
    ─────────────────────────────────────────────────────────
    - apply() 不返回值，所有状态变更通过 ctx 传递
    - 策略间通信使用 ctx.extra 字典（避免子类返回值类型膨胀）
    - 红线熔断策略可设置 ctx.extra["redline_triggered"] = True，
      IntelligenceScorer 据此短路后续策略
    """

    @abstractmethod
    def apply(
        self,
        ctx:    _ScoringContext,
        result: StageTwoResult,
    ) -> None:
        """
        对当前对话执行打分逻辑。

        Parameters
        ----------
        ctx    : 评分工作台（可变状态）
        result : 阶段二输出（只读输入）
        """
        ...
