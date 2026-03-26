# models.py
# ============================================================
# ASR 流水线 —— 核心数据模型定义
# 设计原则：Immutable-first，所有阶段产出字段均为可选，
#           方便流水线按需填充，不强迫一次性计算所有字段。
# ============================================================

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── 枚举：连接状态 ────────────────────────────────────────────

class ConnectionStatus(str, Enum):
    """通话连接状态标签（阶段一输出）。"""
    CONNECTED   = "connected"      # 正常接通
    UNCONNECTED = "unconnected"    # 未接通 / 秒挂
    UNCERTAIN   = "uncertain"      # 置信度不足，保留人工复核


class BotLabel(str, Enum):
    """机器人/真人标签（阶段一输出，阶段二可覆写）。"""
    HUMAN     = "human"
    BOT       = "bot"
    UNCERTAIN = "uncertain"


# ── 子模型：未接通概率分解 ────────────────────────────────────

class UnconnectedFeatures(BaseModel):
    """
    未接通概率的三项原子特征，保留中间计算结果以便调试与审计。
    """
    f_len: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="文本长度特征：max(0, 1 - len/threshold)，越短越接近 1",
    )
    f_ent: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="信息熵特征：基于唯一字符率或 N-gram 交叉熵，越低熵越接近 1",
    )
    f_entity: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="实体否决特征：检测到业务实体词则为 0.0，否则为 1.0",
    )
    p_unconnected: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="融合概率 = f_entity * (0.7*f_len + 0.3*f_ent)",
    )

    model_config = {"frozen": True}


# ── 子模型：机器人启发式特征 ──────────────────────────────────

class BotFeatures(BaseModel):
    """
    阶段一可计算的轻量级机器人特征（无需向量模型）。
    """
    filler_word_ratio: float = Field(
        ...,
        ge=0.0,
        description="语气词占比（嗯/啊/哦/那个/对 等），机器人通常 < 0.02",
    )
    prefix_tokens: str = Field(
        ...,
        max_length=40,
        description="文本前 20 个字符，用于后续批量 GroupBy 去重",
    )
    simhash_value: Optional[int] = Field(
        default=None,
        description="全文 SimHash 值（可选，供批处理引擎碰撞检测使用）",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="文本总 Token 数（用于短文本保护，避免误判 BOT）",
    )

    model_config = {"frozen": True}


# ── 主模型：单条 ASR 记录 ─────────────────────────────────────

class ASRRecord(BaseModel):
    """
    单条 ASR 转写记录的完整数据模型。

    生命周期说明
    ─────────────────────────────────────────────────────────
    raw_text          ← 入库时填充，此后只读，永不修改
    normalized_text   ← 阶段一 动作1 后填充
    lang              ← 阶段一 动作3 LID 后填充
    cleaned_text      ← 阶段一 动作4 ASR 容错后填充
    metadata          ← 各阶段追加写入，最终汇总
    stage_one_done    ← 阶段一全部完成后置 True
    ─────────────────────────────────────────────────────────
    """

    # ── 原始输入 ──────────────────────────────────────────────
    record_id: str = Field(
        ...,
        description="全局唯一记录 ID（由上游系统注入）",
    )
    speaker_id: str = Field(..., description="说话人 ID (如 'A', 'B', '客户')")
    raw_text: str = Field(
        ...,
        min_length=0,
        description="ASR 原始转写文本，绝对不允许被下游逻辑覆写",
    )
    source_lang_hint: Optional[str] = Field(
        default=None,
        description="上游系统提供的语种提示（可为空，LID 会自行判断）",
    )

    # ── 阶段一产出：文本变换 ──────────────────────────────────
    normalized_text: Optional[str] = Field(
        default=None,
        description="NFKC 归一化 + 无意义连续标点处理后的文本",
    )
    lang: Optional[str] = Field(
        default=None,
        description="fastText LID 识别结果，如 'zh', 'en', 'ug'；短文本默认 'zh'",
    )
    lang_confidence: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="LID 置信度（0~1），短文本回退时为 None",
    )
    cleaned_text: Optional[str] = Field(
        default=None,
        description="ASR 容错处理后的最终可用文本（去结巴 + 音似纠错）",
    )

    # ── 阶段一产出：结构化特征 ────────────────────────────────
    unconnected_features: Optional[UnconnectedFeatures] = Field(
        default=None,
        description="未接通三项特征及融合概率",
    )
    connection_status: Optional[ConnectionStatus] = Field(
        default=None,
        description="连接状态判定结果",
    )
    bot_features: Optional[BotFeatures] = Field(
        default=None,
        description="机器人启发式特征（轻量级）",
    )
    bot_label: Optional[BotLabel] = Field(
        default=None,
        description="机器人/真人初步判定（阶段二可覆写）",
    )

    # ── 流水线控制字段 ────────────────────────────────────────
    stage_one_done: bool = Field(
        default=False,
        description="阶段一是否已全部完成",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="各阶段可自由追加的扩展字段（调试信息、来源标注等）",
    )

    # ── 校验规则 ──────────────────────────────────────────────

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_whitespace_only(cls, v: str) -> str:
        """raw_text 允许为空字符串（未接通场景），但不允许仅含不可见字符。"""
        if v != "" and not v.strip():
            raise ValueError("raw_text 不能是纯空白字符串，请传入空字符串代替。")
        return v

    @model_validator(mode="after")
    def normalized_text_must_not_modify_meaning(self) -> "ASRRecord":
        """
        防御性检查：normalized_text 长度不得比 raw_text 短超过 50%，
        避免误删原始内容。
        """
        if (
            self.normalized_text is not None
            and len(self.raw_text) > 10
            and len(self.normalized_text) < len(self.raw_text) * 0.5
        ):
            raise ValueError(
                f"normalized_text 长度 ({len(self.normalized_text)}) 与 "
                f"raw_text ({len(self.raw_text)}) 相差悬殊，疑似意外删除内容。"
            )
        return self
    
    @property
    def effective_text(self) -> str:
        """返回最优可用文本：优先 cleaned_text，回退 raw_text。"""
        return (self.cleaned_text or self.raw_text).strip()

    model_config = {
        "frozen": False,          # 允许流水线各阶段按需填充字段
        "validate_assignment": True,  # 赋值时触发字段级校验
        "extra": "forbid",        # 禁止传入未声明字段，防止上游乱注入
    }
