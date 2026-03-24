# models_stage2.py
# ============================================================
# ASR 流水线阶段二 —— 核心数据模型定义
#
# 依赖阶段一的 ASRRecord（通过 Dummy 占位以保持解耦）。
# 所有阶段二产出均为 frozen=True，计算后不可变，保证可审计性。
# ============================================================

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 占位：最小化 ASRRecord Stub（解耦阶段一依赖）
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# class _ASRRecordStub(BaseModel):
#     """
#     阶段一 ASRRecord 的最小化 Stub。
#     生产环境中替换为 `from models import ASRRecord`。
#     仅需暴露阶段二所需的三个字段。
#     """
#     record_id:    str
#     speaker_id:   str              # 发言方 ID（如 "A" / "B" / "wxid_xxx"）
#     cleaned_text: Optional[str] = None  # 阶段一容错后的文本；None 时退回 raw_text
#     raw_text:     str = ""

#     @property
#     def effective_text(self) -> str:
#         """返回最优可用文本：优先 cleaned_text，回退 raw_text。"""
#         return (self.cleaned_text or self.raw_text).strip()

#     model_config = {"extra": "allow"}  # Stub 宽容，允许传入阶段一完整字段


# 生产切换：取消注释下行并注释上方 Stub
from models import ASRRecord as _ASRRecordStub  # noqa: F401

ASRRecord = _ASRRecordStub  # 全局别名，阶段二代码统一使用 ASRRecord


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 任务一：核心数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 枚举：对话拓扑轨道 ────────────────────────────────────────

class TrackType(str, Enum):
    """
    对话拓扑分流结果。
    SYMMETRIC  : 双方字数接近（40/60 以内），轮次切换快 → 平等协商
    ASYMMETRIC : 一方字数占比 > 80%，轮次少且单轮极长 → 强输出/灌输
    """
    SYMMETRIC  = "symmetric"
    ASYMMETRIC = "asymmetric"


# ── 枚举：角色标签 ────────────────────────────────────────────

class RoleLabel(str, Enum):
    """
    六种业务身份标签。

    非对称轨道
    ──────────
    AGENT   : 主动输出方（推销/引导/诈骗方）
    TARGET  : 被动接收方（客户/受害者）

    对称轨道——有业务驱动
    ──────────────────────
    DRIVER  : 业务驱动方（提案 + 总结 发起者）
    FOLLOWER: 跟随配合方（高顺从度接受方）

    对称轨道——纯聊天
    ──────────────────
    PEER_A  : 平权一方
    PEER_B  : 平权另一方
    """
    AGENT    = "agent"
    TARGET   = "target"
    DRIVER   = "driver"
    FOLLOWER = "follower"
    PEER_A   = "peer_a"
    PEER_B   = "peer_b"


# ── 模型：单轮发言（同源合并后） ──────────────────────────────

class DialogueTurn(BaseModel):
    """
    同源合并后的单轮发言单元。

    一个 DialogueTurn 可能由多条 ASR 碎片合并而来，
    也可能是单条完整发言。
    """
    speaker_id:       str   = Field(..., description="发言方唯一 ID")
    merged_text:      str   = Field(..., description="合并后的完整发言文本")
    word_count:       int   = Field(..., ge=0, description="字符数（中文按字，英文按词）")
    raw_record_count: int   = Field(..., ge=1, description="合并前的 ASR 碎片数量")
    is_backchannel:   bool  = Field(
        default=False,
        description="是否为纯倾听附和轮次（嗯/啊/对/好 等），不含实质信息",
    )
    # 阶段二推理阶段填入的意图标签（IntentRadar 输出）
    intent_labels: list[str] = Field(
        default_factory=list,
        description="该轮次触发的意图标签列表，如 ['proposal', 'emotion']",
    )
    # 供 RoleBinder 使用的细粒度发言位置（对话总轮数中的序号，0-based）
    turn_index: int = Field(default=0, ge=0, description="在完整对话中的轮次序号")

    model_config = {"frozen": True}


# ── 模型：深层互动特征 ────────────────────────────────────────

class InteractionFeatures(BaseModel):
    """
    深层统计互动特征，作为 RoleBinder 的中间产物存档，
    同时作为下游风险评分引擎（阶段三）的原始特征向量。

    所有字段均为 float，量纲归一到 [0, 1]，除特别注明。
    """

    # ── 通用特征（对称 / 非对称轨道均计算）────────────────────

    speaker_word_ratio: dict[str, float] = Field(
        default_factory=dict,
        description="各 speaker_id 的字数占比，key=speaker_id，val ∈ [0,1]，总和=1",
    )
    turn_count_per_speaker: dict[str, int] = Field(
        default_factory=dict,
        description="各 speaker_id 的发言轮次数（含 backchannel）",
    )
    backchannel_rate_per_speaker: dict[str, float] = Field(
        default_factory=dict,
        description="各 speaker_id 的 backchannel 轮次占自身总轮次的比例",
    )

    # ── 对称轨道专属 ────────────────────────────────────────────

    negotiation_ping_pong_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "协商往返率：连续出现「提问-回答-反问」三元组的轮次比例，"
            "高值（>0.4）提示真实业务协商，排除群发广告和机器人。"
        ),
    )
    emotional_grooming_index: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "情绪价值提供指数（per speaker）：情绪锚点命中轮次 / 总轮次。"
            "高值（>0.3）且主要由单一 speaker 贡献，为「杀猪盘」强信号。"
        ),
    )
    compliance_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "顺从度：Follower 对 Driver 提案发出顺从信号的比例。"
            "顺从信号 = compliance 锚点命中 + is_backchannel。"
        ),
    )

    # ── 非对称轨道专属 ──────────────────────────────────────────

    interrogation_rate: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "提问压制率（per speaker）：发言中含疑问语气词的轮次比例。"
            "Agent 通常 > 0.3，用于确认控制节奏方。"
        ),
    )
    resistance_decay: float = Field(
        default=0.0, ge=0.0,
        description=(
            "反抗衰减度：Target 前20%轮次平均字数 / 后20%轮次平均字数的比值。"
            "比值 > 1.5 表示防线显著收缩，为深度套牢信号。"
            "无量纲，不限于1，越大越危险。"
        ),
    )

    model_config = {"frozen": True}


# ── 模型：单说话人角色绑定结果 ───────────────────────────────

class SpeakerRoleResult(BaseModel):
    """单个 speaker 的角色绑定结果。"""
    speaker_id:  str       = Field(..., description="说话人 ID")
    role:        RoleLabel = Field(..., description="绑定的角色标签")
    confidence:  float     = Field(..., ge=0.0, le=1.0, description="判定置信度")
    evidence:    list[str] = Field(
        default_factory=list,
        description="支撑该角色判定的证据摘要（可读文字，用于审计）",
    )

    model_config = {"frozen": True}


# ── 模型：阶段二完整输出 ──────────────────────────────────────

class StageTwoResult(BaseModel):
    """
    阶段二完整输出，聚合拓扑、意图、角色、互动特征。
    作为阶段三（风险评分）的标准化输入。
    """
    conversation_id: str = Field(..., description="会话唯一 ID（由上游注入）")
    track_type:      TrackType = Field(..., description="对话拓扑轨道")

    dialogue_turns: list[DialogueTurn] = Field(
        ..., description="同源合并后的对话轮次序列（含意图标签）"
    )
    speaker_roles: list[SpeakerRoleResult] = Field(
        ..., description="每个 speaker 的角色绑定结果"
    )
    interaction_features: InteractionFeatures = Field(
        ..., description="深层互动特征快照"
    )

    # 流水线元信息
    stage_two_done: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def roles_cover_all_speakers(self) -> "StageTwoResult":
        """每个出现过的 speaker 必须有对应的角色绑定结果。"""
        speakers_in_turns: set[str] = {t.speaker_id for t in self.dialogue_turns}
        speakers_in_roles: set[str] = {r.speaker_id for r in self.speaker_roles}
        missing = speakers_in_turns - speakers_in_roles
        if missing:
            raise ValueError(f"以下 speaker 缺少角色绑定结果：{missing}")
        return self

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "extra": "forbid",
    }
