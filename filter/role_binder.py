# role_binder.py
# ============================================================
# 阶段二：双轨角色绑定与深层互动特征研判
#
# 模块职责
# ─────────────────────────────────────────────────────────────
#  bind()   : 顶层入口，按 TrackType 分流到对应处理器
#  _bind_symmetric()   : 对称轨道 → 纯聊/Driver/Follower 判定
#  _bind_asymmetric()  : 非对称轨道 → Agent/Target 绑定
#  _compute_interaction_features() : 通用特征计算（两轨复用）
#
# V5.1 修复
# ─────────────────────────────────────────────────────────────
# ① _bind_symmetric() 多人通话覆盖（去掉 [:2] 限制，遍历完整 speakers）
# ② _bind_asymmetric() 多方通话兜底（未分配角色统一标记为 TARGET）
# ============================================================

from __future__ import annotations

import statistics
from typing import Optional

from models_stage2 import (
    DialogueTurn,
    InteractionFeatures,
    RoleLabel,
    SpeakerRoleResult,
    TrackType,
)
from intent_radar import IntentRadar
from topology_engine import TopologyAnalyzer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 判定阈值常量（集中管理，方便后续通过配置文件覆盖）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 纯聊天判定
_PURE_CHAT_EMOTION_THRESHOLD:     float = 0.35  # 情绪轮次占比超过此值认为纯聊
_PURE_CHAT_COMPLIANCE_THRESHOLD:  float = 0.10  # 顺从+提案总密度低于此值
_PURE_CHAT_MIN_SPEAKERS:          int   = 2

# Driver/Follower 判定
_DRIVER_PROPOSAL_WEIGHT:          float = 0.6   # 提案对 driver_score 的权重
_DRIVER_CLOSURE_WEIGHT:           float = 0.4   # 总结确认对 driver_score 的权重

# 非对称轨道 Agent 判定
_AGENT_INTERROGATION_THRESHOLD:   float = 0.25  # 提问压制率阈值

# 反抗衰减计算分位（前后各取 20%）
_RESISTANCE_QUANTILE:             float = 0.20

# 协商往返率计算：连续「A→B→A」三元组
_PING_PONG_MIN_TURNS:             int   = 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RoleBinder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RoleBinder:
    """
    双轨角色绑定器。

    依赖注入
    ─────────────────────────────────────────────────────────
    - IntentRadar  : 注入单例，用于意图标签批量推理。
    - TopologyAnalyzer : 注入分析器，用于字数分布辅助计算。

    设计约定
    ─────────────────────────────────────────────────────────
    - 本类无状态，bind() 可并发安全调用。
    - 所有中间特征均返回，不在内部打日志，由调用方决定如何处置。
    """

    def __init__(
        self,
        radar:    IntentRadar,
        topology: TopologyAnalyzer,
    ) -> None:
        self._radar    = radar
        self._topology = topology

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 顶层入口
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def bind(
        self,
        turns:      list[DialogueTurn],
        track_type: TrackType,
    ) -> tuple[list[DialogueTurn], list[SpeakerRoleResult], InteractionFeatures]:
        """
        执行角色绑定与特征研判。

        Parameters
        ----------
        turns      : merge_turns() 的输出（尚未注入意图标签）
        track_type : classify_track() 的输出

        Returns
        -------
        (labeled_turns, role_results, interaction_features)
            labeled_turns       : 已注入 intent_labels 的轮次序列
            role_results        : 每个 speaker 的角色绑定结果
            interaction_features: 深层统计特征快照
        """
        # ── Step 1：批量意图标注（一次推理覆盖所有轮次）──────
        labeled_turns = self._annotate_intents(turns)

        # ── Step 2：计算通用互动特征 ────────────────────────
        ifeats = self._compute_interaction_features(labeled_turns)

        # ── Step 3：按轨道分流角色绑定 ──────────────────────
        if track_type == TrackType.SYMMETRIC:
            roles = self._bind_symmetric(labeled_turns, ifeats)
        else:
            roles = self._bind_asymmetric(labeled_turns, ifeats)

        return labeled_turns, roles, ifeats

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1：意图批量标注
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _annotate_intents(
        self, turns: list[DialogueTurn]
    ) -> list[DialogueTurn]:
        """
        调用 IntentRadar.detect_batch() 为所有轮次注入意图标签。
        backchannel 轮次跳过推理（文本无语义信息），直接赋空列表。
        """
        # 分离非 backchannel 轮次，收集索引和文本
        non_bc_indices: list[int]  = []
        non_bc_texts:   list[str]  = []
        for i, turn in enumerate(turns):
            if not turn.is_backchannel and turn.merged_text.strip():
                non_bc_indices.append(i)
                non_bc_texts.append(turn.merged_text)

        # 批量推理
        intent_results: list[list[str]] = (
            self._radar.detect_batch(non_bc_texts)
            if non_bc_texts else []
        )

        # 将意图标签回填到对应轮次
        index_to_intents: dict[int, list[str]] = {
            idx: labels
            for idx, labels in zip(non_bc_indices, intent_results)
        }

        labeled: list[DialogueTurn] = []
        for i, turn in enumerate(turns):
            labeled.append(
                DialogueTurn(
                    speaker_id       = turn.speaker_id,
                    merged_text      = turn.merged_text,
                    word_count       = turn.word_count,
                    raw_record_count = turn.raw_record_count,
                    is_backchannel   = turn.is_backchannel,
                    turn_index       = turn.turn_index,
                    intent_labels    = index_to_intents.get(i, []),
                )
            )
        return labeled

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 2：通用互动特征计算（两轨复用）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_interaction_features(
        self, turns: list[DialogueTurn]
    ) -> InteractionFeatures:
        """
        计算所有与轨道无关的通用互动特征。
        轨道专属特征（resistance_decay 等）在此处也计算，
        但对不适用的轨道无副作用（下游忽略即可）。
        """
        speakers: list[str] = list(
            dict.fromkeys(t.speaker_id for t in turns)  # 保序去重
        )

        # ── 字数占比 ──────────────────────────────────────────
        word_dist = self._topology.compute_word_distribution(turns)

        # ── 轮次计数 + backchannel 率 ─────────────────────────
        turn_count:    dict[str, int]   = {s: 0 for s in speakers}
        bc_count:      dict[str, int]   = {s: 0 for s in speakers}
        intent_counts: dict[str, dict[str, int]] = {s: {} for s in speakers}

        for turn in turns:
            sid = turn.speaker_id
            turn_count[sid] = turn_count.get(sid, 0) + 1
            if turn.is_backchannel:
                bc_count[sid] = bc_count.get(sid, 0) + 1
            for label in turn.intent_labels:
                intent_counts.setdefault(sid, {})
                intent_counts[sid][label] = intent_counts[sid].get(label, 0) + 1

        bc_rate: dict[str, float] = {
            sid: round(bc_count.get(sid, 0) / max(turn_count.get(sid, 1), 1), 4)
            for sid in speakers
        }

        # ── 协商往返率（Ping-Pong Rate）────────────────────────
        ping_pong_rate = self._compute_ping_pong_rate(turns)

        # ── 情绪价值提供指数（per speaker）───────────────────
        emo_index: dict[str, float] = {
            sid: round(
                intent_counts.get(sid, {}).get("emotion", 0)
                / max(turn_count.get(sid, 1), 1),
                4,
            )
            for sid in speakers
        }

        # ── 顺从度（Compliance Rate）──────────────────────────
        # 定义：Follower 的 compliance 意图轮次 + backchannel 轮次 之和
        # 在此阶段先算出全局值，bind_symmetric 会进一步拆分
        total_non_bc = sum(
            1 for t in turns if not t.is_backchannel
        )
        total_compliance = sum(
            1 for t in turns if "compliance" in t.intent_labels
        )
        compliance_rate = round(
            total_compliance / max(total_non_bc, 1), 4
        )

        # ── 提问压制率（per speaker）──────────────────────────
        interrogation_rate: dict[str, float] = {
            sid: round(
                intent_counts.get(sid, {}).get("interrogation", 0)
                / max(turn_count.get(sid, 1), 1),
                4,
            )
            for sid in speakers
        }

        # ── 反抗衰减度（对非对称轨道有意义）──────────────────
        resistance_decay = self._compute_resistance_decay(turns, speakers)

        return InteractionFeatures(
            speaker_word_ratio            = word_dist,
            turn_count_per_speaker        = turn_count,
            backchannel_rate_per_speaker  = bc_rate,
            negotiation_ping_pong_rate    = ping_pong_rate,
            emotional_grooming_index      = emo_index,
            compliance_rate               = compliance_rate,
            interrogation_rate            = interrogation_rate,
            resistance_decay              = resistance_decay,
        )

    @staticmethod
    def _compute_ping_pong_rate(turns: list[DialogueTurn]) -> float:
        """
        协商往返率：统计连续出现「A→B→A」格局的三元组数量
        占非 backchannel 总轮次的比例。

        高值（>0.4）：真实双向协商，排除单向灌输和机器人群发。
        """
        non_bc = [t for t in turns if not t.is_backchannel]
        if len(non_bc) < _PING_PONG_MIN_TURNS:
            return 0.0

        ping_pong_count: int = 0
        for i in range(1, len(non_bc) - 1):
            if (
                non_bc[i - 1].speaker_id != non_bc[i].speaker_id
                and non_bc[i].speaker_id != non_bc[i + 1].speaker_id
                and non_bc[i - 1].speaker_id == non_bc[i + 1].speaker_id
            ):
                ping_pong_count += 1

        return round(ping_pong_count / max(len(non_bc) - 2, 1), 4)

    @staticmethod
    def _compute_resistance_decay(
        turns: list[DialogueTurn], speakers: list[str]
    ) -> float:
        """
        反抗衰减度：取 Target（字数较少方）在对话前20% vs 后20% 的
        平均字数比值。比值越大，说明前期反抗强、后期沦为纯附和。

        实现说明：
        - 此处计算所有 speakers 中字数最少方（后续 bind_asymmetric
          会用实际 Target 重新计算），作为通用预计算值。
        - 比值无上限，越大越危险。
        """
        if not speakers:
            return 0.0

        # 找字数最少的 speaker（候选 Target）
        word_dist = {
            sid: sum(t.word_count for t in turns if t.speaker_id == sid and not t.is_backchannel)
            for sid in speakers
        }
        if not word_dist:
            return 0.0
        candidate_target = min(word_dist, key=lambda s: word_dist[s])

        target_turns = [
            t for t in turns
            if t.speaker_id == candidate_target and not t.is_backchannel
        ]
        n = len(target_turns)
        if n < 4:
            return 0.0

        cut = max(1, int(n * _RESISTANCE_QUANTILE))
        early_words = [t.word_count for t in target_turns[:cut]]
        late_words  = [t.word_count for t in target_turns[-cut:]]

        early_avg = statistics.mean(early_words) if early_words else 0.0
        late_avg  = statistics.mean(late_words)  if late_words  else 0.0

        if late_avg == 0:
            # 完全归零：极端沦陷信号，返回一个饱和值
            return 5.0 if early_avg > 0 else 0.0

        return round(early_avg / late_avg, 4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3a：对称轨道角色绑定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _bind_symmetric(
        self,
        turns:  list[DialogueTurn],
        ifeats: InteractionFeatures,
    ) -> list[SpeakerRoleResult]:
        """
        对称轨道处理流程：
        1. 判定是否为纯聊天 → PEER_A / PEER_B
        2. 否则：寻找 Driver（提案+总结发起者）→ DRIVER / FOLLOWER
        3. 附加计算：情绪价值指数 + 顺从度（写入 metadata，由调用方使用）
        """
        speakers: list[str] = list(
            dict.fromkeys(t.speaker_id for t in turns)
        )

        # ── 纯聊判定 ──────────────────────────────────────────
        if self._is_pure_chat(turns, ifeats, speakers):
            return [
                SpeakerRoleResult(
                    speaker_id = sid,
                    role       = (RoleLabel.PEER_A if i == 0 else RoleLabel.PEER_B),
                    confidence = 0.85,
                    evidence   = ["高情绪密度", "无业务提案/顺从模式", "双方字数接近"],
                )
                for i, sid in enumerate(speakers)  # 覆盖所有 speaker，防止 pydantic 校验失败
            ]

        # ── Driver / Follower 识别 ─────────────────────────────
        driver_sid, follower_sid, evidence = self._find_driver_follower(
            turns, speakers, ifeats
        )

        results: list[SpeakerRoleResult] = []
        if driver_sid:
            # Driver 的情绪价值指数（高危特征）
            egi = ifeats.emotional_grooming_index.get(driver_sid, 0.0)
            driver_evidence = evidence + [
                f"emotional_grooming_index={egi:.3f}"
            ]
            results.append(SpeakerRoleResult(
                speaker_id = driver_sid,
                role       = RoleLabel.DRIVER,
                confidence = 0.80,
                evidence   = driver_evidence,
            ))

        if follower_sid:
            # Follower 的顺从度（高危特征）
            cr = ifeats.compliance_rate
            follower_evidence = [f"compliance_rate={cr:.3f}"]
            results.append(SpeakerRoleResult(
                speaker_id = follower_sid,
                role       = RoleLabel.FOLLOWER,
                confidence = 0.75,
                evidence   = follower_evidence,
            ))

        # 兜底：无法区分时平权
        if not results:
            results = [
                SpeakerRoleResult(
                    speaker_id = sid,
                    role       = (RoleLabel.PEER_A if i == 0 else RoleLabel.PEER_B),
                    confidence = 0.50,
                    evidence   = ["Driver/Follower 特征不显著，保守标记为平权"],
                )
                for i, sid in enumerate(speakers)  # 覆盖所有 speaker
            ]

        assigned_sids = {r.speaker_id for r in results}
        for sid in speakers:
            if sid not in assigned_sids:
                results.append(SpeakerRoleResult(
                    speaker_id = sid,
                    role       = RoleLabel.PEER_B, # 多余的人统统算作平权方
                    confidence = 0.3,
                    evidence   = ["多方通话/边缘场景兜底：统一标记为平权"]
                ))

        return results

    def _is_pure_chat(
        self,
        turns:    list[DialogueTurn],
        ifeats:   InteractionFeatures,
        speakers: list[str],
    ) -> bool:
        """
        纯聊天判定：满足以下 **全部** 条件才认定为纯聊：
        1. 至少有两个 speaker
        2. 情绪轮次密度（任一方）> _PURE_CHAT_EMOTION_THRESHOLD
        3. 全局 compliance 密度 < _PURE_CHAT_COMPLIANCE_THRESHOLD（无业务顺从）
        4. 无 proposal + closure 意图轮次出现
        """
        if len(speakers) < _PURE_CHAT_MIN_SPEAKERS:
            return False

        # 条件 2：任一方的情绪指数超标
        has_high_emotion = any(
            ifeats.emotional_grooming_index.get(sid, 0.0)
            > _PURE_CHAT_EMOTION_THRESHOLD
            for sid in speakers
        )

        # 条件 3：全局顺从密度极低
        low_compliance = (
            ifeats.compliance_rate < _PURE_CHAT_COMPLIANCE_THRESHOLD
        )

        # 条件 4：没有业务驱动意图
        has_business_intent = any(
            label in ("proposal", "closure")
            for turn in turns
            for label in turn.intent_labels
        )

        return has_high_emotion and low_compliance and not has_business_intent

    @staticmethod
    def _find_driver_follower(
        turns:    list[DialogueTurn],
        speakers: list[str],
        ifeats:   InteractionFeatures,
    ) -> tuple[Optional[str], Optional[str], list[str]]:
        """
        Driver 评分 = 0.6 × (proposal 轮次数) + 0.4 × (closure 轮次数)
        得分最高者为 Driver，其余为 Follower。

        Returns
        -------
        (driver_sid, follower_sid, evidence_list)
        """
        proposal_counts: dict[str, int] = {}
        closure_counts:  dict[str, int] = {}

        for turn in turns:
            sid = turn.speaker_id
            if "proposal" in turn.intent_labels:
                proposal_counts[sid] = proposal_counts.get(sid, 0) + 1
            if "closure" in turn.intent_labels:
                closure_counts[sid] = closure_counts.get(sid, 0) + 1

        driver_scores: dict[str, float] = {
            sid: (
                _DRIVER_PROPOSAL_WEIGHT * proposal_counts.get(sid, 0)
                + _DRIVER_CLOSURE_WEIGHT * closure_counts.get(sid, 0)
            )
            for sid in speakers
        }

        if not any(v > 0 for v in driver_scores.values()):
            return None, None, []

        sorted_speakers = sorted(
            speakers, key=lambda s: driver_scores[s], reverse=True
        )
        driver_sid:   str = sorted_speakers[0]
        follower_sid: str = sorted_speakers[-1] if len(sorted_speakers) > 1 else ""

        evidence: list[str] = [
            f"proposal_count={proposal_counts.get(driver_sid, 0)}",
            f"closure_count={closure_counts.get(driver_sid, 0)}",
            f"driver_score={driver_scores[driver_sid]:.2f}",
        ]

        return driver_sid, follower_sid or None, evidence

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3b：非对称轨道角色绑定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _bind_asymmetric(
        self,
        turns:  list[DialogueTurn],
        ifeats: InteractionFeatures,
    ) -> list[SpeakerRoleResult]:
        """
        非对称轨道处理流程：
        1. Agent（输出方）= 字数占比最大 + 提问压制率最高
        2. Target（接收方）= 字数占比最小 + backchannel 率最高
        3. 提取 Target 的沦陷特征：backchannel_rate + resistance_decay
        """
        speakers: list[str] = list(
            dict.fromkeys(t.speaker_id for t in turns)
        )
        if not speakers:
            return []

        word_dist = ifeats.speaker_word_ratio

        # ── Agent 识别 ────────────────────────────────────────
        # 首要信号：字数占比最大方
        agent_sid = max(word_dist, key=lambda s: word_dist.get(s, 0.0))

        # 次要信号：提问压制率验证（若与字数最大方不符，取提问率最高方）
        intr_rate = ifeats.interrogation_rate
        intr_max_sid = (
            max(intr_rate, key=lambda s: intr_rate.get(s, 0.0))
            if intr_rate else agent_sid
        )
        # 若两个信号一致性强，提升置信度
        agent_confidence: float = (
            0.90 if agent_sid == intr_max_sid else 0.70
        )

        # 若提问压制率超过阈值，进一步确认
        if intr_rate.get(agent_sid, 0.0) < _AGENT_INTERROGATION_THRESHOLD:
            agent_confidence = max(0.60, agent_confidence - 0.10)

        agent_evidence: list[str] = [
            f"word_ratio={word_dist.get(agent_sid, 0.0):.3f}",
            f"interrogation_rate={intr_rate.get(agent_sid, 0.0):.3f}",
        ]

        # ── Target 识别 ───────────────────────────────────────
        # 字数最少方
        target_sid = min(word_dist, key=lambda s: word_dist.get(s, 1.0))

        # 沦陷特征
        bc_rate = ifeats.backchannel_rate_per_speaker.get(target_sid, 0.0)
        r_decay = ifeats.resistance_decay  # 已在 _compute 阶段计算

        target_confidence: float = min(0.95, 0.65 + bc_rate * 0.5)
        target_evidence: list[str] = [
            f"backchannel_rate={bc_rate:.3f}",
            f"resistance_decay={r_decay:.3f}",
            f"word_ratio={word_dist.get(target_sid, 0.0):.3f}",
        ]

        results: list[SpeakerRoleResult] = [
            SpeakerRoleResult(
                speaker_id = agent_sid,
                role       = RoleLabel.AGENT,
                confidence = round(agent_confidence, 3),
                evidence   = agent_evidence,
            ),
        ]
        if target_sid and target_sid != agent_sid:
            results.append(SpeakerRoleResult(
                speaker_id = target_sid,
                role       = RoleLabel.TARGET,
                confidence = round(target_confidence, 3),
                evidence   = target_evidence,
            ))

        assigned_sids = {r.speaker_id for r in results}
        for sid in speakers:
            if sid not in assigned_sids:
                results.append(SpeakerRoleResult(
                    speaker_id = sid,
                    role       = RoleLabel.TARGET, # 多余的人统统算作 Target
                    confidence = 0.3,
                    evidence   = ["多方通话/边缘场景兜底：统一标记为 TARGET"]
                ))

        return results
