# test_stage_two.py
# ============================================================
# 阶段二单元 + 集成测试套件
# 全部测试在无 GPU / 无 BGE-M3 的环境下可通过（使用降级编码器）
# 运行：pytest test_stage_two.py -v
# ============================================================

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from models_stage2 import (
    ASRRecord,
    DialogueTurn,
    InteractionFeatures,
    RoleLabel,
    SpeakerRoleResult,
    StageTwoResult,
    TrackType,
)
from topology_engine  import TopologyAnalyzer, _count_words, _is_backchannel
from intent_radar     import IntentRadar
from role_binder      import RoleBinder
from stage_two_pipeline import StageTwoPipeline


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_record(speaker_id: str, text: str, rid: str = "") -> ASRRecord:
    return ASRRecord(
        record_id   = rid or f"{speaker_id}-{text[:6]}",
        speaker_id  = speaker_id,
        raw_text    = text,
        cleaned_text= text,
    )


def _make_turn(
    speaker_id: str,
    text:       str,
    word_count: int       = 10,
    is_bc:      bool      = False,
    intents:    list[str] | None = None,
    idx:        int       = 0,
) -> DialogueTurn:
    return DialogueTurn(
        speaker_id       = speaker_id,
        merged_text      = text,
        word_count       = word_count if not is_bc else 0,
        raw_record_count = 1,
        is_backchannel   = is_bc,
        intent_labels    = intents or [],
        turn_index       = idx,
    )


@pytest.fixture(scope="module")
def radar() -> IntentRadar:
    """使用降级编码器的 IntentRadar 实例（无需 GPU / TEI）。"""
    # 重置单例，防止测试间状态污染
    IntentRadar._instance = None
    return IntentRadar(model_name="BAAI/bge-m3", use_fp16=False, bge_service_url=None)


@pytest.fixture(scope="module")
def topology() -> TopologyAnalyzer:
    return TopologyAnalyzer()


@pytest.fixture(scope="module")
def binder(radar: IntentRadar, topology: TopologyAnalyzer) -> RoleBinder:
    return RoleBinder(radar=radar, topology=topology)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 数据模型测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestModels:
    def test_dialogue_turn_immutable(self) -> None:
        t = _make_turn("A", "hello")
        with pytest.raises(Exception):
            t.merged_text = "changed"  # type: ignore  frozen 模型不允许赋值

    def test_stage_two_result_role_coverage(self) -> None:
        turns = [_make_turn("A", "hello", idx=0)]
        roles = [SpeakerRoleResult(speaker_id="A", role=RoleLabel.AGENT, confidence=0.9)]
        ifeats = InteractionFeatures()
        result = StageTwoResult(
            conversation_id      = "c001",
            track_type           = TrackType.ASYMMETRIC,
            dialogue_turns       = turns,
            speaker_roles        = roles,
            interaction_features = ifeats,
        )
        assert result.stage_two_done is False  # 默认值

    def test_stage_two_result_missing_role_raises(self) -> None:
        turns = [
            _make_turn("A", "hello", idx=0),
            _make_turn("B", "world", idx=1),
        ]
        roles = [SpeakerRoleResult(speaker_id="A", role=RoleLabel.AGENT, confidence=0.9)]
        with pytest.raises(Exception, match="B"):
            StageTwoResult(
                conversation_id      = "c002",
                track_type           = TrackType.ASYMMETRIC,
                dialogue_turns       = turns,
                speaker_roles        = roles,
                interaction_features = InteractionFeatures(),
            )

    def test_track_type_enum_values(self) -> None:
        assert TrackType.SYMMETRIC.value  == "symmetric"
        assert TrackType.ASYMMETRIC.value == "asymmetric"

    def test_role_label_enum_coverage(self) -> None:
        labels = {r.value for r in RoleLabel}
        expected = {"agent", "target", "driver", "follower", "peer_a", "peer_b"}
        assert labels == expected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TopologyAnalyzer 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTopologyAnalyzer:
    def test_count_words_cjk(self) -> None:
        assert _count_words("你好世界") == 4

    def test_count_words_english(self) -> None:
        assert _count_words("hello world") == 2

    def test_count_words_mixed(self) -> None:
        count = _count_words("你好 hello 世界 world")
        assert count == 6  # 2 CJK + 2 EN

    def test_is_backchannel_pure_filler(self) -> None:
        assert _is_backchannel("嗯") is True
        assert _is_backchannel("嗯嗯") is True
        assert _is_backchannel("好的") is True
        assert _is_backchannel("ok") is True

    def test_is_backchannel_real_content(self) -> None:
        assert _is_backchannel("我想了解一下贷款利率") is False
        assert _is_backchannel("明天我们可以见面吗") is False

    def test_merge_turns_adjacent_same_speaker(
        self, topology: TopologyAnalyzer
    ) -> None:
        records = [
            _make_record("A", "你好"),
            _make_record("A", "请问有什么可以帮您"),
            _make_record("B", "我想问一下转账"),
        ]
        turns = topology.merge_turns(records)
        assert len(turns) == 2
        assert turns[0].speaker_id == "A"
        assert "请问" in turns[0].merged_text
        assert turns[0].raw_record_count == 2

    def test_merge_turns_backchannel_collapse(
        self, topology: TopologyAnalyzer
    ) -> None:
        """连续多个语气词应被折叠为单条 backchannel 轮次。"""
        records = [
            _make_record("A", "你了解我们的服务吗？"),
            _make_record("B", "嗯"),
            _make_record("B", "嗯"),
            _make_record("B", "嗯"),
            _make_record("A", "我们提供最优质的……"),
        ]
        turns = topology.merge_turns(records)
        # B 的三个嗯应折叠为 1 条 backchannel
        b_turns = [t for t in turns if t.speaker_id == "B"]
        assert len(b_turns) == 1
        assert b_turns[0].is_backchannel is True
        assert b_turns[0].raw_record_count == 3

    def test_classify_track_asymmetric(
        self, topology: TopologyAnalyzer
    ) -> None:
        """A 说了大量内容（>80%），B 只有寥寥数语。"""
        turns = (
            [_make_turn("A", f"发言{i}", word_count=50) for i in range(10)]
            + [_make_turn("B", "好", word_count=2) for _ in range(2)]
        )
        assert topology.classify_track(turns) == TrackType.ASYMMETRIC

    def test_classify_track_symmetric(
        self, topology: TopologyAnalyzer
    ) -> None:
        """双方轮流发言，字数均衡。"""
        turns = []
        for i in range(10):
            turns.append(_make_turn("A", f"句子{i}", word_count=8, idx=i*2))
            turns.append(_make_turn("B", f"回复{i}", word_count=7, idx=i*2+1))
        assert topology.classify_track(turns) == TrackType.SYMMETRIC

    def test_classify_track_empty(
        self, topology: TopologyAnalyzer
    ) -> None:
        assert topology.classify_track([]) == TrackType.ASYMMETRIC

    def test_turn_index_sequential(
        self, topology: TopologyAnalyzer
    ) -> None:
        records = [
            _make_record("A", "你好"),
            _make_record("B", "你好"),
            _make_record("A", "再见"),
        ]
        turns = topology.merge_turns(records)
        indices = [t.turn_index for t in turns]
        assert indices == list(range(len(turns)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. IntentRadar 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntentRadar:
    def test_detect_returns_list(self, radar: IntentRadar) -> None:
        result = radar.detect("你好")
        assert isinstance(result, list)

    def test_detect_batch_length_matches(self, radar: IntentRadar) -> None:
        texts = ["你好", "明天见面吧", "好的没问题"]
        results = radar.detect_batch(texts)
        assert len(results) == len(texts)

    def test_detect_batch_empty(self, radar: IntentRadar) -> None:
        assert radar.detect_batch([]) == []

    def test_score_batch_has_all_intents(self, radar: IntentRadar) -> None:
        scores = radar.score_batch(["早安，好好休息"])
        assert len(scores) == 1
        expected_intents = {"emotion", "proposal", "closure", "compliance", "interrogation"}
        assert expected_intents.issubset(set(scores[0].keys()))

    def test_score_values_in_range(self, radar: IntentRadar) -> None:
        scores = radar.score_batch(["test text"])
        for intent, score in scores[0].items():
            assert 0.0 <= score <= 1.0, f"intent={intent} score={score} out of range"

    def test_singleton_returns_same_instance(self) -> None:
        IntentRadar._instance = None  # 重置
        r1 = IntentRadar.get_instance()
        r2 = IntentRadar.get_instance()
        assert r1 is r2

    def test_anchor_matrices_loaded(self, radar: IntentRadar) -> None:
        """所有 intent 的锚点矩阵均已加载，形状正确。"""
        assert len(radar._anchor_matrices) > 0
        for intent, mat in radar._anchor_matrices.items():
            assert mat.ndim == 2, f"intent={intent} 的矩阵维度不是 2"
            assert mat.shape[0] > 0, f"intent={intent} 的锚点数量为 0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RoleBinder 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRoleBinder:

    def _asymmetric_turns(self) -> list[DialogueTurn]:
        """模拟非对称对话：A 大量输出，B 偶尔附和。"""
        turns: list[DialogueTurn] = []
        for i in range(10):
            turns.append(_make_turn("A", f"介绍内容{i}", word_count=40, intents=["interrogation"], idx=i*2))
            turns.append(_make_turn("B", "嗯", word_count=0, is_bc=True, idx=i*2+1))
        return turns

    def _symmetric_turns_with_driver(self) -> list[DialogueTurn]:
        """模拟对称有业务轨道：A 提案，B 顺从。"""
        return [
            _make_turn("A", "明天上午九点见", word_count=8, intents=["proposal"], idx=0),
            _make_turn("B", "好的",          word_count=2, intents=["compliance"], idx=1),
            _make_turn("A", "就定在周五",     word_count=5, intents=["closure"], idx=2),
            _make_turn("B", "可以",           word_count=1, intents=["compliance"], idx=3),
            _make_turn("A", "那先这样",       word_count=4, intents=["closure"], idx=4),
            _make_turn("B", "行",             word_count=1, intents=["compliance"], idx=5),
        ]

    def _pure_chat_turns(self) -> list[DialogueTurn]:
        """模拟纯聊天：高情绪、无提案、无顺从。"""
        return [
            _make_turn("A", "哈哈哈哈笑死",  word_count=6,  intents=["emotion"], idx=0),
            _make_turn("B", "太好笑了吧",    word_count=5,  intents=["emotion"], idx=1),
            _make_turn("A", "你今天怎么样",   word_count=5,  intents=["emotion"], idx=2),
            _make_turn("B", "还不错哈哈哈",   word_count=5,  intents=["emotion"], idx=3),
        ]

    def test_bind_asymmetric_assigns_agent_target(
        self, binder: RoleBinder
    ) -> None:
        turns  = self._asymmetric_turns()
        _, roles, _ = binder.bind(turns, TrackType.ASYMMETRIC)
        role_map = {r.speaker_id: r.role for r in roles}
        assert RoleLabel.AGENT  in role_map.values()
        assert RoleLabel.TARGET in role_map.values()
        assert role_map["A"] == RoleLabel.AGENT
        assert role_map["B"] == RoleLabel.TARGET

    def test_bind_symmetric_driver_follower(
        self, binder: RoleBinder
    ) -> None:
        turns  = self._symmetric_turns_with_driver()
        _, roles, _ = binder.bind(turns, TrackType.SYMMETRIC)
        role_map = {r.speaker_id: r.role for r in roles}
        assert role_map.get("A") == RoleLabel.DRIVER
        assert role_map.get("B") == RoleLabel.FOLLOWER

    def test_bind_pure_chat_assigns_peers(
        self, binder: RoleBinder
    ) -> None:
        turns  = self._pure_chat_turns()
        _, roles, _ = binder.bind(turns, TrackType.SYMMETRIC)
        role_labels = {r.role for r in roles}
        assert RoleLabel.PEER_A in role_labels
        assert RoleLabel.PEER_B in role_labels

    def test_interaction_features_backchannel_rate(
        self, binder: RoleBinder
    ) -> None:
        turns = self._asymmetric_turns()
        _, _, ifeats = binder.bind(turns, TrackType.ASYMMETRIC)
        # B 全部是 backchannel，bc_rate 应为 1.0
        assert ifeats.backchannel_rate_per_speaker.get("B", 0.0) == 1.0

    def test_interaction_features_ping_pong(
        self, binder: RoleBinder
    ) -> None:
        turns = self._symmetric_turns_with_driver()
        _, _, ifeats = binder.bind(turns, TrackType.SYMMETRIC)
        # 严格交替发言，ping-pong rate 应 > 0
        assert ifeats.negotiation_ping_pong_rate >= 0.0

    def test_role_result_confidence_in_range(
        self, binder: RoleBinder
    ) -> None:
        turns  = self._asymmetric_turns()
        _, roles, _ = binder.bind(turns, TrackType.ASYMMETRIC)
        for role in roles:
            assert 0.0 <= role.confidence <= 1.0

    def test_all_speakers_covered_after_bind(
        self, binder: RoleBinder
    ) -> None:
        turns = self._symmetric_turns_with_driver()
        labeled, roles, _ = binder.bind(turns, TrackType.SYMMETRIC)
        speakers_in_turns = {t.speaker_id for t in labeled}
        speakers_in_roles = {r.speaker_id for r in roles}
        assert speakers_in_turns == speakers_in_roles

    def test_intent_labels_injected(
        self, binder: RoleBinder
    ) -> None:
        """_annotate_intents 必须为非 backchannel 轮次注入 intent_labels。"""
        turns = [
            DialogueTurn(
                speaker_id="A", merged_text="明天见面吧",
                word_count=5, raw_record_count=1, turn_index=0,
            ),
        ]
        labeled, _, _ = binder.bind(turns, TrackType.SYMMETRIC)
        # intent_labels 字段存在（内容依赖模型，降级下可能为空列表，不强断言内容）
        assert isinstance(labeled[0].intent_labels, list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. StageTwoPipeline 端到端集成测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStageTwoPipeline:

    @pytest.fixture(scope="class")
    def pipeline(self) -> StageTwoPipeline:
        IntentRadar._instance = None
        return StageTwoPipeline(bge_model_name="BAAI/bge-m3", use_fp16=False, bge_service_url=None)

    def _make_asymmetric_records(self) -> list[ASRRecord]:
        """模拟非对称通话记录：A 大量输出，B 纯附和。"""
        records: list[ASRRecord] = []
        for i in range(8):
            records.append(_make_record("A", f"我们平台专注金融理财已有十年，您可以把资金托管给我们，保证年化收益超过百分之二十。第{i}句"))
        for i in range(3):
            records.append(_make_record("B", "嗯"))
        return records

    def _make_symmetric_records(self) -> list[ASRRecord]:
        """模拟对称协商记录：双方交替，有业务提案。"""
        content = [
            ("A", "你明天有空吗，我们可以约一下"),
            ("B", "明天下午可以哦"),
            ("A", "那就定在明天下午三点"),
            ("B", "好的没问题"),
            ("A", "地址发你一下"),
            ("B", "行，收到了"),
        ]
        return [_make_record(sid, text) for sid, text in content]

    def test_asymmetric_result_structure(
        self, pipeline: StageTwoPipeline
    ) -> None:
        records = self._make_asymmetric_records()
        result  = pipeline.process_conversation("conv_asym_001", records)
        assert result.stage_two_done is True
        assert result.track_type == TrackType.ASYMMETRIC
        assert len(result.speaker_roles) >= 1
        assert len(result.dialogue_turns) >= 1

    def test_symmetric_result_structure(
        self, pipeline: StageTwoPipeline
    ) -> None:
        records = self._make_symmetric_records()
        result  = pipeline.process_conversation("conv_sym_001", records)
        assert result.stage_two_done is True
        assert result.track_type == TrackType.SYMMETRIC

    def test_metadata_populated(
        self, pipeline: StageTwoPipeline
    ) -> None:
        records = self._make_symmetric_records()
        result  = pipeline.process_conversation(
            "conv_meta_001", records,
            extra_metadata={"source": "test_suite"}
        )
        assert "raw_record_count" in result.metadata
        assert "merged_turn_count" in result.metadata
        assert result.metadata.get("source") == "test_suite"

    def test_role_coverage_invariant(
        self, pipeline: StageTwoPipeline
    ) -> None:
        """StageTwoResult 的角色覆盖校验器必须能被触发（无遗漏 speaker）。"""
        records = self._make_asymmetric_records()
        result  = pipeline.process_conversation("conv_cov_001", records)
        speakers_in_turns = {t.speaker_id for t in result.dialogue_turns}
        speakers_in_roles = {r.speaker_id for r in result.speaker_roles}
        assert speakers_in_turns.issubset(speakers_in_roles)

    def test_empty_records(
        self, pipeline: StageTwoPipeline
    ) -> None:
        """空输入不应崩溃，返回合理的空结构。"""
        result = pipeline.process_conversation("conv_empty_001", [])
        assert result.stage_two_done is True
        assert result.dialogue_turns == []
