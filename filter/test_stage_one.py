# test_stage_one.py
# ============================================================
# 阶段一单元测试套件
# 覆盖：归一化、未接通检测、机器人特征、LID 路由、ASR 容错
# 运行：pytest test_stage_one.py -v
# ============================================================

import pytest

from models import ASRRecord, BotLabel, ConnectionStatus
from stage_one_filter import (
    ASRErrorCorrector,
    BotFeatureExtractor,
    LIDRouter,
    Normalizer,
    StageOneFilter,
    UnconnectedDetector,
    _compute_f_ent,
    _compute_f_len,
    _double_metaphone_match,
    _ngram_jaccard,
    _pinyin_similarity,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 数据模型测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestASRRecord:
    def test_basic_creation(self) -> None:
        r = ASRRecord(record_id="r001", raw_text="你好，请问有什么可以帮助您？")
        assert r.normalized_text is None  # 未经流水线处理
        assert r.stage_one_done is False

    def test_whitespace_only_raw_text_raises(self) -> None:
        with pytest.raises(Exception):
            ASRRecord(record_id="r002", raw_text="   ")

    def test_empty_raw_text_allowed(self) -> None:
        r = ASRRecord(record_id="r003", raw_text="")
        assert r.raw_text == ""

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ASRRecord(record_id="r004", raw_text="test", unknown_field="x")  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Normalizer 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNormalizer:
    def setup_method(self) -> None:
        self.norm = Normalizer()

    def test_fullwidth_to_halfwidth(self) -> None:
        result = self.norm.normalize("ｈｅｌｌｏ　ｗｏｒｌｄ")
        assert "ｈ" not in result  # 全角字母应被转换

    def test_repeat_punct_compression(self) -> None:
        result = self.norm.normalize("好的，，，我知道了。。。")
        assert "，，" not in result
        assert "。。" not in result
        assert "，" in result  # 保留单个标点

    def test_original_words_preserved(self) -> None:
        text = "嗯嗯嗯，那个，就是说，我想问一下转账的事情"
        result = self.norm.normalize(text)
        # 原始汉字不应丢失（允许标点压缩）
        assert "转账" in result
        assert "问" in result

    def test_multi_space_compressed(self) -> None:
        result = self.norm.normalize("你好   世界")
        assert "   " not in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 未接通特征函数测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUnconnectedFeatures:
    def test_f_len_short_text(self) -> None:
        # 3 字符 → f_len = 1 - 3/15 = 0.8
        assert abs(_compute_f_len("喂喂喂") - 0.8) < 0.01

    def test_f_len_long_text(self) -> None:
        # 超过阈值，归零
        assert _compute_f_len("这是一段超过十五个字的正常对话内容，不应被判定为未接通。") == 0.0

    def test_f_ent_repetitive_text(self) -> None:
        # "我我我喂" → 去重后 2 个，length=4，min=4
        # f_ent = 1 - (2+ε)/4 ≈ 0.5
        val = _compute_f_ent("我我我喂")
        assert val > 0.3

    def test_f_ent_diverse_text(self) -> None:
        # 多样化文本，熵特征接近 0
        val = _compute_f_ent("你好世界！今天天气不错。")
        assert val < 0.3


class TestUnconnectedDetector:
    def setup_method(self) -> None:
        self.detector = UnconnectedDetector()

    def test_unconnected_short_repetitive(self) -> None:
        _, status = self.detector.compute("喂喂")
        assert status == ConnectionStatus.UNCONNECTED

    def test_connected_business_text(self) -> None:
        _, status = self.detector.compute(
            "您好，我想了解一下关于转账的手续费问题，我的账户余额不足。"
        )
        # 包含业务实体词（转账、账户），f_entity=0 → p=0 → CONNECTED
        assert status == ConnectionStatus.CONNECTED

    def test_features_structure(self) -> None:
        feats, _ = self.detector.compute("嗯")
        assert 0.0 <= feats.p_unconnected <= 1.0
        assert 0.0 <= feats.f_len <= 1.0
        assert 0.0 <= feats.f_ent <= 1.0
        assert feats.f_entity in (0.0, 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 机器人特征提取测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBotFeatureExtractor:
    def setup_method(self) -> None:
        self.extractor = BotFeatureExtractor()

    def test_high_filler_ratio_is_human(self) -> None:
        text = "嗯，那个，就是，啊，对对，我觉得吧"
        feats = self.extractor.extract(text)
        label = self.extractor.infer_bot_label(feats)
        assert label == BotLabel.HUMAN

    def test_low_filler_ratio_is_bot(self) -> None:
        text = "您好，请问您需要办理什么业务？请问您有什么需要帮助的？"
        feats = self.extractor.extract(text)
        label = self.extractor.infer_bot_label(feats)
        assert label == BotLabel.BOT

    def test_prefix_tokens_length(self) -> None:
        feats = self.extractor.extract("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert len(feats.prefix_tokens) <= 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ASR 容错测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestASRErrorCorrector:
    def setup_method(self) -> None:
        self.corrector = ASRErrorCorrector()

    def test_chinese_stutter_removal(self) -> None:
        result = self.corrector.correct("我我我想想想问问问一下", "zh")
        assert "我我" not in result
        assert "想想" not in result
        assert "我" in result  # 保留单个字

    def test_english_stutter_removal(self) -> None:
        result = self.corrector.correct("I I I want to to to go", "en")
        assert "I I" not in result

    def test_pinyin_similarity_homophones(self) -> None:
        # "退款" vs "推款"（ASR 同音错字）
        sim = _pinyin_similarity("退款", "推款")
        assert sim > 0.7

    def test_double_metaphone_match(self) -> None:
        # Smith / Smyth 音似
        assert _double_metaphone_match("Smith", "Smyth") is True
        # 完全不同的词不应匹配
        assert _double_metaphone_match("Smith", "Brown") is False

    def test_ngram_jaccard_similar(self) -> None:
        # 相似字符串
        score = _ngram_jaccard("hello", "hallo")
        assert score > 0.4

    def test_ngram_jaccard_dissimilar(self) -> None:
        score = _ngram_jaccard("hello", "xyz")
        assert score < 0.2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. LIDRouter 测试（降级模式，不依赖真实模型文件）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLIDRouter:
    def setup_method(self) -> None:
        # 故意传入不存在的路径，测试降级行为
        self.router = LIDRouter(model_path="/nonexistent/lid.176.bin")

    def test_short_text_fallback(self) -> None:
        lang, conf = self.router.detect("喂")
        assert lang == "zh"
        assert conf is None

    def test_degraded_mode_returns_default(self) -> None:
        lang, conf = self.router.detect("This is a normal length English sentence.")
        assert lang == "zh"   # 降级模式全部返回默认值
        assert conf is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. StageOneFilter 端到端集成测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStageOneFilter:
    def setup_method(self) -> None:
        self.pipeline = StageOneFilter(fasttext_model_path="/nonexistent/lid.176.bin")

    def _make_record(self, text: str, rid: str = "test001") -> ASRRecord:
        return ASRRecord(record_id=rid, raw_text=text)

    def test_raw_text_never_modified(self) -> None:
        original = "我我我，嗯，，，喂喂喂"
        r = self._make_record(original)
        self.pipeline.process(r)
        assert r.raw_text == original  # 绝对不变

    def test_stage_one_done_flag(self) -> None:
        r = self._make_record("你好，请问有什么可以帮助您的吗？")
        self.pipeline.process(r)
        assert r.stage_one_done is True

    def test_unconnected_marked_correctly(self) -> None:
        r = self._make_record("喂")
        self.pipeline.process(r)
        assert r.connection_status == ConnectionStatus.UNCONNECTED

    def test_all_fields_filled_after_pipeline(self) -> None:
        r = self._make_record("嗯，那个，我想咨询一下转账手续费的问题。")
        self.pipeline.process(r)
        assert r.normalized_text is not None
        assert r.lang is not None
        assert r.cleaned_text is not None
        assert r.bot_features is not None
        assert r.unconnected_features is not None
        assert r.connection_status is not None

    def test_metadata_populated(self) -> None:
        r = self._make_record("谁啊")
        self.pipeline.process(r)
        assert "stage_one" in r.metadata
        assert "p_unconnected" in r.metadata["stage_one"]

    def test_batch_processing(self) -> None:
        records = [
            self._make_record("喂", rid="b001"),
            self._make_record("你好，我想咨询贷款业务。", rid="b002"),
            self._make_record("Hello, I need help with my account.", rid="b003"),
        ]
        results = self.pipeline.process_batch(records)
        assert all(r.stage_one_done for r in results)
        assert len(results) == 3
