# intent_radar.py  ── V5.0 配置驱动版
# ============================================================
# 软语义轨道：BGE-M3 多语言向量锚点雷达
#
# V5.0 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 废弃文件内所有硬编码锚点词表和 _INTENT_THRESHOLD_OVERRIDE
# ② __init__ 时从 config_topics.TOPIC_REGISTRY 动态加载：
#      bge_anchors   → 离线向量化为锚点矩阵
#      threshold     → 写入 _threshold_map 供推理使用
# ③ 新增主题只需在 config_topics.py 追加条目，本文件零修改
# ④ 保留 score_batch() 连续值接口，供共现矩阵使用
# ============================================================

from __future__ import annotations

import threading
from typing import ClassVar, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config_topics import TOPIC_REGISTRY, TopicDefinition

try:
    from FlagEmbedding import BGEM3FlagModel
    _BGEM3_AVAILABLE = True
except ImportError:
    _BGEM3_AVAILABLE = False
    BGEM3FlagModel = None  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 降级编码器（无 GPU / CI 测试环境）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _FallbackEncoder:
    """字符 3-gram 哈希向量，保证接口与 BGE-M3 一致，仅用于测试。"""
    _DIM: int = 4096

    def encode(self, texts: list[str], **kwargs) -> dict[str, np.ndarray]:
        vecs: list[np.ndarray] = []
        for text in texts:
            v = np.zeros(self._DIM, dtype=np.float32)
            for i in range(max(0, len(text) - 2)):
                bucket = abs(hash(text[i:i + 3])) % self._DIM
                v[bucket] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vecs.append(v)
        return {"dense_vecs": np.array(vecs, dtype=np.float32)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IntentRadar —— 配置驱动单例
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntentRadar:
    """
    V5.0 配置驱动的多语言向量锚点雷达，进程内单例，线程安全。

    初始化流程
    ─────────────────────────────────────────────────────────
    1. 从 TOPIC_REGISTRY 中收集所有主题的 bge_anchors
    2. 将全部锚点拼接后一次性批量编码（离线向量化）
    3. 按 topic_id 切片存储为 numpy 矩阵
    4. 记录每个 topic_id 的触发阈值到 _threshold_map

    扩展方法
    ─────────────────────────────────────────────────────────
    在 config_topics.TOPIC_REGISTRY 中追加新主题 → 自动生效，
    无需修改本文件任何代码。
    """

    _instance: ClassVar[Optional["IntentRadar"]] = None
    _lock:     ClassVar[threading.Lock]          = threading.Lock()

    def __init__(
        self,
        model_name:  str   = "BAAI/bge-m3",
        use_fp16:    bool  = True,
        batch_size:  int   = 32,
        registry:    dict[str, TopicDefinition] = TOPIC_REGISTRY,
    ) -> None:
        self._batch_size = batch_size

        # ── 模型加载 ──────────────────────────────────────────
        if _BGEM3_AVAILABLE and BGEM3FlagModel is not None:
            self._model       = BGEM3FlagModel(model_name, use_fp16=use_fp16)
            self._is_fallback = False
        else:
            import warnings
            warnings.warn(
                "[IntentRadar] FlagEmbedding 未安装，使用 N-gram 降级编码器。",
                RuntimeWarning, stacklevel=2,
            )
            self._model       = _FallbackEncoder()
            self._is_fallback = True

        # ── 从配置动态加载阈值和锚点 ─────────────────────────
        # _threshold_map : {topic_id: threshold}
        self._threshold_map: dict[str, float] = {
            tid: td.threshold
            for tid, td in registry.items()
        }

        # 离线向量化：按顺序拼接所有主题的锚点
        self._anchor_matrices: dict[str, np.ndarray] = {}
        self._vectorize_from_registry(registry)

    # ── 单例工厂 ──────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_name: str  = "BAAI/bge-m3",
        use_fp16:   bool = True,
    ) -> "IntentRadar":
        """双重检查锁定，线程安全单例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name = model_name,
                        use_fp16   = use_fp16,
                    )
        return cls._instance

    # ── 离线向量化 ────────────────────────────────────────────

    def _vectorize_from_registry(
        self, registry: dict[str, TopicDefinition]
    ) -> None:
        """
        遍历注册表，将所有 bge_anchors 拼接为大列表后一次批量编码，
        再按 topic_id 切片存入 _anchor_matrices。
        时间复杂度：O(total_anchors)，每个锚点仅编码一次。
        """
        all_sentences: list[str]                  = []
        slice_map:     dict[str, tuple[int, int]] = {}

        for topic_id, topic_def in registry.items():
            if not topic_def.bge_anchors:
                continue
            start = len(all_sentences)
            all_sentences.extend(topic_def.bge_anchors)
            slice_map[topic_id] = (start, len(all_sentences))

        if not all_sentences:
            return

        encoded = self._model.encode(
            all_sentences,
            batch_size=min(self._batch_size, len(all_sentences)),
            max_length=512,
        )
        all_vecs: np.ndarray = encoded["dense_vecs"]

        for topic_id, (start, end) in slice_map.items():
            self._anchor_matrices[topic_id] = all_vecs[start:end]

    # ── 公开推理接口 ──────────────────────────────────────────

    def detect(self, text: str) -> list[str]:
        """单句意图检测，返回触发的 topic_id 列表。"""
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: list[str]) -> list[list[str]]:
        """
        批量意图检测。每个 topic_id 使用其配置中的独立阈值判定。
        """
        if not texts:
            return []

        raw_scores = self._compute_raw_scores(texts)
        results: list[list[str]] = []

        for score_dict in raw_scores:
            triggered: list[str] = [
                topic_id
                for topic_id, score in score_dict.items()
                if score >= self._threshold_map.get(topic_id, 0.72)
            ]
            results.append(triggered)

        return results

    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """
        返回连续相似度得分（未截断）。
        共现矩阵打分器优先调用此接口以获取激活强度。
        """
        return self._compute_raw_scores(texts) if texts else []

    # ── 内部核心计算 ──────────────────────────────────────────

    def _compute_raw_scores(
        self, texts: list[str]
    ) -> list[dict[str, float]]:
        """
        Q(B×d) × A^T(d×n) → sim(B×n)，取行最大值 → (B,) 激活强度。
        """
        encoded     = self._model.encode(
            texts,
            batch_size=min(self._batch_size, len(texts)),
            max_length=512,
        )
        query_vecs: np.ndarray = encoded["dense_vecs"]
        all_scores: list[dict[str, float]] = [{} for _ in texts]

        for topic_id, anchor_mat in self._anchor_matrices.items():
            sim_mat    = cosine_similarity(query_vecs, anchor_mat)
            max_scores = sim_mat.max(axis=1)
            for i, score in enumerate(max_scores):
                all_scores[i][topic_id] = round(float(score), 4)

        return all_scores
    
    # 新增语义检索
    def dynamic_search(
        self, 
        turn_texts: list[str], 
        dynamic_topic: str, 
        default_threshold: float = 0.65
    ) -> dict[str, Any]:
        """
        零样本动态语义检索
        """
        # 鲁棒性设计 1：空值与边界拦截
        clean_topic = dynamic_topic.strip() if dynamic_topic else ""
        if not turn_texts or not clean_topic:
            return {
                "topic_queried": clean_topic,
                "matched": False, 
                "max_score": 0.0,
                "status": "skipped_due_to_empty_input"
            }

        try:
            # 鲁棒性设计 2：BGE-M3 检索专属 Prompt 构造
            # 对于 BGE 模型，Query 必须加 instruction 才能在非对称检索中获得高准确率
            query_instruction = f"为这个句子生成表示以用于检索相关文章：{clean_topic}"
            
            # 执行编码
            query_vec = self._model.encode([query_instruction])[0]  # Shape: (dim,)
            doc_vecs = self._model.encode(turn_texts)               # Shape: (N, dim)

            # 鲁棒性设计 3：安全矩阵运算 (处理单条/多条文本的维度对齐)
            # 确保向量是 numpy array 且处理可能出现的一维情况
            if doc_vecs.ndim == 1:
                doc_vecs = doc_vecs.reshape(1, -1)
                
            # 计算余弦相似度 (BGE encode 默认已 L2 归一化，直接点乘即可)
            similarities = np.dot(doc_vecs, query_vec)

            # 提取最高分及对应文本片段（用于审计防查）
            best_idx = int(np.argmax(similarities))
            max_score = float(similarities[best_idx])
            best_snippet = turn_texts[best_idx][:50] + "..." # 仅截取前50字防日志过大

            return {
                "topic_queried": clean_topic,
                "matched": max_score >= default_threshold,
                "max_score": round(max_score, 4),
                "best_match_snippet": best_snippet if max_score >= default_threshold else None,
                "status": "success"
            }

        except Exception as e:
            # 鲁棒性设计 4：异常隔离。大模型计算失败绝不能抛出异常导致外层 HTTP 500
            logger.error(f"[DynamicSearch] 检索主题 '{clean_topic}' 时发生异常: {str(e)}", exc_info=True)
            return {
                "topic_queried": clean_topic,
                "matched": False,
                "max_score": 0.0,
                "status": "error",
                "error_msg": str(e)
            }
