# intent_radar.py  ── V5.3 配置驱动架构（TEI 服务化增强）
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
#
# V5.1 变更摘要
# ─────────────────────────────────────────────────────────────
# ① 语义滑窗切片：_semantic_chunking() 替代旧的逐句编码，
#   长文本按句号→逗号→滑窗三级切分，Max 聚合相似度
# ② 新增 reload() 公开方法 + _reload_lock 互斥锁，支持线程安全热更新
# ③ 新增 dynamic_search() 零样本动态语义检索（RAG 增强）
#
# V5.3 变更摘要
# ─────────────────────────────────────────────────────────────
# ① BGE-M3 服务化：新增 BgeHttpClient，调用 TEI /embed 端点
# ② 三级降级链：TEI HTTP → 进程内 BGEM3FlagModel → _FallbackEncoder
# ③ 参数变更：bge_model_name → bge_service_url + bge_model_name
# ④ 环境变量：MODEL_BGE_PATH → BGE_SERVICE_URL（默认 http://localhost:8080）
# ============================================================

from __future__ import annotations

import os
import threading
from typing import ClassVar, Optional, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config_topics import TOPIC_REGISTRY, TopicDefinition

try:
    from FlagEmbedding import BGEM3FlagModel
    _BGEM3_AVAILABLE = True
except ImportError:
    _BGEM3_AVAILABLE = False
    BGEM3FlagModel = None  # type: ignore

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

import re
_RE_SENTENCE_BOUNDARY = re.compile(r"[。！？；\n.!?]+")
_RE_CLAUSE_BOUNDARY = re.compile(r"[，、,]")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 语义滑窗切片（Semantic Sliding Window）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _semantic_chunking(text: str, max_chunk_length: int = 80) -> list[str]:
    """
    动态语义切片：优先按句号切分，超长再按逗号切分，极限情况才用滑窗。
    """
    if not text.strip():
        return []

    # 第一层：按大边界（完整句子）切分
    raw_sentences = _RE_SENTENCE_BOUNDARY.split(text)
    chunks = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chunk_length:
            chunks.append(sentence)
        else:
            # 第二层：超长句按小边界（逗号）切分合并
            clauses = _RE_CLAUSE_BOUNDARY.split(sentence)
            current_chunk = ""
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                # 如果加上这个子句还不超长，就合并
                if len(current_chunk) + len(clause) + 1 <= max_chunk_length:
                    current_chunk = f"{current_chunk}，{clause}" if current_chunk else clause
                else:
                    # 如果当前块有内容，先保存
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # 极限兜底：如果单个子句本身就超长，启用大尺度机械滑窗
                    if len(clause) > max_chunk_length:
                        window = 60
                        stride = 30
                        for i in range(0, len(clause), stride):
                            sub_chunk = clause[i:i+window]
                            if len(sub_chunk) >= 10: 
                                chunks.append(sub_chunk)
                        current_chunk = ""
                    else:
                        current_chunk = clause

            if current_chunk:
                chunks.append(current_chunk)

    # 过滤掉极短的无意义切片（< 5个字）
    return [c for c in chunks if len(c) >= 5]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEI HTTP 客户端（BGE-M3 服务化调用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BgeHttpClient:
    """
    调用 HuggingFace TEI (Text Embeddings Inference) 的 /embed 端点，
    获取 BGE-M3 dense 向量。

    接口契约：encode() 返回 {"dense_vecs": np.ndarray}，
    与 BGEM3FlagModel.encode() 的 dense 输出格式一致。

    TEI /embed 请求格式：
        POST /embed  {"inputs": [...], "normalize": true, "truncate": true}
    TEI /embed 响应格式：
        [[0.01, -0.02, ...], [0.03, 0.04, ...]]  (float[][])
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        调用 TEI /embed 端点获取 dense 向量。

        分批发送请求，每批最多 batch_size 条文本。
        返回 {"dense_vecs": np.ndarray}，shape = (len(texts), dim)。
        """
        if not texts:
            return {"dense_vecs": np.array([], dtype=np.float32)}

        all_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._encode_batch(batch)
            all_vecs.append(vecs)

        return {"dense_vecs": np.vstack(all_vecs)} if all_vecs else {
            "dense_vecs": np.array([], dtype=np.float32)
        }

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """单批编码，带重试。"""
        import time

        payload = {
            "inputs": texts,
            "normalize": True,
            "truncate": True,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    f"{self._service_url}/embed",
                    json=payload,
                    timeout=self._timeout,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                # TEI 返回 float[][]，直接转为 numpy
                return np.array(data, dtype=np.float32)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[BgeHttpClient] /embed 请求失败 "
                    f"(尝试 {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避

        raise RuntimeError(
            f"[BgeHttpClient] /embed 请求在 {self._max_retries} 次重试后仍失败: "
            f"{last_error}"
        )

    def health(self) -> bool:
        """检查 TEI 服务健康状态。"""
        try:
            resp = requests.get(
                f"{self._service_url}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False


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
    配置驱动的多语言向量锚点雷达，进程内单例，线程安全。

    初始化流程
    ─────────────────────────────────────────────────────────
    1. 从 TOPIC_REGISTRY 中收集所有主题的 bge_anchors
    2. 将全部锚点拼接后一次性批量编码（离线向量化）
    3. 按 topic_id 切片存储为 numpy 矩阵
    4. 记录每个 topic_id 的触发阈值到 _threshold_map

    V5.1 变更
    ─────────────────────────────────────────────────────────
    - _compute_raw_scores 改用语义滑窗 + Max 聚合（替代全文本编码）
    - 新增 reload() + _reload_lock，支持配置热更新
    - 新增 dynamic_search()，零样本动态语义检索

    V5.3 变更
    ─────────────────────────────────────────────────────────
    - 编码器三级降级链：TEI HTTP → 进程内 BGEM3FlagModel → _FallbackEncoder
    - 新增 bge_service_url 参数，优先调用 TEI 服务
    - 环境变量 BGE_SERVICE_URL 优先于参数

    扩展方法
    ─────────────────────────────────────────────────────────
    在 config_topics.TOPIC_REGISTRY 中追加新主题 → 自动生效，
    无需修改本文件任何代码。
    """

    _instance: ClassVar[Optional["IntentRadar"]] = None
    _lock:     ClassVar[threading.Lock]          = threading.Lock()
    _reload_lock: ClassVar[threading.Lock]       = threading.Lock()

    def __init__(
        self,
        model_name:      str   = "BAAI/bge-m3",
        use_fp16:        bool  = True,
        batch_size:      int   = 32,
        registry:        dict[str, TopicDefinition] = TOPIC_REGISTRY,
        bge_service_url: str | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._registry   = registry  # 保存注册表引用，供 reload() 使用

        # ── 环境变量覆盖 ───────────────────────────────────────
        _bge_service_url = os.getenv("BGE_SERVICE_URL", bge_service_url or "")

        # ── 编码器三级降级链 ──────────────────────────────────
        # Level 1: TEI HTTP 服务（生产环境首选，模型独立部署）
        # Level 2: 进程内 BGEM3FlagModel（TEI 不可用时的降级）
        # Level 3: _FallbackEncoder（无 GPU / CI 测试环境兜底）
        self._encoder_name: str = "fallback"

        if _bge_service_url and _REQUESTS_AVAILABLE:
            try:
                client = BgeHttpClient(service_url=_bge_service_url)
                if client.health():
                    self._model = client
                    self._is_fallback = False
                    self._encoder_name = "tei_http"
                    logger.info(f"[IntentRadar] 使用 TEI HTTP 服务: {_bge_service_url}")
                else:
                    logger.warning(
                        f"[IntentRadar] TEI 服务健康检查失败: {_bge_service_url}，"
                        f"降级到进程内模型"
                    )
            except Exception as e:
                logger.warning(
                    f"[IntentRadar] TEI 客户端初始化异常: {e}，"
                    f"降级到进程内模型"
                )

        # Level 2: 进程内 BGEM3FlagModel（TEI 不可用 或 未配置时）
        if self._encoder_name == "fallback" and _BGEM3_AVAILABLE and BGEM3FlagModel is not None:
            self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
            self._is_fallback = False
            self._encoder_name = "in_process"
            logger.info(f"[IntentRadar] 使用进程内 BGEM3FlagModel: {model_name}")

        # Level 3: _FallbackEncoder（无 GPU / CI 测试环境）
        if self._encoder_name == "fallback":
            import warnings
            warnings.warn(
                "[IntentRadar] TEI 服务不可用且 FlagEmbedding 未安装，"
                "使用 N-gram 降级编码器。",
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
        model_name:      str  = "BAAI/bge-m3",
        use_fp16:        bool = True,
        bge_service_url: str | None = None,
    ) -> "IntentRadar":
        """双重检查锁定，线程安全单例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name      = model_name,
                        use_fp16        = use_fp16,
                        bge_service_url = bge_service_url,
                    )
        return cls._instance

    # ── 热更新 ──────────────────────────────────────────────

    def reload(self, registry: dict[str, TopicDefinition] | None = None) -> None:
        """
        热更新：全量重建锚点矩阵和阈值映射。

        当配置发生变更（如新增/删除主题、修改 bge_anchors 或 threshold）
        时调用此方法。使用互斥锁保证线程安全——在重建期间，
        并发的 detect_batch 调用会等待完成后再读取新矩阵。

        Parameters
        ----------
        registry : 新的注册表。若为 None 则使用初始化时的注册表。
        """
        new_registry = registry if registry is not None else self._registry
        with self._reload_lock:
            self._registry = new_registry
            self._threshold_map = {
                tid: td.threshold
                for tid, td in new_registry.items()
            }
            self._anchor_matrices.clear()
            self._vectorize_from_registry(new_registry)

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
        语义滑窗 + Max 聚合相似度计算。

        算法：
        1. 对每个输入文本执行滑窗切片（短文本不切片）
        2. 所有切片扁平化后一次性批量编码（最大化 GPU 利用率）
        3. 对每个原始文本，取其所有切片与锚点矩阵的相似度 Max 值

        优势：
        - 长语音中核心意图不会被周围废料稀释向量表示
        - 高浓度语义片段能瞬间击穿阈值
        """
        if not texts or not self._anchor_matrices:
            return [{} for _ in texts]

        # ── Step 1: 对每个文本执行滑窗切片 ───────────────────
        all_chunks: list[str] = []              # 扁平化的全部切片
        chunk_map:   list[list[int]] = []       # chunk_map[i] = 该文本对应的切片索引范围 [start, end)

        for text in texts:
            chunks = _semantic_chunking(text)
            start = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_map.append((start, len(all_chunks)))

        if not all_chunks:
            return [{} for _ in texts]

        # ── Step 2: 一次性批量编码（GPU 利用率最大化）───────
        encoded = self._model.encode(
            all_chunks,
            batch_size=min(self._batch_size, len(all_chunks)),
            max_length=512,
        )
        query_vecs: np.ndarray = encoded["dense_vecs"]

        # ── Step 3: 对每个 topic_id 计算相似度矩阵 + Max 聚合 ──
        all_scores: list[dict[str, float]] = [{} for _ in texts]

        for topic_id, anchor_mat in self._anchor_matrices.items():
            # (total_chunks × anchor_count) 相似度矩阵
            sim_mat = cosine_similarity(query_vecs, anchor_mat)  # (C, A)

            # 对每个原始文本，取其所有切片的最大相似度
            for text_idx, (start, end) in enumerate(chunk_map):
                if start >= end:
                    continue
                chunk_sims = sim_mat[start:end]  # (num_chunks_for_this_text, A)
                max_sim = float(chunk_sims.max())
                all_scores[text_idx][topic_id] = round(max_sim, 4)

        return all_scores
    

    # 新增语义检索 (支持 Top-10 滑动窗口完整召回)
    def dynamic_search(
        self, 
        search_chunks: list[str], 
        dynamic_topic: str, 
        default_threshold: float = 0.65,
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        零样本动态语义检索（RAG 增强版）
        """
        clean_topic = dynamic_topic.strip() if dynamic_topic else ""
        if not search_chunks or not clean_topic:
            return {
                "topic_queried": clean_topic,
                "matched": False, 
                "max_score": 0.0,
                "top_matches": [],
                "status": "skipped_due_to_empty_input"
            }

        try:
            # 1. 构造检索向量
            query_instruction = f"为这个句子生成表示以用于检索相关文章：{clean_topic}"
            query_vec = self._model.encode([query_instruction])["dense_vecs"][0]
            
            # 2. 对所有滑动窗口块进行向量化
            encoded_docs = self._model.encode(search_chunks)
            
            # 增加字典解析：兼容 BGE-M3 的字典输出和传统模型的数组输出
            if isinstance(encoded_docs, dict) and 'dense_vecs' in encoded_docs:
                doc_vecs = encoded_docs['dense_vecs']
            else:
                doc_vecs = encoded_docs
                
            if doc_vecs.ndim == 1:   # ✅ 现在 doc_vecs 是纯粹的 Numpy 数组了
                doc_vecs = doc_vecs.reshape(1, -1)
                
            # 3. 计算余弦相似度矩阵
            similarities = np.dot(doc_vecs, query_vec)

            # =========================================================
            # 【RAG 优化方向三：召回阈值以上的 Top-K 完整对话】
            # =========================================================
            # 找出所有大于等于安全阈值的索引
            valid_indices = np.where(similarities >= default_threshold)[0]
            
            top_matches = []
            if len(valid_indices) > 0:
                # 按相似度从高到低排序
                sorted_valid_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
                # 截取前 K 个
                top_indices = sorted_valid_indices[:top_k]
                
                # 组装返回结果，直接返回带说话人的完整上下文
                for idx in top_indices:
                    top_matches.append({
                        "score": round(float(similarities[idx]), 4),
                        "chunk_text": search_chunks[idx]  # 直接返回完整的原汁原味的对话块
                    })

            # 计算全局最高分
            max_score = float(np.max(similarities)) if len(similarities) > 0 else 0.0

            return {
                "topic_queried": clean_topic,
                "matched": len(top_matches) > 0,
                "max_score": round(max_score, 4),
                "top_matches": top_matches,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"[DynamicSearch] 检索主题 '{clean_topic}' 时发生异常: {str(e)}", exc_info=True)
            return {
                "topic_queried": clean_topic,
                "matched": False,
                "max_score": 0.0,
                "top_matches": [],
                "status": "error",
                "error_msg": str(e)
            }
