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
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

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

    V5.4: 迁移 requests → httpx.Client，与 LtpHttpClient 统一 HTTP 库，
          减少依赖项，同时为未来全链路异步化（httpx.AsyncClient）做准备。
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._max_retries = max_retries
        self._degraded = False  # 降级标记：TEI 服务不可用时置 True
        # httpx.Client 内置连接池复用（HTTP/1.1 Keep-Alive + HTTP/2 多路复用），
        # 替代 requests.Session，减少依赖项
        self._client = httpx.Client(
            base_url=self._service_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers={"Content-Type": "application/json"},
        )

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
        """单批编码，带重试。重试耗尽后返回零向量（降级），不阻断流水线。"""
        import time

        payload = {
            "inputs": texts,
            "normalize": True,
            "truncate": True,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post("/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
                # TEI 返回 float[][]，直接转为 numpy
                return np.array(data, dtype=np.float32)

            except httpx.HTTPStatusError as e:
                last_error = e
                # 4xx 不重试，但也不抛异常，直接降级
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f"[BgeHttpClient] /embed 请求被拒 "
                        f"(status={e.response.status_code}): {e.response.text[:200]}"
                    )
                    break
                logger.warning(
                    f"[BgeHttpClient] /embed 请求失败 "
                    f"(尝试 {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    f"[BgeHttpClient] /embed 连接失败 "
                    f"(尝试 {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))

            except Exception as e:
                last_error = e
                logger.error(f"[BgeHttpClient] /embed 未知异常: {e}")
                break

        # ── 降级：返回零向量，不阻断流水线 ──
        # 调用方通过相似度阈值（≥0.65）自然过滤零向量，不会产生误命中
        self._degraded = True
        logger.warning(
            f"[BgeHttpClient] /embed 降级：重试耗尽后返回零向量 "
            f"(batch_size={len(texts)}, last_error={last_error})"
        )
        # 使用 1024 维（BGE-M3 dense 输出维度），零向量经 normalize 后范数为 0，
        # 与任何锚点的余弦相似度均为 0，不会误触发任何意图
        return np.zeros((len(texts), 1024), dtype=np.float32)

    def health(self) -> bool:
        """检查 TEI 服务健康状态。"""
        try:
            resp = self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """关闭同步客户端连接池。"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEI 异步 HTTP 客户端（全链路异步化核心）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AsyncBgeHttpClient:
    """
    调用 HuggingFace TEI /embed 端点的异步客户端。

    使用 httpx.AsyncClient，单线程即可处理数千并发请求，
    彻底消除 ThreadPoolExecutor + 同步 HTTP 的上下文切换开销。

    接口契约与 BgeHttpClient 一致，所有方法均为 async。
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._max_retries = max_retries
        self._degraded = False
        self._client = httpx.AsyncClient(
            base_url=self._service_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers={"Content-Type": "application/json"},
        )

    async def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """异步批量编码，分批发送请求。"""
        if not texts:
            return {"dense_vecs": np.array([], dtype=np.float32)}

        all_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = await self._encode_batch(batch)
            all_vecs.append(vecs)

        return {"dense_vecs": np.vstack(all_vecs)} if all_vecs else {
            "dense_vecs": np.array([], dtype=np.float32)
        }

    async def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """异步单批编码，带重试。重试耗尽后返回零向量（降级），不阻断流水线。"""
        import asyncio

        payload = {
            "inputs": texts,
            "normalize": True,
            "truncate": True,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post("/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return np.array(data, dtype=np.float32)

            except httpx.HTTPStatusError as e:
                last_error = e
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f"[AsyncBgeHttpClient] /embed 请求被拒 "
                        f"(status={e.response.status_code}): {e.response.text[:200]}"
                    )
                    break
                logger.warning(
                    f"[AsyncBgeHttpClient] /embed 请求失败 "
                    f"(尝试 {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    f"[AsyncBgeHttpClient] /embed 连接失败 "
                    f"(尝试 {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

            except Exception as e:
                last_error = e
                logger.error(f"[AsyncBgeHttpClient] /embed 未知异常: {e}")
                break

        # ── 降级：返回零向量，不阻断流水线 ──
        self._degraded = True
        logger.warning(
            f"[AsyncBgeHttpClient] /embed 降级：重试耗尽后返回零向量 "
            f"(batch_size={len(texts)}, last_error={last_error})"
        )
        return np.zeros((len(texts), 1024), dtype=np.float32)

    async def health(self) -> bool:
        """异步健康检查。"""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """关闭异步客户端连接池。"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


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

    V5.4 变更
    ─────────────────────────────────────────────────────────
    - BgeHttpClient 迁移 requests → httpx.Client
    - 熔断降级：微服务失败返回零向量/降级到 RuleBasedFallback

    V5.5 变更
    ─────────────────────────────────────────────────────────
    - 新增 AsyncBgeHttpClient + 异步工厂 get_async_instance()
    - 新增 async detect_batch / dynamic_search_batch / score_batch
    - 同步方法完全保留，向后兼容

    扩展方法
    ─────────────────────────────────────────────────────────
    在 config_topics.TOPIC_REGISTRY 中追加新主题 → 自动生效，
    无需修改本文件任何代码。
    """

    _instance: ClassVar[Optional["IntentRadar"]] = None
    _async_instance: ClassVar[Optional["IntentRadar"]] = None
    _lock:     ClassVar[threading.Lock]          = threading.Lock()
    _async_lock: ClassVar[threading.Lock]        = threading.Lock()
    _reload_lock: ClassVar[threading.Lock]       = threading.Lock()

    def __init__(
        self,
        model_name:      str   = "BAAI/bge-m3",
        use_fp16:        bool  = True,
        batch_size:      int   = 32,
        registry:        dict[str, TopicDefinition] = TOPIC_REGISTRY,
        bge_service_url: str | None = None,
        *,
        _async: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._registry   = registry  # 保存注册表引用，供 reload() 使用
        self._is_async   = _async    # 标记是否使用异步编码器

        # ── 环境变量覆盖 ───────────────────────────────────────
        _bge_service_url = os.getenv("BGE_SERVICE_URL", bge_service_url or "")

        # ── 编码器三级降级链 ──────────────────────────────────
        # Level 1: TEI HTTP 服务（生产环境首选，模型独立部署）
        # Level 2: 进程内 BGEM3FlagModel（TEI 不可用时的降级）
        # Level 3: _FallbackEncoder（无 GPU / CI 测试环境兜底）
        self._encoder_name: str = "fallback"

        if _bge_service_url and _HTTPX_AVAILABLE:
            try:
                if _async:
                    client = AsyncBgeHttpClient(service_url=_bge_service_url)
                    # 异步实例不做健康检查（需要 event loop，__init__ 可能在同步上下文）
                    self._model = client
                    self._is_fallback = False
                    self._encoder_name = "tei_async_http"
                    logger.info(f"[IntentRadar] 使用 TEI 异步 HTTP 服务: {_bge_service_url}")
                else:
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
        if self._is_async and self._encoder_name == "tei_async_http":
            # 异步实例：锚点向量化需要 event loop，延迟到首次调用时执行
            self._anchor_vectors_initialized = False
        else:
            self._vectorize_from_registry(registry)
            self._anchor_vectors_initialized = True

    # ── 单例工厂 ──────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_name:      str  = "BAAI/bge-m3",
        use_fp16:        bool = True,
        bge_service_url: str | None = None,
    ) -> "IntentRadar":
        """双重检查锁定，线程安全单例（同步模式）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name      = model_name,
                        use_fp16        = use_fp16,
                        bge_service_url = bge_service_url,
                    )
        return cls._instance

    @classmethod
    def get_async_instance(
        cls,
        model_name:      str  = "BAAI/bge-m3",
        use_fp16:        bool = True,
        bge_service_url: str | None = None,
    ) -> "IntentRadar":
        """双重检查锁定，线程安全单例（异步模式，使用 AsyncBgeHttpClient）。"""
        if cls._async_instance is None:
            with cls._async_lock:
                if cls._async_instance is None:
                    cls._async_instance = cls(
                        model_name      = model_name,
                        use_fp16        = use_fp16,
                        bge_service_url = bge_service_url,
                        _async          = True,
                    )
        return cls._async_instance

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

    async def _vectorize_from_registry_async(
        self, registry: dict[str, TopicDefinition]
    ) -> None:
        """异步版锚点向量化，在首次异步调用时执行。"""
        all_sentences: list[str] = []
        slice_map: dict[str, tuple[int, int]] = {}

        for topic_id, topic_def in registry.items():
            if not topic_def.bge_anchors:
                continue
            start = len(all_sentences)
            all_sentences.extend(topic_def.bge_anchors)
            slice_map[topic_id] = (start, len(all_sentences))

        if not all_sentences:
            return

        encoded = await self._model.encode(
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

    # ── 异步推理接口 ──────────────────────────────────────────

    async def detect_batch_async(self, texts: list[str]) -> list[list[str]]:
        """异步批量意图检测。仅当 _is_async=True 时可用。"""
        if not texts:
            return []
        raw_scores = await self._compute_raw_scores_async(texts)
        results: list[list[str]] = []
        for score_dict in raw_scores:
            triggered: list[str] = [
                topic_id
                for topic_id, score in score_dict.items()
                if score >= self._threshold_map.get(topic_id, 0.72)
            ]
            results.append(triggered)
        return results

    async def score_batch_async(self, texts: list[str]) -> list[dict[str, float]]:
        """异步批量评分。仅当 _is_async=True 时可用。"""
        return await self._compute_raw_scores_async(texts) if texts else []

    async def _compute_raw_scores_async(
        self, texts: list[str]
    ) -> list[dict[str, float]]:
        """异步版语义滑窗 + Max 聚合相似度计算。"""
        if not texts:
            return [{} for _ in texts]

        # 延迟初始化锚点向量
        if not self._anchor_vectors_initialized:
            await self._vectorize_from_registry_async(self._registry)
            self._anchor_vectors_initialized = True

        if not self._anchor_matrices:
            return [{} for _ in texts]

        # Step 1: 滑窗切片
        all_chunks: list[str] = []
        chunk_map: list[tuple[int, int]] = []
        for text in texts:
            chunks = _semantic_chunking(text)
            start = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_map.append((start, len(all_chunks)))

        if not all_chunks:
            return [{} for _ in texts]

        # Step 2: 异步批量编码
        encoded = await self._model.encode(
            all_chunks,
            batch_size=min(self._batch_size, len(all_chunks)),
            max_length=512,
        )
        query_vecs: np.ndarray = encoded["dense_vecs"]

        # Step 3: CPU 相似度计算（与同步版相同）
        all_scores: list[dict[str, float]] = [{} for _ in texts]
        for topic_id, anchor_mat in self._anchor_matrices.items():
            sim_mat = cosine_similarity(query_vecs, anchor_mat)
            for text_idx, (start, end) in enumerate(chunk_map):
                if start >= end:
                    continue
                chunk_sims = sim_mat[start:end]
                max_sim = float(chunk_sims.max())
                all_scores[text_idx][topic_id] = round(max_sim, 4)

        return all_scores

    async def dynamic_search_batch_async(
        self,
        search_chunks: list[str],
        dynamic_topics: list[str],
        default_threshold: float = 0.65,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """异步版多主题批量语义检索。"""
        clean_topics = [t.strip() for t in dynamic_topics if t and t.strip()]
        if not search_chunks or not clean_topics:
            return {
                "topic_queried": str(clean_topics) if clean_topics else "",
                "matched": False, "max_score": 0.0,
                "top_matches": [], "status": "skipped_due_to_empty_input"
            }

        try:
            # 1. 异步编码文档块
            encoded_docs = await self._model.encode(search_chunks)
            doc_vecs = encoded_docs['dense_vecs'] if isinstance(encoded_docs, dict) and 'dense_vecs' in encoded_docs else encoded_docs
            if doc_vecs.ndim == 1:
                doc_vecs = doc_vecs.reshape(1, -1)

            # 2. 异步编码主题指令
            query_instructions = [
                f"为这个句子生成表示以用于检索相关文章：{t}"
                for t in clean_topics
            ]
            query_vecs = (await self._model.encode(query_instructions))["dense_vecs"]

            # 3. 矩阵乘法（CPU）
            similarity_matrix = np.dot(doc_vecs, query_vecs.T)

            # 4. 逐主题提取最佳匹配
            best_result: dict[str, Any] | None = None
            for col_idx, topic in enumerate(clean_topics):
                similarities = similarity_matrix[:, col_idx]
                valid_indices = np.where(similarities >= default_threshold)[0]
                top_matches = []
                if len(valid_indices) > 0:
                    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
                    for idx in sorted_indices[:top_k]:
                        top_matches.append({
                            "score": round(float(similarities[idx]), 4),
                            "chunk_text": search_chunks[idx]
                        })
                max_score = float(np.max(similarities)) if len(similarities) > 0 else 0.0
                topic_result = {
                    "topic_queried": topic, "matched": len(top_matches) > 0,
                    "max_score": round(max_score, 4), "top_matches": top_matches,
                    "status": "success"
                }
                if best_result is None or max_score > best_result.get("max_score", 0.0):
                    best_result = topic_result

            return best_result  # type: ignore[return-value]

        except Exception as e:
            logger.error(f"[DynamicSearchBatchAsync] 异步批量检索异常 (topics={clean_topics}): {str(e)}", exc_info=True)
            return {
                "topic_queried": str(clean_topics), "matched": False,
                "max_score": 0.0, "top_matches": [],
                "status": "error", "error_msg": str(e)
            }

    async def dynamic_search_async(
        self,
        search_chunks: list[str],
        dynamic_topic: str,
        default_threshold: float = 0.65,
        top_k: int = 10
    ) -> dict[str, Any]:
        """异步版单主题语义检索（便捷接口）。"""
        return await self.dynamic_search_batch_async(
            search_chunks=search_chunks,
            dynamic_topics=[dynamic_topic],
            default_threshold=default_threshold,
            top_k=top_k,
        )
    

    # 新增语义检索 (支持 Top-10 滑动窗口完整召回)
    def dynamic_search(
        self, 
        search_chunks: list[str], 
        dynamic_topic: str, 
        default_threshold: float = 0.65,
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        零样本动态语义检索（RAG 增强版）—— 单主题便捷接口。
        内部委托给 dynamic_search_batch，向后兼容旧调用方。
        """
        batch_result = self.dynamic_search_batch(
            search_chunks=search_chunks,
            dynamic_topics=[dynamic_topic],
            default_threshold=default_threshold,
            top_k=top_k,
        )
        # dynamic_search_batch 返回最佳主题结果，直接透传
        return batch_result

    def dynamic_search_batch(
        self, 
        search_chunks: list[str], 
        dynamic_topics: list[str],
        default_threshold: float = 0.65,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        零样本动态语义检索（多主题批量版，RAG 增强版）

        核心优化：search_chunks 只编码 1 次，所有主题共享同一份文档向量。
        原先 N 个主题 = N 次 GPU 推理 → 现在 1 + 1 次（文档 1 次 + 主题 1 次）。

        返回得分最高的那个主题的结果（内部竞价）。

        Parameters
        ----------
        search_chunks    : 滑动窗口切块后的对话文本列表
        dynamic_topics   : 待检索的主题指令列表
        default_threshold: 语义匹配阈值
        top_k            : 每主题最多返回的匹配数

        Returns
        -------
        dict[str, Any]  得分最高主题的检索结果（与旧 dynamic_search 格式一致）
        """
        # 清洗主题列表
        clean_topics = [t.strip() for t in dynamic_topics if t and t.strip()]
        if not search_chunks or not clean_topics:
            return {
                "topic_queried": str(clean_topics) if clean_topics else "",
                "matched": False,
                "max_score": 0.0,
                "top_matches": [],
                "status": "skipped_due_to_empty_input"
            }

        try:
            # ── 1. 文档块只编码 1 次！（核心省算力点）────────────
            encoded_docs = self._model.encode(search_chunks)
            if isinstance(encoded_docs, dict) and 'dense_vecs' in encoded_docs:
                doc_vecs = encoded_docs['dense_vecs']
            else:
                doc_vecs = encoded_docs

            if doc_vecs.ndim == 1:
                doc_vecs = doc_vecs.reshape(1, -1)

            # ── 2. 批量编码所有主题指令 ──────────────────────────
            query_instructions = [
                f"为这个句子生成表示以用于检索相关文章：{t}"
                for t in clean_topics
            ]
            query_vecs = self._model.encode(query_instructions)["dense_vecs"]

            # ── 3. 矩阵乘法：(N个文本块, Dim) × (Dim, M个Topic) = (N, M) ──
            similarity_matrix = np.dot(doc_vecs, query_vecs.T)  # shape: (N, M)

            # ── 4. 逐主题提取最佳匹配，内部竞价 ──────────────────
            best_result: dict[str, Any] | None = None

            for col_idx, topic in enumerate(clean_topics):
                similarities = similarity_matrix[:, col_idx]  # (N,)

                # 阈值筛选 + Top-K 召回
                valid_indices = np.where(similarities >= default_threshold)[0]
                top_matches = []
                if len(valid_indices) > 0:
                    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
                    for idx in sorted_indices[:top_k]:
                        top_matches.append({
                            "score": round(float(similarities[idx]), 4),
                            "chunk_text": search_chunks[idx]
                        })

                max_score = float(np.max(similarities)) if len(similarities) > 0 else 0.0

                topic_result = {
                    "topic_queried": topic,
                    "matched": len(top_matches) > 0,
                    "max_score": round(max_score, 4),
                    "top_matches": top_matches,
                    "status": "success"
                }

                # 内部竞价：保留得分最高的主题
                if best_result is None or max_score > best_result.get("max_score", 0.0):
                    best_result = topic_result

            return best_result  # type: ignore[return-value]

        except Exception as e:
            logger.error(
                f"[DynamicSearchBatch] 批量检索异常 (topics={clean_topics}): {str(e)}",
                exc_info=True,
            )
            return {
                "topic_queried": str(clean_topics),
                "matched": False,
                "max_score": 0.0,
                "top_matches": [],
                "status": "error",
                "error_msg": str(e)
            }
