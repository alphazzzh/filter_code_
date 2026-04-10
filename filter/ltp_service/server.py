"""
LTP NLP 微服务 — 核心服务文件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
架构要点：
  ① FastAPI lifespan 钩子：LTP 模型全局单例，启动加载、关闭释放
  ② Dynamic Batching：asyncio 队列 + 后台消费协程
     - 请求到达 → 入队挂起（asyncio.Future）
     - 后台协程按「最大等待 50ms / 最大批次 32 条」聚合
     - 批量送入 LTP pipeline → 结果精准分发回各 Future
  ③ 兼容上游 stage_two_pipeline.py 的 _LtpBackend.analyze 返回格式

启动方式：
  uvicorn ltp_service.server:app --host 0.0.0.0 --port 8900

环境变量：
  MODEL_LTP_PATH       LTP 模型路径（默认 /home/zzh/923/model/ltp_small）
  BATCH_MAX_WAIT_MS    批次最大等待毫秒（默认 50）
  BATCH_MAX_SIZE       批次最大条数（默认 32）
  LTP_REQUEST_TIMEOUT_SEC  单请求超时秒数（默认 30）
  LTP_MAX_CONCURRENT   全局并发上限（默认 200）
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import (
    BATCH_MAX_SIZE,
    BATCH_MAX_WAIT_MS,
    LTP_MODEL_PATH,
    LTP_TASKS,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT_SEC,
)

logger = logging.getLogger("ltp_service")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic 请求/响应模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnalyzeRequest(BaseModel):
    """批量分析请求"""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="待分析文本列表，单次最多 64 条",
    )


class SingleResult(BaseModel):
    """单条文本分析结果（与 _LtpBackend.analyze 返回格式对齐）"""
    seg: list[str]                    = Field(description="分词结果")
    pos: list[str]                    = Field(description="词性标注结果")
    dep: list[dict[str, Any]]         = Field(description="依存句法，每项 {head: int, label: str}")
    ner: list[dict[str, Any]]         = Field(description="命名实体，每项 {text: str, label: str}")


class AnalyzeResponse(BaseModel):
    """批量分析响应"""
    results: list[SingleResult]       = Field(description="与请求 texts 等长的结果列表")
    batch_size: int                   = Field(description="本批次实际聚合条数")
    latency_ms: float                 = Field(description="LTP 推理耗时（毫秒）")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    queue_length: int
    active_batches: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dynamic Batching 核心
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _BatchItem:
    """队列中的单个请求条目"""
    __slots__ = ("text", "future")

    def __init__(self, text: str, future: asyncio.Future):
        self.text = text
        self.future = future


class DynamicBatcher:
    """
    基于 asyncio 的异步批处理队列。

    工作流程：
      1. 外部调用 submit(texts) → 为每条文本创建 Future → 挂起等待
      2. 后台 _consumer 协程持续消费队列
      3. 当满足「等待超时」或「批次满」时，聚合当前所有待处理条目
      4. 调用 _predict_batch 统一送入 LTP
      5. 将结果逐一分发回各 Future，唤醒等待者
    """

    def __init__(self, ltp_instance: Any) -> None:
        self._ltp = ltp_instance
        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        self._active_batches: int = 0
        self._shutdown: bool = False

    # ── 启动 / 关闭 ──────────────────────────────────────────

    def start(self) -> None:
        """启动后台消费协程"""
        self._consumer_task = asyncio.create_task(self._consumer())
        logger.info("[Batcher] 后台消费协程已启动 (max_wait=%dms, max_size=%d)",
                     BATCH_MAX_WAIT_MS, BATCH_MAX_SIZE)

    async def stop(self) -> None:
        """优雅关闭：设置标志 → 等待消费协程结束 → 拒绝剩余请求"""
        self._shutdown = True
        if self._consumer_task:
            # 向队列推入哨兵，唤醒可能阻塞的消费者
            await self._queue.put(_BatchItem("", asyncio.get_event_loop().create_future()))
            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._consumer_task.cancel()
                logger.warning("[Batcher] 消费协程关闭超时，已取消")

        # 拒绝队列中所有待处理请求
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(RuntimeError("服务正在关闭"))
            except asyncio.QueueEmpty:
                break
        logger.info("[Batcher] 已关闭")

    # ── 外部接口 ──────────────────────────────────────────────

    async def submit(self, texts: list[str]) -> list[SingleResult]:
        """
        提交一批文本，返回对应的 Future 列表。
        调用方 await 每个 Future 即可获得结果。
        """
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future] = []
        for text in texts:
            future = loop.create_future()
            self._queue.put_nowait(_BatchItem(text, future))
            futures.append(future)

        # 等待所有结果，带超时保护
        results: list[SingleResult] = []
        for fut in futures:
            try:
                result = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT_SEC)
                results.append(result)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"NLP 推理超时（>{REQUEST_TIMEOUT_SEC}s），请稍后重试",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"NLP 推理异常: {e}",
                )
        return results

    @property
    def queue_length(self) -> int:
        return self._queue.qsize()

    @property
    def active_batches(self) -> int:
        return self._active_batches

    # ── 内部消费协程 ─────────────────────────────────────────

    async def _consumer(self) -> None:
        """后台消费：按时间/大小触发批次"""
        while not self._shutdown:
            batch: list[_BatchItem] = []

            # ── 等待第一个请求入队 ──
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch.append(first)
            except asyncio.TimeoutError:
                continue  # 空闲轮转

            # ── 聚合更多请求（超时或批次满即停） ──
            deadline = time.monotonic() + BATCH_MAX_WAIT_MS / 1000.0
            while len(batch) < BATCH_MAX_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    # 跳过关闭哨兵的空文本
                    if item.text == "" and self._shutdown:
                        # 把哨兵之前收集的也处理掉
                        if not item.future.done():
                            item.future.set_result(SingleResult(seg=[], pos=[], dep=[], ner=[]))
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            # ── 执行批量推理 ──
            self._active_batches += 1
            try:
                await self._predict_batch(batch)
            except Exception as e:
                logger.exception("[Batcher] 批量推理失败: %s", e)
                # 将异常分发给所有等待者
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
            finally:
                self._active_batches -= 1

    async def _predict_batch(self, batch: list[_BatchItem]) -> None:
        """
        将批次文本统一送入 LTP pipeline，结果分发回各 Future。
        在线程池中执行，避免阻塞事件循环。
        """
        texts = [item.text for item in batch]
        t0 = time.monotonic()

        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            None, self._run_pipeline, texts
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "[Batcher] 推理完成: batch=%d, latency=%.1fms",
            len(batch), elapsed_ms,
        )

        # 分发结果
        for item, result in zip(batch, raw_results):
            if not item.future.done():
                item.future.set_result(result)

    def _run_pipeline(self, texts: list[str]) -> list[SingleResult]:
        """
        同步调用 LTP pipeline，在线程池中执行。
        兼容新版/旧版 LTP 输出格式。
        """
        output = self._ltp.pipeline(texts, tasks=LTP_TASKS)

        results: list[SingleResult] = []
        n = len(texts)

        for i in range(n):
            # ── 分词 (cws) ──
            seg: list[str] = []
            if output.cws and i < len(output.cws):
                seg = output.cws[i]

            # ── 词性标注 (pos) ──
            pos: list[str] = []
            if output.pos and i < len(output.pos):
                pos = output.pos[i]

            # ── 依存句法 (dep) ──
            dep: list[dict[str, Any]] = []
            if output.dep and i < len(output.dep):
                dep_data = output.dep[i]
                if isinstance(dep_data, dict):
                    # 新版 LTP: {'head': [2, 0], 'label': ['SBV', 'HED']}
                    heads = dep_data.get("head", [])
                    labels = dep_data.get("label", [])
                    for h, l in zip(heads, labels):
                        dep.append({"head": h, "label": l.upper()})
                elif isinstance(dep_data, list):
                    # 旧版 LTP: [{'head': 2, 'label': 'SBV'}, ...]
                    for d in dep_data:
                        if isinstance(d, dict):
                            dep.append({
                                "head": d.get("head", 0),
                                "label": d.get("label", "UNK").upper(),
                            })

            # ── 命名实体识别 (ner) ──
            ner: list[dict[str, Any]] = []
            if output.ner and i < len(output.ner):
                ner_data = output.ner[i]
                if isinstance(ner_data, dict):
                    # 新版 LTP: {'label': ['Nh'], 'text': ['张三'], 'offset': [...]}
                    labels = ner_data.get("label", [])
                    texts_ner = ner_data.get("text", [])
                    for t, l in zip(texts_ner, labels):
                        ner.append({"text": t, "label": l})
                elif isinstance(ner_data, list):
                    # 旧版 LTP: [{'label': 'Nh', 'text': '张三'}, ...]
                    for d in ner_data:
                        if isinstance(d, dict):
                            ner.append({
                                "text": d.get("text", d.get("name", "")),
                                "label": d.get("label", d.get("tag", "")),
                            })

            results.append(SingleResult(seg=seg, pos=pos, dep=dep, ner=ner))

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 应用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 全局状态
_ltp_instance: Any = None
_batcher: DynamicBatcher | None = None
_concurrent_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子：启动时加载模型，关闭时释放资源"""
    global _ltp_instance, _batcher, _concurrent_semaphore

    # ── 启动阶段 ──
    logger.info("[Lifespan] 正在加载 LTP 模型: %s", LTP_MODEL_PATH)
    import os
    if not os.path.isdir(LTP_MODEL_PATH):
        logger.error("[Lifespan] LTP 模型路径不存在: %s", LTP_MODEL_PATH)
        raise RuntimeError(f"LTP 模型路径不存在: {LTP_MODEL_PATH}")

    from ltp import LTP  # type: ignore
    _ltp_instance = LTP(LTP_MODEL_PATH)
    logger.info("[Lifespan] LTP 模型加载完成")

    _batcher = DynamicBatcher(_ltp_instance)
    _batcher.start()

    _concurrent_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    logger.info("[Lifespan] 服务就绪 (max_concurrent=%d)", MAX_CONCURRENT_REQUESTS)

    yield

    # ── 关闭阶段 ──
    logger.info("[Lifespan] 正在关闭服务...")
    if _batcher:
        await _batcher.stop()
    _ltp_instance = None
    logger.info("[Lifespan] 资源已释放")


app = FastAPI(
    title="LTP NLP 微服务",
    description="基于 LTP 4.x 的分词/词性/依存句法/NER HTTP 微服务，内置 Dynamic Batching",
    version="1.0.0",
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 中间件：全局并发限制 + 请求日志
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.middleware("http")
async def concurrency_limit_middleware(request: Request, call_next):
    """全局并发信号量保护，超限直接 503"""
    if _concurrent_semaphore is None:
        return await call_next(request)

    acquired = _concurrent_semaphore.locked() and _concurrent_semaphore._value <= 0
    # 非阻塞快速检查
    try:
        await asyncio.wait_for(
            _concurrent_semaphore.acquire(), timeout=0.1
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"detail": "服务过载，请稍后重试"},
        )

    try:
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("[Middleware] %s %s → %d (%.1fms)",
                     request.method, request.url.path, response.status_code, elapsed_ms)
        return response
    finally:
        _concurrent_semaphore.release()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API 路由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    批量 NLP 分析接口。

    请求示例：
        POST /analyze
        {"texts": ["你好世界", "今天天气不错"]}

    响应示例：
        {
          "results": [
            {"seg": ["你好", "世界"], "pos": ["i", "n"], "dep": [...], "ner": [...]},
            ...
          ],
          "batch_size": 2,
          "latency_ms": 12.3
        }
    """
    if _batcher is None:
        raise HTTPException(status_code=503, detail="服务未就绪，模型尚未加载")

    t0 = time.monotonic()
    results = await _batcher.submit(req.texts)
    elapsed_ms = (time.monotonic() - t0) * 1000

    return AnalyzeResponse(
        results=results,
        batch_size=len(req.texts),
        latency_ms=round(elapsed_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点"""
    return HealthResponse(
        status="ok" if _ltp_instance is not None else "loading",
        model_loaded=_ltp_instance is not None,
        queue_length=_batcher.queue_length if _batcher else 0,
        active_batches=_batcher.active_batches if _batcher else 0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启动入口（开发用，生产请用 uvicorn 命令）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn
    from .config import HOST, PORT

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
    uvicorn.run(
        "ltp_service.server:app",
        host=HOST,
        port=PORT,
        workers=1,          # LTP 模型全局单例，多 worker 需多进程部署
        log_level="info",
        access_log=True,
    )
