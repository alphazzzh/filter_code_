"""
LTP 微服务 HTTP 客户端
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用于替代 stage_two_pipeline.py 中进程内 _LtpBackend，
将 NLP 推理卸载到独立微服务，实现横向扩容。

用法：
    from ltp_service.client import LtpHttpClient

    backend = LtpHttpClient(base_url="http://localhost:8900")
    result = backend.analyze("你好世界")
    # → {"tokens": ["你好", "世界"], "dep": [...], "ner": [...], "pos": [...]}

与 _LtpBackend.analyze 返回格式兼容，可直接注入 StageTwoPipeline。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("ltp_service.client")


class LtpHttpClient:
    """
    LTP 微服务 HTTP 客户端。

    特性：
      - httpx 同步+异步双模式（同步用 httpx.Client，异步用 httpx.AsyncClient）
      - 自动重试 + 超时保护
      - 返回格式与 _LtpBackend.analyze 对齐
      - 支持 /health 健康检查

    V5.5 变更
    ─────────────────────────────────────────────────────────
    - 新增 AsyncClient，提供 analyze_async / analyze_batch_async 方法
    - 同步方法完全保留，向后兼容
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._base_url = (base_url or os.getenv(
            "LTP_SERVICE_URL", "http://localhost:8900"
        )).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        # 同步客户端（不依赖 Event Loop，保持单例）
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        # 异步客户端：按 Event Loop 绑定，跨循环绝对安全
        self._async_clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

    @property
    def name(self) -> str:
        return "ltp_http"

    # ── 同步方法（完全保留，向后兼容）─────────────────────────

    def analyze(self, text: str) -> dict[str, Any]:
        """
        单条文本分析，返回格式与 _LtpBackend.analyze 对齐：
          {"tokens": [...], "dep": [...], "ner": [...], "pos": [...]}
        """
        result = self._call_analyze([text])
        if not result:
            return {"tokens": [], "dep": [], "ner": [], "pos": []}
        return self._convert_single(result[0])

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量文本分析，返回等长结果列表"""
        results = self._call_analyze(texts)
        return [self._convert_single(r) for r in results]

    # ── 异步方法（V5.5 新增）─────────────────────────────────

    def _get_async_client(self) -> httpx.AsyncClient:
        """获取当前 Event Loop 绑定的 httpx.AsyncClient（跨循环绝对安全）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("必须在异步上下文中获取 AsyncClient")

        if loop not in self._async_clients:
            # 顺手清理已关闭 loop 的残留，防止内存泄漏
            closed_loops = [l for l in self._async_clients if l.is_closed()]
            for l in closed_loops:
                del self._async_clients[l]

            self._async_clients[loop] = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=5.0),
            )

        return self._async_clients[loop]

    async def analyze_async(self, text: str) -> dict[str, Any]:
        """异步单条文本分析。"""
        result = await self._call_analyze_async([text])
        if not result:
            return {"tokens": [], "dep": [], "ner": [], "pos": []}
        return self._convert_single(result[0])

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]:
        """异步批量文本分析。"""
        results = await self._call_analyze_async(texts)
        return [self._convert_single(r) for r in results]

    async def _call_analyze_async(self, texts: list[str]) -> list[dict[str, Any]]:
        """异步调用 /analyze 接口，带重试。"""
        client = self._get_async_client()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post("/analyze", json={"texts": texts})
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
            except httpx.HTTPStatusError as e:
                last_exc = e
                logger.warning(
                    "[AsyncClient] /analyze 返回 %d (attempt %d/%d): %s",
                    e.response.status_code,
                    attempt + 1, self._max_retries + 1,
                    e.response.text[:200],
                )
                if 400 <= e.response.status_code < 500:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "[AsyncClient] 连接失败 (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, e,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_exc = e
                logger.error("[AsyncClient] 未知异常: %s", e)
                raise

        raise ConnectionError(
            f"LTP 微服务异步调用失败（重试 {self._max_retries} 次）: {last_exc}"
        )

    async def health_async(self) -> dict[str, Any]:
        """异步健康检查。"""
        try:
            client = self._get_async_client()
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # ── 健康检查 ──────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """调用 /health 端点，返回服务状态"""
        try:
            resp = self._client.get("/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # ── 内部实现 ──────────────────────────────────────────────

    def _call_analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """调用 /analyze 接口，带重试"""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    "/analyze",
                    json={"texts": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
            except httpx.HTTPStatusError as e:
                last_exc = e
                logger.warning(
                    "[Client] /analyze 返回 %d (attempt %d/%d): %s",
                    e.response.status_code,
                    attempt + 1, self._max_retries + 1,
                    e.response.text[:200],
                )
                # 4xx 不重试
                if 400 <= e.response.status_code < 500:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "[Client] 连接失败 (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, e,
                )
            except Exception as e:
                last_exc = e
                logger.error("[Client] 未知异常: %s", e)
                raise

        raise ConnectionError(
            f"LTP 微服务调用失败（重试 {self._max_retries} 次）: {last_exc}"
        )

    @staticmethod
    def _convert_single(result: dict[str, Any]) -> dict[str, Any]:
        """
        将微服务 SingleResult 格式转换为 _LtpBackend.analyze 返回格式。

        微服务格式:
          {
            "seg": ["你好", "世界"],
            "pos": ["i", "n"],
            "dep": [{"head": 2, "label": "SBV"}, ...],
            "ner": [{"text": "张三", "label": "Nh"}, ...]
          }

        _LtpBackend 格式:
          {
            "tokens": ["你好", "世界"],
            "dep": [(1, "SBV"), ...],       # (head_index_0based, label)
            "ner": [("张三", "Nh"), ...],   # (text, label)
            "pos": ["i", "n"]              # V5.3 新增
          }
        """
        # tokens ← seg
        tokens = result.get("seg", [])

        # pos 直接透传
        pos = result.get("pos", [])

        # dep: [{head: int, label: str}] → [(head-1, label)]
        dep_raw = result.get("dep", [])
        dep = []
        for d in dep_raw:
            if isinstance(d, dict):
                head = d.get("head", 0)
                label = d.get("label", "UNK")
                # 微服务返回的 head 是 1-based（LTP 原始），
                # _LtpBackend.analyze 转为 0-based: head - 1
                dep.append((head - 1, label))

        # ner: [{text: str, label: str}] → [(text, label)]
        ner_raw = result.get("ner", [])
        ner = []
        for n in ner_raw:
            if isinstance(n, dict):
                ner.append((n.get("text", ""), n.get("label", "")))

        return {"tokens": tokens, "pos": pos, "dep": dep, "ner": ner}

    # ── 生命周期 ──────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        """关闭所有 Event Loop 绑定的异步客户端连接池。"""
        for loop, client in list(self._async_clients.items()):
            try:
                if not loop.is_closed():
                    await client.aclose()
            except Exception:
                pass  # loop 已关闭，丢弃引用即可
        self._async_clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()
        await self.aclose()
