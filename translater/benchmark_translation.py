"""
翻译微服务高并发压测脚本

用法：
    # 启动翻译服务
    cd translater && python api_translation.py

    # 另起终端运行压测
    python translater/benchmark_translation.py

    # 自定义参数
    python translater/benchmark_translation.py --concurrency 100 --total 500 --url http://localhost:8000

依赖：
    pip install httpx aiofire
    # 或
    pip install httpx  # 仅需 httpx，无额外强依赖

压测维度：
    1. 纯文本模式（content 为字符串，触发 CPU 卸载路径）
    2. 预解析模式（content 为 DialogueTurn 列表，跳过 CPU 卸载）
    3. 混合模式（随机选择两种模式）

指标输出：
    - 总耗时 / QPS
    - P50 / P90 / P99 / P999 延迟
    - 状态码分布（200 / 429 / 504 / 5xx）
    - 批处理命中率（通过日志估算）
"""

import asyncio
import argparse
import json
import random
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import httpx
except ImportError:
    raise SystemExit("请先安装 httpx：pip install httpx")


# ================================================================
# 测试数据模板
# ================================================================

SAMPLE_DIALOGUE_TEXT = """\
张三说：Hello, how are you doing today?
李四说：I'm doing great, thanks for asking! How about you?
张三说：Pretty good. I wanted to discuss the project timeline.
李四说：Sure, let's go over it. When is the deadline?
张三说：The deadline is next Friday. We need to finish the API integration.
李四说：That's tight. Let me check the current progress and get back to you.
张三说：OK, please also review the test coverage report I sent yesterday.
李四说：Will do. I'll send you my feedback by end of day.
"""

SAMPLE_DIALOGUE_LIST = [
    {"id": "bench_0001", "speaker": "Alice", "content": "The server is down again, can you check?"},
    {"id": "bench_0002", "speaker": "Bob",   "content": "I'll look into it right away."},
    {"id": "bench_0003", "speaker": "Alice", "content": "Please also check the database connection pool."},
    {"id": "bench_0004", "speaker": "Bob",   "content": "Sure, I suspect it might be a connection leak."},
    {"id": "bench_0005", "speaker": "Alice", "content": "Keep me posted on what you find."},
    {"id": "bench_0006", "speaker": "Bob",   "content": "Will do. I'll run diagnostics first."},
]


def build_request_payload(session_id: str, mode: str = "mixed") -> dict:
    """
    构建翻译请求 payload

    Args:
        session_id: 会话 ID
        mode: "text"（纯文本）/ "list"（预解析列表）/ "mixed"（随机）
    """
    if mode == "mixed":
        mode = random.choice(["text", "list"])

    if mode == "text":
        content = SAMPLE_DIALOGUE_TEXT
    else:
        content = SAMPLE_DIALOGUE_LIST

    return {
        "session_id": session_id,
        "data": {
            "session_id": session_id,
            "content": content,
            "language": "en",
            "start_time": "2026-04-14 10:00:00",
            "end_time": "2026-04-14 10:05:00",
            "duration": 300.0,
            "caller_number": 8613800138000,
            "called_number": 8613900139000,
            "caller_country_code": 86,
            "called_country_code": 86,
            "file": f"/data/calls/{session_id}.wav",
            "create_time": "2026-04-14 10:06:00",
            "cp": "test_cp",
        },
    }


# ================================================================
# 压测结果统计
# ================================================================

@dataclass
class BenchResult:
    """压测结果容器"""
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    status_codes: dict = field(default_factory=dict)
    latencies: List[float] = field(default_factory=list)  # 单位：秒
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def qps(self) -> float:
        elapsed = self.end_time - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (statistics.mean(self.latencies) * 1000) if self.latencies else 0.0

    def percentile_ms(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        idx = min(idx, len(sorted_lat) - 1)
        return sorted_lat[idx] * 1000

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  翻译服务压测报告",
            "=" * 60,
            f"  总请求数：      {self.total_requests}",
            f"  成功数：        {self.success_count}",
            f"  失败数：        {self.error_count}",
            f"  成功率：        {self.success_count / max(self.total_requests, 1) * 100:.1f}%",
            "-" * 60,
            f"  总耗时：        {self.end_time - self.start_time:.2f}s",
            f"  QPS：           {self.qps:.1f}",
            "-" * 60,
            f"  平均延迟：      {self.avg_latency_ms:.1f}ms",
            f"  P50 延迟：      {self.percentile_ms(50):.1f}ms",
            f"  P90 延迟：      {self.percentile_ms(90):.1f}ms",
            f"  P99 延迟：      {self.percentile_ms(99):.1f}ms",
            f"  P999 延迟：     {self.percentile_ms(99.9):.1f}ms",
            "-" * 60,
            "  状态码分布：",
        ]
        for code, count in sorted(self.status_codes.items()):
            lines.append(f"    {code}: {count} ({count / max(self.total_requests, 1) * 100:.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)


# ================================================================
# 压测核心逻辑
# ================================================================

async def single_request(
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    mode: str,
    result: BenchResult,
    semaphore: asyncio.Semaphore,
) -> None:
    """发送单个翻译请求并记录结果"""
    async with semaphore:
        payload = build_request_payload(session_id, mode)
        start = time.monotonic()

        try:
            resp = await client.post(url, json=payload, timeout=60.0)
            latency = time.monotonic() - start
            status = resp.status_code

            result.latencies.append(latency)
            result.status_codes[status] = result.status_codes.get(status, 0) + 1

            if 200 <= status < 300:
                result.success_count += 1
            else:
                result.error_count += 1
                # 打印前几条错误详情
                if result.error_count <= 5:
                    try:
                        body = resp.json()
                        logger_func = print
                        logger_func(f"  ❌ 请求失败 session_id={session_id} status={status} "
                                    f"message={body.get('message', 'N/A')}")
                    except Exception:
                        pass

        except Exception as exc:
            latency = time.monotonic() - start
            result.latencies.append(latency)
            result.error_count += 1
            result.status_codes["EXCEPTION"] = result.status_codes.get("EXCEPTION", 0) + 1
            if result.error_count <= 5:
                print(f"  ❌ 请求异常 session_id={session_id}: {exc}")


async def run_benchmark(
    url: str,
    concurrency: int,
    total: int,
    mode: str,
    ramp_up: float,
) -> BenchResult:
    """
    执行压测

    Args:
        url: 翻译服务地址（如 http://localhost:8000/api/translate）
        concurrency: 最大并发数
        total: 总请求数
        mode: 请求模式（text/list/mixed）
        ramp_up: 爬坡时间（秒），0 = 瞬时全部发出
    """
    result = BenchResult(total_requests=total)
    semaphore = asyncio.Semaphore(concurrency)

    endpoint = f"{url}/api/translate"
    print(f"\n🚀 压测启动：{endpoint}")
    print(f"   并发数={concurrency}  总请求={total}  模式={mode}  爬坡={ramp_up}s\n")

    # 构建所有请求任务
    tasks = []
    result.start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i in range(total):
            session_id = f"bench_{i:06d}"

            # 爬坡控制：均匀分散请求启动时间
            if ramp_up > 0 and i > 0:
                delay = ramp_up * i / total
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                single_request(client, endpoint, session_id, mode, result, semaphore)
            )
            tasks.append(task)

        # 等待全部请求完成
        await asyncio.gather(*tasks)

    result.end_time = time.monotonic()
    return result


# ================================================================
# 渐进式压测（阶梯加压）
# ================================================================

async def run_staircase_benchmark(
    url: str,
    steps: List[int],
    duration_per_step: float,
    mode: str,
) -> None:
    """
    阶梯加压压测：逐步提高并发，观察系统在哪个水位开始劣化

    Args:
        url: 服务地址
        steps: 并发数阶梯，如 [10, 50, 100, 200, 500]
        duration_per_step: 每个阶梯持续秒数
        mode: 请求模式
    """
    print("\n" + "=" * 60)
    print("  🏔️  阶梯加压压测模式")
    print("=" * 60)

    for idx, concurrency in enumerate(steps):
        print(f"\n--- 阶梯 {idx + 1}/{len(steps)}：并发={concurrency}，持续 {duration_per_step}s ---")

        # 估算该阶梯需要多少请求（按 QPS 估算，每秒约 concurrency 个请求）
        total = int(concurrency * duration_per_step)
        if total < 1:
            total = 1

        result = await run_benchmark(
            url=url,
            concurrency=concurrency,
            total=total,
            mode=mode,
            ramp_up=0,  # 阶梯内瞬时发出
        )

        # 简要输出每阶梯的关键指标
        print(f"  QPS={result.qps:.1f}  "
              f"P50={result.percentile_ms(50):.0f}ms  "
              f"P99={result.percentile_ms(99):.0f}ms  "
              f"成功={result.success_count}  "
              f"429={result.status_codes.get(429, 0)}  "
              f"504={result.status_codes.get(504, 0)}")

        # 如果 429 超过 30%，建议停止
        reject_rate = result.status_codes.get(429, 0) / max(result.total_requests, 1)
        if reject_rate > 0.3:
            print(f"\n  ⚠️  429 拒绝率 {reject_rate * 100:.0f}% > 30%，系统已过载，建议停止加压")
            break

        # 阶梯间休息
        if idx < len(steps) - 1:
            print("  阶梯间冷却 3s...")
            await asyncio.sleep(3)


# ================================================================
# CLI 入口
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="翻译服务高并发压测工具")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="翻译服务地址（默认 http://localhost:8000）")
    parser.add_argument("-c", "--concurrency", type=int, default=50,
                        help="最大并发数（默认 50）")
    parser.add_argument("-n", "--total", type=int, default=200,
                        help="总请求数（默认 200）")
    parser.add_argument("--mode", choices=["text", "list", "mixed"], default="mixed",
                        help="请求模式：text(纯文本) / list(预解析) / mixed(混合，默认)")
    parser.add_argument("--ramp-up", type=float, default=0,
                        help="爬坡时间（秒），0=瞬时全部发出（默认 0）")
    parser.add_argument("--staircase", action="store_true",
                        help="启用阶梯加压模式（忽略 -c/-n，使用内置阶梯）")
    parser.add_argument("--stair-steps", type=str, default="10,50,100,200,500",
                        help="阶梯并发数，逗号分隔（默认 10,50,100,200,500）")
    parser.add_argument("--stair-duration", type=float, default=10,
                        help="每阶梯持续秒数（默认 10）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.staircase:
        steps = [int(s.strip()) for s in args.stair_steps.split(",")]
        asyncio.run(run_staircase_benchmark(
            url=args.url,
            steps=steps,
            duration_per_step=args.stair_duration,
            mode=args.mode,
        ))
    else:
        result = asyncio.run(run_benchmark(
            url=args.url,
            concurrency=args.concurrency,
            total=args.total,
            mode=args.mode,
            ramp_up=args.ramp_up,
        ))
        print(result.summary())


if __name__ == "__main__":
    main()
