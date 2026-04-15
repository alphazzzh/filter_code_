"""
翻译微服务 API —— 高并发防御性重构版

核心改造：
1. 跨请求动态批处理 (Dynamic Batching)：asyncio.Queue + background_batch_worker
2. CPU 密集型任务卸载：parse_transcript_to_turns 通过 ThreadPoolExecutor 隔离
3. 全局限流与快速拒绝 (Load Shedding)：MAX_ACTIVE_REQUESTS 水位线 + 429
4. SLA 超时熔断：asyncio.wait_for 45s + 504
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from models_translation import TranslateRequest, TranslateResponse, DialogueTurn
from translator_engine import LLMTranslatorService

# ================================================================
# 全局常量
# ================================================================
MAX_BATCH_SIZE = 20          # 批处理最大 turns 数
MAX_WAIT_TIME = 1.5          # 批处理最大等待时间（秒）
MAX_ACTIVE_REQUESTS = 500    # 全局并发水位线
SLA_TIMEOUT_SECONDS = 45     # 接口整体 SLA 超时（秒）

# ================================================================
# 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api_translation")

# ================================================================
# CPU 密集型任务卸载：全局线程池
# ================================================================
_parse_executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="parse_worker",
)


def parse_transcript_to_turns(raw_text: str, session_id: str) -> List[DialogueTurn]:
    """
    带记忆状态的翻译单元格解析器。
    直接将长文本解析为翻译引擎所需的 DialogueTurn 对象。

    ⚠️ 此函数包含大量正则匹配，属于 CPU 密集型任务，
       禁止在主协程中直接调用，必须通过 run_in_executor 卸载。
    """
    records = []
    lines = raw_text.strip().splitlines()
    pattern = re.compile(r"^(.+?)说[：:,，；;]\s*(.*)$")

    current_speaker = None
    current_text_blocks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            if current_speaker is not None and current_text_blocks:
                joined_text = " ".join(current_text_blocks).strip()
                if joined_text:
                    records.append(DialogueTurn(
                        id=f"{session_id}_{len(records):04d}",
                        speaker=current_speaker,
                        content=joined_text,
                    ))
            current_speaker = match.group(1).strip()
            current_text_blocks = [match.group(2).strip()]
        else:
            if current_speaker is not None:
                current_text_blocks.append(line)
            else:
                current_speaker = "Unknown"
                current_text_blocks = [line]

    # 处理最后一段
    if current_speaker is not None and current_text_blocks:
        joined_text = " ".join(current_text_blocks).strip()
        if joined_text:
            records.append(DialogueTurn(
                id=f"{session_id}_{len(records):04d}",
                speaker=current_speaker,
                content=joined_text,
            ))

    return records


# ================================================================
# 全局状态
# ================================================================
# 翻译引擎单例
_translator = LLMTranslatorService()

# 动态批处理队列：元素为 (dialogue_turns, asyncio.Future)
_batch_queue: asyncio.Queue[Tuple[List[DialogueTurn], asyncio.Future]] = asyncio.Queue()

# 并发水位计数器
_active_request_count = 0
_active_request_lock = asyncio.Lock()


# ================================================================
# 后台批处理 Worker
# ================================================================
async def background_batch_worker() -> None:
    """
    后台常驻任务：从队列中收集请求，达到批量阈值或等待超时后，
    合并所有 turns 为一次 LLM 调用，再将结果拆解回各 Future。

    ⚠️ 此 Worker 绝对不能因单次异常退出死循环，必须 catch-all。
    """
    logger.info("[BatchWorker] 启动，MAX_BATCH_SIZE=%d, MAX_WAIT_TIME=%.1fs", MAX_BATCH_SIZE, MAX_WAIT_TIME)

    while True:
        batch: List[Tuple[List[DialogueTurn], asyncio.Future]] = []

        try:
            # --------------------------------------------------
            # 阶段 1：阻塞等待第一个请求（无超时，一直等）
            # --------------------------------------------------
            first_item = await _batch_queue.get()
            batch.append(first_item)

            # --------------------------------------------------
            # 阶段 2：在 MAX_WAIT_TIME 窗口内尽可能多收集
            # --------------------------------------------------
            deadline = asyncio.get_event_loop().time() + MAX_WAIT_TIME
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(_batch_queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # --------------------------------------------------
            # 阶段 3：合并所有 turns，一次 LLM 调用
            # --------------------------------------------------
            total_turns = sum(len(turns) for turns, _ in batch)
            logger.info(
                "[BatchWorker] 批次就绪：请求数=%d, 总turns=%d",
                len(batch), total_turns,
            )

            # 将所有请求的 turns 合并为一个列表
            merged_turns: List[DialogueTurn] = []
            for turns, _ in batch:
                merged_turns.extend(turns)

            # 调用翻译引擎（一次 LLM 调用）
            translated_items = await _translator.translate(merged_turns)

            # --------------------------------------------------
            # 阶段 4：按 id 拆解结果，回填各 Future
            # --------------------------------------------------
            # 构建 id -> TranslatedItem 的映射
            translation_map = {item.id: item for item in translated_items}

            for turns, future in batch:
                if future.done():
                    # Future 已被 SLA 超时取消，跳过
                    logger.warning("[BatchWorker] Future 已完成（可能被超时取消），跳过结果回填")
                    continue

                # 从 translation_map 中提取该请求对应的结果
                request_result = []
                for turn in turns:
                    item = translation_map.get(turn.id)
                    if item is not None:
                        request_result.append(item)
                    else:
                        # 兜底：LLM 没返回该 id 的翻译
                        from models_translation import TranslatedItem
                        request_result.append(TranslatedItem(id=turn.id, content=""))
                        logger.warning("[BatchWorker] 缺少 id=%s 的翻译结果，兜底空字符串", turn.id)

                future.set_result(request_result)

        except asyncio.CancelledError:
            # 应用关闭时被取消，优雅退出
            logger.info("[BatchWorker] 收到取消信号，优雅退出")
            # 把已收集的请求都设为异常，避免它们永远挂起
            for _, future in batch:
                if not future.done():
                    future.set_exception(RuntimeError("服务正在关闭"))
            break

        except Exception as exc:
            # ⚠️ 绝对不能退出死循环！把异常注入所有 Future
            logger.exception("[BatchWorker] 批处理异常，将异常注入 %d 个 Future: %s", len(batch), exc)
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)

            # 短暂休眠避免空转（仅异常时）
            await asyncio.sleep(0.5)


# ================================================================
# FastAPI Lifespan（替代 on_event）
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动后台 Worker，关闭时清理"""
    worker_task = asyncio.create_task(background_batch_worker())
    logger.info("[Lifespan] background_batch_worker 已启动")

    yield

    # 关闭阶段：取消 Worker
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("[Lifespan] background_batch_worker 已停止")

    # 关闭线程池
    _parse_executor.shutdown(wait=False)
    logger.info("[Lifespan] parse_executor 已关闭")


# ================================================================
# FastAPI 应用实例
# ================================================================
app = FastAPI(title="Text Translation API", version="2.0.0", lifespan=lifespan)


# ================================================================
# 主路由
# ================================================================
@app.post("/api/translate", response_model=TranslateResponse)
async def translate_endpoint(raw_request: Request) -> JSONResponse:
    """
    翻译微服务主路由 —— 高并发防御版

    防御层顺序：
    1. 全局限流（Load Shedding）→ 429
    2. 请求体解析 → 400
    3. 参数校验 → 400
    4. CPU 任务卸载（parse_transcript_to_turns）
    5. 入队批处理 + SLA 超时 → 504
    """
    global _active_request_count
    session_id: str = "unknown"

    # ========================================================
    # Layer 0: 全局限流（快速拒绝，防止内存打爆）
    # ========================================================
    async with _active_request_lock:
        if _active_request_count >= MAX_ACTIVE_REQUESTS:
            logger.warning("[LoadShed] 并发超限 %d/%d，快速拒绝 session_id=%s",
                           _active_request_count, MAX_ACTIVE_REQUESTS, session_id)
            return JSONResponse(
                status_code=429,
                content={
                    "session_id": session_id,
                    "status": 429,
                    "message": "服务过载，请稍后重试",
                    "translated": [],
                },
            )
        _active_request_count += 1

    try:
        # ========================================================
        # Layer 1: 尝试解析请求体
        # ========================================================
        try:
            raw_body = await raw_request.json()
            session_id = raw_body.get("session_id", "unknown")
        except Exception as parse_exc:
            logger.error("[Request] 请求体 JSON 解析失败：%s", parse_exc)
            return JSONResponse(
                status_code=400,
                content={
                    "session_id": session_id,
                    "status": 400,
                    "message": f"请求体格式错误: {parse_exc}",
                    "translated": [],
                },
            )

        try:
            # ========================================================
            # Layer 2: 严格参数校验
            # ========================================================
            req = TranslateRequest(**raw_body)

            # ========================================================
            # Layer 3: 智能路由 + CPU 任务卸载
            # ========================================================
            if isinstance(req.data.content, str):
                # ⚠️ parse_transcript_to_turns 是 CPU 密集型，
                # 必须卸载到线程池，绝不在主协程裸奔
                loop = asyncio.get_running_loop()
                dialogue_turns = await loop.run_in_executor(
                    _parse_executor,
                    parse_transcript_to_turns,
                    req.data.content,
                    req.session_id,
                )
            else:
                dialogue_turns = req.data.content
                # ⚠️ 批处理防串号：客户端传来的结构化 ID 可能跨请求重复
                # （如请求 A 的 id="1" 和请求 B 的 id="1" 合批后字典覆盖）
                # 必须用 session_id 加盐，保证全局唯一；返回前再剥离
                for turn in dialogue_turns:
                    turn.id = f"{req.session_id}___{turn.id}"

            # 空对话直接返回
            if not dialogue_turns:
                return JSONResponse(status_code=200, content={
                    "session_id": req.session_id,
                    "status": 200,
                    "message": "OK",
                    "translated": [],
                })

            # ========================================================
            # Layer 4: 入队批处理 + SLA 超时熔断
            # ========================================================
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            await _batch_queue.put((dialogue_turns, future))

            try:
                translated_items = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=SLA_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error("[SLA] 翻译超时 session_id=%s，超过 %ds", session_id, SLA_TIMEOUT_SECONDS)
                # Future 可能还在队列中等待，或 Worker 正在处理
                # 不取消 Future（让 Worker 自然跳过已完成的），
                # 但这里直接返回 504
                return JSONResponse(
                    status_code=504,
                    content={
                        "session_id": session_id,
                        "status": 504,
                        "message": f"翻译服务处理超时（{SLA_TIMEOUT_SECONDS}s）",
                        "translated": [],
                    },
                )
            except Exception as batch_exc:
                # Worker 通过 set_exception 注入的异常
                logger.error("[BatchWorker] 批处理异常传导 session_id=%s：%s", session_id, batch_exc)
                return JSONResponse(
                    status_code=502,
                    content={
                        "session_id": session_id,
                        "status": 502,
                        "message": f"翻译后端异常: {batch_exc}",
                        "translated": [],
                    },
                )

            # ========================================================
            # Layer 5: 封装标准成功响应
            # ========================================================
            # ⚠️ 剥离批处理加盐前缀，恢复客户端原始 ID
            for item in translated_items:
                if "___" in item.id:
                    item.id = item.id.split("___", 1)[-1]

            return JSONResponse(status_code=200, content={
                "session_id": req.session_id,
                "status": 200,
                "message": "OK",
                "translated": [item.dict() for item in translated_items],
            })

        except ValueError as ve:
            logger.warning("[Validate] 参数校验失败 session_id=%s：%s", session_id, ve)
            return JSONResponse(
                status_code=400,
                content={
                    "session_id": session_id,
                    "status": 400,
                    "message": str(ve),
                    "translated": [],
                },
            )

        except Exception as e:
            logger.exception("[Internal] 翻译服务内部错误 session_id=%s：%s", session_id, e)
            return JSONResponse(
                status_code=500,
                content={
                    "session_id": session_id,
                    "status": 500,
                    "message": str(e),
                    "translated": [],
                },
            )

    finally:
        # ========================================================
        # 安全释放并发计数器
        # ========================================================
        async with _active_request_lock:
            _active_request_count -= 1


# ================================================================
# 入口
# ================================================================
if __name__ == "__main__":
    uvicorn.run(
        "api_translation:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        # 生产环境建议配合 uvicorn worker 数量
        # workers=4,
    )
