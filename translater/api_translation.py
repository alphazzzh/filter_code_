import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 导入分离出去的模型和引擎
from models_translation import TranslateRequest, TranslateResponse
from translator_engine import LLMTranslatorService

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("api_translation")

app = FastAPI(title="Text Translation API", version="1.0.0")

# 初始化翻译大脑（单例模式）
_translator = LLMTranslatorService()

@app.post("/api/translate", response_model=TranslateResponse)
async def translate_endpoint(raw_request: Request) -> JSONResponse:
    """
    翻译微服务主路由
    """
    session_id: str = "unknown"

    # ========================================================
    # 0. 尝试解析请求体，剥离出 session_id 用于异常日志追踪
    # ========================================================
    try:
        raw_body = await raw_request.json()
        session_id = raw_body.get("session_id", "unknown")
    except Exception as parse_exc:
        logger.error("请求体 JSON 解析失败：%s", parse_exc)
        return JSONResponse(
            status_code=400, 
            content={
                "session_id": session_id, 
                "status": 400, 
                "message": f"请求体格式错误: {parse_exc}", 
                "translated": []
            }
        )

    try:
        # ========================================================
        # 1. 严格参数校验 (此步骤会自动调用 Pydantic 中的 JSON 字符串解析器)
        # ========================================================
        req = TranslateRequest(**raw_body)
        
        # ========================================================
        # 2. 抛给后端算法引擎执行整段上下文翻译
        # ========================================================
        translated_items = await _translator.translate(req.data.content)

        # ========================================================
        # 3. 封装标准成功响应
        # ========================================================
        return JSONResponse(status_code=200, content={
            "session_id": req.session_id,
            "status": 200,
            "message": "OK",
            "translated": [item.dict() for item in translated_items]
        })

    except ValueError as ve:
        # 捕获业务校验错误 (400 参数缺失、类型不合法等)
        logger.warning("参数校验失败 session_id=%s：%s", session_id, ve)
        return JSONResponse(
            status_code=400, 
            content={
                "session_id": session_id, 
                "status": 400, 
                "message": str(ve), 
                "translated": []
            }
        )

    except Exception as e:
        # 捕获大模型崩溃等内部致命错误 (500)
        logger.exception("翻译服务内部错误 session_id=%s：%s", session_id, e)
        return JSONResponse(
            status_code=500, 
            content={
                "session_id": session_id, 
                "status": 500, 
                "message": str(e), 
                "translated": []
            }
        )

if __name__ == "__main__":
    # 启动服务
    uvicorn.run("api_translation:app", host="0.0.0.0", port=8000, reload=False)