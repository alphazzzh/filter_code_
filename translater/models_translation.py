import json
from typing import Any, List
from pydantic import BaseModel, validator

# ---------------------------------------------------------
# 1. 内部核心结构定义
# ---------------------------------------------------------
class DialogueTurn(BaseModel):
    """单条对话轮次，解析自 content 数组"""
    id: str
    speaker: Any          
    content: str

# ---------------------------------------------------------
# 2. 严格入参模型定义 (所有字段均为必填)
# ---------------------------------------------------------
class CallData(BaseModel):
    """话单的 data 字段，所有字段严格必填"""
    session_id: str
    content: List[DialogueTurn]          
    language: str
    start_time: str
    end_time: str
    duration: float
    caller_number: int
    called_number: int
    caller_country_code: int
    called_country_code: int
    file: str
    create_time: str                     
    cp: str

    @validator("content", pre=True)
    def parse_stringified_content(cls, v: Any) -> List[dict]:
        """
        兼容上游系统传来的字符串化 JSON 数组：
        如果上游传来的是 "[{...}, {...}]"，在此处拦截并转为 List[dict]
        """
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as exc:
                raise ValueError(f"content 字段无法解析为 JSON：{exc}") from exc
                
            if not isinstance(parsed, list):
                raise ValueError(f"content 解析后应为 JSON 数组，实际得到 {type(parsed).__name__}")
            return parsed
            
        raise ValueError(f"content 字段类型不合法：期望 str 或 list，实际为 {type(v).__name__}")

class TranslateRequest(BaseModel):
    """主请求入口模型"""
    session_id: str
    data: CallData

# ---------------------------------------------------------
# 3. 严格出参模型定义
# ---------------------------------------------------------
class TranslatedItem(BaseModel):
    """单条翻译结果"""
    id: str
    content: str

class TranslateResponse(BaseModel):
    """主响应出口模型"""
    session_id: str
    status: int
    message: str
    translated: List[TranslatedItem]