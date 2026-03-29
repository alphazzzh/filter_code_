import json
from typing import Any, List, Union
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
    content: Union[str, List[DialogueTurn]]         
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
    def parse_stringified_content(cls, v: Any) -> Union[str, List[dict]]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass # 👇 2. 忽略 JSON 解析错误，说明它是 A说B说的纯文本
            
            # 👇 3. 直接原样返回字符串，交由 API 层的智能路由处理
            return v
            
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