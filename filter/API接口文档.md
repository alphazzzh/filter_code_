# ASR 情报清洗与结构化抽取 API 文档

## 1. 接口概述
- **接口名称**: 语音对话情报分析与结构化提取接口
- **接口描述**: 接收 ASR（自动语音识别）转写后的多轮对话文本记录，执行启发式特征提取、拓扑轨道路由、意图识别与业务角色绑定，最终返回结构化的危险情报打分、身份标签和证据链。
- **请求协议**: HTTP
- **请求方式**: POST
- **数据格式**: 请求和响应体均为 JSON 格式 (`application/json`)。
- **请求路由**: `http://<ip>:<port>/api/v1/intelligence/analyze`

---

## 2. 请求说明

### 请求头 (Headers)
| Header | 值 | 必填 | 说明 |
| --- | --- | --- | --- |
| `Content-Type` | `application/json` | 是 | 指定请求体格式为 JSON |

### 入参说明 (Body)
| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `conversation_id` | String | 是 | 业务侧生成的会话唯一标识 ID。 |
| `records` | Array of Object | 是 | 会话中的 ASR 转写记录片段列表（按对话时间顺序排列）。 |
| `records[].record_id` | String | 是 | 单条 ASR 片段的唯一 ID。 |
| `records[].speaker_id` | String | 是 | 说话人标识 ID（例如 "A"、"B"、"坐席"、"客户"等）。 |
| `records[].raw_text` | String | 是 | ASR 原始转写文本。允许为空字符串，但不能是纯不可见字符。 |
| `records[].source_lang_hint`| String | 否 | 上游系统提供的语种提示（如 "zh", "en"），供底层 LID 识别参考。 |

### 入参示例
```json
{
  "conversation_id": "conv_20260323_001",
  "records": [
    {
      "record_id": "r_001",
      "speaker_id": "A",
      "raw_text": "您好，我这里是公安局经侦支队，您的账户涉嫌洗钱案件。"
    },
    {
      "record_id": "r_002",
      "speaker_id": "B",
      "raw_text": "啊？什么？"
    },
    {
      "record_id": "r_003",
      "speaker_id": "A",
      "raw_text": "你听清楚，你的账户现在马上就要被冻结了，必须立刻配合我们操作。"
    }
  ]
}
```

---

## 3. 返回参数 (Response)

### 3.1 根层级参数 (Level 1)

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `status` | Int | 接口调用状态。`200`: 成功；`400`: 参数错误；`500`: 服务内部错误。 |
| `message` | String | 接口调用失败原因，调用成功固定返回 `"OK"`。 |
| `session_id` | String | 透传回上游的会话标识符。 |
| `data` | Object | 核心业务产出内容（风控情报结果）。当请求非法或系统异常时为 `null`。 |

### 3.2 `data` 层级参数 (核心业务产出)
*(此部分保持不变，包含 final_score, tags, roles, dynamic_search 等)*

### 3.3 成功返回报文示例

```json
{
  "status": 200,
  "message": "OK",
  "session_id": "CALL_20260324_99812",
  "data": {
    "conversation_id": "CALL_20260324_99812",
    "final_score": 50,
    "tags": [
      "low_value_casual_chat"
    ],
    "dynamic_search": {
      "topic_queried": "海外房产投资",
      "matched": true,
      "max_score": 0.8215,
      "status": "success"
    },
    "...": "其余业务字段省略"
  }
}

{
  "status": 400,
  "message": "Content array is empty",
  "session_id": "CALL_20260324_99812",
  "data": null
}