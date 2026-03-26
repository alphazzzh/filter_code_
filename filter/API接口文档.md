# ASR 情报清洗与结构化抽取 API 文档

## 1. 接口概述
- **接口名称**: 语音对话情报分析与结构化提取接口
- **接口描述**: 接收 ASR（自动语音识别）转写后的多轮对话文本记录，执行启发式特征提取、拓扑轨道路由、意图识别与业务角色绑定，最终返回结构化的危险情报打分、身份标签和证据链。
- **请求协议**: HTTP
- **请求方式**: POST
- **数据格式**: 请求和响应体均为 JSON 格式 (`application/json`)。
- **请求路由**: `http://<ip>:<port>/api/v1/analyze`

---

## 2. 请求说明

### 请求头 (Headers)
| Header | 值 | 必填 | 说明 |
| --- | --- | --- | --- |
| `Content-Type` | `application/json` | 是 | 指定请求体格式为 JSON |

### 入参说明 (Body)
| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `session_id` | String | 是 | 业务侧生成的会话唯一标识 ID。 |
| `data` | Object | 是 | 对话数据载荷，包含 content 和可选字段。 |
| `data.content` | String / Array | 是 | 对话内容。可以是 JSON 字符串（将被解析为数组），或直接传递 TurnContent 数组。 |
| `data.content[].id` | String | 是 | 单条对话片段的唯一 ID。 |
| `data.content[].speaker` | String / Int | 是 | 说话人标识（例如 "A"、"B"、"坐席"、"客户"等）。 |
| `data.content[].content` | String | 是 | 该说话人的 ASR 转写文本。 |
| `data.language` | String | 否 | 上游系统提供的语种提示（如 "zh"、"en"），供 LID 识别参考。 |
| `dynamic_topic` | String | 否 | 动态查询主题（如 "海外房产投资"），用于 BGE-M3 向量相似度检索。 |

### 入参示例
```json
{
  "session_id": "CALL_20260326_001",
  "data": {
    "content": [
      {
        "id": "r_001",
        "speaker": "A",
        "content": "您好，我这里是公安局经侦支队，您的账户涉嫌洗钱案件。"
      },
      {
        "id": "r_002",
        "speaker": "B",
        "content": "啊？什么？"
      },
      {
        "id": "r_003",
        "speaker": "A",
        "content": "你听清楚，你的账户现在马上就要被冻结了，必须立刻配合我们操作。"
      }
    ]
  },
  "dynamic_topic": null
}
```

---

## 3. 返回参数 (Response)

### 3.1 根层级参数 (Level 1)

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `status` | Int | 接口调用状态。`200`: 成功；`206`: 部分成功（Stage2 超时/降级）；`400`: 参数错误；`429`: 服务过载；`500`: 服务内部错误。 |
| `message` | String | 接口调用结果描述，调用成功固定返回 `"OK"`。 |
| `session_id` | String | 透传回上游的会话标识符。 |
| `data` | Object / null | 核心业务产出内容（风控情报结果）。当请求非法或系统异常时为 `null`。 |

### 3.2 `data` 层级参数 (核心业务产出)

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `conversation_id` | String | 会话 ID（透传 session_id）。 |
| `final_score` | Int | 最终情报风险评分，范围 `[0, 100]`。`50` 为基线，`≥ 60` 建议关注，`≥ 80` 高危预警。 |
| `tags` | Array\<String\> | **经过降维压制后的**最终标签集合。语义高层标签（如诈骗意图）优先，底层噪音标签（如闲聊/碎片）已被压制清除。 |
| `tags_suppressed` | Array\<String\> | **被压制清除的**底层噪音标签（仅审计用）。当高危意图存在时，这些标签被判定为语义冲突而移除。空数组表示无压制。 |
| `track_type` | String | 对话拓扑轨道类型。`"ASYMMETRIC"`: 非对称（一方主导）；`"SYMMETRIC"`: 对称（双方均衡）。 |
| `roles` | Object | 角色绑定结果。`{speaker_id: role_label}`，role_label 可选值：`"AGENT"`（骗子）、`"TARGET"`（受害者）、`"DRIVER"`（情绪驱动者）、`"PEER_A"/"PEER_B"`（对等方）。 |
| `nlp_features_summary` | Object | NLP 硬特征布尔摘要（如 `has_imperative_syntax: true`、`has_drug_quantity: false`）。 |
| `interaction_summary` | Object | 对话交互特征摘要，包含 `ping_pong_rate`（博弈交互率）、`compliance_rate`（顺从率）、`resistance_decay`（抵抗衰减值）、`word_distribution`（字数分布）。 |
| `score_breakdown` | Array\<Object\> | 完整的打分明细链，每条记录包含 `delta`（分值变化）、`tag`（标签）、`reason`（原因描述）。 |
| `bot_confidence` | Object | 机器人置信度检测结果（见 3.3）。 |
| `voicemail_detection` | Object | 无效通话检测结果（见 3.3）。 |
| `topology_metrics` | Object | 拓扑特征度量（见 3.3）。 |
| `language_detection` | Object | 语种识别结果（见 3.3）。仅当 LID 模型可用时返回。 |
| `dynamic_search` | Object | 动态主题检索结果（仅当请求携带 `dynamic_topic` 时返回）。包含 `topic_queried`、`matched`、`max_score`、`status`。 |

### 3.3 嵌套对象字段说明

#### `bot_confidence` (机器人置信度)
| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `bot_score` | Int | 机器人置信度得分 `[0, 100]`。`> 80` 判定为 BOT。 |
| `bot_label` | String | 判定结果：`"BOT"` 或 `"HUMAN"`。 |

#### `voicemail_detection` (无效通话检测)
| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `voicemail_score` | Int | 无效通话置信度得分 `[0, 100]`。`> 80` 判定为无效通话。 |
| `is_voicemail` | Bool | 是否判定为无效通话（语音信箱/未接通/机器盲播）。 |

#### `topology_metrics` (拓扑特征)
| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `is_decoupled` | Bool | 是否处于「解耦盲说」状态（一方在说、另一方没在听）。 |
| `filler_word_rate` | Float | 语气词占比（0.0~1.0）。极低值（<0.005）为机器人特征信号。 |

#### `language_detection` (语种识别)
| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `detected_language` | String | fasttext LID 检测的语种代码（如 `"zh"`, `"en"`, `"ja"`, `"ko"`）。特殊值：`"unknown"`（预测失败）、`"empty"`（空文本）、`"unavailable"`（模型未加载）。 |
| `lid_confidence` | Float | LID 预测置信度（0.0~1.0）。 |

### 3.4 标签降维压制机制说明

系统采用**层级化标签体系（Hierarchical Tagging）**，输出前自动执行标签降维压制（Tag Suppression）：

- **规则**：当 `tags` 中存在任何高危意图标签（诈骗/涉毒/暴力/极端思想/AI 机器人融合等），所有底层噪音标签（OOD_NOISE_TAGS）将被自动清除。
- **目的**：防止语义冲突。例如，一通命中电商诈骗的电话，即使受害者骂人导致总分降低，最终标签也应是 `["suspicious_fake_cs", "scam_attempt_rejected"]`，而不会出现 `"global_business_sparse"` 等噪音标签。
- **审计**：被压制的标签会出现在 `tags_suppressed` 字段中，供人工审计和模型迭代使用。
- **关键状态标签**：
  - `scam_attempt_rejected`：诈骗未遂。受害者已识破并拒绝配合，情报价值极高。

### 3.5 成功返回报文示例

#### 示例 1：高危诈骗（含标签压制）
```json
{
  "status": 200,
  "message": "OK",
  "session_id": "CALL_20260326_001",
  "data": {
    "conversation_id": "CALL_20260326_001",
    "final_score": 60,
    "tags": [
      "suspicious_fake_cs",
      "scam_attempt_rejected"
    ],
    "tags_suppressed": [
      "casual_chat_extremely_sparse",
      "global_business_sparse"
    ],
    "track_type": "ASYMMETRIC",
    "roles": {
      "A": "AGENT",
      "B": "TARGET"
    },
    "nlp_features_summary": {
      "has_imperative_syntax": true,
      "nlp_backend": "ltp"
    },
    "interaction_summary": {
      "ping_pong_rate": 0.05,
      "compliance_rate": 0.0,
      "resistance_decay": 2.1,
      "word_distribution": {"A": 0.85, "B": 0.15}
    },
    "score_breakdown": [
      {"delta": 15, "tag": "suspicious_fake_cs", "reason": "单项意图命中 [e_commerce_cs]（无硬特征强化）"},
      {"delta": -35, "tag": "fraud_failed_target_resisted", "reason": "检测到高危风险，但受害者明确拒绝/脱战（抵抗率=0.40或触发绝对识破），案件降级为未遂线索"},
      {"delta": -3, "tag": "low_value_casual_chat", "reason": "命中低价值噪声主题 [casual_chat]，降权处理"}
    ],
    "bot_confidence": {
      "bot_score": 0,
      "bot_label": "HUMAN"
    },
    "voicemail_detection": {
      "voicemail_score": 0,
      "is_voicemail": false
    },
    "topology_metrics": {
      "is_decoupled": false,
      "filler_word_rate": 0.032
    },
    "language_detection": {
      "detected_language": "zh",
      "lid_confidence": 0.9876
    }
  }
}
```

#### 示例 2：AI 机器人批量外呼诈骗
```json
{
  "status": 200,
  "message": "OK",
  "session_id": "CALL_20260326_002",
  "data": {
    "conversation_id": "CALL_20260326_002",
    "final_score": 80,
    "tags": [
      "ai_scam_bot_ecommerce",
      "suspicious_fake_cs",
      "fake_cs_screen_share_trap"
    ],
    "tags_suppressed": [],
    "track_type": "ASYMMETRIC",
    "roles": {"A": "AGENT", "B": "TARGET"},
    "bot_confidence": {
      "bot_score": 90,
      "bot_label": "BOT"
    },
    "topology_metrics": {
      "is_decoupled": true,
      "filler_word_rate": 0.001
    },
    "language_detection": {
      "detected_language": "zh",
      "lid_confidence": 0.9521
    }
  }
}
```

#### 示例 3：正常闲聊（无高危信号，无压制）
```json
{
  "status": 200,
  "message": "OK",
  "session_id": "CALL_20260326_003",
  "data": {
    "conversation_id": "CALL_20260326_003",
    "final_score": 30,
    "tags": [
      "low_value_casual_chat",
      "casual_chat_extremely_sparse"
    ],
    "tags_suppressed": [],
    "track_type": "SYMMETRIC",
    "roles": {"A": "PEER_A", "B": "PEER_B"},
    "bot_confidence": {
      "bot_score": 0,
      "bot_label": "HUMAN"
    },
    "topology_metrics": {
      "is_decoupled": false,
      "filler_word_rate": 0.045
    },
    "language_detection": {
      "detected_language": "zh",
      "lid_confidence": 0.9912
    }
  }
}
```

#### 示例 4：Stage2 超时降级 (206)
```json
{
  "status": 206,
  "message": "Partial Content: Stage2 Timeout",
  "session_id": "CALL_20260326_004",
  "data": {
    "final_score": 50,
    "tags": ["stage2_timeout_degraded"],
    "_error": "SLA Timeout"
  }
}
```

#### 示例 5：服务过载 (429)
```json
{
  "status": 429,
  "message": "Server is at full capacity. Please try again later.",
  "session_id": "CALL_20260326_005",
  "data": null
}
```
