## 🚀 V5.1 风控与翻译引擎 LangGraph 接入指南
你好！V5.1 版本的 ASR 风控引擎与翻译引擎已全面重构为 LangGraph 标准节点（Node）。底层繁杂的 NLP 模型调度、并发控制（CPU 线程池隔离 & GPU 显存保护）、超时熔断以及多语言识别等逻辑已全部封装在黑盒内部。你只需要按照本指南，将这两个节点挂载到你的 StateGraph 中即可完成极速接入。
# ⚙️ 1. 环境变量配置 (Deployment Config)
在服务启动前，请确保环境中配置了以下变量（节点会自动读取，提供默认兜底）：
环境变量名默认值说明
MODEL_BGE_PATHBAAI/bge-m3BGE-M3 向量模型路径（支持 HuggingFace ID 或本地路径）
MODEL_LTP_PATHLTP/smallLTP 依存句法模型本地文件夹路径
MODEL_LID_PATHmodels/lid.176.bin(可选) Fasttext 多语言识别模型路径
RISK_LOG_DIRlogs[重要] 审计诊断日志的本地输出目录
# 📜 2. 状态契约 (Global State Schema)
为了极致优化 LangGraph 的流转性能，防范大对象序列化导致的阻塞或大模型幻觉，风控节点的 State 做了极致瘦身。请在你的全局 State 中包含以下字段：Pythonfrom typing import TypedDict, Optional, List, Dict, Any

class GlobalSessionState(TypedDict):
    # ==================================
    # 共享输入字段 (前端/网关传入)
    # ==================================
    session_id: str
    raw_content: str               # 格式如 "A说:...\nB说:..." 的纯文本
    
    # ==================================
    # 风控节点专用 (Risk Node)
    # ==================================
    dynamic_topic: Optional[str]   # (可选) RAG 动态检索主题
    final_score: Optional[int]     # [输出] 最终风险打分 (0-100)，供你做条件路由
    risk_error: Optional[str]      # [输出] 记录节点是否发生崩溃/超时降级

    # ==================================
    # 翻译节点专用 (Translation Node)
    # ==================================
    target_language: Optional[str] # (可选) 目标语言，默认 "en"
    translated_turns: Optional[List[Dict[str, Any]]] # [输出] 翻译后的结构化数组
    translation_error: Optional[str]                 # [输出] 翻译错误信息
🔔 架构提示：明细数据去哪了？你可能会发现 State 里没有 tags、roles、bot_confidence 等详细信息。设计考量：这些高维诊断数据（全量字典）已由风控节点在内部通过独立 CPU 线程池异步安全地追加写入到了本地 .jsonl 文件中（默认路径：logs/risk_audit_details.jsonl），绝对不会阻塞你的 Event Loop。运维建议：请直接使用 Filebeat / Logstash 采集该 .jsonl 文件入库 ES，以供后续报表分析。
# 📦 3. 节点初始化与编排示例
⚠️ 核心注意：RiskControlNode 内部包含了庞大的本地模型（BGE、LTP 等），绝对不能在每次请求时实例化！ 
必须在 FastAPI / 服务启动的生命周期（lifespan）内只实例化一次。以下是完整的组装示例代码：

Pythonfrom langgraph.graph import StateGraph, END
# 导入封装好的黑盒节点
from filter.langgraph_risk_node import RiskControlNode
from translater.langgraph_translation_node import TranslationNode

# ---------------------------------------------------------
# 1. 全局初始化 (在服务启动时执行一次即可)
# ---------------------------------------------------------
print("正在加载风控与翻译模型...")
risk_node_instance = RiskControlNode()  # 自动读取环境变量加载模型
translation_node_instance = TranslationNode(
    base_url="http://内网IP:8000/v1",  # 你的 LLM 推理接口
    model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"
)
print("模型加载完毕！")

# ---------------------------------------------------------
# 2. 构建工作流 (Graph Builder)
# ---------------------------------------------------------
workflow = StateGraph(GlobalSessionState)

# 将实例的 process 方法挂载为图节点
workflow.add_node("analyze_risk", risk_node_instance.process)
workflow.add_node("translate_text", translation_node_instance.process)

# ---------------------------------------------------------
# 3. 边编排 (Edges & Routing)
# 你可以根据业务需要，选择【串行】或【并行】执行
# ---------------------------------------------------------

# 【方案 A：并行流转】(推荐！极致性能)
# 从入口同时发给风控和翻译，两者底层互不干扰，耗时取最长者
workflow.set_entry_point("analyze_risk")
workflow.set_entry_point("translate_text")

workflow.add_edge("analyze_risk", "decision_router")
workflow.add_edge("translate_text", "decision_router")

# 【条件路由逻辑示例】
def route_by_risk_score(state: GlobalSessionState):
    """根据极简的 final_score 决定后续动作"""
    score = state.get("final_score", 50)
    if score >= 80:
        return "trigger_alert"  # 阻断或报警
    return END

workflow.add_conditional_edges("decision_router", route_by_risk_score)

# ... 添加你的其他业务节点 ...

# 4. 编译并使用
app_graph = workflow.compile()

# 运行时调用示例：
# await app_graph.ainvoke({
#     "session_id": "req-10086",
#     "raw_content": "A说:把验证码给我\nB说:好的",
#     "target_language": "en"
# })
🤝 协作边界与排障节点崩溃或超时：如果风控节点触发 30 秒限流熔断或 CUDA 报错，它会自我吞没异常，安全返回 final_score: 50，并在 risk_error 字段写入错误原因，绝不会阻断整个 Graph 的运行。修改判定规则：如果运营需要新增诈骗拦截词或修改打分权重，由算法同学在 config_topics.py 中修改即可，图的结构与契约无需任何改动。