import asyncio
import json
import os

# 导入你封装好的两个节点
from filter.langgraph_risk_node import RiskControlNode
from translater.langgraph_translation_node import TranslationNode

# ==========================================
# 1. 准备你的 Mock 数据 (用你手头搬运出来的数据)
# ==========================================
MOCK_DATA_SCAM = {
    "session_id": "test_scam_001",
    "raw_content": "A说：您好，这里是京东金融客服，恭喜您获得了一笔五万元的备用金额度\nB说：哦？我没申请过啊\nA说：这是系统随机发放的，现在只需要您点击链接激活一下，需要支付一百元的服务费\nB说：好的我看看",
    "dynamic_topic": "finance",
    "target_language": "en" # 顺便测试翻译节点
}

MOCK_DATA_BOT = {
    "session_id": "test_bot_002",
    "raw_content": "A说：您好，这里是快递服务中心，您有一个快递需要签收，请回复1确认，回复2改期\nB说：1\nA说：好的，您的快递将在今天下午送达，祝您生活愉快。",
    "target_language": "en"
}


async def test_nodes():
    print("="*50)
    print("🚀 开始初始化节点 (预热大模型)...")
    print("="*50)
    
    # ⚠️ 注意：如果你在外网没有 LTP 本地模型，这里可能会报警告并降级为规则匹配，这是正常的！
    # BGE-M3 会自动从 HuggingFace 下载缓存（如果你有外网的话）
    risk_node = RiskControlNode(
        bge_model_name="BAAI/bge-m3", 
        ltp_model_path="LTP/small", # 如果外网没有这个目录，会走平滑降级
        log_dir="test_logs"         # 测试时把日志写在明确的地方
    )
    
    translation_node = TranslationNode(
        # 换成你在外网能调通的大模型 API，比如硅基流动、Kimi 等
        # base_url="...", 
        # model_name="...",
        log_dir="test_logs"
    )

    print("\n" + "="*50)
    print("🧪 测试案例 1：京东备用金诈骗 (风控 + 翻译)")
    print("="*50)
    
    # 1. 测风控节点
    print(f"正在分析: {MOCK_DATA_SCAM['session_id']}")
    risk_result = await risk_node.process(MOCK_DATA_SCAM)
    print(f"✅ 风控节点返回 State: {json.dumps(risk_result, ensure_ascii=False, indent=2)}")
    
    # 2. 测翻译节点
    # print(f"正在翻译: {MOCK_DATA_SCAM['session_id']}")
    # trans_result = await translation_node.process(MOCK_DATA_SCAM)
    # print(f"✅ 翻译节点返回 State: {json.dumps(trans_result, ensure_ascii=False, indent=2)}")

    print("\n" + "="*50)
    print("🧪 测试案例 2：快递机器人")
    print("="*50)
    risk_result_bot = await risk_node.process(MOCK_DATA_BOT)
    print(f"✅ 风控节点返回 State: {json.dumps(risk_result_bot, ensure_ascii=False, indent=2)}")

    print("\n" + "="*50)
    print("📂 验证本地 JSONL 落盘数据")
    print("="*50)
    
    # 为了防止 I/O 线程池还没写完主线程就退出了，稍微等 0.5 秒
    await asyncio.sleep(0.5) 
    
    log_file = "test_logs/risk_audit_details.jsonl"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"成功在 {log_file} 中找到了 {len(lines)} 条落盘记录！")
            print("最后一条记录的预览：")
            last_record = json.loads(lines[-1])
            print(f" - Session ID: {last_record['session_id']}")
            print(f" - Final Score: {last_record['final_score']}")
            print(f" - Tags: {last_record['tags']}")
            print(f" - 包含完整的 risk_details 抽屉数据：{'voicemail_detection' in last_record}")
    else:
        print("❌ 哎呀，没有找到本地日志文件，请检查 I/O 逻辑！")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(test_nodes())