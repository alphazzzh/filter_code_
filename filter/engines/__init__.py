"""
engines/  ── 独立引擎包

每个引擎封装一个完整的业务判定逻辑，与打分策略解耦。
引擎可被多个策略复用（例如 BotConfidenceEngine 同时被
BotIntentFusionStrategy 和 filter_node 使用）。
"""
