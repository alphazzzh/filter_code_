"""
LTP NLP 微服务配置
──────────────────────────────────────────────────────
所有可调参数集中管理，避免散落在业务代码中。
修改方式：环境变量 > 本文件默认值
"""

import os

# ── 模型路径 ──────────────────────────────────────────────────
LTP_MODEL_PATH: str = os.getenv("MODEL_LTP_PATH", "/home/zzh/923/model/ltp_small")

# ── 服务绑定 ──────────────────────────────────────────────────
HOST: str = os.getenv("LTP_SERVICE_HOST", "0.0.0.0")
PORT: int = int(os.getenv("LTP_SERVICE_PORT", "8900"))

# ── Dynamic Batching 参数 ────────────────────────────────────
#   最大等待时间（秒）：从第一个请求入队开始计时，超时即提交批次
BATCH_MAX_WAIT_MS: int = int(os.getenv("BATCH_MAX_WAIT_MS", "50"))
#   最大批次大小：累积到该数量即提交批次（即使未超时）
BATCH_MAX_SIZE: int = int(os.getenv("BATCH_MAX_SIZE", "32"))

# ── LTP 任务列表 ─────────────────────────────────────────────
#   cws=分词  pos=词性  dep=依存句法  ner=命名实体
LTP_TASKS: list[str] = ["cws", "pos", "dep", "ner"]

# ── 超时 & 限流 ──────────────────────────────────────────────
# 单次请求最长等待秒数（队列积压保护）
REQUEST_TIMEOUT_SEC: float = float(os.getenv("LTP_REQUEST_TIMEOUT_SEC", "30"))
# 全局并发上限（超过直接 503）
MAX_CONCURRENT_REQUESTS: int = int(os.getenv("LTP_MAX_CONCURRENT", "200"))
