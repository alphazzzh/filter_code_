#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LTP 微服务启动脚本（开发用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 用法：
#   bash start.sh              # 使用默认配置启动
#   MODEL_LTP_PATH=/path/to/model bash start.sh   # 指定模型路径
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 环境变量默认值
export MODEL_LTP_PATH="${MODEL_LTP_PATH:-/home/zzh/923/model/ltp_small}"
export LTP_SERVICE_HOST="${LTP_SERVICE_HOST:-0.0.0.0}"
export LTP_SERVICE_PORT="${LTP_SERVICE_PORT:-8900}"
export BATCH_MAX_WAIT_MS="${BATCH_MAX_WAIT_MS:-50}"
export BATCH_MAX_SIZE="${BATCH_MAX_SIZE:-32}"
export LTP_REQUEST_TIMEOUT_SEC="${LTP_REQUEST_TIMEOUT_SEC:-30}"
export LTP_MAX_CONCURRENT="${LTP_MAX_CONCURRENT:-200}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LTP NLP 微服务启动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 模型路径:    $MODEL_LTP_PATH"
echo " 监听地址:    $LTP_SERVICE_HOST:$LTP_SERVICE_PORT"
echo " 批次等待:    ${BATCH_MAX_WAIT_MS}ms"
echo " 批次大小:    $BATCH_MAX_SIZE"
echo " 并发上限:    $LTP_MAX_CONCURRENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 确保在项目根目录下启动（使 ltp_service 包可导入）
cd "$PROJECT_ROOT"

exec uvicorn ltp_service.server:app \
    --host "$LTP_SERVICE_HOST" \
    --port "$LTP_SERVICE_PORT" \
    --workers 1 \
    --log-level info \
    --access-log
