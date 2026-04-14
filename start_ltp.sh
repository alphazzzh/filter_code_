#!/bin/bash

# --- 配置区域 ---
# 1. 宿主机上的模型路径（对应您 server.py 里的默认路径）
MODEL_DIR="/home/zzh/923/model/ltp_small"
# 2. 对外暴露端口
HOST_PORT=8900
# 3. 容器名称
CONTAINER_NAME="ltp-service-server"
# 4. 镜像名称
IMAGE_NAME="ltp-service:v1"

echo "=========================================="
echo "    启动 LTP NLP Docker 服务"
echo "=========================================="

# 检查模型目录
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 错误: 找不到模型目录 [$MODEL_DIR]"
    exit 1
fi

# 1. 构建镜像（如果代码改动了，需要重新构建）
echo "🛠️ 正在构建镜像..."
cd "$(dirname "$0")" # 进入当前脚本所在目录
docker build -t $IMAGE_NAME -f Dockerfile .

# 2. 清理旧容器
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "🔄 清理旧容器..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
fi

# 3. 启动容器
echo "🚀 正在启动容器..."
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p $HOST_PORT:8900 \
  -v "$MODEL_DIR:/app/model" \
  -e MODEL_LTP_PATH="/app/model" \
  $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "🎉 LTP 服务已在 Docker 后台启动！"
    echo "📍 接口地址: http://127.0.0.1:${HOST_PORT}/analyze"
    echo "📜 查看日志命令: docker logs -f $CONTAINER_NAME"
else
    echo "❌ 启动异常，请检查 Docker 日志。"
fi