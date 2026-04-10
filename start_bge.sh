#!/bin/bash

# ==========================================
# TEI (BGE-M3) 向量模型离线启动脚本 (多卡优化版)
# ==========================================

# --- 配置区域 ---

# 1. 指定使用的 GPU 序号 (非常重要)
# 留空或写 "all" 表示使用所有卡
# 写 "device=0" 表示只使用第 0 张卡
# 写 "device=1,2" 表示同时使用第 1 和第 2 张卡
GPU_DEVICE="device=0" 

# 2. 宿主机上 BGE-M3 模型权重的绝对路径
MODEL_DIR="/home/zzh/models/bge-m3"

# 3. 对外暴露的服务端口
HOST_PORT=20097

# 4. 容器名称
CONTAINER_NAME="bge-tei-server"

# 5. 镜像名称 (确认与显卡架构匹配，例如 4090/L40 用 89-1.9)
IMAGE_NAME="ghcr.io/huggingface/text-embeddings-inference:89-1.9"

# ------------------------------------------

echo "=========================================="
echo "    启动 BGE-M3 TEI 向量服务"
echo "    分配显卡: $GPU_DEVICE"
echo "=========================================="

# 安全检查
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 启动失败: 找不到模型目录 [$MODEL_DIR]"
    exit 1
fi

# 清理旧容器
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "🔄 检测到同名旧容器，正在清理..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
fi

# 启动命令 (注意 --gpus 参数的改变)
echo "🚀 正在挂载 GPU 并启动容器..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus "$GPU_DEVICE" \
  --restart unless-stopped \
  -p $HOST_PORT:80 \
  -v "$MODEL_DIR:/data" \
  $IMAGE_NAME \
  --model-id /data \
  --max-client-batch-size 128 \
  --max-batch-tokens 16384

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 服务启动成功！仅使用了物理机的 [$GPU_DEVICE]"
    echo "📍 接口地址: http://127.0.0.1:${HOST_PORT}/embed"
else
    echo "❌ 启动异常，请检查卡号是否正确或驱动状态。"
fi