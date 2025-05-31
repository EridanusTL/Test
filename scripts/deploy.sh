#!/bin/bash

# 本地文件路径（可以是单个文件或文件夹）
LOCAL_PATH="$(pwd)/install_arm64/"

# 远程板端路径
REMOTE_PATH="/root/yxh-test/y1a/"

# 板端用户、IP和密码
REMOTE_USER="root"
REMOTE_IP="192.168.0.1"
REMOTE_PASSWORD="123"

# 检查文件是否存在
if [ ! -e "$LOCAL_PATH" ]; then
    echo "错误: 本地路径 $LOCAL_PATH 不存在。"
    exit 1
fi

# 输出传输信息
echo "开始将 $LOCAL_PATH 传输到 $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH"

# 创建远程目录（如果不存在）
sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_IP" "mkdir -p $REMOTE_PATH"

# 使用 sshpass 和 scp 传输文件，并强制实时输出进度
export SSHPASS="$REMOTE_PASSWORD"
sshpass -e scp -r -v "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_IP:$REMOTE_PATH" | tee /dev/tty

# 检查传输是否成功
if [ $? -eq 0 ]; then
    echo "文件成功部署到板端: $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH"
else
    echo "文件传输失败，请检查连接和路径。"
    exit 1
fi