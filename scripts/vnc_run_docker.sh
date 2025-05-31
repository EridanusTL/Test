#!/bin/bash

docker_image=andrewdiyi/ros2-desktop-vnc:foxy
id=ros-foxy-vnc

# 架构检测逻辑
detect_platform() {
    local arch=$(uname -m)
    case $arch in
        x86_64)    echo "linux/amd64" ;;
        aarch64)   echo "linux/arm64" ;;
        armv7l)    echo "linux/arm/v7" ;;
        *)         echo "unknown" ;;
    esac
}

# 自动设置平台参数
DOCKER_PLATFORM=$(detect_platform)

# 特殊处理macOS
if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        DOCKER_PLATFORM="linux/arm64"  # Apple Silicon
    else
        DOCKER_PLATFORM="linux/amd64"  # Intel Mac
    fi
fi

# 平台参数组装
platform_args=()
if [[ "$DOCKER_PLATFORM" != "unknown" ]]; then
    platform_args+=(--platform "$DOCKER_PLATFORM")
    echo "[INFO] 检测到系统架构: $DOCKER_PLATFORM"
else
    echo "[WARN] 未知架构，Docker将自动选择平台"
fi

# NVIDIA显卡检测
gpu_args=()
if [[ "${DOCKER_PLATFORM%%/*}" == "linux" ]] && \
   command -v nvidia-smi &>/dev/null && \
   nvidia-smi &>/dev/null
then
    gpu_args+=(--gpus all)
    echo "[INFO] 检测到NVIDIA显卡，已启用GPU支持"
fi

PROJECT_NAME=$(basename "$(pwd)")
HOME=/home/gac

docker run \
    -d \
    -p 6081:80 \
    --security-opt seccomp=unconfined \
    --shm-size=1024m \
    --name=$id \
    --rm \
    "${platform_args[@]}" \
    --hostname $(hostname) \
    --volume $(pwd):$HOME/workspace/$PROJECT_NAME \
    --volume $(pwd)/scripts/bashrc_vnc:$HOME/.bashrc \
    "${gpu_args[@]}" \
    $docker_image