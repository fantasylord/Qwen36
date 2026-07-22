FROM debian:bookworm-slim

# 安装git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /models

# 从GitHub克隆模型仓库
RUN git clone https://github.com/fantasylord/Qwen36.git qwen3.6-27b-aggressive-awq

# 验证文件结构
RUN ls -la /models/qwen3.6-27b-aggressive-awq/
