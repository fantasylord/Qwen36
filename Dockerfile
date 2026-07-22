FROM debian:bookworm-slim

WORKDIR /models
#
# 直接从构建上下文复制模型文件
COPY . /models/qwen3.6-27b-aggressive-awq

# 验证文件结构
RUN ls -la /models/qwen3.6-27b-aggressive-awq/
