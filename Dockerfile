# 司机卡券 AI 审批助手 —— 后端一体化镜像（主服务 + Mock 司机域微服务共用）
# 主服务: uvicorn main:app :8000    Mock 微服务: uvicorn driver_service_mock:app :8082
FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app/backend
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    TZ=Asia/Shanghai

# 先装 CPU 版 torch（避免默认源拉 2GB+ 的 CUDA 版），再装其余依赖，利用分层缓存
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
RUN chmod +x /app/backend/docker-entrypoint.sh || true

EXPOSE 8000 8082

# FastAPI 自带 /health 可作为存活探针
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health'); exit(0)"

ENTRYPOINT ["sh", "/app/backend/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
