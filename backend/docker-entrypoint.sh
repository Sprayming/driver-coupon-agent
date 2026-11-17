#!/bin/sh
# 容器启动前置:SQLite 平台库不存在时先播种(幂等保护:仅首次执行),再 exec 真正命令。
set -e

DB_FILE="${AGENT_DB_PATH:-/app/backend/data/rag_platform.db}"
if [ ! -f "$DB_FILE" ]; then
  echo "[entrypoint] 初始化种子数据库: $DB_FILE"
  python -c "from store.seed import seed; seed()"
fi

exec "$@"
