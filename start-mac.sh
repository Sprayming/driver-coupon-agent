#!/bin/bash
# 司机卡券 AI 审批助手：macOS 一键启动
# 端口：Agent 后端 8000 / Mock 司机域 8082 / 调试前端 5173
# 注意：小哲电商项目的 Agent 后端和前端也用 8000 和 5173，
#       两个项目不能同时跑，切换前先执行 ./stop-mac.sh
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/Users/sprayming/Documents/Trae-CN/temp_20261116_01"
LOG="$WORKSPACE/.tools/logs"
mkdir -p "$LOG"

PY="$ROOT/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"

port_up() { lsof -ti "tcp:$1" >/dev/null 2>&1; }

[ -x "$PY" ] || { echo "缺少 .venv，先建环境：uv venv --python 3.13 .venv && uv pip install -r backend/requirements.txt"; exit 1; }

echo "==> 1/3 Mock 司机域微服务 (8082)"
if port_up 8082; then
  echo "    已在运行"
else
  cd "$ROOT/backend" || exit 1
  "$PY" -m uvicorn driver_service_mock:app --host 127.0.0.1 --port 8082 > "$LOG/driver-mock.log" 2>&1 &
  echo "    启动中，日志 $LOG/driver-mock.log"
fi

echo "==> 2/3 Agent 审批后端 (8000)"
if port_up 8000; then
  echo "    已在运行"
else
  cd "$ROOT/backend" || exit 1
  "$PY" main.py > "$LOG/driver-agent.log" 2>&1 &
  echo "    启动中，日志 $LOG/driver-agent.log"
fi

echo "==> 3/3 审批对话台前端 (5173)"
if port_up 5173; then
  echo "    已在运行"
else
  cd "$ROOT/frontend" || exit 1
  npm run dev > "$LOG/driver-frontend.log" 2>&1 &
  echo "    启动中，日志 $LOG/driver-frontend.log"
fi

sleep 3
echo
echo "审批对话台：http://localhost:5173"
echo "Agent 后端：http://localhost:8000/docs"
echo "Mock 服务： http://localhost:8082/docs"
echo "管理后台账号：admin / demo123456（publisher）"
