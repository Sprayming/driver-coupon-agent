#!/bin/bash
# 停止司机卡券项目本地服务
kill_port() {
  local port="$1" name="$2" pids
  pids=$(lsof -ti "tcp:$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill 2>/dev/null
    echo "已停止 $name ($port)"
  else
    echo "$name ($port) 未在运行"
  fi
}

kill_port 5173 "审批对话台前端"
kill_port 8082 "Mock 司机域服务"
kill_port 8000 "Agent 审批后端"
