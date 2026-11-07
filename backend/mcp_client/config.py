"""L24 MCP 客户端配置加载层。

Agent 靠配置文件显式声明要连哪些 MCP Server（不是自动扫描）。
支持三种部署形态：
  - local: 进程内工具（把现有 TOOL_SPECS 暴露成 MCP Server，课程演示 + 本地回退）
  - stdio: 本地子进程（command + args），Server 由 Client 拉起，随 Client 退出
  - http:  远程独立部署（url），Server 长期运行，多 Client 共享

配置优先级：
  1. 显式传入的 path
  2. 环境变量 DRIVER_MCP_CONFIG
  3. backend/mcp_servers.json
  4. 文件不存在时回退到「只有一个 local Server」的内置默认配置
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config.settings import BACKEND_DIR


DEFAULT_MCP_CONFIG_PATH = BACKEND_DIR / "mcp_servers.json"
MCP_CONFIG_ENV_VAR = "DRIVER_MCP_CONFIG"


class MCPServerConfig(BaseModel):
    """单个 MCP Server 的连接声明。"""

    name: str
    type: Literal["local", "stdio", "http"] = "local"
    enabled: bool = True

    # local：进程内工具模块（module 里的 TOOL_SPECS）
    module: str | None = None

    # stdio：本地子进程
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http：远程独立部署
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    timeout_seconds: float = 10.0

    def validate_for_type(self) -> None:
        """按 type 校验必填字段，配置写错时启动即报，不等到调用才发现。"""
        if self.type == "local" and not self.module:
            raise ValueError(f"MCP Server {self.name} 是 local 类型，必须配置 module。")
        if self.type == "stdio" and not self.command:
            raise ValueError(f"MCP Server {self.name} 是 stdio 类型，必须配置 command。")
        if self.type == "http" and not self.url:
            raise ValueError(f"MCP Server {self.name} 是 http 类型，必须配置 url。")


def resolve_config_path(path: str | Path | None = None) -> Path:
    """按优先级解析配置文件路径。"""
    if path is not None:
        return Path(path)
    env_path = os.environ.get(MCP_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_MCP_CONFIG_PATH


def load_mcp_config(path: str | Path | None = None) -> list[MCPServerConfig]:
    """读取配置文件，返回已启用的 Server 清单。

    文件不存在时不报错，回退到内置的 local 默认配置——
    这样没接远程 Server 的开发环境也能照常跑通工具链。
    """
    config_path = resolve_config_path(path)
    if not config_path.exists():
        return [_default_local_server()]

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    servers_raw = raw.get("mcpServers") or {}
    if not isinstance(servers_raw, dict):
        raise ValueError("mcpServers 必须是对象，形如 {\"server-name\": {...}}")

    configs: list[MCPServerConfig] = []
    for name, item in servers_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"MCP Server {name} 的配置必须是对象。")
        config = MCPServerConfig(name=name, **item)
        if not config.enabled:
            continue
        config.validate_for_type()
        configs.append(config)

    if not configs:
        # 全部 disabled 时不给空池，回退 local，避免 Agent 完全没工具可用。
        return [_default_local_server()]
    return configs


def _default_local_server() -> MCPServerConfig:
    """内置默认：把进程内 TOOL_SPECS 当作一个 MCP Server 暴露。"""
    return MCPServerConfig(
        name="driver-local",
        type="local",
        module="tools.contracts",
    )
