"""L24：MCP 工具目录——汇总所有 MCP Server 的工具绑定信息。

和小哲电商 mcp_catalog/catalog.py 对齐：
- MCPCatalog：读取 mcp_servers.json，汇总所有 MCP 工具
- binding_summary：给 agent/调试台看的工具清单

本课次演化：
- L24 mcp-catalog：MCP 工具目录 + binding summary
- L41 final-rehearsal：终态 MCP 集成

被谁调用：agents/customer_service_agent.py（工具路由时）或调试台展示
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPToolBinding:
    """单个 MCP 工具绑定。"""
    server_name: str
    tool_name: str
    description: str
    namespace: str = ""


@dataclass
class MCPCatalog:
    """MCP 工具目录。"""
    servers: dict[str, Any] = field(default_factory=dict)
    tools: list[MCPToolBinding] = field(default_factory=list)

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "MCPCatalog":
        """从 mcp_servers.json 加载 MCP 配置。

        被谁调用：模块加载时或 agent 初始化时
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "mcp_servers.json")
        if not os.path.exists(config_path):
            return cls()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return cls()

        catalog = cls()
        servers = config.get("mcpServers", config.get("servers", {}))
        for server_name, server_cfg in servers.items():
            catalog.servers[server_name] = server_cfg
            # 从配置里提取工具列表（如果有）
            tools = server_cfg.get("tools", [])
            for tool in tools:
                catalog.tools.append(MCPToolBinding(
                    server_name=server_name,
                    tool_name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    namespace=server_cfg.get("namespace", server_name),
                ))
        return catalog

    def binding_summary(self) -> list[dict[str, str]]:
        """返回工具绑定摘要（给调试台展示用）。

        被谁调用：api/admin_routes.py 或调试台
        """
        return [
            {
                "server": t.server_name,
                "tool": t.tool_name,
                "namespace": t.namespace,
                "description": t.description[:80],
            }
            for t in self.tools
        ]


# 模块级单例
_mcp_catalog: MCPCatalog | None = None


def get_mcp_catalog() -> MCPCatalog:
    """获取 MCPCatalog 单例。"""
    global _mcp_catalog
    if _mcp_catalog is None:
        _mcp_catalog = MCPCatalog.from_config()
    return _mcp_catalog
