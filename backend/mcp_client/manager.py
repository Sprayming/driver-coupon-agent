"""L24 MCP 客户端管理层：发现、连接、刷新工具池、路由调用。

Agent 侧要做的四件事：
  1. connect_all()    按配置逐个连接 Server（stdio 拉子进程 / http 建连），并完成 initialize 握手
  2. refresh_tools()  并发 tools/list，按 Server 名加命名空间前缀，合并成统一工具池
  3. call_tool()      拆前缀 → 路由回对应 Server → 归一化 content 数组
  4. read_resource()  读取工具的边界文档

命名空间设计：
  不同团队的工具很容易重名（两个团队都可能叫 get_status），
  所以暴露名统一为 "{server}__{tool}"，内部维护 exposed → (server, original) 的路由表。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from api.schemas import ChatRequest, ToolSpec
from mcp_client.config import MCPServerConfig, load_mcp_config
from mcp_client.transport import HttpTransport, JsonRpcTransport, LocalTransport, StdioTransport


@dataclass
class MCPToolRef:
    """工具池里的一条：暴露名、归属 Server、原始名、MCP 原始定义、转换后的 ToolSpec。"""

    exposed_name: str
    server: str
    original_name: str
    definition: dict[str, Any]
    spec: ToolSpec


@dataclass
class MCPServerState:
    """单个 Server 的连接状态，供调试台观察。"""

    config: MCPServerConfig
    transport: JsonRpcTransport | None = None
    server_info: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    tool_count: int = 0
    error: str | None = None


def _to_tool_spec(definition: dict[str, Any], exposed_name: str) -> ToolSpec:
    """MCP tools/list 条目 → 项目内部 ToolSpec。

    标准字段从 MCP 原生形状取（inputSchema / annotations.readOnlyHint），
    扩展字段从 _meta 取（riskLevel / domain / boundary / resourceUris / promptIds）。
    """
    schema = definition.get("inputSchema") or {}
    properties = schema.get("properties") or {}
    annotations = definition.get("annotations") or {}
    meta = definition.get("_meta") or {}
    return ToolSpec(
        name=exposed_name,
        description=definition.get("description", ""),
        required=list(schema.get("required") or []),
        parameters_schema={
            key: (value.get("description", "") if isinstance(value, dict) else str(value))
            for key, value in properties.items()
        },
        domain=meta.get("domain", "unknown"),
        read_only=bool(annotations.get("readOnlyHint", True)),
        risk_level=meta.get("riskLevel", "low"),
        boundary=list(meta.get("boundary") or []),
        resource_uris=list(meta.get("resourceUris") or []),
        prompt_ids=list(meta.get("promptIds") or []),
    )


class MCPClientManager:
    """管理所有 MCP Server 的连接与工具池。"""

    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self._configs = configs if configs is not None else load_mcp_config()
        self._servers: dict[str, MCPServerState] = {}
        self._tools: dict[str, MCPToolRef] = {}
        self._resources: dict[str, dict[str, Any]] = {}
        self._errors: list[str] = []
        self._ready = False

    # ── 连接与握手 ──

    async def connect_all(self) -> None:
        """并发连接所有启用的 Server，逐个完成 initialize 握手。"""
        await asyncio.gather(*[self._connect_one(cfg) for cfg in self._configs])

    async def _connect_one(self, cfg: MCPServerConfig) -> None:
        state = MCPServerState(config=cfg)
        self._servers[cfg.name] = state
        try:
            transport = self._build_transport(cfg)
            await transport.start()
            state.transport = transport
            # 握手：声明协议版本、客户端能力，换回 Server 能力
            result = await transport.request(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                    "clientInfo": {"name": "driver-agent", "version": "1.0.0"},
                },
            )
            state.server_info = result.get("serverInfo") or {}
            state.capabilities = result.get("capabilities") or {}
            await transport.notify("notifications/initialized")
        except Exception as exc:  # 单个 Server 挂掉不能拖垮整个 Agent
            state.error = f"{type(exc).__name__}: {exc}"
            self._errors.append(f"[{cfg.name}] {state.error}")

    @staticmethod
    def _build_transport(cfg: MCPServerConfig) -> JsonRpcTransport:
        if cfg.type == "stdio":
            return StdioTransport(
                command=cfg.command or "",
                args=cfg.args,
                env=cfg.env,
                timeout_seconds=cfg.timeout_seconds,
            )
        if cfg.type == "http":
            return HttpTransport(
                url=cfg.url or "",
                headers=cfg.headers,
                timeout_seconds=cfg.timeout_seconds,
            )
        return LocalTransport(module=cfg.module or "tools.contracts")

    # ── 工具发现 ──

    async def refresh_tools(self) -> None:
        """并发 tools/list，按 Server 名加命名空间前缀，合并成统一工具池。"""
        connected = [s for s in self._servers.values() if s.transport is not None]
        results = await asyncio.gather(
            *[s.transport.request("tools/list") for s in connected],  # type: ignore[union-attr]
            return_exceptions=True,
        )
        self._tools.clear()
        for state, result in zip(connected, results):
            if isinstance(result, Exception):
                state.error = f"{type(result).__name__}: {result}"
                self._errors.append(f"[{state.config.name}] tools/list 失败：{state.error}")
                continue
            tools = (result or {}).get("tools") or []
            state.tool_count = len(tools)
            for definition in tools:
                original_name = definition.get("name", "")
                if not original_name:
                    continue
                exposed_name = self._exposed_name(state.config.name, original_name)
                self._tools[exposed_name] = MCPToolRef(
                    exposed_name=exposed_name,
                    server=state.config.name,
                    original_name=original_name,
                    definition=definition,
                    spec=_to_tool_spec(definition, exposed_name),
                )

    def _exposed_name(self, server: str, tool_name: str) -> str:
        """命名空间：{server}__{tool}。极端重名时追加序号，保证池内唯一。"""
        candidate = f"{server}__{tool_name}"
        if candidate not in self._tools:
            return candidate
        index = 2
        while f"{candidate}#{index}" in self._tools:
            index += 1
        return f"{candidate}#{index}"

    # ── 调用与读取 ──

    async def call_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """拆前缀 → 路由回对应 Server → 归一化 MCP content 数组。"""
        ref = self._tools.get(exposed_name)
        if ref is None:
            return {"isError": True, "content": [], "error": f"工具 {exposed_name} 不在工具池里。"}

        state = self._servers[ref.server]
        if state.transport is None:
            return {"isError": True, "content": [], "error": f"Server {ref.server} 未连接。"}

        # 进程内传输需要 ChatRequest 上下文才能执行工具
        if isinstance(state.transport, LocalTransport):
            state.transport.context = context or {}

        try:
            result = await state.transport.request(
                "tools/call",
                {"name": ref.original_name, "arguments": arguments},
            )
        except Exception as exc:
            return {"isError": True, "content": [], "error": f"{type(exc).__name__}: {exc}"}

        content = result.get("content") or []
        return {
            "isError": bool(result.get("isError")),
            "content": content,
            "text": self._extract_text(content),
        }

    @staticmethod
    def _extract_text(content: list[dict[str, Any]]) -> str:
        """MCP content 是数组（text/image/resource 混合），这里只取文本并合并。"""
        parts = [
            str(item.get("text", ""))
            for item in content
            if item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """按 URI 找归属 Server，再走 resources/read。"""
        route = self._resolve_resource(uri)
        if route is None:
            return {"contents": [], "error": f"没有 Server 声明资源 {uri}"}
        server, transport = route
        state = self._servers[server]
        if state.transport is None:
            return {"contents": [], "error": f"Server {server} 未连接。"}
        try:
            result = await transport.request("resources/read", {"uri": uri})
        except Exception as exc:
            return {"contents": [], "error": f"{type(exc).__name__}: {exc}"}
        return result

    def _resolve_resource(self, uri: str) -> tuple[str, JsonRpcTransport] | None:
        """资源归属判断：工具定义里声明过这个 URI 的 Server 就是归属方。"""
        for ref in self._tools.values():
            if uri in ref.spec.resource_uris:
                transport = self._servers[ref.server].transport
                if transport is not None:
                    return ref.server, transport
        # 兜底：交给第一个连上的 Server（全局边界资源这类不属于单个工具的 URI）
        for name, state in self._servers.items():
            if state.transport is not None:
                return name, state.transport
        return None

    # ── 对外视图 ──

    def tool_pool(self) -> dict[str, MCPToolRef]:
        """暴露名 → 工具引用，调试台和 Agent 都能读。"""
        return dict(self._tools)

    def tool_specs(self) -> list[ToolSpec]:
        """合并后的工具池（暴露名带命名空间前缀），给模型的工具清单。"""
        return [ref.spec for ref in self._tools.values()]

    def local_tool_specs(self) -> list[ToolSpec]:
        """本地 Server 的工具（原始名，不带前缀），兼容现有本地执行链路。"""
        specs: list[ToolSpec] = []
        for ref in self._tools.values():
            state = self._servers.get(ref.server)
            if state and state.config.type == "local":
                specs.append(ref.spec.model_copy(update={"name": ref.original_name}))
        return specs

    def summary(self) -> dict[str, Any]:
        """调试台视图：Server 连接状态 + 工具池路由表 + 错误。"""
        return {
            "connected_servers": [
                {
                    "name": state.config.name,
                    "type": state.config.type,
                    "server_info": state.server_info,
                    "capabilities": list(state.capabilities.keys()),
                    "tool_count": state.tool_count,
                    "error": state.error,
                }
                for state in self._servers.values()
            ],
            "tool_pool": {
                exposed: {"server": ref.server, "original_name": ref.original_name}
                for exposed, ref in self._tools.items()
            },
            "errors": list(self._errors),
        }

    # ── 同步桥接：chat() 是同步方法 ──

    async def bootstrap(self) -> None:
        """一次完成「连接 + 握手 + 拉工具池」。"""
        if self._ready:
            return
        await self.connect_all()
        await self.refresh_tools()
        self._ready = True

    def ensure_ready_sync(self) -> bool:
        """同步入口：连接失败返回 False，让调用方回退到本地静态工具清单。"""
        if self._ready:
            return True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，可以直接起一个
            asyncio.run(self.bootstrap())
            return self._ready
        # 已有事件循环（异步调用方）不能嵌套 asyncio.run，交由调用方 await bootstrap()
        return False

    async def close(self) -> None:
        for state in self._servers.values():
            if state.transport is not None:
                try:
                    await state.transport.close()
                except Exception:
                    pass
        self._servers.clear()
        self._tools.clear()
        self._ready = False


def build_manager_with_request(configs: list[MCPServerConfig] | None = None) -> MCPClientManager:
    """便捷构造器：保留函数名便于后续注入请求上下文。"""
    return MCPClientManager(configs)
