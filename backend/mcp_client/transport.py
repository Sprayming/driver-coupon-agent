"""L24 MCP 客户端传输层：JSON-RPC 2.0 over stdio / HTTP / 进程内。

三个传输实现共用同一套请求格式，Agent 侧不需要关心 Server 用什么语言实现：
  - StdioTransport: 拉起子进程，按行分隔的 JSON-RPC 帧走 stdin/stdout
  - HttpTransport:  POST JSON-RPC，兼容 application/json 和 text/event-stream(SSE)
  - LocalTransport: 进程内工具直接暴露成 MCP Server，课程演示和本地回退用
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from typing import Any, Protocol

import httpx


PROTOCOL_VERSION = "2025-06-18"

# 全局请求 id 生成器（JSON-RPC 要求同一连接内 id 唯一，全局递增够用）
_request_ids = itertools.count(1)


def _next_id() -> int:
    return next(_request_ids)


def _envelope(method: str, params: dict[str, Any] | None, request_id: int | None) -> dict[str, Any]:
    """构造 JSON-RPC 2.0 报文：有 id 是请求，无 id 是通知。"""
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        payload["params"] = params
    if request_id is not None:
        payload["id"] = request_id
    return payload


class JsonRpcTransport(Protocol):
    """传输层接口：Manager 只依赖这四个方法。"""

    async def start(self) -> None: ...
    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]: ...
    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None: ...
    async def close(self) -> None: ...


class _TransportError(RuntimeError):
    """传输层异常：连接断开、超时、协议错误。"""


class StdioTransport:
    """stdio 传输：Server 是 Client 拉起的子进程，生命周期跟随 Client。"""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None, timeout_seconds: float = 10.0) -> None:
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._timeout = timeout_seconds
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        merged_env = {**os.environ, **self._env}
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        # stderr 必须有人读，否则子进程写满缓冲区会卡死
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                return

    async def _write(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise _TransportError("stdio 传输未启动。")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        await self._proc.stdin.drain()

    async def _read_matching(self, request_id: int) -> dict[str, Any]:
        """读到 id 匹配的响应为止，途中的通知直接丢弃。"""
        if not self._proc or not self._proc.stdout:
            raise _TransportError("stdio 传输未启动。")
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=self._timeout)
            if not line:
                raise _TransportError("MCP Server 子进程已关闭 stdout。")
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue  # 非 JSON 行（启动日志）跳过
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise _TransportError(str(message["error"]))
            return message.get("result") or {}

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = _next_id()
        await self._write(_envelope(method, params, request_id))
        return await self._read_matching(request_id)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._write(_envelope(method, params, None))

    async def close(self) -> None:
        if self._stderr_task:
            self._stderr_task.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()


class HttpTransport:
    """Streamable HTTP 传输：Server 独立部署，Client 只是 HTTP 客户端。"""

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 timeout_seconds: float = 10.0) -> None:
        self._url = url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise _TransportError("HTTP 传输未启动。")
        response = await self._client.post(self._url, json=payload, headers=self._headers)
        response.raise_for_status()
        if "text/event-stream" in response.headers.get("content-type", ""):
            return self._parse_sse(response.text)
        return response.json()

    @staticmethod
    def _parse_sse(body: str) -> dict[str, Any]:
        """从 SSE 流里取第一条带 result/error 的 data 行。"""
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if not chunk:
                continue
            message = json.loads(chunk)
            if "result" in message or "error" in message:
                if "error" in message:
                    raise _TransportError(str(message["error"]))
                return message.get("result") or {}
        raise _TransportError("SSE 响应里没有找到 result。")

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message = await self._post(_envelope(method, params, _next_id()))
        if "error" in message:
            raise _TransportError(str(message["error"]))
        return message.get("result") or {}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._post(_envelope(method, params, None))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class LocalTransport:
    """进程内传输：把 module 里的 TOOL_SPECS 暴露成 MCP Server。

    课程演示和本地回退用。协议格式和远程 Server 完全一致，
    所以 Agent 侧代码不用区分「本地工具」和「远程工具」。
    """

    def __init__(self, module: str = "tools.contracts") -> None:
        self._module = module
        self._tools: dict[str, Any] = {}
        self._resources: dict[str, dict[str, str]] = {}
        self._prompts: dict[str, dict[str, str]] = {}
        # 执行工具需要 ChatRequest 上下文，由 Manager 在调用前注入
        self.context: dict[str, Any] | None = None

    async def start(self) -> None:
        specs = self._load_specs()
        self._tools = specs
        self._resources = {}
        self._prompts = {
            "prompt://driver/tool-observation": {
                "name": "prompt://driver/tool-observation",
                "title": "工具 Observation 口径",
                "content": "把工具返回压缩成事实摘要，保留必要字段和 omitted_fields；driverId 和 bankAccount 必须过滤。",
            },
        }
        for spec in specs.values():
            for uri in spec.resource_uris:
                self._resources[uri] = {
                    "uri": uri,
                    "title": f"{spec.name} 边界",
                    "text": "\n".join(spec.boundary),
                }

    def _load_specs(self) -> dict[str, Any]:
        import importlib

        module = importlib.import_module(self._module)
        return dict(getattr(module, "TOOL_SPECS"))

    # ── MCP 方法实现 ──

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False},
                    "prompts": {},
                },
                "serverInfo": {"name": "driver-local", "version": "1.0.0"},
            }
        if method == "tools/list":
            return {"tools": [self._to_mcp_tool(spec) for spec in self._tools.values()]}
        if method == "tools/call":
            return self._call_tool(params.get("name", ""), params.get("arguments") or {})
        if method == "resources/list":
            return {"resources": [{"uri": r["uri"], "name": r["title"]} for r in self._resources.values()]}
        if method == "resources/read":
            uri = params.get("uri", "")
            resource = self._resources.get(uri)
            if resource is None:
                return {"contents": []}
            return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": resource["text"]}]}
        if method == "prompts/list":
            return {"prompts": list(self._prompts.values())}
        raise _TransportError(f"local 传输不支持方法 {method}")

    @staticmethod
    def _to_mcp_tool(spec: Any) -> dict[str, Any]:
        """ToolSpec → MCP tools/list 条目。

        标准字段用 MCP 原生形状（inputSchema / annotations.readOnlyHint），
        扩展字段放 _meta，避免污染标准字段。
        """
        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": desc}
                    for key, desc in spec.parameters_schema.items()
                },
                "required": list(spec.required),
            },
            "annotations": {"readOnlyHint": spec.read_only},
            "_meta": {
                "domain": spec.domain,
                "riskLevel": spec.risk_level,
                "boundary": list(spec.boundary),
                "resourceUris": list(spec.resource_uris),
                "promptIds": list(spec.prompt_ids),
            },
        }

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行工具，结果包装成 MCP 的 content 数组形状。"""
        from api.schemas import ToolAction
        from tools.tool_runtime import execute_tool_action

        request = (self.context or {}).get("request")
        if request is None:
            return {
                "content": [{"type": "text", "text": "缺少运行时上下文，无法执行工具。"}],
                "isError": True,
            }
        action = ToolAction(tool_name=name, arguments=arguments, reason="mcp local transport")
        observation = execute_tool_action(action, request)
        payload = {
            "status": observation.status,
            "summary": observation.summary,
            "data": observation.data,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": observation.status == "error",
        }

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        return None

    async def close(self) -> None:
        return None
