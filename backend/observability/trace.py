"""L37：全链路 Trace 存储 + 事件标准化。

和小哲电商 observability/trace.py 对齐：
- TraceStore：按 session_id 存储本轮所有事件（意图识别 / 工具调用 / RAG 检索 / 模型调用 / 成本）
- TraceEventNormalizer：把不同来源的事件标准化成统一结构（递归脱敏 + 分类）
- 查询接口：get_trace / list_traces

本课次演化：
- L37 trace-store：TraceStore + TraceEventNormalizer
- L41 final-rehearsal：终态全链路 Trace

被谁调用：agents/customer_service_agent.py chat()（每个阶段都 add）、api/routes.py /trace 端点
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TraceEvent:
    """单个 Trace 事件。"""
    event_type: str  # intent / tool_call / rag_retrieval / model_call / cost / security / clarification
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""


class TraceEventNormalizer:
    """事件标准化器——把不同来源的事件统一成 TraceEvent。

    核心能力：
    1. 递归脱敏：driverId / bankAccount / phone 等敏感字段替换成 ***
    2. 分类：根据 event_type 决定保留哪些字段
    3. 截断：超长字符串截断到 500 字符
    """

    SENSITIVE_KEYS = {"driverid", "bankaccount", "phone", "mobile", "idcard", "password", "token"}

    def normalize(self, event_type: str, data: dict[str, Any], session_id: str = "") -> TraceEvent:
        """标准化事件。"""
        sanitized = self._sanitize(data)
        return TraceEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=sanitized,
            session_id=session_id,
        )

    def _sanitize(self, data: Any) -> Any:
        """递归脱敏 + 截断。"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key.lower() in self.SENSITIVE_KEYS:
                    result[key] = "***"
                else:
                    result[key] = self._sanitize(value)
            return result
        if isinstance(data, list):
            return [self._sanitize(item) for item in data[:20]]  # 列表最多 20 条
        if isinstance(data, str) and len(data) > 500:
            return data[:500] + "...[truncated]"
        return data


class TraceStore:
    """全链路 Trace 存储。

    线程安全的内存存储：{session_id: [TraceEvent, ...]}
    """

    def __init__(self) -> None:
        self._traces: dict[str, list[TraceEvent]] = {}
        self._lock = threading.Lock()
        self._normalizer = TraceEventNormalizer()

    def add(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        """添加一个 Trace 事件。"""
        event = self._normalizer.normalize(event_type, data, session_id)
        with self._lock:
            self._traces.setdefault(session_id, []).append(event)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        """获取某个会话的所有 Trace 事件（返回 dict 列表方便序列化）。"""
        with self._lock:
            events = self._traces.get(session_id, [])
            return [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "data": e.data,
                }
                for e in events
            ]

    def list_sessions(self, limit: int = 50) -> list[str]:
        """列出最近的会话 ID。"""
        with self._lock:
            return list(self._traces.keys())[-limit:]

    def clear(self, session_id: str | None = None) -> None:
        """清空某个会话或全部 Trace。"""
        with self._lock:
            if session_id:
                self._traces.pop(session_id, None)
            else:
                self._traces.clear()


# 模块级单例
_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    """获取 TraceStore 单例。"""
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore()
    return _trace_store
