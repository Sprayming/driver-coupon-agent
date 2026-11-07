"""FastAPI 路由层，只负责 HTTP 接入和调用 Agent，不承载业务判断。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.schemas import ChatRequest, ChatResponse


class ChatResumeRequest(BaseModel):
    """L30-L31：HITL 人工审批恢复请求。"""
    session_id: str
    approval_id: str
    approved: bool
    approver: str = ""
    comment: str = ""


def create_router(agent_provider: Any) -> APIRouter:
    """创建路由，对外暴露 /health、/chat 和调试台页面。"""

    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        """返回后端是否已启动。"""

        return {"status": "ok", "service": "driver-coupon-refund-agent"}

    @router.get("/", response_class=FileResponse)
    def debug_console() -> FileResponse:
        """返回调试台前端页面。"""
        import os
        return FileResponse(os.path.join(os.path.dirname(__file__), "..", "debug_console.html"))

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """接收聊天请求，并返回回答、意图和成本摘要。"""

        # token 观察在 cost/observer.py，路由层不混入成本计算细节。
        return agent_provider().chat(request)

    @router.post("/chat/resume")
    def chat_resume(request: ChatResumeRequest) -> dict[str, Any]:
        """L30-L31：HITL 人工审批后恢复暂停的退款 Workflow。"""
        from workflows.resume import resume_refund_workflow
        return resume_refund_workflow(
            thread_id=request.session_id,
            approval_id=request.approval_id,
            approved=request.approved,
            approver=request.approver,
            comment=request.comment,
        )

    @router.get("/trace/{session_id}")
    def get_trace(session_id: str) -> dict[str, Any]:
        """L37：查询某个会话的全链路 Trace 事件。"""
        from observability.trace import get_trace_store
        events = get_trace_store().get(session_id)
        return {"session_id": session_id, "event_count": len(events), "events": events}

    return router
