"""FastAPI 路由层，只负责 HTTP 接入和调用 Agent，不承载业务判断。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalCaseResult,
    EvalRunRequest,
    EvalRunResponse,
    FailureAttribution,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackSubmitResponse,
)
from config import settings
from evals.runner import EvalRunner
from feedback.attribution import FailureAttributor, build_backfilled_case
from mcp_catalog.catalog import get_mcp_catalog
from state.session_state import BACKFILLED_CASES, FEEDBACK_RECORDS
from tools.catalog import DRIVER_TOOL_CATALOG

SERVICE_NAME = "driver-coupon-refund-agent"
# 与 main.py 的 FastAPI title 保持一致，不写课程课次号。
SERVICE_VERSION = "v0.1"


class ChatResumeRequest(BaseModel):
    """L30-L31：HITL 人工审批恢复请求。

    coupon_id 用来拼审批线程键（session_id:coupon_id）——发起时就是这么拼的，
    恢复时少传就会找不到对应的 checkpoint。只带订单号发起的单子可以不传。
    """
    session_id: str
    approval_id: str
    # 作废（cancel=True）时这个字段没有意义，给它默认值免得调用方还要随便填一个。
    approved: bool = False
    approver: str = ""
    comment: str = ""
    coupon_id: str | None = None
    cancel: bool = False


def _trace_payload(session_id: str) -> dict[str, Any]:
    """Trace 查询的统一返回体，/trace 与 /sessions/{session_id}/trace 共用。"""

    from observability.trace import get_trace_store

    events = get_trace_store().get(session_id)
    return {"session_id": session_id, "event_count": len(events), "events": events}


def _build_capabilities() -> dict[str, Any]:
    """汇总能力清单：开关取 config.settings 真实值，工具和 MCP 绑定取真实目录。"""

    mcp_catalog = get_mcp_catalog()
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "domain": "driver_coupon_refund",
        "capabilities": {
            "rbac_enabled": settings.RBAC_ENABLED,
            "rerank_enabled": settings.env_flag_enabled(settings.RERANK_ENABLED_ENV),
            "retrieval_cache_enabled": settings.RAG_CACHE_ENABLED,
            "knowledge_lifecycle_enabled": settings.KNOWLEDGE_LIFECYCLE_ENABLED,
            "bm25_enabled": settings.BM25_ENABLED,
            "rag_fusion_enabled": settings.RAG_FUSION_ENABLED,
            "gate_enabled": settings.GATE_ENABLED,
            "intent_provider": settings.INTENT_PROVIDER,
        },
        "tools": [
            {
                "name": spec.name,
                "domain": spec.domain,
                "risk_level": spec.risk_level,
                "read_only": spec.read_only,
            }
            for spec in DRIVER_TOOL_CATALOG.list_tools()
        ],
        "tool_binding": DRIVER_TOOL_CATALOG.binding_summary(None, "low").model_dump(),
        "mcp": {
            "server_count": len(mcp_catalog.servers),
            "servers": sorted(mcp_catalog.servers),
            "bindings": mcp_catalog.binding_summary(),
        },
    }


def _to_eval_response(
    summary: Any, *, case_id: str | None, cases_path: str
) -> EvalRunResponse:
    """把 EvalSummary 数据类转成 HTTP 契约，字段一一对应不造数。"""

    return EvalRunResponse(
        total=summary.total,
        passed=summary.passed,
        failed=summary.failed,
        pass_rate=summary.pass_rate,
        summary={
            "cases_path": cases_path,
            "case_id": case_id,
            "matched_cases": summary.total,
        },
        results=[
            EvalCaseResult(
                case_id=result.case_id,
                passed=result.passed,
                actual_intent=result.actual_intent,
                actual_next_action=result.actual_next_action,
                failures=result.failures,
            )
            for result in summary.results
        ],
    )


def create_router(agent_provider: Any) -> APIRouter:
    """创建路由，对外暴露 /health、/chat 等聊天接口。"""

    router = APIRouter()
    # 评测执行器只实例化一次，agent 注入方式与 /chat 保持一致（都走 agent_provider）。
    eval_runner = EvalRunner(agent_provider)
    eval_runner.backfilled_cases = BACKFILLED_CASES
    failure_attributor = FailureAttributor()

    @router.get("/health")
    def health() -> dict[str, str]:
        """返回后端是否已启动。"""

        return {"status": "ok", "service": "driver-coupon-refund-agent"}

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        """返回当前后端能力清单（配置开关 / 工具目录 / MCP 绑定）。"""

        return _build_capabilities()

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """接收聊天请求，并返回回答、意图和成本摘要。"""

        # token 观察在 cost/observer.py，路由层不混入成本计算细节。
        return agent_provider().chat(request)

    @router.post("/chat/resume")
    def chat_resume(request: ChatResumeRequest) -> dict[str, Any]:
        """L30-L31：HITL 人工审批后恢复暂停的退款 Workflow。"""
        from workflows.after_sale_graph import build_thread_id
        from workflows.resume import resume_refund_workflow
        return resume_refund_workflow(
            thread_id=build_thread_id(request.session_id, request.coupon_id),
            approval_id=request.approval_id,
            approved=request.approved,
            approver=request.approver,
            comment=request.comment,
            cancel=request.cancel,
        )

    @router.get("/trace/{session_id}")
    def get_trace(session_id: str) -> dict[str, Any]:
        """L37：查询某个会话的全链路 Trace 事件。"""

        return _trace_payload(session_id)

    @router.get("/sessions/{session_id}/trace")
    def get_session_trace(session_id: str) -> dict[str, Any]:
        """L41 对齐路径：与 /trace/{session_id} 返回同一份数据，旧路径继续保留。"""

        return _trace_payload(session_id)

    @router.post("/eval/run", response_model=EvalRunResponse)
    def run_eval(request: EvalRunRequest | None = None) -> EvalRunResponse:
        """L41 对齐：按 case_id 跑司机域回归评测；未指定用例时不执行。"""

        case_id = request.case_id if request else None
        if not case_id:
            return EvalRunResponse(
                summary={
                    "message": "未指定 case_id，未执行评测；请传 {\"case_id\": \"<用例ID>\"}",
                    "cases_path": eval_runner.cases_path,
                    "backfilled_case_count": len(eval_runner.backfilled_cases),
                }
            )
        summary = eval_runner.run(case_id=case_id)
        return _to_eval_response(summary, case_id=case_id, cases_path=eval_runner.cases_path)

    @router.post("/feedback/submit", response_model=FeedbackSubmitResponse)
    def submit_feedback(request: FeedbackRequest) -> FeedbackSubmitResponse:
        """L41 对齐：提交反馈 → 取 Trace → 归因 → 回填回归 case → 落库。"""

        events = _trace_payload(request.session_id)["events"]
        eval_report: EvalRunResponse | None = None
        eval_result = None
        if request.case_id:
            summary = eval_runner.run(case_id=request.case_id)
            eval_report = _to_eval_response(
                summary, case_id=request.case_id, cases_path=eval_runner.cases_path
            )
            eval_result = summary.results[0] if summary.results else None

        attribution = failure_attributor.attribute(request.user_comment or request.rating, events)
        base_case = (
            next(
                (case for case in eval_runner.load_cases() if case.case_id == request.case_id),
                None,
            )
            if request.case_id
            else None
        )
        backfilled_case = build_backfilled_case(request, attribution, base_case, events)
        BACKFILLED_CASES.append(backfilled_case)

        record = FeedbackRecord(
            feedback_id=f"fb-{len(FEEDBACK_RECORDS) + 1:03d}",
            session_id=request.session_id,
            case_id=request.case_id,
            rating=request.rating,
            user_comment=request.user_comment,
            runtime_user_id=request.runtime_user_id,
            trace_event_names=[str(event.get("event_type", "")) for event in events],
            eval_failure_categories=[
                failure.split(":", 1)[0] for failure in (eval_result.failures if eval_result else [])
            ],
            attributions=[
                FailureAttribution(
                    category=attribution.category,
                    reason=attribution.reason,
                    confidence=attribution.confidence,
                    suggested_fix=attribution.suggested_fix,
                )
            ],
            backfilled_case=backfilled_case,
        )
        FEEDBACK_RECORDS.append(record)
        return FeedbackSubmitResponse(record=record, eval_report=eval_report)

    return router
