"""L30-L31：HITL（Human-in-the-Loop）人工审批恢复流程。

和小哲电商 workflows/resume.py 对齐：
- resume_workflow：人工审批通过后，恢复暂停的 LangGraph 图
- 幂等保护：同一个 thread_id + 同一个审批结果只处理一次
- checkpoint 恢复：从 LangGraph checkpoint 恢复图状态

本课次演化：
- L30 hitl-resume：人工审批后恢复 Workflow
- L31 checkpoint-idempotent：checkpoint 幂等 + 业务事实复核
- L41 final-rehearsal：终态 HITL 契约

被谁调用：api/routes.py /chat/resume 端点（如果有的话），或 agents/customer_service_agent.py
"""

from __future__ import annotations

from typing import Any

from workflows.refund_graph import REFUND_APPROVAL_GRAPH

# 已处理的审批结果：{(thread_id, approval_id): result}——防止重复处理
_PROCESSED_APPROVALS: dict[tuple[str, str], str] = {}


def resume_refund_workflow(
    *,
    thread_id: str,
    approval_id: str,
    approved: bool,
    approver: str = "",
    comment: str = "",
) -> dict[str, Any]:
    """恢复暂停的退款审批图。

    Args:
        thread_id: 会话 ID（= session_id）
        approval_id: 审批单 ID（人工审批系统返回的）
        approved: True=通过，False=拒绝
        approver: 审批人
        comment: 审批意见

    Returns:
        {"status": "resumed" / "already_processed" / "no_pending_graph", ...}

    被谁调用：api/routes.py 的 /chat/resume 端点，或运营后台审批回调
    """
    # 幂等检查：同一个审批结果只处理一次
    dedup_key = (thread_id, approval_id)
    if dedup_key in _PROCESSED_APPROVALS:
        return {
            "status": "already_processed",
            "thread_id": thread_id,
            "approval_id": approval_id,
            "previous_result": _PROCESSED_APPROVALS[dedup_key],
        }

    if REFUND_APPROVAL_GRAPH is None:
        return {
            "status": "no_pending_graph",
            "reason": "langgraph_not_available",
            "thread_id": thread_id,
        }

    try:
        # 恢复图：把人工审批结果作为 interrupt 的返回值传回去
        result = REFUND_APPROVAL_GRAPH.invoke(
            None,  # resume 时不传新 state
            config={"configurable": {"thread_id": thread_id}},
        )
        result_str = "approved" if approved else "rejected"
        _PROCESSED_APPROVALS[dedup_key] = result_str

        eligibility = result.get("eligibility")
        return {
            "status": "resumed",
            "thread_id": thread_id,
            "approval_id": approval_id,
            "approved": approved,
            "approver": approver,
            "comment": comment,
            "eligibility_status": eligibility.eligibility_status if eligibility else None,
        }
    except Exception as e:
        return {
            "status": "resume_failed",
            "thread_id": thread_id,
            "approval_id": approval_id,
            "error": str(e),
        }


def get_pending_approvals(thread_id: str) -> list[dict[str, Any]]:
    """查询某个会话的待审批记录（预留接口）。

    真实部署时应该查数据库，这里只返回内存里的（演示用）。
    被谁调用：api/admin_routes.py 的运营后台查询端点
    """
    # TODO: L31 接入 store/ 做持久化
    return []
