"""L30-L31：HITL（Human-in-the-Loop）人工审批恢复流程。

和小哲电商 workflows/resume.py 对齐：
- resume_workflow：人工审批通过后，恢复暂停的 LangGraph 图
- 幂等保护：同一个 thread_id + 同一个审批结果只处理一次（落库，不依赖进程内存）
- checkpoint 恢复：从 SqliteSaver 持久化的 checkpoint 恢复图状态

幂等的两层：
1. 抢占：`approval_results` 主键 (thread_id, approval_id) + INSERT OR IGNORE，
   并发打进来只有一个能插入成功；
2. 失败释放：恢复过程抛异常时删掉占位，允许人工重试，而不是把单子永久标记成处理中。

被谁调用：api/routes.py /chat/resume 端点，或运营后台审批回调
"""

from __future__ import annotations

from typing import Any

from store import approvals
from integrations.driver_service_client import driver_coupon_fact_from_service
from workflows.after_sale_graph import get_refund_graph

try:
    from langgraph.types import Command

    _COMMAND_AVAILABLE = True
except Exception:
    _COMMAND_AVAILABLE = False


def _already_processed(thread_id: str, approval_id: str, row: Any) -> dict[str, Any]:
    return {
        "status": "already_processed",
        "thread_id": thread_id,
        "approval_id": approval_id,
        "previous_status": row["status"] if row is not None else "unknown",
        "previous_approved": bool(row["approved"]) if row is not None and row["approved"] is not None else None,
    }


def _recheck_coupon_state(values: dict[str, Any]) -> dict[str, Any]:
    """恢复前复核券状态，防止审批等待期间业务事实漂移。

    审批可能要等几小时：这期间券可能被核销、过期或被转走。冻结状态在 checkpoint
    的 state 里（发起时写入），恢复前重新查一次当前状态，不一致就不恢复。

    三种结果：
    - passed=False / business_fact_drift：状态变了 → 拒绝恢复，单子仍挂在审批节点
    - passed=False / coupon_not_found：券已不在该司机名下 → 拒绝恢复
    - passed=True  / recheck_unavailable：后端不可用 → 放行，但显式标记，审计可见
    """

    coupon_id = str(values.get("coupon_id") or "").strip()
    driver_id = str(values.get("runtime_user_id") or "").strip()
    frozen_status = str(values.get("coupon_status") or "")

    # 没冻结到必要信息（老 checkpoint / 测试直调）→ 不阻断，但标明跳过了复核。
    if not coupon_id or not driver_id:
        return {"passed": True, "reason": "recheck_skipped", "mismatches": {}}

    coupons, user_matched = driver_coupon_fact_from_service(driver_id)
    if not user_matched:
        return {
            "passed": False,
            "reason": "coupon_owner_mismatch",
            "mismatches": {"driver_id": {"frozen": driver_id, "current": None}},
        }
    if coupons is None:
        # 后端不可用：不阻断（审批已等很久，网络抖动不该让单子永久卡住），
        # 但把 recheck_unavailable 带回去，别让「没复核」看起来像「复核通过」。
        return {"passed": True, "reason": "recheck_unavailable", "mismatches": {}}

    items = coupons if isinstance(coupons, list) else []
    current = next(
        (
            item
            for item in items
            if str(item.get("coupon_id") or item.get("couponId") or "").upper()
            == coupon_id.upper()
        ),
        None,
    )
    if current is None:
        return {
            "passed": False,
            "reason": "coupon_not_found",
            "mismatches": {"coupon_id": {"frozen": coupon_id, "current": None}},
        }

    # 只比状态：状态决定资格（unused 可退 / verified 看 48h / expired 不可退），
    # 比 verified_at 字符串会因格式差异误报。
    current_status = str(current.get("status") or "")
    mismatches: dict[str, Any] = {}
    if current_status != frozen_status:
        mismatches["coupon_status"] = {"frozen": frozen_status, "current": current_status}

    return {
        "passed": not mismatches,
        "reason": None if not mismatches else "business_fact_drift",
        "mismatches": mismatches,
    }


def resume_refund_workflow(
    *,
    thread_id: str,
    approval_id: str,
    approved: bool,
    approver: str = "",
    comment: str = "",
    cancel: bool = False,
) -> dict[str, Any]:
    """恢复暂停的退款审批图；cancel=True 时改为作废这张审批单。

    Args:
        thread_id: 会话 ID（= session_id，也是 checkpoint 的 thread 键）
        approval_id: 审批单 ID（人工审批系统返回的）
        approved: True=通过，False=拒绝
        approver: 审批人
        comment: 审批意见
        cancel: True=作废挂起的单子。用驳回载荷把图推过暂停点，从而清掉 checkpoint
            的挂起状态；不做业务复核（状态漂移恰恰是作废的常见原因）。

    Returns:
        {"status": "resumed" / "cancelled" / "already_processed" / "no_pending_graph"
         / "blocked" / "resume_failed", ...}
    """

    graph = get_refund_graph()
    if graph is None or not _COMMAND_AVAILABLE:
        return {
            "status": "no_pending_graph",
            "reason": "langgraph_not_available",
            "thread_id": thread_id,
        }

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 没有挂起的图：可能是重复回调，也可能是会话根本没发起过审批。
    if snapshot is None or not snapshot.next:
        previous = approvals.get(thread_id, approval_id)
        if previous is not None:
            return _already_processed(thread_id, approval_id, previous)
        return {
            "status": "no_pending_graph",
            "reason": "no_pending_checkpoint",
            "thread_id": thread_id,
        }

    # 恢复前复核：审批可能等了数小时，券状态可能已被核销/过期/转走。
    # 冻结状态在 checkpoint 的 state 里，这里重新查一次当前状态；不一致就不恢复。
    # 作废不做复核：状态漂移本身就是最常见的作废理由。
    recheck: dict[str, Any] = {
        "passed": True,
        "reason": "recheck_skipped_for_cancel",
        "mismatches": {},
    }
    if not cancel:
        recheck = _recheck_coupon_state(dict(snapshot.values or {}))
        if not recheck["passed"]:
            return {
                "status": "blocked",
                "thread_id": thread_id,
                "approval_id": approval_id,
                "reason": recheck["reason"],
                "business_recheck": recheck,
            }

    # 抢占：同一个审批单只允许处理一次（跨进程、跨重启都成立）。
    if not approvals.claim(thread_id, approval_id):
        return _already_processed(thread_id, approval_id, approvals.get(thread_id, approval_id))

    decision_comment = f"cancelled: {comment}" if cancel and comment else comment
    if cancel and not comment:
        decision_comment = "cancelled"

    try:
        result = graph.invoke(
            Command(
                resume={
                    # 作废走驳回载荷：图照常越过暂停点走到 END，checkpoint 不再有 next。
                    "approved": False if cancel else approved,
                    "approver": approver,
                    "comment": decision_comment,
                }
            ),
            config=config,
        )
    except Exception as e:  # noqa: BLE001 - 恢复失败要释放占位，让运维可以重试
        approvals.release(thread_id, approval_id)
        return {
            "status": "resume_failed",
            "thread_id": thread_id,
            "approval_id": approval_id,
            "error": str(e),
        }

    approvals.finish(
        thread_id,
        approval_id,
        approved=False if cancel else approved,
        approver=approver,
        comment=decision_comment,
    )

    eligibility = result.get("eligibility")
    decision = result.get("approval_decision") or {}
    return {
        "status": "cancelled" if cancel else "resumed",
        "thread_id": thread_id,
        "approval_id": approval_id,
        "approved": False if cancel else approved,
        "approver": approver,
        "comment": comment,
        "approval_decision": decision,
        "eligibility_status": eligibility.eligibility_status if eligibility else None,
        "business_recheck": recheck,
    }


def get_pending_approvals(thread_id: str) -> list[dict[str, Any]]:
    """查询某个会话的审批记录（含处理中与已处理）。

    数据来自 approval_results 表，不再返回内存占位。
    """

    return [dict(row) for row in approvals.list_for_thread(thread_id)]
