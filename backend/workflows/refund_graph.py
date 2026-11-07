"""L27-L29：卡券退款审批 LangGraph Workflow。

和小哲电商 workflows/after_sale_graph.py 对齐，但司机域是卡券退款（不是电商订单退款）：

退款 Workflow 有两种：
1. REFUND_APPROVAL_GRAPH：未使用卡券退款 → 3 节点（查状态 → 资格判断 → pause_for_human_approval）
2. (暂不实现) RETURN_GRAPH：卡券退货（司机域一般不退实物，预留扩展）

业务规则（通用版，可在 refund_policy.py 调整）：
- 未使用 + 未核销 → eligible=True → 转人工审批
- 已使用 → eligible=False → 阻断
- 已核销 → eligible=False → 阻断
- 过期 → eligible=False → 阻断

本课次演化：
- L27 langgraph-intro：StateGraph + TypedDict + 线性节点串
- L28 refund-approval：退款审批图（3 节点 + pause_for_human_approval）
- L29 return-window：签收退货图（司机域无实物退货，预留）
- L41 final-rehearsal：终态售后图 + checkpoint/resume 契约

被谁调用：agents/customer_service_agent.py chat() 高风险退款分支（L26 升级）
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from policies.refund_policy import assess_refund_boundary, HighRiskAssessment

try:
    # LangGraph 是可选依赖：没装也不阻断主流程（降级为纯资格判断）
    from langgraph.graph import StateGraph, END
    from langgraph.types import interrupt

    _LANGGRAPH_AVAILABLE = True
except Exception:
    _LANGGRAPH_AVAILABLE = False

    # 没装 langgraph 时提供占位类型，避免模块 import 失败
    class StateGraph:
        def __init__(self, *a, **k):
            pass

        def add_node(self, *a, **k):
            return self

        def add_edge(self, *a, **k):
            return self

        def set_entry_point(self, *a, **k):
            return self

        def compile(self, *a, **k):
            return None

    END = "__end__"

    def interrupt(_value):
        return None


class RefundApprovalState(TypedDict, total=False):
    """退款审批图的状态契约。

    和小哲电商 RefundApprovalState 对齐，但司机域用 coupon_id / coupon_status：
    - coupon_id: 卡券号
    - coupon_status: 卡券状态（unused / used / redeemed / expired）
    - eligibility: 退款资格判断结果
    - approval_pending: 是否等待人工审批
    - thread_id: 会话 ID（用于 checkpoint 恢复）
    """
    coupon_id: str
    coupon_status: str | None
    eligibility: HighRiskAssessment | None
    approval_pending: bool
    thread_id: str


def _assess_refund_eligibility(state: RefundApprovalState) -> RefundApprovalState:
    """节点 2：根据卡券状态判断退款资格。

    直接复用 policies/refund_policy.py 的 assess_refund_boundary，
    保证 LangGraph 图内的判断逻辑和 chat() 里的判断逻辑完全一致。

    被谁调用：REFUND_APPROVAL_GRAPH 内部节点
    """
    coupon_id = state.get("coupon_id", "")
    coupon_status = state.get("coupon_status")

    # 直接复用 refund_policy.assess_refund_boundary
    eligibility = assess_refund_boundary(
        coupon_status=coupon_status,
        verified_at=None,  # LangGraph 图暂不处理 verified_at 时间窗（预留）
        coupon_id=coupon_id,
        order_id=None,
    )

    # 只有 needs_human_approval 才需要 pause
    approval_pending = eligibility.needs_human_approval and eligibility.eligibility_status != "blocked"

    return {
        "coupon_id": coupon_id,
        "coupon_status": coupon_status,
        "eligibility": eligibility,
        "approval_pending": approval_pending,
        "thread_id": state.get("thread_id", ""),
    }


def _pause_for_human_approval(state: RefundApprovalState) -> RefundApprovalState:
    """节点 3：暂停等待人工审批（LangGraph interrupt）。

    只有 approval_pending=True 时才会走到这个节点。
    interrupt() 会让图暂停，直到外部调用 resume_workflow() 恢复。

    被谁调用：REFUND_APPROVAL_GRAPH 内部节点
    """
    eligibility = state.get("eligibility")
    if state.get("approval_pending") and eligibility is not None:
        # LangGraph interrupt：图暂停，等待人工审批结果
        # 真实部署时这里会返回审批链接 / 通知运营后台
        interrupt({
            "type": "human_approval_required",
            "coupon_id": state.get("coupon_id"),
            "eligibility_status": eligibility.eligibility_status,
            "reasons": eligibility.reasons,
        })
    return state


def _build_refund_graph():
    """构建退款审批 LangGraph 图。

    节点：fetch_status → assess_eligibility → pause_for_human_approval → END
    （fetch_status 节点由 chat() 在 invoke 前填充 state，图里只做资格判断 + 暂停）

    被谁调用：模块加载时执行，生成 REFUND_APPROVAL_GRAPH 单例
    """
    graph = StateGraph(RefundApprovalState)
    graph.add_node("assess_eligibility", _assess_refund_eligibility)
    graph.add_node("pause_for_human_approval", _pause_for_human_approval)
    graph.set_entry_point("assess_eligibility")

    # 条件边：需要人工审批 → pause；否则 → END
    def _route_after_assess(state: RefundApprovalState) -> str:
        if state.get("approval_pending"):
            return "pause_for_human_approval"
        return END

    graph.add_conditional_edges("assess_eligibility", _route_after_assess, {
        "pause_for_human_approval": "pause_for_human_approval",
        END: END,
    })
    graph.add_edge("pause_for_human_approval", END)

    return graph.compile() if _LANGGRAPH_AVAILABLE else None


# 模块级单例：退款审批图
REFUND_APPROVAL_GRAPH = _build_refund_graph()


def run_refund_approval(
    *,
    coupon_id: str,
    coupon_status: str | None,
    thread_id: str,
) -> HighRiskAssessment:
    """执行退款审批图。

    如果 LangGraph 可用 → 跑完整图（含 pause_for_human_approval）
    如果 LangGraph 不可用 → 降级为直接调 _assess_refund_eligibility

    被谁调用：agents/customer_service_agent.py chat() 高风险退款分支
    """
    initial_state: RefundApprovalState = {
        "coupon_id": coupon_id,
        "coupon_status": coupon_status,
        "eligibility": None,
        "approval_pending": False,
        "thread_id": thread_id,
    }

    if REFUND_APPROVAL_GRAPH is None:
        # LangGraph 不可用 → 降级为纯资格判断
        result_state = _assess_refund_eligibility(initial_state)
        return result_state["eligibility"]

    # LangGraph 可用 → 跑图（如果有 pause，会在 interrupt 处暂停）
    result_state = REFUND_APPROVAL_GRAPH.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    eligibility = result_state.get("eligibility")
    if eligibility is None:
        # 理论上不会到这里，防御性兜底
        return HighRiskAssessment(
            eligibility_status="needs_clarification",
            reasons=["退款资格判断异常，请联系客服。"],
            suggested_action="ask_clarification",
            risk_level="medium",
        )
    return eligibility
