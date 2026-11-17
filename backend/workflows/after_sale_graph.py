"""L27-L29：卡券退款审批 LangGraph Workflow。

和小哲电商 workflows/after_sale_graph.py 对齐，但司机域是卡券退款（不是电商订单退款）：

退款 Workflow 有两种：
1. REFUND_APPROVAL_GRAPH：未使用卡券退款 → 2 节点（资格判断 → pause_for_human_approval）
2. (暂不实现) RETURN_GRAPH：卡券退货（司机域一般不退实物，预留扩展）

业务规则（通用版，可在 refund_policy.py 调整）：
- 未使用 + 未核销 → eligible=True → 转人工审批
- 已使用 → eligible=False → 阻断
- 已核销 → eligible=False → 阻断
- 过期 → eligible=False → 阻断

checkpoint 设计（L31 补齐）：
- 图编译时挂 SqliteSaver，interrupt 会把「挂在哪、state 是什么」持久化；
- 进程重启后仍能按同一 thread_id 恢复，而不是只在内存里假装暂停；
- checkpoint 与业务表共用同一个 SQLite 文件，部署只需要挂一个卷。

被谁调用：agents/customer_service_agent.py chat() 高风险退款分支（L26 升级）
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from config import settings
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

        def add_conditional_edges(self, *a, **k):
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

    - coupon_id / coupon_status：退款对象与当前状态
    - verified_at：核销时间（已核销券算 48h 退款窗口用，必须随状态一起进图）
    - order_id：订单号（用户只给订单号时，靠它通过「缺编号」这一关）
    - runtime_user_id：发起审批的司机，恢复前复核券归属与状态用
    - eligibility：退款资格判断结果
    - approval_pending：是否等待人工审批
    - thread_id：会话 ID（checkpoint 的 thread 键）
    - approval_decision：人工审批回传的决定（resume 时写入）
    """
    coupon_id: str
    coupon_status: str | None
    verified_at: str | None
    order_id: str | None
    runtime_user_id: str
    eligibility: HighRiskAssessment | None
    approval_pending: bool
    thread_id: str
    approval_decision: dict[str, Any] | None


def _assess_refund_eligibility(state: RefundApprovalState) -> RefundApprovalState:
    """节点 1：根据卡券状态判断退款资格。

    直接复用 policies/refund_policy.py 的 assess_refund_boundary，
    保证 LangGraph 图内的判断逻辑和 chat() 里的判断逻辑完全一致。
    """

    coupon_id = state.get("coupon_id", "")
    coupon_status = state.get("coupon_status")
    verified_at = state.get("verified_at")
    order_id = state.get("order_id")

    eligibility = assess_refund_boundary(
        coupon_status=coupon_status,
        verified_at=verified_at,
        coupon_id=coupon_id,
        order_id=order_id,
    )

    # 只有「可申请退款」才进人工审批。
    # 已核销/已使用（not_eligible）、证据不足（blocked）、缺参数（needs_clarification）
    # 都应该直接回复，挂起等审批没有意义，还会占住 checkpoint。
    approval_pending = eligibility.eligibility_status == "eligible_for_application"

    return {
        "coupon_id": coupon_id,
        "coupon_status": coupon_status,
        "verified_at": verified_at,
        "order_id": order_id,
        "runtime_user_id": state.get("runtime_user_id", ""),
        "eligibility": eligibility,
        "approval_pending": approval_pending,
        "thread_id": state.get("thread_id", ""),
    }


def _pause_for_human_approval(state: RefundApprovalState) -> RefundApprovalState:
    """节点 2：暂停等待人工审批（LangGraph interrupt）。

    只有 approval_pending=True 时才会走到这个节点。
    interrupt() 抛出 GraphInterrupt，Pregel 把当前状态写入 checkpointer 后挂起；
    人工审批回调通过 Command(resume=...) 把决定送回，作为这里的返回值。
    """

    eligibility = state.get("eligibility")
    if state.get("approval_pending") and eligibility is not None:
        decision = interrupt({
            "type": "human_approval_required",
            "coupon_id": state.get("coupon_id"),
            "thread_id": state.get("thread_id"),
            "eligibility_status": eligibility.eligibility_status,
            "reasons": eligibility.reasons,
        })
        return {"approval_decision": decision if isinstance(decision, dict) else {"raw": decision}}
    return {}


def _build_refund_graph(checkpointer: Any = None):
    """构建退款审批 LangGraph 图。

    条件边决定两种走向：需要人工审批 → pause 节点；否则 → END。
    checkpointer 传入 SqliteSaver 后，interrupt 才有可恢复的现场。
    """

    graph = StateGraph(RefundApprovalState)
    graph.add_node("assess_eligibility", _assess_refund_eligibility)
    graph.add_node("pause_for_human_approval", _pause_for_human_approval)
    graph.set_entry_point("assess_eligibility")

    def _route_after_assess(state: RefundApprovalState) -> str:
        if state.get("approval_pending"):
            return "pause_for_human_approval"
        return END

    # 第三个参数是路径映射：路由函数的返回值 → 实际节点名。
    graph.add_conditional_edges("assess_eligibility", _route_after_assess, {
        "pause_for_human_approval": "pause_for_human_approval",
        END: END,
    })
    graph.add_edge("pause_for_human_approval", END)

    if not _LANGGRAPH_AVAILABLE:
        return None
    return graph.compile(checkpointer=checkpointer)


# 按 DB 路径缓存编译结果：测试切库后能拿到新的 saver，不会复用旧连接。
_GRAPH_CACHE: dict[str, Any] = {}
# 显式持有连接，避免 SqliteSaver 内部连接被回收。
_CHECKPOINTER_CONNECTIONS: dict[str, sqlite3.Connection] = {}


def _build_checkpointer(db_path: str) -> Any:
    """建 SQLite checkpointer；包没装就退回无持久化（条件边仍然工作）。"""

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        return None
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(connection)
        saver.setup()
    except Exception:
        return None
    _CHECKPOINTER_CONNECTIONS[db_path] = connection
    return saver


def get_refund_graph():
    """按当前 DB 路径懒加载退款审批图（带 checkpoint）。"""

    if not _LANGGRAPH_AVAILABLE:
        return None
    db_path = str(settings.DB_PATH)
    cached = _GRAPH_CACHE.get(db_path)
    if cached is not None:
        return cached
    graph = _build_refund_graph(_build_checkpointer(db_path))
    if graph is not None:
        _GRAPH_CACHE[db_path] = graph
    return graph


def reset_graph_cache() -> None:
    """测试用：切库后清掉编译缓存与连接。"""

    for connection in _CHECKPOINTER_CONNECTIONS.values():
        try:
            connection.close()
        except Exception:
            pass
    _CHECKPOINTER_CONNECTIONS.clear()
    _GRAPH_CACHE.clear()


def build_thread_id(session_id: str, coupon_id: str | None) -> str:
    """审批线程键：会话 + 卡券。

    同一会话里不同卡券必须各自独立挂起，否则第二单会复用第一单的资格结论
    （checkpoint 按 thread_id 存，一个会话只有一个槽）。
    没有卡券号时退回会话级（高风险分支也可能只带订单号）。

    发起（chat）与恢复（/chat/resume）必须用同一个公式，否则找不到存档。
    """

    coupon_id = (coupon_id or "").strip()
    return f"{session_id}:{coupon_id}" if coupon_id else session_id


def has_pending_approval(thread_id: str) -> bool:
    """该线程是否真的有挂起的审批（interrupt 已把状态写进 checkpoint）。

    回答文案依赖这个判断：LangGraph 不可用或 checkpointer 没建成时，图不会留下
    挂起状态，这时不能对用户说“已进入审批流程”——否则 /chat/resume 永远查不到单子。
    """

    graph = get_refund_graph()
    if graph is None:
        return False
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:  # noqa: BLE001 - 查询失败按「没有挂起」处理，不能让回答 500
        return False
    return bool(snapshot is not None and snapshot.next)


def run_refund_approval(
    *,
    coupon_id: str,
    coupon_status: str | None,
    thread_id: str,
    verified_at: str | None = None,
    order_id: str | None = None,
    runtime_user_id: str = "",
) -> HighRiskAssessment:
    """执行退款审批图。

    LangGraph 可用 → 跑图（需要审批会在 pause 节点中断并写入 checkpoint）
    LangGraph 不可用 → 降级为直接调 _assess_refund_eligibility

    verified_at：已核销券的核销时间，必须透传进图——否则 48 小时退款窗口
    在图路径上永远算不出来，会被误判成 not_eligible。
    order_id：用户只给订单号时也要透传，否则策略层会误判成「缺编号」而反复追问。
    runtime_user_id：冻结进 state，恢复前用它复核券归属和状态是否漂移。

    同一会话已有待审批的图时不重复发起，直接复用上次的资格结论（避免一单多审批）。
    """

    initial_state: RefundApprovalState = {
        "coupon_id": coupon_id,
        "coupon_status": coupon_status,
        "verified_at": verified_at,
        "order_id": order_id,
        "runtime_user_id": runtime_user_id,
        "eligibility": None,
        "approval_pending": False,
        "thread_id": thread_id,
    }

    graph = get_refund_graph()
    if graph is None:
        result_state = _assess_refund_eligibility(initial_state)
        return result_state["eligibility"]

    config = {"configurable": {"thread_id": thread_id}}

    # 已有挂起的审批：直接复用，不覆盖 checkpoint。
    snapshot = graph.get_state(config)
    if snapshot is not None and snapshot.next:
        pending = (snapshot.values or {}).get("eligibility")
        if pending is not None:
            return pending

    result_state = graph.invoke(initial_state, config=config)
    eligibility = result_state.get("eligibility")
    if eligibility is None:
        # interrupt 时返回的可能是挂起快照，兜底从 checkpoint 再取一次。
        snapshot = graph.get_state(config)
        eligibility = (snapshot.values or {}).get("eligibility") if snapshot else None
    if eligibility is None:
        return HighRiskAssessment(
            eligibility_status="needs_clarification",
            reasons=["退款资格判断异常，请联系客服。"],
            suggested_action="ask_clarification",
            risk_level="medium",
        )
    return eligibility
