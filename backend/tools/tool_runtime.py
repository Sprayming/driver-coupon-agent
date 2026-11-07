"""L17/L18 司机域：工具执行层。

L17：summary 函数做「字段取值 → 可读句子」的确定性转换（deterministic_answer）。
L18：新增 validate_tool_action / execute_tool_action，把 LLM 产出的 ToolAction
     转成 BusinessFactNeed 交給 BusinessFactService.lookup 跑三道安全闸，
     再把 BusinessFactResult 转回 ToolObservation。安全闸一道没动。
"""

from __future__ import annotations

from typing import Any

from api.schemas import BusinessFactNeed, ChatRequest, ToolAction, ToolObservation
from tools.contracts import TOOL_SPECS


# L18：工具名 → BusinessFactNeed.kind 的映射（7 个工具对 7 种事实）
_TOOL_NAME_TO_KIND: dict[str, str] = {
    "get_activity_progress": "driver_activity",
    "get_driver_order_count": "driver_order",
    "get_activity_qualification": "driver_qualification",
    "get_refund_status": "driver_refund",
    "get_coupon_status": "driver_coupon",
    "get_verification_record": "driver_verification",
    "get_income_summary": "driver_income",
}


def validate_tool_action(action: ToolAction) -> ToolObservation | None:
    """校验 LLM 给出的工具名和必填参数。返回 None 表示通过。"""
    spec = TOOL_SPECS.get(action.tool_name)
    if spec is None:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary=f"工具 {action.tool_name} 不存在。",
        )
    missing = [field for field in spec.required if not action.arguments.get(field)]
    if missing:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary=f"工具参数缺失：{', '.join(missing)}。",
            data={"missing": missing, "parameters_schema": spec.parameters_schema},
        )
    return None


def _tool_action_to_need(action: ToolAction) -> BusinessFactNeed:
    """把 LLM 的 ToolAction 转成 BusinessFactService 认识的 BusinessFactNeed。"""
    kind = _TOOL_NAME_TO_KIND.get(action.tool_name, "unknown")
    args = action.arguments
    return BusinessFactNeed(
        kind=kind,
        requires_realtime=True,
        activity_id=args.get("activity_id"),
        time_range=args.get("time_range"),
        order_id=args.get("order_id"),
        coupon_id=args.get("coupon_id"),
        reason=f"LangChain 选择工具 {action.tool_name}",
    )


def execute_tool_action(action: ToolAction, request: ChatRequest) -> ToolObservation:
    """执行后端受控的只读工具调用（三道安全闸在 BusinessFactService.lookup 内）。

    LLM 只能决定「调哪个工具、传什么参数」，权限/归属/查不到全由后端闸卡死。
    """
    validation_error = validate_tool_action(action)
    if validation_error:
        return validation_error

    if action.tool_name not in _TOOL_NAME_TO_KIND:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary=f"工具 {action.tool_name} 未注册到司机域执行层。",
        )

    need = _tool_action_to_need(action)
    # 复用 L17 的 BusinessFactService：三道闸（缺参数/越权/查不到）原样保留。
    # 懒导入避免 services.business_facts ↔ tools.tool_runtime 循环导入。
    from services.business_facts import BusinessFactService
    fact_result = BusinessFactService().lookup(need, request)

    status = "success" if fact_result.found else "error"
    return ToolObservation(
        tool_name=action.tool_name,
        status=status,
        summary=fact_result.summary,
        data={
            "found": fact_result.found,
            "user_matched": fact_result.user_matched,
            "source": fact_result.source,
            **fact_result.facts,
        },
    )


def summarize_driver_activity(activity_fact: dict[str, Any]) -> str:
    """把司机活动完成度事实整理成可直接回答用户的摘要（三态）。

    期望字段（driver_service_client 接真实接口后按后端实际字段调整）：
      - targetOrders: 目标单量
      - completedOrders: 已完成单量
      - activityName / activityId: 活动标识
      - deadline: 截止时间（可选）
    """
    target = activity_fact.get("targetOrders") or activity_fact.get("target") or 0
    completed = activity_fact.get("completedOrders") or activity_fact.get("completed") or 0
    name = str(activity_fact.get("activityName") or activity_fact.get("activityId") or "该活动")
    deadline = str(activity_fact.get("deadline") or activity_fact.get("endTime") or "").strip()

    # 三态判定（小哲 L17 的 deterministic_answer：代码做决策，不交给 LLM）
    if not target:
        return f"{name} 暂时没查到目标单量数据，无法判断你是否达标。"

    suffix = f"（截止 {deadline}）" if deadline else ""
    if completed >= target:
        return f"{name}{suffix}：你已完成 {completed} 单，达标了！"
    remaining = target - completed
    return f"{name}{suffix}：目标 {target} 单，你已完成 {completed} 单，还差 {remaining} 单。"


def summarize_driver_order_count(order_fact: dict[str, Any], time_range: str) -> str:
    """把司机单量事实整理成可读摘要。

    期望字段：totalOrders（总单量），可扩展分时段（peakOrders / offPeakOrders 等）。
    """
    total = order_fact.get("totalOrders") or order_fact.get("orderCount") or 0
    if not total:
        return f"{time_range}暂时没查到你的单量数据。"
    extra_parts: list[str] = []
    for key, label in [("peakOrders", "高峰单量"), ("nightOrders", "夜间单量")]:
        val = order_fact.get(key)
        if val:
            extra_parts.append(f"{label} {val} 单")
    extra = f"，其中 {'，'.join(extra_parts)}" if extra_parts else ""
    return f"{time_range}你共完成 {total} 单{extra}。"


# ── L17 新增四类 summary ──


def summarize_driver_refund(refund: dict[str, Any]) -> str:
    """单条退款记录摘要。"""
    status_text = refund.get("statusText") or refund.get("status") or "未知"
    amount = refund.get("amount", 0)
    order_id = refund.get("orderId", "")
    applied = refund.get("appliedAt", "")
    completed = refund.get("completedAt") or "处理中"
    bank = refund.get("bankAccount", "")
    reason = refund.get("rejectReason") or ""

    if refund.get("status") == "refunded":
        return f"订单 {order_id} 的退款 {amount} 元已到账（{completed}，到账账户 {bank}）。"
    if refund.get("status") == "processing":
        return f"订单 {order_id} 的退款 {amount} 元正在处理中（申请于 {applied}，当前还未到账）。"
    if refund.get("status") == "rejected":
        return f"订单 {order_id} 的退款 {amount} 元已驳回{'（原因：' + reason + '）' if reason else ''}。"
    return f"订单 {order_id} 的退款状态：{status_text}，金额 {amount} 元。"


def summarize_driver_refund_list(refunds: list[dict[str, Any]]) -> str:
    """多条退款记录摘要（最近三条）。"""
    if not refunds:
        return "最近没有退款记录。"
    lines = [summarize_driver_refund(r) for r in refunds]
    return "你的最近三条退款：\n" + "\n".join(f"  - {line}" for line in lines)


def summarize_driver_coupons(coupons: list[dict[str, Any]]) -> str:
    """司机卡券列表摘要。"""
    unused = [c for c in coupons if c.get("status") == "unused"]
    verified = [c for c in coupons if c.get("status") == "verified"]
    expired = [c for c in coupons if c.get("status") == "expired"]

    parts: list[str] = []
    if unused:
        parts.append(f"{len(unused)} 张未使用：" + "、".join(
            f"{c.get('name', '卡券')}（{c.get('validFrom')} ~ {c.get('validTo')}）"
            for c in unused
        ))
    if verified:
        parts.append(f"{len(verified)} 张已核销")
    if expired:
        parts.append(f"{len(expired)} 张已过期")

    if not parts:
        return "你当前没有卡券。"
    return f"你共有 {len(coupons)} 张卡券：" + "；".join(parts) + "。"


def summarize_driver_verification(verification: dict[str, Any]) -> str:
    """卡券核销记录摘要。"""
    coupon_id = verification.get("couponId", "")
    coupon_name = verification.get("couponName", coupon_id)
    verified = verification.get("verified", False)
    if not verified:
        return f"卡券 {coupon_name}（{coupon_id}）还没有被核销过。"
    verified_at = verification.get("verifiedAt", "")
    order_id = verification.get("orderId", "")
    amount = verification.get("amount", 0)
    return f"卡券 {coupon_name} 已于 {verified_at} 在订单 {order_id} 中核销（金额 {amount} 元）。"


def summarize_driver_income(income: dict[str, Any]) -> str:
    """司机收入统计摘要。"""
    time_range = income.get("timeRange", "本月")
    total = income.get("totalIncome", 0)
    coupon = income.get("couponIncome", 0)
    percent = income.get("couponIncomePercent", 0)
    orders = income.get("orders", 0)
    return (
        f"{time_range}你总收入 {total} 元，完成 {orders} 单；"
        f"其中卡券贡献 {coupon} 元，占 {percent}%。"
    )
