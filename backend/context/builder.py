"""L32-L34：会话上下文构建器。

和小哲电商 context/builder.py 对齐，但司机域用 coupon_id / activity_id / order_id：

核心函数：
- remember_pending_clarification：上一轮缺参数 → 写入 Session Memory
- consume_pending_clarification：这一轮用户补了参数 → 恢复原意图
- find_context_coupon：从三个来源选当前卡券号（显式 > Runtime Context > Session Memory）
- update_memory：把验证过的实体写回 Session Memory

本课次演化：
- L32 session-memory：pending_clarification 读写闭环
- L33 runtime-context：从 Runtime Context 拿候选列表
- L34 context-builder：上下文组装 + 历史压缩
- L35 context-compression：历史消息压缩

被谁调用：agents/customer_service_agent.py chat() 主流程
"""

from __future__ import annotations

from typing import Any

from api.schemas import ChatRequest
from state.session_state import current_memory

# 澄清状态过期窗口：超过 N 轮没补参数就算失效
PENDING_CLARIFICATION_TURN_WINDOW = 3


def remember_pending_clarification(
    *,
    session_id: str,
    runtime_user_id: str,
    intent: str,
    required_field: str,
    current_message_count: int,
) -> None:
    """把"上一轮缺什么参数"写入 Session Memory。

    Args:
        session_id: 会话 ID
        runtime_user_id: 司机 ID
        intent: 上一轮识别出的意图（补完参数后恢复用）
        required_field: 缺的参数名（coupon_id / activity_id / order_id）
        current_message_count: 当前是第几轮

    被谁调用：agents/customer_service_agent.py chat() 阶段 3（产生澄清时）
    """
    memory = current_memory(session_id, runtime_user_id)
    memory["pending_clarification"] = {
        "intent": intent,
        "required_fields": [required_field],
        "status": "waiting_for_user",
        "created_message_count": current_message_count,
        "expires_after_message_count": current_message_count + PENDING_CLARIFICATION_TURN_WINDOW,
    }


def consume_pending_clarification(
    *,
    session_id: str,
    runtime_user_id: str,
    current_intent: str,
    supplied_fields: dict[str, str | None],
    current_message_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """消费 Session Memory 里的 pending_clarification。

    五层检查：
    ① 没有 pending_clarification → 返回 (None, None)
    ② 过期了（消息计数 > expires_after）→ 清掉，返回 (None, None)
    ③ 用户说了新意图（不是 general_chat）→ 取消旧澄清
    ④ required_fields 还没全补 → 等待，返回 (None, None)
    ⑤ 全补完 → 成功恢复，返回 (resolved, transition)

    Args:
        session_id: 会话 ID
        runtime_user_id: 司机 ID
        current_intent: 这一轮 classify_intent 的结果
        supplied_fields: 用户这一轮补了哪些参数 {coupon_id: "CP..."}
        current_message_count: 当前是第几轮

    Returns:
        (resolved_dict, transition_dict) 或 (None, None)
        resolved_dict 包含 intent / required_fields / supplied_fields

    被谁调用：agents/customer_service_agent.py chat() 阶段 2（意图识别之后）
    """
    memory = current_memory(session_id, runtime_user_id)
    pending = memory.get("pending_clarification")

    # ① 没有待澄清
    if not isinstance(pending, dict):
        return None, None

    # ② 过期了
    if current_message_count > pending.get("expires_after_message_count", 0):
        memory.pop("pending_clarification", None)
        return None, None

    # ③ 用户说了新意图（不是 general_chat）→ 取消旧澄清
    if current_intent and current_intent != "general_chat":
        memory.pop("pending_clarification", None)
        return None, None

    # ④ 检查 required_fields 是否全补了
    required = pending.get("required_fields", [])
    supplied = {k: v for k, v in supplied_fields.items() if v}
    missing = [f for f in required if not supplied.get(f)]

    if missing:
        return None, None

    # ⑤ 全补完 → 成功恢复
    resolved = {
        "intent": pending["intent"],
        "required_fields": required,
        "supplied_fields": supplied,
    }
    transition = {
        "from": "waiting_for_user",
        "to": "resolved",
        "resolved_message_count": current_message_count,
    }
    memory.pop("pending_clarification", None)
    return resolved, transition


def find_context_coupon(request: ChatRequest, explicit_coupon_id: str | None) -> str | None:
    """从三个来源选当前卡券号：显式 > Runtime Context > Session Memory。

    优先级：
    1. explicit_coupon_id（用户消息里直接提的 CP...）
    2. request.runtime_context 里的当前卡券列表
    3. Session Memory 里 verified_coupon_id（上一轮验证过的）

    被谁调用：agents/customer_service_agent.py chat() 阶段 3
    """
    if explicit_coupon_id:
        return explicit_coupon_id

    # Runtime Context 里的当前卡券
    runtime_context = request.runtime_context or {}
    current_coupons = runtime_context.get("currentCoupons") or runtime_context.get("current_coupons")
    if isinstance(current_coupons, list) and current_coupons:
        first = current_coupons[0]
        if isinstance(first, dict):
            cid = first.get("coupon_id") or first.get("couponId") or first.get("id")
            if cid:
                return str(cid)

    # Session Memory 里上一轮验证过的
    verified = current_memory(request.session_id, request.runtime_user_id).get("verified_coupon_id")
    return verified


def find_context_activity(request: ChatRequest, explicit_activity_id: str | None) -> str | None:
    """从三个来源选当前活动 ID：显式 > Runtime Context > Session Memory。

    被谁调用：agents/customer_service_agent.py chat() 阶段 3
    """
    if explicit_activity_id:
        return explicit_activity_id

    runtime_context = request.runtime_context or {}
    current_activities = runtime_context.get("currentActivities") or runtime_context.get("current_activities")
    if isinstance(current_activities, list) and current_activities:
        first = current_activities[0]
        if isinstance(first, dict):
            aid = first.get("activity_id") or first.get("activityId") or first.get("id")
            if aid:
                return str(aid)

    verified = current_memory(request.session_id, request.runtime_user_id).get("verified_activity_id")
    return verified


def update_memory(
    *,
    session_id: str,
    runtime_user_id: str,
    intent: str,
    verified_coupon_id: str | None = None,
    verified_activity_id: str | None = None,
    verified_order_id: str | None = None,
    user_message: str = "",
) -> dict[str, Any]:
    """把验证过的业务实体写回 Session Memory。

    被谁调用：agents/customer_service_agent.py chat() 阶段 6
    """
    memory = current_memory(session_id, runtime_user_id)
    memory["last_intent"] = intent
    memory["last_user_message"] = user_message
    if verified_coupon_id:
        memory["verified_coupon_id"] = verified_coupon_id
    if verified_activity_id:
        memory["verified_activity_id"] = verified_activity_id
    if verified_order_id:
        memory["verified_order_id"] = verified_order_id
    return memory
