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

import re
from typing import Any

from api.schemas import ChatRequest, HistoryMessage
from cost.governance import estimate_tokens
from safety.source_guard import inspect_source
from state.session_state import current_memory
from tools.runtime_context import mask_sensitive_text

# 澄清状态过期窗口：超过 N 轮没补参数就算失效
PENDING_CLARIFICATION_TURN_WINDOW = 3

# L34/L35：历史只保留最近窗口 + 与当前卡券相关的轮次，超出 token 预算的直接丢弃。
RECENT_WINDOW_SIZE = 4
MAX_HISTORY_TOKENS = 80


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
    ② 过期了（消息计数 > expires_after）→ 清掉，返回 (None, {"reason": "expired", ...})
    ③ 用户说了新意图（不是 general_chat）→ 取消旧澄清，返回 (None, {"reason": "superseded", ...})
    ④ required_fields 还没全补 → 等待，返回 (None, {"reason": "waiting", ...})
    ⑤ 全补完 → 成功恢复，返回 (resolved, {"reason": "resolved", ...})

    Args:
        session_id: 会话 ID
        runtime_user_id: 司机 ID
        current_intent: 这一轮 classify_intent 的结果
        supplied_fields: 用户这一轮补了哪些参数 {coupon_id: "CP..."}
        current_message_count: 当前是第几轮

    Returns:
        (resolved_dict, transition_dict)
        - 没有 pending 时 transition 为 None（保持原语义，调用方无需处理）
        - 其余情况的 transition 带 reason，供调用方写 Trace（L41 对齐 tool_clarification_expired）
        resolved_dict 包含 intent / required_fields / supplied_fields

    被谁调用：agents/customer_service_agent.py chat() 阶段 2（意图识别之后）
    """
    memory = current_memory(session_id, runtime_user_id)
    pending = memory.get("pending_clarification")

    # ① 没有待澄清
    if not isinstance(pending, dict):
        return None, None

    required = pending.get("required_fields", []) or []

    # ② 过期了
    if current_message_count > pending.get("expires_after_message_count", 0):
        created_count = pending.get("created_message_count", current_message_count)
        expired_intent = pending.get("intent")
        memory.pop("pending_clarification", None)
        return None, {
            "reason": "expired",
            "intent": expired_intent,
            "required_fields": required,
            "age": current_message_count - created_count,
        }

    # ③ 用户说了明确的新意图 → 取消旧澄清。
    # 必须同时排除 unknown：用户只补一个卡券号（CP123456）时 classify_intent 返回 unknown，
    # 若把它当成「换了话题」，澄清在第 ⑤ 层之前就被取消，恢复永远走不到。
    if current_intent and current_intent not in ("general_chat", "unknown"):
        pending_intent = pending.get("intent")
        memory.pop("pending_clarification", None)
        return None, {
            "reason": "superseded",
            "pending_intent": pending_intent,
            "current_intent": current_intent,
        }

    # ④ 检查 required_fields 是否全补了
    supplied = {k: v for k, v in supplied_fields.items() if v}
    missing = [f for f in required if not supplied.get(f)]

    if missing:
        return None, {"reason": "waiting", "required_fields": required}

    # ⑤ 全补完 → 成功恢复
    resolved = {
        "intent": pending["intent"],
        "required_fields": required,
        "supplied_fields": supplied,
    }
    transition = {
        "reason": "resolved",
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


def build_context(
    request: ChatRequest,
    explicit_coupon_id: str | None,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """按 explicit > Runtime Context > Session Memory 选卡券，并压缩历史消息。

    返回三元组，和 L41 的 `build_context` 同形：
    (chosen_coupon_id, context_report, compression_report)

    安全口径：
    - 模型只看到脱敏后的历史行，原始手机号/身份证/银行卡不进入上下文报告
    - 用户自称的等级、退款进度不能覆盖 Runtime Context 或 Workflow 事实，只记为冲突
    """

    memory = current_memory(request.session_id, request.runtime_user_id)
    page = request.runtime_context or {}
    page_coupon_id = (
        page.get("current_coupon_id")
        or page.get("currentCouponId")
        or _first_coupon_id(page)
    )
    chosen_coupon_id = find_context_coupon(request, explicit_coupon_id)
    conflicts: list[str] = []
    if (
        page_coupon_id
        and memory.get("verified_coupon_id")
        and page_coupon_id != memory["verified_coupon_id"]
        and not explicit_coupon_id
    ):
        conflicts.append("coupon_id: 页面 Runtime Context 与 Session Memory 冲突，采用页面卡券。")
    if any(term in request.user_message for term in ("我是VIP", "我是 VIP", "我是黑卡", "我是会员")):
        if (request.runtime_driver_level or "unknown").lower() not in {"vip", "black", "黑卡", "member"}:
            conflicts.append("driver_level: 用户自称与 Runtime Context 冲突，采用系统等级。")
    if any(term in request.user_message for term in ("已经退款", "已经到账", "审批通过", "主管同意", "客服说可以退")):
        conflicts.append("refund_state: 用户说法不能覆盖 Workflow 审批与退款状态。")

    kept_history, dropped_history = _compress_history(request.history_messages, chosen_coupon_id)
    # 模型只看非敏感运行标签；driver_id 只在服务端工具层使用。
    model_context = [
        f"[runtime_context/trusted] driver_level={request.runtime_driver_level or 'unknown'}, "
        f"risk_level={request.runtime_risk_level or 'unknown'}",
    ]
    source_reports: list[dict[str, Any]] = []
    user_safe, user_reason = inspect_source(request.user_message)
    source_reports.append({"source": "user_message", "tainted": not user_safe, "reason": user_reason})
    for item in kept_history:
        redacted = _redact_history(item.content)
        item_safe, item_reason = inspect_source(redacted)
        source_reports.append(
            {"source": "history_messages", "tainted": not item_safe, "reason": item_reason}
        )
        # L41 对齐：被污染的历史消息不能原文进模型上下文，整段替换成 ***
        model_context.append(
            f"[history/session] {item.role}: {redacted if item_safe else '***'}"
        )
    if chosen_coupon_id:
        model_context.append(f"[coupon_reference/session] chosen_coupon_id={chosen_coupon_id}")

    context_report = {
        "schema_version": "context_build_report_v1",
        "sources": ["runtime_context", "session_memory", "history_messages", "user_message"],
        "trust_order": [
            "runtime_context",
            "verified_tool_fact",
            "session_memory",
            "history_messages",
            "user_message",
        ],
        "chosen_coupon_id": chosen_coupon_id,
        "conflict_resolutions": conflicts,
        "model_context": model_context,
        "source_safety": {
            "tainted": any(report["tainted"] for report in source_reports),
            "tainted_sources": sorted(
                {report["source"] for report in source_reports if report["tainted"]}
            ),
            "reports": source_reports,
        },
    }
    compression_report = {
        "schema_version": "context_compression_v1",
        "recent_window_size": RECENT_WINDOW_SIZE,
        "token_budget": MAX_HISTORY_TOKENS,
        "input_count": len(request.history_messages),
        "kept_count": len(kept_history),
        "dropped_count": len(dropped_history),
        "kept_indexes": [
            index
            for index, item in enumerate(request.history_messages)
            if any(item is kept for kept in kept_history)
        ],
        "strategy": "recent_window_plus_coupon_relevance",
        "relevance_score": {
            "matching_coupon_reference": 100,
            "recent_window": 80,
            "older_history": 20,
        },
    }
    return chosen_coupon_id, context_report, compression_report


def _first_coupon_id(runtime_context: dict[str, Any]) -> str | None:
    """从 Runtime Context 的 currentCoupons 里取第一张卡券号。"""

    coupons = runtime_context.get("currentCoupons") or runtime_context.get("current_coupons")
    if not isinstance(coupons, list) or not coupons:
        return None
    first = coupons[0]
    if not isinstance(first, dict):
        return None
    coupon_id = first.get("coupon_id") or first.get("couponId") or first.get("id")
    return str(coupon_id) if coupon_id else None


def _compress_history(
    history: list[HistoryMessage], coupon_id: str | None
) -> tuple[list[HistoryMessage], list[HistoryMessage]]:
    """只保留最近窗口 + 命中当前卡券的轮次，并受 token 预算限制。"""

    recent_start = max(0, len(history) - RECENT_WINDOW_SIZE)
    selected_indexes = set(range(recent_start, len(history)))
    if coupon_id:
        selected_indexes.update(
            index for index, item in enumerate(history) if coupon_id in item.content
        )

    kept: list[HistoryMessage] = []
    used_tokens = 0
    for index in sorted(selected_indexes, reverse=True):
        item = history[index]
        tokens = estimate_tokens(item.content)
        if used_tokens + tokens <= MAX_HISTORY_TOKENS or (coupon_id and coupon_id in item.content):
            kept.append(item)
            used_tokens += tokens
    kept.reverse()
    kept_ids = {id(item) for item in kept}
    return kept, [item for item in history if id(item) not in kept_ids]


def _redact_history(content: str) -> str:
    """历史消息进入模型上下文前先做隐私脱敏。"""

    return mask_sensitive_text(content)


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
