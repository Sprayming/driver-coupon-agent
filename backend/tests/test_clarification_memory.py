"""澄清状态机（pending_clarification）的单测。

这块钱此前零覆盖，导致「用户只补一个卡券号被误判成换话题」的缺陷长期没被发现。
三条用例锁定 consume_pending_clarification 的核心行为：
1. unknown 不算新意图，补完参数要能恢复；
2. 明确的新业务意图（如投诉）要取消旧澄清；
3. 超过轮次窗口后即使补了参数也不能恢复。
"""

from __future__ import annotations

from context.builder import (
    PENDING_CLARIFICATION_TURN_WINDOW,
    consume_pending_clarification,
    remember_pending_clarification,
)
from state.session_state import current_memory

INTENT = "refund_request"
FIELD = "coupon_id"
COUPON = "CP123456"


def _remember(session_id: str, user_id: str, message_count: int = 1) -> None:
    remember_pending_clarification(
        session_id=session_id,
        runtime_user_id=user_id,
        intent=INTENT,
        required_field=FIELD,
        current_message_count=message_count,
    )


def test_unknown_intent_still_restores_clarification():
    """只补一个卡券号时 classify_intent 返回 unknown，不能被当成「换了话题」。"""

    _remember("s-unknown", "u-unknown")

    resolved, transition = consume_pending_clarification(
        session_id="s-unknown",
        runtime_user_id="u-unknown",
        current_intent="unknown",
        supplied_fields={FIELD: COUPON},
        current_message_count=2,
    )

    assert resolved is not None
    assert resolved["intent"] == INTENT
    assert resolved["supplied_fields"] == {FIELD: COUPON}
    assert transition is not None
    assert transition["to"] == "resolved"


def test_new_business_intent_cancels_clarification():
    """用户换成投诉，旧的退款澄清必须取消，不能再用旧意图覆盖本轮判断。"""

    _remember("s-switch", "u-switch")

    resolved, transition = consume_pending_clarification(
        session_id="s-switch",
        runtime_user_id="u-switch",
        current_intent="complaint",
        supplied_fields={},
        current_message_count=2,
    )

    assert resolved is None
    # L41 对齐：被新意图取代时 transition 带 reason，供上层写 Trace。
    assert transition is not None
    assert transition["reason"] == "superseded"
    assert transition["pending_intent"] == INTENT
    assert transition["current_intent"] == "complaint"
    assert "pending_clarification" not in current_memory("s-switch", "u-switch")


def test_expired_clarification_is_dropped():
    """超过轮次窗口后即使补了参数也不能恢复，避免跨天误认。"""

    _remember("s-expired", "u-expired", message_count=1)

    resolved, transition = consume_pending_clarification(
        session_id="s-expired",
        runtime_user_id="u-expired",
        current_intent="unknown",
        supplied_fields={FIELD: COUPON},
        current_message_count=1 + PENDING_CLARIFICATION_TURN_WINDOW + 1,
    )

    assert resolved is None
    # L41 对齐：过期时 transition 带 reason/age/required_fields，供上层写 tool_clarification_expired。
    assert transition is not None
    assert transition["reason"] == "expired"
    assert transition["intent"] == INTENT
    assert transition["required_fields"] == [FIELD]
    assert transition["age"] == PENDING_CLARIFICATION_TURN_WINDOW + 1
    assert "pending_clarification" not in current_memory("s-expired", "u-expired")
