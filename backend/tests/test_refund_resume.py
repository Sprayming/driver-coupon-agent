"""退款审批图的 checkpoint / interrupt / 幂等恢复验证（L31）。

覆盖四件事：
1. 需要审批时图真的挂起（checkpoint 里有 next，不是跑完返回）；
2. resume 能把人工决定送回图里，并且图能走到终态；
3. 同一个 approval_id 只处理一次（落库幂等，不依赖进程内存）；
4. 模拟重启（清空连接与编译缓存）后，挂起的审批仍能恢复。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config import settings
from store.db import init_db
from workflows import after_sale_graph
from workflows.resume import get_pending_approvals, resume_refund_workflow

THREAD = "sess-refund-1"


@pytest.fixture()
def refund_db(tmp_path, monkeypatch):
    path = tmp_path / "refund.db"
    monkeypatch.setattr(settings, "DB_PATH", path)
    init_db(path)
    after_sale_graph.reset_graph_cache()
    yield path
    after_sale_graph.reset_graph_cache()


def _config() -> dict:
    return {"configurable": {"thread_id": THREAD}}


def test_pause_writes_checkpoint_and_resume_completes(refund_db):
    assessment = after_sale_graph.run_refund_approval(
        coupon_id="COUPON-1", coupon_status="unused", thread_id=THREAD
    )
    assert assessment.needs_human_approval is True

    graph = after_sale_graph.get_refund_graph()
    # 关键断言：图停在 pause 节点，而不是直接跑完。
    assert graph.get_state(_config()).next == ("pause_for_human_approval",)

    result = resume_refund_workflow(
        thread_id=THREAD, approval_id="APV-1", approved=True, approver="lead"
    )
    assert result["status"] == "resumed"
    assert result["approval_decision"]["approved"] is True
    # 恢复后不应再有待执行节点。
    assert graph.get_state(_config()).next == ()


def test_same_approval_only_processed_once(refund_db):
    after_sale_graph.run_refund_approval(coupon_id="COUPON-2", coupon_status="unused", thread_id=THREAD)

    first = resume_refund_workflow(thread_id=THREAD, approval_id="APV-2", approved=False)
    second = resume_refund_workflow(thread_id=THREAD, approval_id="APV-2", approved=False)

    assert first["status"] == "resumed"
    assert second["status"] == "already_processed"
    assert second["previous_approved"] is False


def test_claim_lease_blocks_duplicate_within_window(refund_db):
    from store import approvals

    assert approvals.claim(THREAD, "APV-LEASE") is True
    # 租约内重复回调：仍然判为已占用，不重复执行。
    assert approvals.claim(THREAD, "APV-LEASE") is False


def test_expired_claim_can_be_taken_over(refund_db):
    import sqlite3

    from store import approvals

    assert approvals.claim(THREAD, "APV-LEASE") is True
    # 模拟上个进程 claim 后被 kill -9：把 created_at 挪到租约窗口之前。
    with sqlite3.connect(refund_db) as connection:
        connection.execute(
            "UPDATE approval_results SET created_at = datetime('now', '-10 minutes') "
            "WHERE approval_id = ?",
            ("APV-LEASE",),
        )
    # 过期占位允许接管，接管后租约重新计时。
    assert approvals.claim(THREAD, "APV-LEASE") is True
    assert approvals.claim(THREAD, "APV-LEASE") is False


def test_done_record_is_never_taken_over(refund_db):
    import sqlite3

    from store import approvals

    assert approvals.claim(THREAD, "APV-DONE") is True
    approvals.finish(THREAD, "APV-DONE", approved=True)
    # 即使把时间改到很久以前，已完成的结果也不能被再次执行。
    with sqlite3.connect(refund_db) as connection:
        connection.execute(
            "UPDATE approval_results SET created_at = datetime('now', '-30 days') "
            "WHERE approval_id = ?",
            ("APV-DONE",),
        )
    assert approvals.claim(THREAD, "APV-DONE") is False


def test_resume_without_pending_returns_no_graph(refund_db):
    result = resume_refund_workflow(thread_id="never-started", approval_id="APV-X", approved=True)
    assert result["status"] == "no_pending_graph"
    assert result["reason"] == "no_pending_checkpoint"


def test_pending_survives_restart(refund_db):
    after_sale_graph.run_refund_approval(coupon_id="COUPON-3", coupon_status="unused", thread_id=THREAD)

    # 模拟进程重启：丢掉连接与编译缓存，重新按同一 DB 建图。
    after_sale_graph.reset_graph_cache()

    result = resume_refund_workflow(
        thread_id=THREAD, approval_id="APV-3", approved=True, approver="ops"
    )
    assert result["status"] == "resumed"
    assert result["eligibility_status"] == "eligible_for_application"


def test_pending_lookup_reads_database(refund_db):
    after_sale_graph.run_refund_approval(coupon_id="COUPON-4", coupon_status="unused", thread_id=THREAD)
    resume_refund_workflow(thread_id=THREAD, approval_id="APV-4", approved=True)

    rows = get_pending_approvals(THREAD)
    assert [row["approval_id"] for row in rows] == ["APV-4"]
    assert rows[0]["status"] == "done"


def test_blocked_coupon_does_not_pause(refund_db):
    assessment = after_sale_graph.run_refund_approval(
        coupon_id="COUPON-5", coupon_status="redeemed", thread_id="sess-blocked"
    )
    assert assessment.eligibility_status == "not_eligible"
    # 策略层恒定 needs_human_approval=True，真正决定是否挂起的是图的路由。
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-blocked"}}
    )
    assert snapshot.next == ()


def _hours_ago(hours: int) -> str:
    """构造与 refund_policy 同口径的 naively 本地时间字符串（不带时区）。"""

    return (datetime.now() - timedelta(hours=hours)).replace(microsecond=0).isoformat(sep=" ")


def test_verified_coupon_within_48h_pauses(refund_db):
    """已核销但在 48h 内 → 可申请退款 → 图必须停在审批节点。

    verified_at 不透传进图时，这里会落成 not_eligible（政策层拿不到核销时间），
    所以这条同时是「透传」的回归测试。
    """

    assessment = after_sale_graph.run_refund_approval(
        coupon_id="COUPON-48H",
        coupon_status="verified",
        verified_at=_hours_ago(1),
        thread_id="sess-48h",
    )
    assert assessment.eligibility_status == "eligible_for_application"
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-48h"}}
    )
    assert snapshot.next == ("pause_for_human_approval",)


def test_verified_coupon_beyond_48h_does_not_pause(refund_db):
    """已核销超 48h → 不可退，且不应占住 checkpoint。"""

    assessment = after_sale_graph.run_refund_approval(
        coupon_id="COUPON-OLD",
        coupon_status="verified",
        verified_at=_hours_ago(72),
        thread_id="sess-old",
    )
    assert assessment.eligibility_status == "not_eligible"
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-old"}}
    )
    assert snapshot.next == ()


def test_verified_coupon_without_timestamp_is_not_eligible(refund_db):
    """已核销但没拿到核销时间 → 仍按不可退处理（保守，不猜）。"""

    assessment = after_sale_graph.run_refund_approval(
        coupon_id="COUPON-NOTIME",
        coupon_status="verified",
        verified_at=None,
        thread_id="sess-notime",
    )
    assert assessment.eligibility_status == "not_eligible"


# ---------------------------------------------------------------------------
# P1：恢复前业务事实复核（审批等待期间券状态漂移）
# ---------------------------------------------------------------------------


def _start_pending_refund(thread_id: str, coupon_id: str = "CP-DRIFT", status: str = "unused") -> None:
    after_sale_graph.run_refund_approval(
        coupon_id=coupon_id,
        coupon_status=status,
        thread_id=thread_id,
        runtime_user_id="driver_001",
    )


def test_recheck_blocks_when_coupon_status_drifted(refund_db, monkeypatch):
    """发起时未核销，审批期间被核销 → 拒绝恢复，单子仍挂在审批节点。"""

    from store import approvals
    from workflows import resume as resume_module

    _start_pending_refund("sess-drift")
    monkeypatch.setattr(
        resume_module,
        "driver_coupon_fact_from_service",
        lambda driver_id: ([{"couponId": "CP-DRIFT", "status": "verified"}], True),
    )

    result = resume_module.resume_refund_workflow(
        thread_id="sess-drift", approval_id="APV-DRIFT", approved=True
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "business_fact_drift"
    assert result["business_recheck"]["mismatches"]["coupon_status"] == {
        "frozen": "unused",
        "current": "verified",
    }
    # 没有抢占、没有恢复：图仍停在 pause，可以等人工重新确认后重试。
    assert approvals.get("sess-drift", "APV-DRIFT") is None
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-drift"}}
    )
    assert snapshot.next == ("pause_for_human_approval",)


def test_recheck_blocks_when_coupon_not_found(refund_db, monkeypatch):
    """审批期间券已不在该司机名下 → 拒绝恢复。"""

    from workflows import resume as resume_module

    _start_pending_refund("sess-gone")
    monkeypatch.setattr(
        resume_module,
        "driver_coupon_fact_from_service",
        lambda driver_id: ([{"couponId": "CP-OTHER", "status": "unused"}], True),
    )

    result = resume_module.resume_refund_workflow(
        thread_id="sess-gone", approval_id="APV-GONE", approved=True
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "coupon_not_found"


def test_recheck_passes_when_status_unchanged(refund_db, monkeypatch):
    """状态没变 → 正常恢复，并把复核结果带回去。"""

    from workflows import resume as resume_module

    _start_pending_refund("sess-same")
    monkeypatch.setattr(
        resume_module,
        "driver_coupon_fact_from_service",
        lambda driver_id: ([{"couponId": "CP-DRIFT", "status": "unused"}], True),
    )

    result = resume_module.resume_refund_workflow(
        thread_id="sess-same", approval_id="APV-SAME", approved=True
    )
    assert result["status"] == "resumed"
    assert result["business_recheck"]["passed"] is True
    assert result["business_recheck"]["mismatches"] == {}


def test_recheck_unavailable_does_not_block_but_is_marked(refund_db, monkeypatch):
    """后端不可用 → 不阻断，但必须显式标记成 recheck_unavailable（不是 passed）。"""

    from workflows import resume as resume_module

    _start_pending_refund("sess-down")
    monkeypatch.setattr(
        resume_module,
        "driver_coupon_fact_from_service",
        lambda driver_id: (None, True),
    )

    result = resume_module.resume_refund_workflow(
        thread_id="sess-down", approval_id="APV-DOWN", approved=True
    )
    assert result["status"] == "resumed"
    assert result["business_recheck"]["reason"] == "recheck_unavailable"


# ---------------------------------------------------------------------------
# P2：审批线程键按「会话 + 卡券」隔离
# ---------------------------------------------------------------------------


def test_build_thread_id_scopes_by_coupon():
    assert after_sale_graph.build_thread_id("s1", "CP-1") == "s1:CP-1"
    # 没有卡券号（只带订单号发起）→ 退回会话级
    assert after_sale_graph.build_thread_id("s1", None) == "s1"
    assert after_sale_graph.build_thread_id("s1", "  ") == "s1"


def test_two_coupons_in_same_session_pause_independently(refund_db):
    """同一会话的第二张券不再复用第一张的结论，各自有独立存档。"""

    first = after_sale_graph.run_refund_approval(
        coupon_id="CP-A",
        coupon_status="unused",
        thread_id=after_sale_graph.build_thread_id("s1", "CP-A"),
    )
    second = after_sale_graph.run_refund_approval(
        coupon_id="CP-B",
        coupon_status="redeemed",
        thread_id=after_sale_graph.build_thread_id("s1", "CP-B"),
    )

    assert first.eligibility_status == "eligible_for_application"
    assert second.eligibility_status == "not_eligible"
    assert second.coupon_id == "CP-B"

    graph = after_sale_graph.get_refund_graph()
    assert graph.get_state({"configurable": {"thread_id": "s1:CP-A"}}).next == (
        "pause_for_human_approval",
    )
    # 不可退的券不占存档
    assert graph.get_state({"configurable": {"thread_id": "s1:CP-B"}}).next == ()


def test_resume_route_uses_coupon_scoped_thread(refund_db, monkeypatch):
    """恢复接口必须用和发起时相同的公式拼线程键。"""

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routes import create_router
    from workflows import resume as resume_module

    captured: dict = {}

    def fake_resume(**kwargs):
        captured.update(kwargs)
        return {"status": "resumed"}

    monkeypatch.setattr(resume_module, "resume_refund_workflow", fake_resume)

    app = FastAPI()
    app.include_router(create_router(lambda: None))
    response = TestClient(app).post(
        "/chat/resume",
        json={
            "session_id": "s1",
            "approval_id": "APV-P2",
            "approved": True,
            "coupon_id": "CP-A",
        },
    )

    assert response.status_code == 200
    assert captured["thread_id"] == "s1:CP-A"


# ---------------------------------------------------------------------------
# 编号提取与卡券匹配：不允许裸数字、不允许拿第一条兜底
# ---------------------------------------------------------------------------


def test_identifiers_require_prefix():
    """卡券号里的数字段不能被当成订单号，卡券号本身也不能被截断。"""

    from agents.customer_service_agent import _COUPON_ID_PATTERN, _ORDER_ID_PATTERN

    text = "CP-20261015-001 这张券要退款"
    coupon_match = _COUPON_ID_PATTERN.search(text)
    assert coupon_match is not None
    assert coupon_match.group(0) == "CP-20261015-001"
    assert _ORDER_ID_PATTERN.search(text) is None

    order_text = "订单 O-99283 要退款"
    order_match = _ORDER_ID_PATTERN.search(order_text)
    assert order_match is not None
    assert order_match.group(0) == "O-99283"
    assert _COUPON_ID_PATTERN.search(order_text) is None


def test_find_coupon_never_falls_back_to_first():
    """匹配不到就返回 None，不能拿列表第一条顶替。"""

    from agents.customer_service_agent import find_coupon

    coupons = [{"couponId": "CP-A", "status": "unused"}]
    assert find_coupon(coupons, "CP-B") is None
    assert find_coupon(coupons, "CP-A")["status"] == "unused"
    # 后端给 snake_case 时也要能匹配
    assert find_coupon([{"coupon_id": "cp-a", "status": "expired"}], "CP-A")["status"] == "expired"


def test_order_only_request_is_blocked_not_clarification(refund_db):
    """只给订单号 → 不再判定为「缺编号」，而是拿不到卡券状态 → blocked。"""

    assessment = after_sale_graph.run_refund_approval(
        coupon_id="",
        coupon_status=None,
        order_id="O-99283",
        thread_id="sess-order",
    )
    assert assessment.eligibility_status == "blocked"
    assert assessment.order_id == "O-99283"
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-order"}}
    )
    assert snapshot.next == ()


def test_unknown_coupon_without_status_is_blocked(refund_db):
    """给了卡券号但查不到状态 → blocked，且不占 checkpoint。"""

    assessment = after_sale_graph.run_refund_approval(
        coupon_id="CP-UNKNOWN",
        coupon_status=None,
        thread_id="sess-unknown",
    )
    assert assessment.eligibility_status == "blocked"
    snapshot = after_sale_graph.get_refund_graph().get_state(
        {"configurable": {"thread_id": "sess-unknown"}}
    )
    assert snapshot.next == ()


# ---------------------------------------------------------------------------
# 审批是否真的建立：决定回答能不能说“已进入审批”
# ---------------------------------------------------------------------------


def test_has_pending_approval_reflects_checkpoint(refund_db):
    thread_id = after_sale_graph.build_thread_id("s1", "CP-P")
    assert after_sale_graph.has_pending_approval(thread_id) is False

    after_sale_graph.run_refund_approval(
        coupon_id="CP-P", coupon_status="unused", thread_id=thread_id
    )
    assert after_sale_graph.has_pending_approval(thread_id) is True


def test_has_pending_approval_false_when_graph_unavailable(refund_db, monkeypatch):
    """LangGraph 不可用时没有 checkpoint，回答不能声称已进入审批。"""

    monkeypatch.setattr(after_sale_graph, "get_refund_graph", lambda: None)
    assert after_sale_graph.has_pending_approval("s1:CP-X") is False


# ---------------------------------------------------------------------------
# 作废挂起的审批单（P3）：不再永久悬挂
# ---------------------------------------------------------------------------


def test_cancel_clears_pending_without_business_effect(refund_db):
    from store import approvals
    from workflows import resume as resume_module

    thread_id = after_sale_graph.build_thread_id("s1", "CP-C")
    after_sale_graph.run_refund_approval(
        coupon_id="CP-C", coupon_status="unused", thread_id=thread_id
    )
    assert after_sale_graph.has_pending_approval(thread_id) is True

    result = resume_module.resume_refund_workflow(
        thread_id=thread_id,
        approval_id="APV-C",
        approved=False,
        cancel=True,
        comment="券已被核销",
    )

    assert result["status"] == "cancelled"
    assert result["approved"] is False
    # 图上不再有挂起节点：后续重复回调会走 already_processed。
    assert after_sale_graph.has_pending_approval(thread_id) is False

    row = approvals.get(thread_id, "APV-C")
    assert row["status"] == "done"
    assert row["approved"] == 0
    assert row["comment"] == "cancelled: 券已被核销"


def test_cancel_skips_business_recheck(refund_db, monkeypatch):
    """状态漂移恰恰是作废的理由，不能被复核挡住。"""

    from workflows import resume as resume_module

    thread_id = after_sale_graph.build_thread_id("s1", "CP-D")
    after_sale_graph.run_refund_approval(
        coupon_id="CP-D", coupon_status="unused", thread_id=thread_id
    )
    monkeypatch.setattr(
        resume_module,
        "_recheck_coupon_state",
        lambda values: {"passed": False, "reason": "business_fact_drift", "mismatches": {}},
    )

    result = resume_module.resume_refund_workflow(
        thread_id=thread_id, approval_id="APV-D", approved=False, cancel=True
    )
    assert result["status"] == "cancelled"


def test_cancel_twice_is_idempotent(refund_db):
    from workflows import resume as resume_module

    thread_id = after_sale_graph.build_thread_id("s1", "CP-E")
    after_sale_graph.run_refund_approval(
        coupon_id="CP-E", coupon_status="unused", thread_id=thread_id
    )

    first = resume_module.resume_refund_workflow(
        thread_id=thread_id, approval_id="APV-E", approved=False, cancel=True
    )
    second = resume_module.resume_refund_workflow(
        thread_id=thread_id, approval_id="APV-E", approved=False, cancel=True
    )

    assert first["status"] == "cancelled"
    assert second["status"] == "already_processed"


def test_resume_route_passes_cancel_flag(refund_db, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routes import create_router
    from workflows import resume as resume_module

    captured: dict = {}

    def fake_resume(**kwargs):
        captured.update(kwargs)
        return {"status": "cancelled"}

    monkeypatch.setattr(resume_module, "resume_refund_workflow", fake_resume)

    app = FastAPI()
    app.include_router(create_router(lambda: None))
    response = TestClient(app).post(
        "/chat/resume",
        json={
            "session_id": "s1",
            "approval_id": "APV-C",
            # 作废时故意不传 approved，验证默认值兜底。
            "coupon_id": "CP-C",
            "cancel": True,
        },
    )

    assert response.status_code == 200
    assert captured["cancel"] is True
    assert captured["approved"] is False
    assert captured["thread_id"] == "s1:CP-C"
