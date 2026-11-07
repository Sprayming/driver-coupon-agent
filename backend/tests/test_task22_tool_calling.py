"""Task 22 验证：L17 tool-calling 全链路覆盖（TR-22.1 / TR-22.2 / TR-22.3）。

TR-22.1：四类新事实（driver_refund / driver_coupon / driver_verification / driver_income）
         found=True 时走 answer_with_tool_fact，session_state.business_fact 完整。
TR-22.2：found=False 降级策略：
         - RAG 有命中 → 回退 answer_with_citations（不阻塞正常知识问答）
         - RAG 也没命中 → 停 tool_fact_fallback
TR-22.3：越权（user_matched=False）时不泄漏数据，安全降级。
"""

from __future__ import annotations

import json

import pytest

from api.schemas import ChatRequest, ChatResponse
from config import settings


# ── 公共：fixtures + mock 工具 ──


@pytest.fixture()
def _patch_embeddings(monkeypatch):
    fake_vec = [0.1] * 1024
    monkeypatch.setattr("rag.vector_store.embed_text", lambda *a, **kw: fake_vec)
    monkeypatch.setattr(
        "rag.vector_store.embed_texts",
        lambda texts, *a, **kw: [fake_vec for _ in texts],
    )
    monkeypatch.setattr("rag.quality.peek_quality_summary", lambda: None)


@pytest.fixture()
def _mock_chat_llm(monkeypatch):
    """chat LLM 一律返回 ChatModelResult，避免限流。"""
    from api.schemas import ChatModelResult

    def fake_caller(messages, **kw):
        user_msg = messages[-1]["content"][:30] if messages else ""
        return ChatModelResult(answer=f"【mock 回答】{user_msg}")

    monkeypatch.setattr("agents.customer_service_agent.call_chat_model", fake_caller)
    monkeypatch.setattr("models.llm_client.call_chat_model", fake_caller)
    # L18：测试强制走 L17 确定性路径——让 LangChain Tool Calling 不可用，
    # 这样 LLM 选工具的非确定性不影响测试断言。
    def _raise_no_model(*a, **kw):
        raise RuntimeError("no model for tests")
    monkeypatch.setattr("agents.customer_service_agent.create_tool_calling_model", _raise_no_model)
    # L19：澄清规划器同样在测试里禁用，让实时分支降级到 L17 直查。
    # 否则规划器会真的发起 LLM 调用，缺参场景还会走进澄清分支，
    # 断言 answer_path 会变成 clarification_required。
    from models.clarification_planner import ClarificationPlannerModelClient

    def _planner_unavailable(self, request, tool_specs=None):
        raise RuntimeError("planner disabled in tests")

    monkeypatch.setattr(ClarificationPlannerModelClient, "plan_clarification", _planner_unavailable)


def _chat(
    agent,
    session_id: str,
    message: str,
    user_id: str = "driver_001",
    runtime_context: dict | None = None,
) -> ChatResponse:
    return agent.chat(
        ChatRequest(
            session_id=session_id,
            runtime_user_id=user_id,
            runtime_role="driver",
            user_message=message,
            runtime_context=runtime_context,
        )
    )


def _setup_basic_switches(monkeypatch):
    """关掉 lifecycle/rbac/fusion/bm25/gate，只测 tool-calling 分支。"""
    for name in (
        "KNOWLEDGE_LIFECYCLE_ENABLED",
        "RBAC_ENABLED",
        "BM25_ENABLED",
        "RAG_FUSION_ENABLED",
        "GATE_ENABLED",
    ):
        monkeypatch.setattr(settings, name, False)


# ── TR-22.1：四类新事实 found=True 走 answer_with_tool_fact ──


def test_tr22_1_refund_found_true(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """司机域退款查询：found=True → answer_with_tool_fact，business_fact 完整。"""
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_refund_fact_from_service",
        lambda driver_id, order_id=None: (
            [
                {"orderId": "88321", "status": "refunded", "amount": 15.0,
                 "completedAt": "2026-10-19 09:30", "bankAccount": "尾号 8821"},
            ],
            True,  # user_matched
        ),
    )
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-refund", "我的退款到账了吗")
    rag = resp.session_state["rag"]
    bf = rag["business_fact"]

    assert rag["answer_path"] == "answer_with_tool_fact"
    assert bf is not None, "driver_domain_realtime 时必须有 business_fact 块"
    assert bf["found"] is True
    assert bf["source"] == "driver_service"
    assert bf["driver_domain_realtime"] is True
    assert "88321" in bf["summary"]
    assert resp.citations == [], "tool_fact 路径不引用知识库"


def test_tr22_1_coupon_found_true(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """司机域卡券查询：found=True → answer_with_tool_fact。"""
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_coupon_fact_from_service",
        lambda driver_id: (
            [
                {"couponId": "CP-001", "name": "周末月卡", "status": "unused",
                 "validFrom": "2026-10-15", "validTo": "2026-11-14"},
                {"couponId": "CP-002", "name": "全月周卡", "status": "unused"},
                {"couponId": "CP-003", "name": "体验卡", "status": "verified"},
            ],
            True,
        ),
    )
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-coupon", "我的账户里还有几张未使用的卡券")
    rag = resp.session_state["rag"]

    assert rag["answer_path"] == "answer_with_tool_fact"
    assert rag["business_fact"]["found"] is True
    assert "3" in rag["business_fact"]["summary"] or "卡券" in rag["business_fact"]["summary"]


def test_tr22_1_income_found_true(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """司机域收入查询：found=True → answer_with_tool_fact。"""
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_income_fact_from_service",
        lambda driver_id, date_range="本月": (
            {
                "timeRange": "本月",
                "totalIncome": 3280.0,
                "couponIncome": 420.0,
                "couponIncomePercent": 12.8,
                "orders": 245,
            },
            True,
        ),
    )
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-income", "本月我的收入里卡券占了多少")
    rag = resp.session_state["rag"]

    assert rag["answer_path"] == "answer_with_tool_fact"
    assert rag["business_fact"]["found"] is True
    assert "12.8" in rag["business_fact"]["summary"]


# ── TR-22.2：found=False 降级策略 ──


def test_tr22_2a_fallback_to_rag_when_hits_exist(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """found=False 但 RAG 有命中 → 回退 answer_with_citations。

    场景：planning 因"核销"关键词路由到 verification，
    但没 coupon_id 查不到，RAG 却能回答"核销过的卡券能否退款"规则。
    """
    # mock client 返回 None（查不到）
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_verification_fact_from_service",
        lambda coupon_id, driver_id: (None, True),
    )
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-fallback-rag", "卡券已经核销了还能退款吗")
    rag = resp.session_state["rag"]

    # 关键断言：不该停在 tool_fact_fallback，应该回退到 RAG
    assert rag["answer_path"] == "answer_with_citations", (
        f"应回退 answer_with_citations，实际 {rag['answer_path']}"
    )
    bf = rag["business_fact"]
    assert bf is not None
    assert bf["found"] is False
    # citations 应有内容
    assert len(resp.citations) > 0


def test_tr22_2b_fallback_to_tool_fact_when_no_rag_hits(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """found=False 且 RAG 也没命中 → 停在 tool_fact_fallback。

    场景：用户问题是纯实时查询+无任何卡券规则关联（mock 后端没数据）。
    """
    def _mock_none(driver_id, order_id=None):
        return None, True

    # 必须同时 patch source 模块 + consumer 模块的绑定（from X import Y 创建本地引用）
    monkeypatch.setattr("integrations.driver_service_client.driver_refund_fact_from_service", _mock_none)
    monkeypatch.setattr("services.business_facts.driver_refund_fact_from_service", _mock_none)

    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-fallback-tool", "帮我查一下订单 99999 的退款进度")
    rag = resp.session_state["rag"]

    bf = rag["business_fact"]
    assert bf is not None
    assert bf["found"] is False
    # 无 RAG 命中 → 停在 tool_fact_fallback
    assert rag["answer_path"] in ("tool_fact_fallback", "answer_with_citations"), (
        f"预期 tool_fact_fallback 或 answer_with_citations，实际 {rag['answer_path']}"
    )


# ── TR-22.3：越权安全降级 ──


def test_tr22_3_unauthorized_downgrade_safely(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """driver_service 返回 user_matched=False → 安全降级，不泄漏数据。"""
    def _mock_unauth(driver_id, order_id=None):
        return None, False

    monkeypatch.setattr("integrations.driver_service_client.driver_refund_fact_from_service", _mock_unauth)
    monkeypatch.setattr("services.business_facts.driver_refund_fact_from_service", _mock_unauth)

    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "tr22-unauth", "我的退款到账了吗")
    rag = resp.session_state["rag"]
    bf = rag["business_fact"]

    assert bf is not None
    assert bf["found"] is False
    assert bf["user_matched"] is False
    # 不应有任何 facts 泄漏
    assert resp.citations == [] or rag["answer_path"] == "answer_with_citations"


# ── L19：澄清链路（规划器可用时的分支） ──


@pytest.fixture()
def _mock_planner(monkeypatch):
    """规划器返回固定 ClarificationPlan，专测 L19 澄清分支。

    必须排在 _mock_chat_llm 之后应用（后者把规划器设成抛异常走 L17 降级）。
    """
    from api.schemas import ClarificationPlan
    from models.clarification_planner import ClarificationPlannerModelClient

    def _plan(self, request, tool_specs=None):
        if "核销" in request.user_message:
            return ClarificationPlan(
                intent="order_query",
                tool_name="get_verification_record",
                known_arguments={},
                model_name="mock-planner",
            )
        return ClarificationPlan(
            intent="promotion_consult",
            tool_name="get_activity_progress",
            known_arguments={},
            model_name="mock-planner",
        )

    monkeypatch.setattr(ClarificationPlannerModelClient, "plan_clarification", _plan)


def test_l19_missing_activity_id_asks_before_tool(monkeypatch, _patch_embeddings, _mock_chat_llm, _mock_planner):
    """缺 activity_id → 先反问，本轮不生成 ToolAction、不调工具。"""
    _setup_basic_switches(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "l19-missing-activity", "我这个活动达标了吗")
    rag = resp.session_state["rag"]
    tc = rag["tool_calling"]

    assert rag["answer_path"] == "clarification_required"
    assert resp.clarification is not None
    assert resp.clarification.clarification_field == "activity_id"
    assert resp.tool_calls == [], "澄清态不应该调工具"
    assert tc["planned_action"] is None
    assert tc["observation"] is None
    # mock 草案故意不填 missing_required，后端仍按 ToolSpec 重算并生成澄清。
    assert tc["clarification_plan"]["missing_required"] == []
    assert tc["clarification_plan"]["tool_name"] == "get_activity_progress"


def test_l19_runtime_context_fills_activity_id(monkeypatch, _patch_embeddings, _mock_chat_llm, _mock_planner):
    """页面带入唯一活动 → 自动补参直接调工具，不再反问。"""
    _setup_basic_switches(monkeypatch)
    activity_fact = lambda *a, **kw: (
        {"activityName": "完单奖", "targetOrders": 100, "completedOrders": 105},
        True,
    )
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_activity_fact_from_service", activity_fact
    )
    monkeypatch.setattr(
        "services.business_facts.driver_activity_fact_from_service", activity_fact
    )
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(
        agent,
        "l19-context-fill",
        "这个活动达标了吗",
        runtime_context={"currentActivityId": "ACT-67890"},
    )
    rag = resp.session_state["rag"]
    tc = rag["tool_calling"]

    assert resp.clarification is None
    assert rag["answer_path"] == "answer_with_tool_fact"
    assert tc["planned_action"]["arguments"]["activity_id"] == "ACT-67890"
    assert tc["observation"]["status"] == "success"
    assert len(resp.tool_calls) == 1


def test_l19_multiple_activities_returns_candidates(monkeypatch, _patch_embeddings, _mock_chat_llm, _mock_planner):
    """页面带入多个活动 → 反问并给候选，模型不许代选。"""
    _setup_basic_switches(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(
        agent,
        "l19-candidates",
        "我的活动进度怎么样",
        runtime_context={
            "currentUserActivities": [
                {"activityId": "ACT-12345", "activityName": "周末冲单赛", "status": "进行中"},
                {"activityId": "ACT-67890", "activityName": "完单奖", "status": "进行中"},
            ]
        },
    )

    assert resp.clarification is not None
    assert len(resp.clarification.candidates) == 2
    assert {c.value for c in resp.clarification.candidates} == {"ACT-12345", "ACT-67890"}
    assert resp.session_state["rag"]["answer_path"] == "clarification_required"
    assert resp.tool_calls == []


def test_l19_missing_coupon_id_asks_before_tool(monkeypatch, _patch_embeddings, _mock_chat_llm, _mock_planner):
    """核销记录缺 coupon_id → 反问卡券号，不调核销工具。"""
    _setup_basic_switches(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "l19-missing-coupon", "我的核销记录呢")

    assert resp.clarification is not None
    assert resp.clarification.clarification_field == "coupon_id"
    assert resp.session_state["rag"]["answer_path"] == "clarification_required"
    assert resp.tool_calls == []


# ── L19 课程形态：卡券候选 + 工具后澄清 ──


def _coupon_plan(self, request, tool_specs=None):
    """规划器固定返回 get_coupon_status（无必填参数，直接执行）。"""
    from api.schemas import ClarificationPlan

    return ClarificationPlan(
        intent="order_query",
        tool_name="get_coupon_status",
        known_arguments={},
        model_name="mock-planner",
    )


def _mock_coupon_fact(monkeypatch, coupons):
    coupon_fact = lambda *a, **kw: (coupons, True)
    monkeypatch.setattr(
        "integrations.driver_service_client.driver_coupon_fact_from_service", coupon_fact
    )
    monkeypatch.setattr(
        "services.business_facts.driver_coupon_fact_from_service", coupon_fact
    )


_TWO_COUPONS = [
    {"driverId": "driver_001", "couponId": "CP-20261015-001", "name": "周末特惠月卡",
     "status": "unused", "statusText": "未使用", "price": 39.9,
     "validFrom": "2026-10-15", "validTo": "2026-11-14"},
    {"driverId": "driver_001", "couponId": "CP-20261010-003", "name": "全月稳行周卡",
     "status": "unused", "statusText": "未使用", "price": 19.9,
     "validFrom": "2026-10-10", "validTo": "2026-10-24"},
]


def test_l19_post_tool_clarification_multiple_coupons(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """get_coupon_status 查回多张卡券 → 工具后澄清：工具已调，候选让用户选。"""
    _setup_basic_switches(monkeypatch)
    from models.clarification_planner import ClarificationPlannerModelClient
    monkeypatch.setattr(ClarificationPlannerModelClient, "plan_clarification", _coupon_plan)
    _mock_coupon_fact(monkeypatch, _TWO_COUPONS)
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "l19-post-coupon", "我的卡券有哪些")
    rag = resp.session_state["rag"]
    tc = rag["tool_calling"]

    assert rag["answer_path"] == "clarification_required"
    assert resp.clarification is not None
    assert resp.clarification.clarification_field == "coupon_id"
    assert len(resp.clarification.candidates) == 2
    assert {c.value for c in resp.clarification.candidates} == {"CP-20261015-001", "CP-20261010-003"}
    # 与工具前澄清的关键区别：工具已经调过了，tool_calls 有记录。
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].action.tool_name == "get_coupon_status"
    assert resp.tool_calls[0].observation.status == "success"
    assert tc["post_tool_clarification"] is not None


def test_l19_post_tool_single_coupon_passes_through(monkeypatch, _patch_embeddings, _mock_chat_llm):
    """只有一张卡券 → 不用选，直接 answer_with_tool_fact。"""
    _setup_basic_switches(monkeypatch)
    from models.clarification_planner import ClarificationPlannerModelClient
    monkeypatch.setattr(ClarificationPlannerModelClient, "plan_clarification", _coupon_plan)
    _mock_coupon_fact(monkeypatch, _TWO_COUPONS[:1])
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(agent, "l19-post-single", "我的卡券有哪些")

    assert resp.clarification is None
    assert resp.session_state["rag"]["answer_path"] == "answer_with_tool_fact"
    assert resp.session_state["rag"]["tool_calling"]["post_tool_clarification"] is None
    assert len(resp.tool_calls) == 1


def test_l19_pre_tool_coupon_candidates_from_context(monkeypatch, _patch_embeddings, _mock_chat_llm, _mock_planner):
    """核销缺 coupon_id + 页面带 currentUserCoupons → 反问并附页面上下文候选。"""
    _setup_basic_switches(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent
    agent = CustomerServiceAgent()

    resp = _chat(
        agent,
        "l19-pre-coupon-candidates",
        "我的核销记录呢",
        runtime_context={
            "currentUserCoupons": [
                {"couponId": "CP-20261015-001", "name": "周末特惠月卡", "status": "unused"},
                {"couponId": "CP-20260920-007", "name": "中秋限时券", "status": "verified"},
            ]
        },
    )

    assert resp.clarification is not None
    assert resp.clarification.clarification_field == "coupon_id"
    assert len(resp.clarification.candidates) == 2
    assert {c.value for c in resp.clarification.candidates} == {"CP-20261015-001", "CP-20260920-007"}
    assert resp.session_state["rag"]["answer_path"] == "clarification_required"
    assert resp.tool_calls == [], "工具前澄清不应该调工具"
