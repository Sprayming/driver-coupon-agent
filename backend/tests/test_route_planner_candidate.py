"""路由模型候选（对标 L41）：模型只提议，代码裁决。

覆盖四条硬约束：intent 一致、工具白名单、风险只升不降、候选失败退回规则基线。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from api.schemas import RoutePlanCandidate, ToolCandidate
from models import router_client
from planner import task_planner as planner_module
from planner.task_planner import TaskPlanner


@pytest.fixture(autouse=True)
def _reset_router_breaker(monkeypatch):
    """熔断状态是模块级变量，每个用例前清零，避免失败用例污染后续用例。"""

    monkeypatch.setattr(router_client, "_last_failure_at", 0.0)


def _fake_tool(name: str) -> ToolCandidate:
    return ToolCandidate(
        name=name,
        domain="coupon",
        allowed_in_light_path=True,
        risk_level="low",
        reason="test",
    )


# ---------------------------------------------------------------------------
# 候选契约：自相矛盾的输出在构造时就被拒绝
# ---------------------------------------------------------------------------


def test_candidate_rejects_tools_without_needs_business_tools():
    with pytest.raises(ValidationError):
        RoutePlanCandidate(
            intent="driver_coupon", required_tools=["get_coupon_status"], needs_business_tools=False
        )


def test_candidate_rejects_workflow_with_low_risk():
    with pytest.raises(ValidationError):
        RoutePlanCandidate(intent="driver_refund", requires_workflow=True, risk_level="low")


def test_candidate_rejects_workflow_without_workflow_first():
    with pytest.raises(ValidationError):
        RoutePlanCandidate(
            intent="driver_refund",
            requires_workflow=True,
            risk_level="high",
            fallback_policy="default",
        )


def test_candidate_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        RoutePlanCandidate(intent="general_chat", confidence=5.0)


def test_candidate_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        RoutePlanCandidate(intent="general_chat", confidence=-0.1)


# ---------------------------------------------------------------------------
# 收敛：白名单 + 风险只升不降
# ---------------------------------------------------------------------------


def test_merge_adds_domain_for_intent_with_tools():
    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("今日单量多少", {})
    assert rule_plan.intent == "driver_order"

    candidate = RoutePlanCandidate(
        intent="driver_order",
        needs_business_tools=True,
        required_tools=["get_driver_order_count"],
        knowledge_domains=["promotion"],
        risk_level="medium",
        confidence=0.8,
    )

    merged = planner._merge_candidate(
        rule_plan, candidate, [_fake_tool("get_driver_order_count")]
    )

    assert merged.source == "classifier"
    assert "get_driver_order_count" in merged.required_tools
    assert merged.knowledge_domains == ["promotion"]
    assert merged.needs_business_tools is True
    # 风险等级仍由基线决定，模型给的 medium 不生效。
    assert merged.risk_level == rule_plan.risk_level


def test_merge_cannot_add_tool_to_intent_without_tools():
    """general_chat 这类没有工具语义的意图，即使工具在目录里也不许补。"""

    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("你好", {})
    assert rule_plan.intent == "general_chat"

    candidate = RoutePlanCandidate(
        intent="general_chat",
        needs_business_tools=True,
        required_tools=["get_coupon_status"],
        confidence=0.9,
    )

    merged = planner._merge_candidate(rule_plan, candidate, [_fake_tool("get_coupon_status")])

    assert merged.required_tools == []
    assert merged.needs_business_tools is False


def test_merge_drops_tool_outside_catalog():
    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("你好", {})
    candidate = RoutePlanCandidate(
        intent="general_chat",
        needs_business_tools=True,
        required_tools=["invented_tool"],
        confidence=0.8,
    )

    merged = planner._merge_candidate(rule_plan, candidate, [])

    assert merged.required_tools == []


def test_merge_never_lowers_risk_or_closes_workflow():
    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("我要退款 CP-20260101-001", {})
    assert rule_plan.risk_level == "high"
    assert rule_plan.requires_workflow is True

    candidate = RoutePlanCandidate(
        intent="driver_refund", risk_level="low", requires_workflow=False, confidence=0.9
    )
    merged = planner._merge_candidate(rule_plan, candidate, [])

    assert merged.risk_level == "high"
    assert merged.requires_workflow is True
    assert merged.fallback_policy == "workflow_first"


def test_merge_cannot_raise_risk_or_open_workflow():
    """模型不能把普通咨询抬成高风险、更不能自己打开 workflow。"""

    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("你好", {})
    assert rule_plan.requires_workflow is False

    candidate = RoutePlanCandidate(
        intent="general_chat",
        risk_level="high",
        requires_workflow=True,
        fallback_policy="workflow_first",
        confidence=0.95,
    )
    merged = planner._merge_candidate(rule_plan, candidate, [])

    assert merged.risk_level == rule_plan.risk_level
    assert merged.requires_workflow is False
    assert merged.needs_rag == rule_plan.needs_rag


def test_merge_falls_back_when_intent_mismatch():
    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("你好", {})
    candidate = RoutePlanCandidate(intent="driver_refund", confidence=0.9)

    merged = planner._merge_candidate(rule_plan, candidate, [])

    assert merged.source == "rules_fallback"
    assert merged.intent == rule_plan.intent


def test_merge_falls_back_below_model_confidence_floor():
    planner = TaskPlanner()
    rule_plan = planner.plan_by_rules("你好", {})
    candidate = RoutePlanCandidate(intent="general_chat", confidence=0.5)

    merged = planner._merge_candidate(rule_plan, candidate, [])

    assert merged.source == "rules_fallback"


# ---------------------------------------------------------------------------
# 模型客户端：失败一律返回 None
# ---------------------------------------------------------------------------


def test_no_api_key_returns_none_without_calling_model(monkeypatch):
    monkeypatch.setattr(router_client, "api_key_is_missing", lambda _key: True)

    def _boom(_messages, **_kwargs):
        raise AssertionError("没有 Key 时不应该调用模型")

    result = router_client.plan_route_candidate(
        "你好", TaskPlanner().plan_by_rules("你好", {}), llm_caller=_boom
    )
    assert result is None


def test_parses_code_fenced_json(monkeypatch):
    monkeypatch.setattr(router_client, "api_key_is_missing", lambda _key: False)
    payload = {"intent": "general_chat", "knowledge_domains": ["promotion"], "confidence": 0.9}
    answer = "```json\n" + json.dumps(payload) + "\n```"

    result = router_client.plan_route_candidate(
        "你好",
        TaskPlanner().plan_by_rules("你好", {}),
        llm_caller=lambda _messages, **_kwargs: SimpleNamespace(answer=answer),
    )

    assert result is not None
    assert result.knowledge_domains == ["promotion"]


def test_unknown_domain_is_filtered(monkeypatch):
    monkeypatch.setattr(router_client, "api_key_is_missing", lambda _key: False)
    payload = {"intent": "general_chat", "knowledge_domains": ["promotion", "made_up"], "confidence": 0.9}

    result = router_client.plan_route_candidate(
        "你好",
        TaskPlanner().plan_by_rules("你好", {}),
        llm_caller=lambda _messages, **_kwargs: SimpleNamespace(answer=json.dumps(payload)),
    )

    assert result is not None
    assert result.knowledge_domains == ["promotion"]


def test_invalid_json_returns_none(monkeypatch):
    monkeypatch.setattr(router_client, "api_key_is_missing", lambda _key: False)

    result = router_client.plan_route_candidate(
        "你好",
        TaskPlanner().plan_by_rules("你好", {}),
        llm_caller=lambda _messages, **_kwargs: SimpleNamespace(answer="不是 JSON"),
    )
    assert result is None


def test_router_passes_short_timeout(monkeypatch):
    """路由是前置判断，超时必须远小于回答链路的 90 秒。"""

    monkeypatch.setattr(router_client, "api_key_is_missing", lambda _key: False)
    seen: dict = {}

    def _caller(_messages, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(answer='{"intent": "general_chat", "confidence": 0.9}')

    router_client.plan_route_candidate(
        "你好", TaskPlanner().plan_by_rules("你好", {}), llm_caller=_caller
    )

    assert seen["timeout"] == router_client.ROUTER_TIMEOUT_SECONDS
    assert seen["timeout"] < 10


# ---------------------------------------------------------------------------
# 知识域真正接进检索：只放宽、不收窄
# ---------------------------------------------------------------------------


class _IntentStub:
    """pre_retrieval_plan 只需要 intent 字段。"""

    def __init__(self, intent: str) -> None:
        self.intent = intent


def test_widen_topics_adds_domain_topics():
    from rag.planning import pre_retrieval_plan, widen_topics_for_domains
    from rag.query_rewrite import rewrite_retrieval_query

    message = "卡券怎么用"
    rewrite = rewrite_retrieval_query(message, "product_consult")
    plan = pre_retrieval_plan(message, _IntentStub("product_consult"), rewrite)
    assert plan.allowed_topics == ["product_consult"]

    widened = widen_topics_for_domains(plan, ["refund"])

    assert "refund_request" in widened.allowed_topics
    # 只放宽：原有主题必须保留
    assert "product_consult" in widened.allowed_topics


def test_widen_topics_never_narrows():
    from rag.planning import pre_retrieval_plan, widen_topics_for_domains
    from rag.query_rewrite import rewrite_retrieval_query

    message = "卡券怎么用"
    rewrite = rewrite_retrieval_query(message, "product_consult")
    plan = pre_retrieval_plan(message, _IntentStub("product_consult"), rewrite)

    assert widen_topics_for_domains(plan, None).allowed_topics == plan.allowed_topics
    assert widen_topics_for_domains(plan, ["made_up_domain"]).allowed_topics == plan.allowed_topics


# ---------------------------------------------------------------------------
# 端到端：低置信才问模型，高置信不问
# ---------------------------------------------------------------------------


def test_plan_uses_candidate_only_when_low_confidence(monkeypatch):
    calls: list[str] = []

    def _fake_candidate(user_message, rule_plan, **_kwargs):
        calls.append(user_message)
        return RoutePlanCandidate(
            intent=rule_plan.intent,
            knowledge_domains=["promotion"],
            confidence=0.9,
        )

    monkeypatch.setattr(planner_module, "plan_route_candidate", _fake_candidate)
    planner = TaskPlanner()

    low_plan, _ = planner.plan("你好")
    assert low_plan.source == "classifier"
    assert calls == ["你好"]

    # 高置信：退款带券号，规则直接返回，不调模型。
    high_plan, _ = planner.plan("我要退款 CP-20260101-001")
    assert high_plan.confidence >= 0.85
    assert calls == ["你好"]
