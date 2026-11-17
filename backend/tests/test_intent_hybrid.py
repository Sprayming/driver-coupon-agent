"""意图供应商开关单测：默认 rule 行为不变；hybrid 只在低置信时兜底，失败回退规则。"""

from __future__ import annotations

import pytest

from api.schemas import IntentResult
from config import settings
from rag import intent_semantic


def _fake(intent: str) -> IntentResult:
    return IntentResult(intent=intent, matched_keywords=[], explanation="fake semantic")


@pytest.fixture()
def hybrid(monkeypatch):
    monkeypatch.setattr(settings, "INTENT_PROVIDER", "hybrid")


def test_rule_provider_does_not_escalate(monkeypatch):
    """默认 rule：即使规则判错，也不触发语义兜底。"""

    monkeypatch.setattr(settings, "INTENT_PROVIDER", "rule")
    called: list[str] = []
    monkeypatch.setattr(
        intent_semantic, "semantic_classify", lambda message: called.append(message)
    )

    result, escalated = intent_semantic.classify_intent_with_provider(
        "不用退款了，帮我查一下订单"
    )

    assert result.intent == "refund_request"  # 规则的真实误判，保留以求可解释
    assert escalated is False
    assert called == []


def test_hybrid_escalates_negation(hybrid, monkeypatch):
    monkeypatch.setattr(
        intent_semantic, "semantic_classify", lambda message: _fake("order_query")
    )

    result, escalated = intent_semantic.classify_intent_with_provider(
        "不用退款了，帮我查一下订单"
    )

    assert result.intent == "order_query"
    assert escalated is True


def test_hybrid_escalates_unknown_and_paraphrase(hybrid, monkeypatch):
    monkeypatch.setattr(
        intent_semantic, "semantic_classify", lambda message: _fake("refund_request")
    )

    result, escalated = intent_semantic.classify_intent_with_provider("钱退给我")

    assert result.intent == "refund_request"
    assert escalated is True


def test_hybrid_escalates_keyword_collision(hybrid, monkeypatch):
    monkeypatch.setattr(
        intent_semantic, "semantic_classify", lambda message: _fake("refund_request")
    )

    result, escalated = intent_semantic.classify_intent_with_provider(
        "这张券我不想要了，能退吗？"
    )

    assert result.intent == "refund_request"
    assert escalated is True


def test_hybrid_keeps_strong_rule_without_llm(hybrid, monkeypatch):
    """强关键词命中（退款）不需要付额外延迟。"""

    called: list[str] = []
    monkeypatch.setattr(
        intent_semantic, "semantic_classify", lambda message: called.append(message)
    )

    result, escalated = intent_semantic.classify_intent_with_provider("我要退款")

    assert result.intent == "refund_request"
    assert escalated is False
    assert called == []


def test_hybrid_falls_back_when_semantic_fails(hybrid, monkeypatch):
    """语义兜底超时/报错返回 None 时，保留规则结果且不抛异常。"""

    monkeypatch.setattr(intent_semantic, "semantic_classify", lambda message: None)

    result, escalated = intent_semantic.classify_intent_with_provider("钱退给我")

    assert result.intent == "unknown"
    assert escalated is False
