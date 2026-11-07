"""Task 14 验证：RAG-Fusion 多查询 + RRF（TR-14.1 / TR-14.2）。

TR-14.1：开：调试信息含子查询与 RRF；关：行为同 L16；LLM 失败回退不报错；
TR-14.2：融合结果去重且排序与 RRF 公式一致。
"""

from __future__ import annotations

import json

import pytest

from api.schemas import ChatRequest, KnowledgeChunk, KnowledgeHit
from config import settings
from rag import index_cache
from rag.fusion import generate_sub_queries, rrf_fuse
from store.seed import seed


@pytest.fixture()
def db_with_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "task14.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    index_cache.reset_index_and_cache()
    yield db_path
    index_cache.reset_index_and_cache()


def _patch_embeddings(monkeypatch):
    fake_vec = [0.1] * 1024
    monkeypatch.setattr("rag.vector_store.embed_text", lambda *a, **kw: fake_vec)
    monkeypatch.setattr(
        "rag.vector_store.embed_texts",
        lambda texts, *a, **kw: [fake_vec for _ in texts],
    )
    monkeypatch.setattr("rag.quality.peek_quality_summary", lambda: None)


def _make_chunk(cid: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=cid,
        title=cid,
        source_path=f"{cid}.md",
        section="s1",
        keywords=[],
        text=f"content of {cid}",
    )


def test_tr14_2_rrf_fuse_dedup_and_order():
    """RRF 融合：同一 chunk 在多个列表中合并，排序符合 1/(k+rank) 累加。"""

    c1 = _make_chunk("c1")
    c2 = _make_chunk("c2")
    c3 = _make_chunk("c3")

    list_a = [
        KnowledgeHit(chunk=c1, score=0.9, sources=["vector"]),
        KnowledgeHit(chunk=c2, score=0.8, sources=["vector"]),
    ]
    list_b = [
        KnowledgeHit(chunk=c2, score=0.95, sources=["keyword"]),
        KnowledgeHit(chunk=c3, score=0.7, sources=["keyword"]),
    ]
    list_c = [
        KnowledgeHit(chunk=c1, score=0.85, sources=["bm25"]),
        KnowledgeHit(chunk=c2, score=0.6, sources=["bm25"]),
    ]

    fused = rrf_fuse([list_a, list_b, list_c], k=60)

    # 去重：3 个不同 chunk。
    ids = [h.chunk.chunk_id for h in fused]
    assert len(ids) == 3
    assert set(ids) == {"c1", "c2", "c3"}

    # c2 在三个列表中排名都靠前（rank 1+1+2），RRF 分最高。
    # c1: rank 1 + 不存在 + rank 1 = 1/61 + 1/61
    # c2: rank 2 + rank 1 + rank 2 = 1/62 + 1/61 + 1/62
    # c3: 不存在 + rank 2 + 不存在 = 1/62
    expected_scores = {
        "c1": 1 / 61 + 1 / 61,
        "c2": 1 / 62 + 1 / 61 + 1 / 62,
        "c3": 1 / 62,
    }
    # 归一化后排序：c2 > c1 > c3。
    assert ids[0] == "c2"
    assert ids[1] == "c1"
    assert ids[2] == "c3"
    # 每个 fused hit 的 score 是归一化后的 RRF 分（0-1）。
    max_raw = max(expected_scores.values())
    for h in fused:
        expected = round(min(1.0, expected_scores[h.chunk.chunk_id] / max_raw), 3)
        assert h.score == expected


def test_tr14_1_sub_queries_generated_on(db_with_seed, monkeypatch):
    """开启 fusion 后，调试信息含子查询清单。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", True)
    _patch_embeddings(monkeypatch)

    # mock LLM 返回 3 个子查询。
    fake_sub = ["退款规则细节", "退款适用条件", "退款例外情形"]

    def fake_caller(messages, **kw):
        from models.llm_client import ChatModelResult

        return ChatModelResult(answer=json.dumps(fake_sub))

    monkeypatch.setattr("models.llm_client.call_chat_model", fake_caller)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    req = ChatRequest(
        session_id="s1",
        runtime_user_id="driver-001",
        runtime_role="driver",
        user_message="怎么退款",
    )
    resp = agent.chat(req)
    rag = resp.session_state["rag"]
    assert rag["fusion_enabled"] is True
    assert len(rag["fusion_sub_queries"]) == 3
    assert rag["fusion_sub_queries"] == fake_sub


def test_tr14_1_fusion_off_behavior_same_as_l16(db_with_seed, monkeypatch):
    """关闭 fusion 时，fusion_sub_queries 为空，行为同 L16。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", False)
    _patch_embeddings(monkeypatch)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    req = ChatRequest(
        session_id="s2",
        runtime_user_id="driver-001",
        runtime_role="driver",
        user_message="怎么退款",
    )
    resp = agent.chat(req)
    rag = resp.session_state["rag"]
    assert rag["fusion_enabled"] is False
    assert rag["fusion_sub_queries"] == []


def test_tr14_1_llm_failure_falls_back_to_single_query(db_with_seed, monkeypatch):
    """LLM 失败（抛异常）时自动回退单查询，不报错。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", True)
    _patch_embeddings(monkeypatch)

    def failing_caller(messages, **kw):
        raise RuntimeError("LLM 服务不可用")

    monkeypatch.setattr("models.llm_client.call_chat_model", failing_caller)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    req = ChatRequest(
        session_id="s3",
        runtime_user_id="driver-001",
        runtime_role="driver",
        user_message="怎么退款",
    )
    # 不抛异常。
    resp = agent.chat(req)
    rag = resp.session_state["rag"]
    # 回退到单查询（原始 rewritten_query）。
    assert len(rag["fusion_sub_queries"]) == 1


def test_tr14_1_invalid_json_falls_back(db_with_seed, monkeypatch):
    """LLM 返回非法 JSON 时回退单查询。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", True)
    _patch_embeddings(monkeypatch)

    def bad_json_caller(messages, **kw):
        from models.llm_client import ChatModelResult

        return ChatModelResult(answer="这不是 JSON")

    monkeypatch.setattr("models.llm_client.call_chat_model", bad_json_caller)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    req = ChatRequest(
        session_id="s4",
        runtime_user_id="driver-001",
        runtime_role="driver",
        user_message="怎么退款",
    )
    resp = agent.chat(req)
    assert len(resp.session_state["rag"]["fusion_sub_queries"]) == 1
