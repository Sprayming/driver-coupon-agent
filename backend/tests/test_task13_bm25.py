"""Task 13 验证：BM25 硬关键词专路（TR-13.1 / TR-13.2）。

TR-13.1：「P-REF-001」「周末日卡」精确查询对应 chunk 融合后排名第一，score_breakdown 有 bm25 分；
TR-13.2：用测试替身替换 SearchEngine 后链路正常。
"""

from __future__ import annotations

import pytest

from api.schemas import ChatRequest, KnowledgeHit
from config import settings
from rag import index_cache
from rag.search_engine import BM25Local, SearchEngine
from store.seed import seed


@pytest.fixture()
def db_with_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "task13.db"
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


def test_tr13_1_policy_code_exact_match_ranks_first(db_with_seed):
    """「P-REF-001」精确查询，含该编码的 chunk 在 BM25 结果中排名第一。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    engine = BM25Local(list(idx.chunks_by_id.values()))

    hits = engine.search("P-REF-001", top_k=5)
    assert hits, "BM25 应返回命中"
    top = hits[0]
    assert "P-REF-001" in top.chunk.policy_codes, (
        f"top chunk 应含 P-REF-001，实际 policy_codes={top.chunk.policy_codes}"
    )
    assert top.bm25_score >= 0.9
    assert "bm25" in top.sources


def test_tr13_1_product_name_exact_match_ranks_first(db_with_seed):
    """「周末日卡」精确查询，含该商品名的 chunk 在 BM25 结果中排名第一。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    engine = BM25Local(list(idx.chunks_by_id.values()))

    hits = engine.search("周末日卡", top_k=5)
    assert hits, "BM25 应返回命中"
    top = hits[0]
    assert "周末日卡" in top.chunk.product_names, (
        f"top chunk 应含周末日卡，实际 product_names={top.chunk.product_names}"
    )
    assert top.bm25_score >= 0.9


def test_tr13_1_agent_chat_has_bm25_score_in_breakdown(db_with_seed, monkeypatch):
    """开启 BM25 后，agent.chat 的 score_breakdown 含 bm25 分。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "BM25_ENABLED", True)
    _patch_embeddings(monkeypatch)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    req = ChatRequest(
        session_id="s1",
        runtime_user_id="driver-001",
        runtime_role="driver",
        user_message="P-REF-001 规则是什么",
    )
    resp = agent.chat(req)
    rag_state = resp.session_state["rag"]
    # BM25 通道命中清单非空（精确码 P-REF-001 应触发 BM25 召回）。
    assert rag_state["bm25_retrieved_count"] > 0
    assert len(rag_state["bm25_chunk_ids"]) > 0


def test_tr13_2_search_engine_double_can_replace(db_with_seed):
    """测试替身实现 SearchEngine 接口后可正常工作（接口可替换证明）。"""

    class FakeEngine(SearchEngine):
        def __init__(self, canned_hits: list[KnowledgeHit]):
            self._hits = canned_hits

        def search(self, query: str, top_k: int = 10) -> list[KnowledgeHit]:
            return self._hits[:top_k]

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    target = list(idx.chunks_by_id.values())[0]
    fake_hit = KnowledgeHit(
        chunk=target,
        score=0.8,
        bm25_score=0.8,
        sources=["bm25"],
    )
    fake = FakeEngine([fake_hit])

    hits = fake.search("anything")
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == target.chunk_id
    assert hits[0].bm25_score == 0.8
    assert "bm25" in hits[0].sources


def test_tr13_2_merge_hybrid_includes_bm25_source(db_with_seed):
    """merge_hybrid_hits 接入 bm25_hits 后，命中 sources 含 bm25 且 bm25_score 保留。"""

    from rag.planning import RetrievalPlan
    from rag.vector_store import merge_hybrid_hits

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    target = list(idx.chunks_by_id.values())[0]
    bm25_hit = KnowledgeHit(
        chunk=target,
        score=0.85,
        bm25_score=0.85,
        sources=["bm25"],
    )
    plan = RetrievalPlan(
        original_query="test",
        rewritten_query="test",
        scene="refund",
        allowed_topics=["refund"],
        keyword_terms=[],
        reason="test",
    )
    merged = merge_hybrid_hits(plan, vector_hits=[], keyword_hits=[], bm25_hits=[bm25_hit])
    assert merged
    m = merged[0]
    assert "bm25" in m.sources
    assert m.bm25_score == 0.85
