"""Task 10 验证：草稿/未发布读取边界（TR-10.1 / TR-10.2）。

TR-10.1：historical chunk 在 release 索引中存在，检索进池，非历史意图下被软降权；
        历史意图下放行不降权——行为与 L16 方案Y 一致。
TR-10.2：draft 独有句子（DRAFT-X9）不在任何 release 索引中，任何问法 0 召回。
"""

from __future__ import annotations

import pytest

from config import settings
from rag import index_cache
from rag.knowledge_base import should_include_chunk_for_query
from rag.reranker import REASON_WEIGHTS, explain_rerank_reasons
from store.seed import seed


@pytest.fixture()
def db_with_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "task10.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    index_cache.reset_index_and_cache()
    yield db_path
    index_cache.reset_index_and_cache()


def test_tr10_2_draft_clause_absent_from_release_indexes(db_with_seed):
    """draft 独有句子 DRAFT-X9 不在 stable/candidate 任何 release 索引中（物理排除）。"""

    for release_id in ("rel-0001", "rel-0002"):
        idx = index_cache.get_release_index(release_id, db_path=db_with_seed)
        all_text = " ".join(chunk.text for chunk in idx.chunks_by_id.values())
        assert "DRAFT-X9" not in all_text, (
            f"{release_id} 索引不应包含草稿独句 DRAFT-X9"
        )


def test_tr10_2_draft_clause_zero_recall(db_with_seed):
    """任何问法检索 draft 独句都 0 召回（release 索引根本没有这个 chunk）。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    chunk_ids = set(idx.chunks_by_id.keys())
    # draft 版本 doc-coupon-refund-v2 不在 rel-0001 的 release_items 中，
    # 其切片 chunk_id 不会出现在索引里。
    assert not any("v2" in cid for cid in chunk_ids)


def test_tr10_1_historical_chunk_present_in_release_index(db_with_seed):
    """historical chunk 存在于 stable release 索引中（方案Y 宽进，不硬过滤）。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    historical_chunks = [
        chunk for chunk in idx.chunks_by_id.values()
        if chunk.effective_status == "historical"
    ]
    assert len(historical_chunks) > 0, "stable 索引应包含 historical chunk（2025 月卡折扣券）"


def test_tr10_1_historical_chunk_enters_candidate_pool(db_with_seed):
    """should_include_chunk_for_query 对 historical 返回 True（方案Y 召回层宽进）。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    historical_chunks = [
        chunk for chunk in idx.chunks_by_id.values()
        if chunk.effective_status == "historical"
    ]
    for chunk in historical_chunks:
        # 非历史意图：仍进池（return True），由 reranker 软降权。
        assert should_include_chunk_for_query(chunk, asks_for_history=False) is True
        # 历史意图：同样进池。
        assert should_include_chunk_for_query(chunk, asks_for_history=True) is True


def test_tr10_1_historical_soft_demoted_for_non_history_query(db_with_seed):
    """非历史意图下 historical chunk 被 reranker 软降权（方案Y 核心）。"""

    idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    historical_chunk = next(
        chunk for chunk in idx.chunks_by_id.values()
        if chunk.effective_status == "historical"
    )

    from api.schemas import KnowledgeHit

    hit = KnowledgeHit(chunk=historical_chunk, score=0.5, vector_score=0.5, sources=["vector"])

    # 非历史问题（如"怎么退款"）：历史片段应被降权。
    reasons_normal = explain_rerank_reasons("怎么申请退款", hit, asks_for_history=False)
    assert "历史卡券降权" in reasons_normal
    demotion_weight = REASON_WEIGHTS["历史卡券降权"]
    assert demotion_weight < 0, "历史片段降权权重应为负"

    # 历史问题（如"2025 月卡折扣券还有吗"）：历史片段放行不降权。
    reasons_history = explain_rerank_reasons("2025 年月卡折扣券还有吗", hit, asks_for_history=True)
    assert "用户明确问历史，历史片段放行不降权" in reasons_history
    assert "历史卡券降权" not in reasons_history
