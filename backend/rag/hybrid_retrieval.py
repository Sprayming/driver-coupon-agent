"""混合检索门面（对标小哲 L41 `rag/hybrid_retrieval.py`）。

L41 用一个 `retrieve_knowledge` 收口「向量 + 关键词 + BM25 + 融合 + 精排」。
本项目底层仍拆成 vector_store / search_engine / fusion / index_cache 四个模块，
本文件只做编排门面：Agent 只依赖这里，不再直接拼三路召回。

职责边界：
- 召回与融合在本模块完成
- 缓存读写仍在 `rag/index_cache.py`，由 Agent 决定何时写缓存
- 低置信判断留在 `rag/quality.py`，因为它同时服务评测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.schemas import KnowledgeHit, KnowledgeIndex, RerankOutcome, RetrievalPlan
from config import settings
from rag.fusion import generate_sub_queries, rrf_fuse
from rag.index_cache import get_knowledge_index as _get_knowledge_index
from rag.index_cache import reset_index_and_cache as _reset_index_and_cache
from rag.query_rewrite import normalize_query
from rag.reranker import rerank_candidates
from rag.search_engine import BM25Local, SearchEngine
from rag.vector_store import (
    get_vector_store,
    keyword_retrieve,
    merge_candidates,
    merge_hybrid_hits,
    retrieve_by_vector,
)

# 与 L41 同名导出：查询归一化、索引读取、缓存重置都属于检索层，不属于 Agent。
__all__ = [
    "HybridRetrievalResult",
    "get_knowledge_index",
    "get_vector_store",
    "normalize_query",
    "reset_hybrid_index_and_cache",
    "retrieve_knowledge",
]


@dataclass
class HybridRetrievalResult:
    """一轮混合检索的完整中间产物，供调试台和缓存层使用。"""

    raw_hits: list[KnowledgeHit] = field(default_factory=list)
    rewritten_hits: list[KnowledgeHit] = field(default_factory=list)
    keyword_hits: list[KnowledgeHit] = field(default_factory=list)
    bm25_hits: list[KnowledgeHit] = field(default_factory=list)
    fusion_sub_queries: list[str] = field(default_factory=list)
    vector_candidates: list[KnowledgeHit] = field(default_factory=list)
    candidates: list[KnowledgeHit] = field(default_factory=list)
    reranked_hits: list[KnowledgeHit] = field(default_factory=list)
    rerank_outcome: RerankOutcome | None = None


def get_knowledge_index() -> KnowledgeIndex:
    """返回当前进程内知识索引（L41 同款入口，实现仍在 rag/index_cache.py）。"""

    return _get_knowledge_index()


def reset_hybrid_index_and_cache() -> None:
    """重置知识索引与检索缓存（L41 同款入口，实现仍在 rag/index_cache.py）。"""

    _reset_index_and_cache()


def retrieve_knowledge(
    *,
    user_message: str,
    rewritten_query: str,
    plan: RetrievalPlan,
    index: KnowledgeIndex,
    embedding_client: Any,
    allowed_kb_ids: set[str] | None = None,
    allowed_audiences: set[str] | None = None,
) -> HybridRetrievalResult:
    """执行宽召回 → 三路融合 → 精排，返回全链路中间产物。"""

    # 5. 第一次检索：用户原话，只作改写前的对比基线，不参与回答。
    raw_hits = retrieve_by_vector(
        user_message,
        top_k=settings.CANDIDATE_K,
        embedding_client=embedding_client,
        allowed_topics=plan.allowed_topics,
        index=index,
        allowed_kb_ids=allowed_kb_ids,
        allowed_audiences=allowed_audiences,
    )

    # 6. 第二次检索：改写后问题，真正用于回答。
    rewritten_hits = retrieve_by_vector(
        rewritten_query,
        top_k=settings.CANDIDATE_K,
        embedding_client=embedding_client,
        allowed_topics=plan.allowed_topics,
        index=index,
        allowed_kb_ids=allowed_kb_ids,
        allowed_audiences=allowed_audiences,
    )

    # 7. 关键词通道：补向量召回够不着的长尾精确词。
    keyword_hits = keyword_retrieve(
        plan,
        embedding_client=embedding_client,
        index=index,
        allowed_kb_ids=allowed_kb_ids,
        allowed_audiences=allowed_audiences,
    )

    vector_candidates = merge_candidates(raw_hits, rewritten_hits)

    # Task 13：BM25 硬关键词专路。
    bm25_engine: SearchEngine | None = None
    if settings.BM25_ENABLED:
        bm25_engine = BM25Local(list(index.chunks_by_id.values()))

    def _bm25_for(query: str) -> list[KnowledgeHit]:
        if bm25_engine is None:
            return []
        hits = bm25_engine.search(query, top_k=settings.CANDIDATE_K)
        if allowed_kb_ids is not None or allowed_audiences is not None:
            hits = [
                hit
                for hit in hits
                if (
                    allowed_kb_ids is None
                    or not hit.chunk.kb_id
                    or hit.chunk.kb_id in allowed_kb_ids
                )
                and (
                    allowed_audiences is None
                    or hit.chunk.audience in allowed_audiences
                )
            ]
        return hits

    bm25_hits = _bm25_for(rewritten_query)

    # Task 14：RAG-Fusion——多子查询各跑 hybrid 后 RRF 融合。
    fusion_sub_queries: list[str] = []
    if settings.RAG_FUSION_ENABLED:
        fusion_sub_queries = generate_sub_queries(
            rewritten_query, count=settings.FUSION_QUERY_COUNT
        )
        # 原查询的融合结果也算一个列表一起进 RRF：子查询会把语义重心拆散，
        # 不保留原始召回的话，只在原查询里排第 1 的片段会被「多列表中等排名」的碎片挤掉。
        sub_ranked: list[list[KnowledgeHit]] = [
            merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)
        ]
        for sub_query in fusion_sub_queries:
            sub_vector = retrieve_by_vector(
                sub_query,
                top_k=settings.CANDIDATE_K,
                embedding_client=embedding_client,
                allowed_topics=plan.allowed_topics,
                index=index,
                allowed_kb_ids=allowed_kb_ids,
                allowed_audiences=allowed_audiences,
            )
            # 关键词通道依赖 plan.keyword_terms（不随子查询变化），复用循环外结果，
            # 否则同一批片段会在每个列表重复出现，被 RRF 累加多倍权重。
            sub_ranked.append(
                merge_hybrid_hits(plan, sub_vector, keyword_hits, _bm25_for(sub_query))
            )
        candidates = rrf_fuse(sub_ranked)
    else:
        candidates = merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)

    rerank_outcome = rerank_candidates(rewritten_query, candidates)
    return HybridRetrievalResult(
        raw_hits=raw_hits,
        rewritten_hits=rewritten_hits,
        keyword_hits=keyword_hits,
        bm25_hits=bm25_hits,
        fusion_sub_queries=fusion_sub_queries,
        vector_candidates=vector_candidates,
        candidates=candidates,
        reranked_hits=rerank_outcome.hits,
        rerank_outcome=rerank_outcome,
    )
