"""RAG 质量检查与低置信判断（L12）。"""

from __future__ import annotations

import json

from api.schemas import (
    KnowledgeHit,
    RagQualityCase,
    RagQualityCaseResult,
    RagQualitySummary,
)
from config import settings
from config.settings import CANDIDATE_K, FINAL_TOP_K, LOW_CONFIDENCE_THRESHOLD, QUALITY_CASES_PATH
from embeddings.client import EmbeddingClient
from rag.vector_store import retrieve_by_vector, merge_candidates, keyword_retrieve, merge_hybrid_hits
# Task 15：评测链路也用 release 索引 + RBAC，确保多角色/越权检测真实生效。
from rag.index_cache import get_release_index
from store.releases import get_allowed_kb_ids, get_allowed_audiences
# L13：评测和线上走同一条「归一化 -> 意图 -> 查询改写 -> 检索」链路，否则测的不是真实表现。
from rag.intent import classify_intent
from rag.query_rewrite import normalize_query, rewrite_retrieval_query
# L14：线上链路又加了 reranker，评测必须同样走「合并候选 -> 精排」，
# 否则评测还在测 L13 的老链路，分数对不上线上表现。
from rag.reranker import rerank_candidates
# L15：线上链路再加检索前计划和关键词通道，评测同步走 hybrid 全链路。
from rag.planning import pre_retrieval_plan

def load_quality_cases() -> list[RagQualityCase]:
    """读取固定问题集。"""
    with QUALITY_CASES_PATH.open(encoding="utf-8") as file:
        return [RagQualityCase.model_validate(item) for item in json.load(file)]

def is_low_confidence(hits: list[KnowledgeHit]) -> bool:
    """看最高分够不够支撑回答。"""
    if not hits:
        return True
    return hits[0].score < LOW_CONFIDENCE_THRESHOLD

def evaluate_quality_case(
    case: RagQualityCase,
    embedding_client: EmbeddingClient | None = None,
    index_override=None,
) -> RagQualityCaseResult:
    """跑一条用例，算 recall@k、precision@k 和是否兜底。

    Task 15 扩展：
    - 用 case.runtime_role 决定 RBAC 过滤（driver/agent/admin 可见范围不同）；
    - 若命中 case.forbidden_chunk_ids → forbidden_hit=True 且 passed=False；
    - evaluate_quality_case 链路同步线上（release 索引 + RBAC + hybrid）。

    Task 16 扩展：
    - index_override：发布门禁传入候选快照的临时索引，避免依赖已发布的 release 索引。
    """
    # 1. 获取索引（release 或全局；index_override 优先）。
    if index_override is not None:
        index = index_override
    elif settings.KNOWLEDGE_LIFECYCLE_ENABLED:
        index = get_release_index("rel-0001")
    else:
        from rag.index_cache import get_knowledge_index
        index = get_knowledge_index()

    # 2. RBAC 过滤参数（按 case.runtime_role）。
    allowed_kb_ids = None
    allowed_audiences = None
    if settings.RBAC_ENABLED:
        allowed_kb_ids = get_allowed_kb_ids(case.runtime_role)
        allowed_audiences = get_allowed_audiences(case.runtime_role)

    # 3. 评测链路（L13-L15 全链路 + Task 11 RBAC + Task 13 BM25）。
    intent_result = classify_intent(normalize_query(case.question))
    rewrite = rewrite_retrieval_query(case.question, intent_result.intent)
    plan = pre_retrieval_plan(case.question, intent_result, rewrite)

    raw_hits = retrieve_by_vector(
        case.question, top_k=CANDIDATE_K, embedding_client=embedding_client,
        allowed_topics=plan.allowed_topics, index=index,
        allowed_kb_ids=allowed_kb_ids, allowed_audiences=allowed_audiences,
    )
    rewritten_hits = retrieve_by_vector(
        rewrite.rewritten_query, top_k=CANDIDATE_K, embedding_client=embedding_client,
        allowed_topics=plan.allowed_topics, index=index,
        allowed_kb_ids=allowed_kb_ids, allowed_audiences=allowed_audiences,
    )
    keyword_hits = keyword_retrieve(
        plan, embedding_client=embedding_client, index=index,
        allowed_kb_ids=allowed_kb_ids, allowed_audiences=allowed_audiences,
    )

    # Task 13：BM25 专路。
    from rag.search_engine import BM25Local
    bm25_hits: list[KnowledgeHit] = []
    if settings.BM25_ENABLED:
        bm25_engine = BM25Local(list(index.chunks_by_id.values()))
        bm25_hits = bm25_engine.search(rewrite.rewritten_query, top_k=CANDIDATE_K)
        if allowed_kb_ids is not None or allowed_audiences is not None:
            bm25_hits = [
                h for h in bm25_hits
                if (allowed_kb_ids is None or not h.chunk.kb_id or h.chunk.kb_id in allowed_kb_ids)
                and (allowed_audiences is None or h.chunk.audience in allowed_audiences)
            ]

    vector_candidates = merge_candidates(raw_hits, rewritten_hits)
    candidates = merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)
    reranked_hits = rerank_candidates(rewrite.rewritten_query, candidates).hits
    fallback = is_low_confidence(reranked_hits)
    retrieved_ids = [] if fallback else [hit.chunk.chunk_id for hit in reranked_hits[:FINAL_TOP_K]]

    expected = set(case.expected_chunk_ids)
    retrieved = set(retrieved_ids)
    forbidden = set(case.forbidden_chunk_ids)
    forbidden_hit = bool(forbidden & retrieved)

    if case.must_fallback:
        # 本来就该兜底。
        passed = fallback and not retrieved_ids and not forbidden_hit
        return RagQualityCaseResult(
            case_id=case.case_id,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=case.expected_chunk_ids,
            recall_at_k=1.0 if (fallback and not forbidden_hit) else 0.0,
            precision_at_k=1.0 if (fallback and not forbidden_hit) else 0.0,
            fallback=fallback,
            forbidden_hit=forbidden_hit,
            runtime_role=case.runtime_role,
            passed=passed,
        )

    matched = expected & retrieved
    recall = len(matched) / max(len(expected), 1)
    precision = len(matched) / max(len(retrieved), 1)
    # Task 15：命中 forbidden → 强制失败。
    passed = (recall > 0 and not fallback and not forbidden_hit)
    return RagQualityCaseResult(
        case_id=case.case_id,
        retrieved_chunk_ids=retrieved_ids,
        expected_chunk_ids=case.expected_chunk_ids,
        recall_at_k=round(recall, 3),
        precision_at_k=round(precision, 3),
        fallback=fallback,
        forbidden_hit=forbidden_hit,
        runtime_role=case.runtime_role,
        passed=passed,
    )

def run_rag_quality_check(
    cases: list[RagQualityCase] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> RagQualitySummary:
    """跑完整套固定问题集，返回汇总。"""
    quality_cases = cases if cases is not None else load_quality_cases()
    results = [
        evaluate_quality_case(case, embedding_client=embedding_client)
        for case in quality_cases
    ]
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    average_recall = sum(result.recall_at_k for result in results) / max(total, 1)
    average_precision = sum(result.precision_at_k for result in results) / max(total, 1)
    return RagQualitySummary(
        total_cases=total,
        passed_cases=passed,
        average_recall_at_k=round(average_recall, 3),
        average_precision_at_k=round(average_precision, 3),
        results=results,
    )

# 固定问题集和知识库在一个进程生命周期内都不变，质量结果就不必每轮重算。
_QUALITY_SUMMARY_CACHE: dict[str, RagQualitySummary] = {} #目的是让每个进程只算一次，避免重复计算

def get_quality_summary(
    embedding_client: EmbeddingClient | None = None,
) -> RagQualitySummary:
    """带进程内缓存的 run_rag_quality_check。仅用于后台预热，不进请求路径。"""
    key = str(QUALITY_CASES_PATH)
    if key not in _QUALITY_SUMMARY_CACHE:
        _QUALITY_SUMMARY_CACHE[key] = run_rag_quality_check(embedding_client=embedding_client)
    return _QUALITY_SUMMARY_CACHE[key]


def peek_quality_summary() -> RagQualitySummary | None:
    """非阻塞读取质量摘要：缓存没就绪就返回 None，绝不触发重算。"""
    return _QUALITY_SUMMARY_CACHE.get(str(QUALITY_CASES_PATH))