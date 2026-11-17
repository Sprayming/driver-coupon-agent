"""检索链路消融实验。

逐项关闭召回通道与重排层，对比 hit@2 与 MRR@8，
用来回答「四路召回和规则重排各自贡献多少」这个问题。

用法（在 backend 目录下）：
    python -m evals.retrieval_ablation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import settings
from rag.fusion import generate_sub_queries, rrf_fuse
from rag.index_cache import get_knowledge_index
from rag.intent import classify_intent
from rag.planning import pre_retrieval_plan
from rag.query_rewrite import rewrite_retrieval_query
from rag.reranker import rerank_candidates
from rag.search_engine import BM25Local
from rag.vector_store import (
    keyword_retrieve,
    merge_candidates,
    merge_hybrid_hits,
    retrieve_by_vector,
)

_BACKEND_DIR = Path(__file__).resolve().parents[1]
# 优先用专门为消融构造的扩样本集；没有时回退到质量回归的原始 8 条。
CASES_PATH = (
    _BACKEND_DIR / "rag_ablation_cases.json"
    if (_BACKEND_DIR / "rag_ablation_cases.json").exists()
    else _BACKEND_DIR / "rag_quality_cases.json"
)

# 与 settings 保持一致的常量（避免 import 路径差异导致脚本跑不起来）。
CANDIDATE_K = 8
FINAL_TOP_K = 2


def load_cases() -> list[dict]:
    """只保留有期望片段的用例——没有期望的兜底题无法算命中率。"""
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return [case for case in raw if case.get("expected_chunk_ids")]


def retrieve(
    case: dict,
    *,
    use_rewrite: bool = True,
    use_keyword: bool = True,
    use_bm25: bool = True,
    use_rerank: bool = True,
    course_style: bool = False,
    use_fusion: bool = False,
) -> list:
    """按开关组合跑一遍检索链路，返回排序后的候选。

    course_style=True 时模拟课程 L41：单查询、向量 + 关键词取 max、
    双路都命中加 0.08，直接截 4 条出结果（无 BM25、无 rerank、无候选池分层）。
    """
    message = case["question"]
    intent_result = classify_intent(message)
    rewrite = rewrite_retrieval_query(message, intent_result.intent)
    plan = pre_retrieval_plan(message, intent_result, rewrite)
    index = get_knowledge_index()

    raw_hits = retrieve_by_vector(
        message,
        top_k=4 if course_style else CANDIDATE_K,
        allowed_topics=plan.allowed_topics,
        index=index,
    )
    if course_style:
        keyword_hits = keyword_retrieve(plan, index=index)
        by_id: dict[str, object] = {}
        for hit in [*raw_hits, *keyword_hits]:
            by_id.setdefault(hit.chunk.chunk_id, hit)
        vector_scores = {hit.chunk.chunk_id: hit.vector_score for hit in raw_hits}
        keyword_scores = {hit.chunk.chunk_id: hit.keyword_score for hit in keyword_hits}
        ranked: list[tuple[float, str]] = []
        for chunk_id in set(vector_scores) | set(keyword_scores):
            vector_score = vector_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            score = max(vector_score, keyword_score) + (0.08 if vector_score and keyword_score else 0.0)
            ranked.append((score, chunk_id))
        ranked.sort(reverse=True)
        return [by_id[chunk_id] for _, chunk_id in ranked[:4]]

    rewritten_hits = (
        retrieve_by_vector(
            rewrite.rewritten_query,
            top_k=CANDIDATE_K,
            allowed_topics=plan.allowed_topics,
            index=index,
        )
        if use_rewrite
        else []
    )
    vector_candidates = merge_candidates(raw_hits, rewritten_hits)

    keyword_hits = keyword_retrieve(plan, index=index) if use_keyword else []

    bm25_hits = []
    if use_bm25:
        engine = BM25Local(list(index.chunks_by_id.values()))
        bm25_hits = engine.search(rewrite.rewritten_query, top_k=CANDIDATE_K)

    if use_fusion:
        # 复刻修复后的 RAG_FUSION：原查询列表也进 RRF；关键词通道复用不随子查询变化的结果。
        sub_queries = generate_sub_queries(
            rewrite.rewritten_query, count=getattr(settings, "FUSION_QUERY_COUNT", 3)
        )
        engine = BM25Local(list(index.chunks_by_id.values()))
        sub_ranked = [merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)]
        for sub_query in sub_queries:
            sub_vec = retrieve_by_vector(
                sub_query, top_k=CANDIDATE_K, allowed_topics=plan.allowed_topics, index=index
            )
            sub_bm25 = engine.search(sub_query, top_k=CANDIDATE_K) if use_bm25 else []
            sub_ranked.append(merge_hybrid_hits(plan, sub_vec, keyword_hits, sub_bm25))
        candidates = rrf_fuse(sub_ranked)
    else:
        candidates = merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)
    if use_rerank:
        return rerank_candidates(rewrite.rewritten_query, candidates).hits
    return candidates


def score(hits: list, expected: list[str]) -> tuple[bool, float]:
    """hit@2：期望片段是否出现在最终前 2；MRR：期望片段倒数排名的均值。"""
    ids = [hit.chunk.chunk_id for hit in hits]
    hit_at_k = any(chunk_id in ids[:FINAL_TOP_K] for chunk_id in expected)
    rank = next((i + 1 for i, chunk_id in enumerate(ids) if chunk_id in expected), None)
    return hit_at_k, (1.0 / rank if rank else 0.0)


CONFIGS: list[tuple[str, dict]] = [
    ("full", {}),
    ("rag_fusion", {"use_fusion": True}),
    ("course_baseline", {"course_style": True}),
    ("no_rewrite", {"use_rewrite": False}),
    ("no_keyword", {"use_keyword": False}),
    ("no_bm25", {"use_bm25": False}),
    ("no_rerank", {"use_rerank": False}),
    ("vector_only", {"use_rewrite": False, "use_keyword": False, "use_bm25": False, "use_rerank": False}),
]


def main() -> None:
    cases = load_cases()
    print(f"用例 {len(cases)} 条（仅含有期望片段的题）\n")
    print(f"{'配置':<14}{'hit@2':>8}{'MRR@8':>9}")
    print("-" * 33)
    for name, flags in CONFIGS:
        hit_total = 0
        mrr_total = 0.0
        for case in cases:
            hits = retrieve(case, **flags)
            hit_at_k, mrr = score(hits, case["expected_chunk_ids"])
            hit_total += int(hit_at_k)
            mrr_total += mrr
        count = len(cases)
        print(f"{name:<14}{hit_total / count:>7.1%}{mrr_total / count:>9.3f}")
    print("\n说明：hit@2 = 期望片段出现在最终前 2 条；MRR@8 = 期望片段倒数排名的均值。")


def detail() -> None:
    """逐题对比 full 与 rag_fusion 的最终排名差异，用于定位噪声来源。"""
    cases = load_cases()
    print(f"{'case':<16}{'full 前2':<44}{'fusion 前2'}")
    print("-" * 110)
    for case in cases:
        plain = [hit.chunk.chunk_id for hit in retrieve(case)][:FINAL_TOP_K]
        fused = [hit.chunk.chunk_id for hit in retrieve(case, use_fusion=True)][:FINAL_TOP_K]
        if plain == fused:
            continue
        expected = ",".join(case["expected_chunk_ids"])
        print(f"{case['case_id']:<16}{str(plain):<44}{fused}  expected={expected}")


if __name__ == "__main__":
    if "--detail" in sys.argv:
        detail()
    else:
        main()
