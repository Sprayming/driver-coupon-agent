"""第 10 课向量库和相似度检索。

L15 在本文件新增 hybrid 召回的两个关键词侧函数：
- keyword_retrieve：关键词通道，补足向量召回够不着的长尾精确词；
- merge_hybrid_hits：向量候选池 + 关键词召回的融合打分，产出喂给 reranker 的候选池。
"""

from __future__ import annotations

import math
import weakref

from api.schemas import KnowledgeChunk, KnowledgeHit, KnowledgeIndex, RetrievalPlan, VectorRecord
from config.settings import CANDIDATE_K, KEYWORD_TOP_K, RETRIEVAL_SCORE_THRESHOLD, TOP_K
from embeddings.client import DEFAULT_EMBEDDING_CLIENT, EmbeddingClient, embed_text, embed_texts
from rag.index_cache import get_knowledge_index
from rag.knowledge_base import load_knowledge_chunks, query_asks_for_history, should_include_chunk_for_query
from rag.planning import topic_allowed


# L16：缓存结构从「client -> 向量库」升级为「client -> {索引版本 -> 向量库}」。
# 旧结构（注释保留，不删）：weakref.WeakKeyDictionary[EmbeddingClient, list[VectorRecord]]
# 升级原因：旧 key 只有 client，知识 md 改了不重启进程，旧向量会无感知地一直用；
# 现在 key 带索引版本（知识内容 sha256 指纹），知识一变向量自动重算。
_VECTOR_STORE_CACHE: weakref.WeakKeyDictionary[EmbeddingClient, dict[str, list[VectorRecord]]] = weakref.WeakKeyDictionary()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的 cosine similarity，并显式拒绝维度不一致。"""

    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：left={len(left)}, right={len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


def build_vector_store(
    chunks: list[KnowledgeChunk] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[VectorRecord]:
    """把知识 chunk 批量向量化，构建进程内向量库。"""

    source_chunks = chunks if chunks is not None else load_knowledge_chunks()
    embedding_texts = [
        " ".join([chunk.title, chunk.section, " ".join(chunk.keywords), chunk.text])
        for chunk in source_chunks
    ]
    embeddings = embed_texts(embedding_texts, embedding_client)
    return [
        VectorRecord(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(source_chunks, embeddings)
    ]


def get_vector_store(
    embedding_client: EmbeddingClient | None = None,
    index: KnowledgeIndex | None = None,
) -> list[VectorRecord]:
    """按 embedding 客户端和索引版本缓存知识向量库。

    index：Task 9 传入 release 专属索引；None 时回退 L16 全局单索引。
    """

    client = embedding_client or DEFAULT_EMBEDDING_CLIENT
    # Task 9：优先用传入的 release 索引；None 时走 L16 全局单索引（开关关闭时的旧路径）。
    resolved_index = index if index is not None else get_knowledge_index()
    version_store = _VECTOR_STORE_CACHE.setdefault(client, {})
    if resolved_index.version not in version_store:
        # L16：chunk 从索引快照取（和检索缓存同一份知识来源），缓存 key 带索引版本——
        # 知识 md 一改指纹就变，下一轮请求自动用新 chunk 重建向量，旧向量作废。
        version_store[resolved_index.version] = build_vector_store(
            chunks=list(resolved_index.chunks_by_id.values()), embedding_client=client
        )
    return version_store[resolved_index.version]


def retrieve_by_vector(
    query: str,
    top_k: int = CANDIDATE_K,
    threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    embedding_client: EmbeddingClient | None = None,
    allowed_topics: list[str] | None = None,
    index: KnowledgeIndex | None = None,
    # Task 11：RBAC 过滤。None=不过滤（L16 旧路径或测试）；传入时只保留该角色可见的 chunk。
    allowed_kb_ids: set[str] | None = None,
    allowed_audiences: set[str] | None = None,
) -> list[KnowledgeHit]:
    """向量化用户问题，并按相似度召回 top_k 知识片段。

    index：Task 9 传入 release 专属索引；None 时回退 L16 全局单索引。
    allowed_kb_ids/allowed_audiences：Task 11 RBAC 过滤，只保留角色可见的 chunk。
    """

    query_embedding = embed_text(query, embedding_client)
    asks_for_history = query_asks_for_history(query)
    hits: list[KnowledgeHit] = []
    for record in get_vector_store(embedding_client=embedding_client, index=index):
        chunk = record.chunk
        if not should_include_chunk_for_query(chunk, asks_for_history):
            continue
        # Task 11：RBAC 硬过滤——kb 不在授权集合或 audience 不可见的 chunk 直接跳过。
        if allowed_kb_ids is not None and chunk.kb_id and chunk.kb_id not in allowed_kb_ids:
            continue
        if allowed_audiences is not None and chunk.audience not in allowed_audiences:
            continue
        # L15：主题硬过滤叠加在 effective_status 宽进（方案Y）之上——
        # 方案Y 解决「时效误杀」，本过滤解决「主题串场」，两者维度不同不冲突。
        # topic 为空的 chunk 保守放行（和 planning.topic_allowed 同一口径）。
        if allowed_topics is not None and chunk.topic and chunk.topic not in allowed_topics:
            continue
        score = cosine_similarity(query_embedding, record.embedding)
        if score >= threshold:
            rounded = round(score, 3)
            # L13：KnowledgeHit 只有 score。L14 补上 vector_score，
            # 让「初召回分」和「重排后的最终分」能分开看，reranker 也才有基准分可以加减。
            # hits.append(KnowledgeHit(chunk=record.chunk, score=round(score, 3)))
            # L15：再补 sources=["vector"]，hybrid 融合时靠它区分本片段来自哪一路召回。
            hits.append(
                KnowledgeHit(chunk=record.chunk, score=rounded, vector_score=rounded, sources=["vector"])
            )
    return sorted(hits, key=lambda hit: hit.vector_score, reverse=True)[:top_k]


def merge_candidates(*candidate_groups: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """合并多个查询（原话 + 改写）召回的候选片段，供 reranker 精排。

    规则：
    - 按 chunk_id 去重，同一个 chunk 只留一次；
    - 重复时保留较高向量分的那一条（两次召回谁更确信就用谁的分）；
    - 合并后仍按向量分排序并截到 CANDIDATE_K，避免候选池被某一次召回撑爆。
    """

    merged: dict[str, KnowledgeHit] = {}
    for group in candidate_groups:
        for hit in group:
            current = merged.get(hit.chunk.chunk_id)
            if current is None or hit.vector_score > current.vector_score:
                merged[hit.chunk.chunk_id] = hit
    return sorted(merged.values(), key=lambda hit: hit.vector_score, reverse=True)[:CANDIDATE_K]


# —— L15 关键词通道 ——
# 稀有词表：出现即高区分度的长尾精确词，命中权重 ×2（BM25 的直觉：稀有词比泛词更能锁定规则）。
# 只影响打分不影响匹配，词表全集在 planning.SCENE_KEYWORD_TERMS。
KEYWORD_RARE_TERMS = {"12315", "时段卡", "夜卡", "原路退回", "未到账", "到账时效", "转人工", "2025", "双11", "赠券"}
# 课程版归一化除数：把几个关键词累计命中的分数压回 0-1，方便和向量分放在同一视图里比较。
KEYWORD_SCORE_NORMALIZER = 8


def keyword_retrieve(
    plan: RetrievalPlan,
    embedding_client: EmbeddingClient | None = None,
    index: KnowledgeIndex | None = None,
    allowed_kb_ids: set[str] | None = None,
    allowed_audiences: set[str] | None = None,
) -> list[KnowledgeHit]:
    """关键词召回：用 plan.keyword_terms 撞 chunk 的 keywords/text，补足向量召回够不着的长尾精确词。

    index：Task 9 传入 release 专属索引；None 时回退 L16 全局单索引。
    allowed_kb_ids/allowed_audiences：Task 11 RBAC 过滤，只保留角色可见的 chunk。
    """

    hits: list[KnowledgeHit] = []
    for record in get_vector_store(embedding_client=embedding_client, index=index):
        chunk = record.chunk
        # Task 11：RBAC 硬过滤。
        if allowed_kb_ids is not None and chunk.kb_id and chunk.kb_id not in allowed_kb_ids:
            continue
        if allowed_audiences is not None and chunk.audience not in allowed_audiences:
            continue
        if not topic_allowed(chunk, plan):
            continue
        matched = [term for term in plan.keyword_terms if term in chunk.keywords or term in chunk.text]
        if not matched:
            continue
        raw_score = sum(2.0 if term in KEYWORD_RARE_TERMS else 1.0 for term in set(matched))
        score = round(min(1.0, raw_score / KEYWORD_SCORE_NORMALIZER), 3)
        hits.append(
            KnowledgeHit(
                chunk=chunk,
                score=score,
                keyword_score=score,
                sources=["keyword"],
                matched_keywords=matched,
            )
        )
    return sorted(hits, key=lambda hit: hit.keyword_score, reverse=True)[:KEYWORD_TOP_K]


# merge 层小权重（量级和 reranker 的 REASON_WEIGHTS 一致：单项不超过 0.25，定位是微调不是推翻）。
# 用户拍板（L15）：merge 加小权重、reranker 保持现状；时效信号两层都有，用质量用例观察是否误伤。
SCENE_ROUTE_BOOST = 0.08
FRESHNESS_BOOST_ACTIVE = 0.06
FRESHNESS_BOOST_HISTORICAL = -0.10


def merge_hybrid_hits(
    plan: RetrievalPlan,
    vector_hits: list[KnowledgeHit],
    keyword_hits: list[KnowledgeHit],
    bm25_hits: list[KnowledgeHit] | None = None,
) -> list[KnowledgeHit]:
    """合并向量/关键词/BM25 三路召回：按 chunk_id 去重，叠加场景/时效小权重。

    和课程版的三处刻意偏离（注释备查，不删）：
    1) 课程版在 merge 里把「问题含"当前" + expired 片段」直接丢弃；项目不采用——
       promotion 意图改写会补"当前有效/现行规则"，问 2025 历史活动的问题 rewritten_query 也含"当前"，
       直接丢弃会把方案Y 放行进池的历史片段误杀。时效一律走下面的 ± 小权重 + reranker 软降权。
    2) 课程版 unknown 场景也会对全主题加 route_boost（无区分度）；项目 unknown 场景不加。
    3) 课程版 merge 后直接截 FINAL_TOP_K=2（课程 L15 这条线没有 reranker）；项目截 CANDIDATE_K，
       保持候选池宽度交给 L14 reranker 精排，最终回答条数仍由 FINAL_TOP_K 控制。
    """

    merged: dict[str, KnowledgeHit] = {}
    for hit in [*vector_hits, *keyword_hits, *(bm25_hits or [])]:
        chunk_id = hit.chunk.chunk_id
        current = merged.get(chunk_id)
        if current is None:
            merged[chunk_id] = hit
            continue
        # 同一 chunk 被多路召回时：各路分各取 max，来源和命中词合并去重。
        merged[chunk_id] = current.model_copy(
            update={
                "vector_score": max(current.vector_score, hit.vector_score),
                "keyword_score": max(current.keyword_score, hit.keyword_score),
                "bm25_score": max(current.bm25_score, hit.bm25_score),
                "sources": list(dict.fromkeys([*current.sources, *hit.sources])),
                "matched_keywords": list(dict.fromkeys([*current.matched_keywords, *hit.matched_keywords])),
            }
        )

    final_hits: list[KnowledgeHit] = []
    for hit in merged.values():
        route_boost = (
            SCENE_ROUTE_BOOST
            if plan.scene != "unknown" and hit.chunk.topic and hit.chunk.topic in plan.allowed_topics
            else 0.0
        )
        freshness_boost = (
            FRESHNESS_BOOST_ACTIVE if hit.chunk.effective_status == "active" else FRESHNESS_BOOST_HISTORICAL
        )
        # Hybrid 是互补召回：任一路有强证据都该留下，再叠加场景路由和时效信号。
        base_score = max(hit.vector_score, hit.keyword_score, hit.bm25_score)
        score = round(max(0.0, min(1.0, base_score + route_boost + freshness_boost)), 3)
        # 召回来源原因先写进 rerank_reasons，reranker 精排时会把自己的加减分原因追加在后面，
        # 调试台能看到一条片段从进池到精排的完整理由链。
        reasons = [f"{source}召回" for source in hit.sources]
        if route_boost:
            reasons.append(f"{plan.scene}场景主题命中")
        if hit.chunk.effective_status == "historical":
            reasons.append("历史片段时效降权")
        final_hits.append(hit.model_copy(update={"score": score, "rerank_reasons": reasons}))
    return sorted(final_hits, key=lambda hit: hit.score, reverse=True)[:CANDIDATE_K]
