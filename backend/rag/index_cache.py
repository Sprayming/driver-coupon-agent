"""L16 知识索引和 RAG 检索缓存。

索引缓存的是知识结构（版本指纹 + 倒排表），检索缓存只缓存稳定知识的检索结果；
最终回答和实时业务问题（订单/物流/退款进度/库存）不能缓存。
用户拍板（L16）：检索缓存放 rerank 之后，命中时连商业 rerank API 一起省。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from api.schemas import KnowledgeChunk, KnowledgeIndex, RetrievalCacheEntry, RetrievalPlan
from config.settings import KNOWLEDGE_LIFECYCLE_ENABLED, RAG_CACHE_ENABLED, RAG_CACHE_MAX_ENTRIES
from rag.knowledge_base import load_knowledge_chunks
from rag.planning import is_realtime_business_query, normalize_query


KNOWLEDGE_INDEX: KnowledgeIndex | None = None
# OrderedDict 实现 LRU：命中 move_to_end，超上限淘汰最旧（课程版是无限 dict，用户拍板加上限）。
RAG_RETRIEVAL_CACHE: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()

# Task 9：按 release 隔离的索引缓存。key=release_id，value=该 release 快照的 KnowledgeIndex。
# 发布/回滚时由 on_release_change 清空，保证旧快照索引不会被新请求复用。
RELEASE_INDEX_CACHE: dict[str, KnowledgeIndex] = {}


def build_knowledge_index(
    chunks: list[KnowledgeChunk], version_prefix: str = ""
) -> KnowledgeIndex:
    """根据知识片段构建索引版本和倒排表。

    version_prefix：Task 9 release 索引传入 release_id（如 "rel-0001"），
    让索引 version 带上 release 标识，检索缓存 key 自然按 release 隔离。
    """

    payload = json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    inverted: dict[str, list[str]] = {}
    for chunk in chunks:
        for keyword in chunk.keywords:
            inverted.setdefault(keyword, []).append(chunk.chunk_id)
    version = f"{version_prefix}:idx-{fingerprint}" if version_prefix else f"idx-{fingerprint}"
    return KnowledgeIndex(
        version=version,
        fingerprint=fingerprint,
        chunk_count=len(chunks),
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
        inverted_index=inverted,
    )


def get_knowledge_index() -> KnowledgeIndex:
    """懒加载当前进程内知识索引。"""

    global KNOWLEDGE_INDEX
    if KNOWLEDGE_INDEX is None:
        KNOWLEDGE_INDEX = build_knowledge_index(load_knowledge_chunks())
    return KNOWLEDGE_INDEX


def rebuild_knowledge_index(chunks: list[KnowledgeChunk] | None = None) -> KnowledgeIndex:
    """重建索引时清空 RAG 检索缓存，避免旧索引产生的缓存继续被复用。

    这就是本课的核心安全点：知识 md 一改 → 指纹变 → 旧缓存全部失效。
    """

    global KNOWLEDGE_INDEX
    source_chunks = chunks if chunks is not None else load_knowledge_chunks()
    KNOWLEDGE_INDEX = build_knowledge_index(source_chunks)
    RAG_RETRIEVAL_CACHE.clear()
    return KNOWLEDGE_INDEX


def reset_index_and_cache() -> None:
    """清空索引和检索缓存，方便测试或手动重建。"""

    global KNOWLEDGE_INDEX
    KNOWLEDGE_INDEX = None
    RAG_RETRIEVAL_CACHE.clear()
    RELEASE_INDEX_CACHE.clear()


def get_release_index(
    release_id: str, db_path: Any = None
) -> KnowledgeIndex:
    """按 release_id 获取（懒构建并缓存）该 release 快照的知识索引。

    Task 9：聊天链路按 driver_id 解析到 release 后，用本函数取对应索引，
    替代 L16 的全局单索引。索引 version 带 release_id 前缀，检索缓存自然隔离。
    """

    if release_id not in RELEASE_INDEX_CACHE:
        # 延迟导入避免循环依赖（releases → index_cache → releases）。
        from store.releases import load_release_chunks

        chunks = load_release_chunks(release_id, db_path=db_path)
        RELEASE_INDEX_CACHE[release_id] = build_knowledge_index(
            chunks, version_prefix=release_id
        )
    return RELEASE_INDEX_CACHE[release_id]


def cache_key_for(plan: RetrievalPlan, index: KnowledgeIndex) -> str:
    """为检索计划和索引版本生成缓存 key。

    key 三要素：索引版本（知识没变）+ 场景（路由没变）+ 归一化改写后问题（问题等价）。
    索引版本在 key 里，是「缓存不答旧规则」的根本保证。
    """

    payload = f"{index.version}|{plan.scene}|{normalize_query(plan.rewritten_query)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_policy_for(plan: RetrievalPlan) -> dict[str, Any]:
    """判断本轮检索结果能否缓存。

    两条红线（课程同款）：
    1. 实时业务问题（我的订单/退款进度/库存…）不能缓存——个人当前状态不是稳定知识；
    2. unknown 场景不缓存——场景识别失败的问题连检索范围都不稳定，缓存命中率没有意义。
    缓存范围永远只是检索命中（retrieval_hits_only），最终回答永不缓存。
    """

    realtime = is_realtime_business_query(plan.original_query)
    cacheable = not realtime and plan.scene != "unknown"
    reason = (
        "稳定知识检索结果可以缓存。"
        if cacheable
        else "实时订单、物流、库存或退款进度不能缓存为知识库答案。"
        if realtime
        else "未识别到稳定知识场景，检索结果不缓存。"
    )
    return {
        "cacheable": cacheable,
        "scope": "retrieval_hits_only",
        "reason": reason,
    }


def cache_get(key: str) -> RetrievalCacheEntry | None:
    """读缓存：总开关关闭或未命中返回 None；命中时把条目移到最新位置（LRU）。"""

    if not RAG_CACHE_ENABLED:
        return None
    entry = RAG_RETRIEVAL_CACHE.get(key)
    if entry is not None:
        RAG_RETRIEVAL_CACHE.move_to_end(key)
    return entry


def cache_put(key: str, entry: RetrievalCacheEntry) -> None:
    """写缓存：总开关关闭直接跳过；超上限淘汰最旧条目。"""

    if not RAG_CACHE_ENABLED:
        return
    RAG_RETRIEVAL_CACHE[key] = entry
    RAG_RETRIEVAL_CACHE.move_to_end(key)
    while len(RAG_RETRIEVAL_CACHE) > RAG_CACHE_MAX_ENTRIES:
        RAG_RETRIEVAL_CACHE.popitem(last=False)


# —— Task 8：发布/回滚后的缓存失效接线 ——

def prewarm_active_release_indexes() -> None:
    """为当前活跃 stable/candidate 预构建索引与向量库（发布后/启动预热，方案 A+B）。

    懒构建的「索引 + 全量 chunk embedding」原本由第一个请求的司机买单（实测约 30s）；
    预热把这笔成本转移到发布动作与启动期，司机首问直接命中热缓存。
    - KNOWLEDGE_LIFECYCLE_ENABLED 读模块级常量：测试 monkeypatch settings 时预热自然跳过，
      避免测试意外触发真实 embedding API；
    - 任何失败由调用方容错（_emit_release_change 吞异常 / warmup 打印留空），不影响主流程。
    """

    if not KNOWLEDGE_LIFECYCLE_ENABLED:
        return
    # 局部 import 避免模块级循环依赖（store.releases ← rag.index_cache 经监听器间接相连）。
    from rag.vector_store import get_vector_store
    from store.releases import list_releases

    started_at = time.perf_counter()
    warmed: list[str] = []
    for release in list_releases():
        if release.status not in ("stable", "candidate"):
            continue
        index = get_release_index(release.release_id)
        # 真正触发全量 chunk embedding（向量库按索引版本缓存，内容没变时秒回）。
        get_vector_store(index=index)
        warmed.append(f"{release.release_id}({release.status},{index.chunk_count}chunks)")
    print(
        f"[prewarm] release 索引+向量预热完成 {warmed}，"
        f"用时 {time.perf_counter() - started_at:.1f}s",
        flush=True,
    )


def on_release_change(event: dict) -> None:
    """release 变更监听器：发布新 stable 或回滚时清空 L16 全局索引与检索缓存。

    Task 8 阶段：先清空全局单索引（扩展 L16 clear 机制），保证回滚后首请求 cache_hit=False；
    Task 9 会把这里升级为「按 release_id 隔离失效」，旧路径注释保留。

    发布后预热（2026-10-18 用户拍板，方案 A）：失效后同步为新活跃 stable/candidate
    重建索引与向量库——发布请求多等几秒，换司机首问从约 30s 降到毫秒级。
    """

    reset_index_and_cache()
    prewarm_active_release_indexes()
