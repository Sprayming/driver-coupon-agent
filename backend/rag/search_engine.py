"""Task 13：BM25 硬关键词专路（政策编号/商品名精确命中通道）。

SearchEngine 抽象接口 + BM25Local 实现；可替换为测试替身。
BM25 分经归一化叠加到 KnowledgeHit.bm25_score，sources=["bm25"]。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from api.schemas import KnowledgeChunk, KnowledgeHit

# 精确命中 boost：query 含 chunk 的 policy_code 或 product_name 时，
# BM25 通道直接给高分（让精确码/商品名查询的对应 chunk 排到第一）。
EXACT_MATCH_BOOST = 0.9


class SearchEngine(ABC):
    """检索引擎抽象接口。BM25 专路和测试替身都实现 search()。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[KnowledgeHit]:
        """按 query 召回 top_k 条命中。"""
        ...


class BM25Local(SearchEngine):
    """基于 rank_bm25 的本地 BM25 实现，叠加 policy_codes/product_names 精确命中高分。"""

    def __init__(self, chunks: Sequence[KnowledgeChunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:  # pragma: no cover - rank_bm25 应在依赖中
            BM25Okapi = None

        self._chunks = list(chunks)
        self._tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized) if (BM25Okapi and self._tokenized) else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：中文按字、英文/数字按词。"""
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 10) -> list[KnowledgeHit]:
        if not self._bm25:
            return []

        query_tokens = self._tokenize(query)
        raw_scores = self._bm25.get_scores(query_tokens)
        hits: list[KnowledgeHit] = []

        for chunk, raw in zip(self._chunks, raw_scores):
            # 精确命中识别：query 含 chunk 的任一 policy_code / product_name → 高分。
            exact_boost = 0.0
            q_lower = query.lower()
            for code in chunk.policy_codes:
                if code and code.lower() in q_lower:
                    exact_boost = max(exact_boost, EXACT_MATCH_BOOST)
            for name in chunk.product_names:
                if name and name.lower() in q_lower:
                    exact_boost = max(exact_boost, EXACT_MATCH_BOOST)

            # BM25 原始分归一化到 0-1（用 raw/(raw+1) 的饱和曲线）。
            normalized = raw / (raw + 1.0) if raw > 0 else 0.0
            bm25_score = min(1.0, max(normalized, exact_boost))
            if bm25_score <= 0.0:
                continue

            hits.append(
                KnowledgeHit(
                    chunk=chunk,
                    score=round(bm25_score, 3),
                    bm25_score=round(bm25_score, 3),
                    sources=["bm25"],
                )
            )

        return sorted(hits, key=lambda h: h.bm25_score, reverse=True)[:top_k]
