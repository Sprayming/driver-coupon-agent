"""Task 14：RAG-Fusion 多查询 + RRF 融合。

LLM 生成 N 个多样化子查询，各跑 hybrid 检索后用 Reciprocal Rank Fusion 合并。
失败自动回退单查询，不报错。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from api.schemas import KnowledgeHit

logger = logging.getLogger(__name__)

# RRF 常数 k（课程/论文常用 60）。
RRF_K = 60

FUSION_SYSTEM_PROMPT = (
    "你是一个查询改写助手。给定用户问题，生成 {count} 个多样化的子查询，"
    "覆盖不同角度（如规则细节、适用条件、例外情形）。"
    "只返回 JSON 数组，不要其他文字。"
)


def generate_sub_queries(
    query: str,
    count: int = 3,
    llm_caller: Callable[..., Any] | None = None,
) -> list[str]:
    """用 LLM 生成多样化子查询。LLM 失败或 JSON 解析失败时回退到单查询 [query]。"""

    if llm_caller is None:
        from models.llm_client import call_chat_model

        llm_caller = call_chat_model

    messages = [
        {"role": "system", "content": FUSION_SYSTEM_PROMPT.format(count=count)},
        {"role": "user", "content": query},
    ]
    try:
        result = llm_caller(messages)
        text = (result.answer or "").strip()
        # 清理可能的 ```json 代码块包裹。
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [q for q in parsed if q][:count]
    except Exception as exc:  # noqa: BLE001 - 回退设计：任何异常都回退单查询。
        logger.warning("RAG-Fusion 子查询生成失败，回退单查询: %s", exc)
    return [query]


def rrf_fuse(
    ranked_lists: list[list[KnowledgeHit]], k: int = RRF_K
) -> list[KnowledgeHit]:
    """Reciprocal Rank Fusion：合并多个排序列表。

    score(d) = Σ 1/(k + rank_i(d))，rank 从 1 开始。
    最终 score 归一化到 0-1，sources 追加 "fusion"。
    """

    scores: dict[str, float] = {}
    chunk_map: dict[str, KnowledgeHit] = {}

    for ranked in ranked_lists:
        for rank_idx, hit in enumerate(ranked, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_idx)
            if cid not in chunk_map:
                chunk_map[cid] = hit

    if not scores:
        return []

    max_score = max(scores.values())
    fused: list[KnowledgeHit] = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        hit = chunk_map[cid]
        fused.append(
            hit.model_copy(
                update={
                    "score": round(min(1.0, rrf_score / max_score), 3),
                    "sources": list(dict.fromkeys([*hit.sources, "fusion"])),
                }
            )
        )
    return fused
