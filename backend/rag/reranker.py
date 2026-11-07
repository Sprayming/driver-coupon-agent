"""第 14 课卡券版 reranker 重排。

课程主线：召回负责「宽」，重排负责「精」。
- 默认模式：本地轻量规则模拟 reranker 的排序信号（本质就是加减分），零网络延迟；
- 商业模式：配好 Key 并打开开关后，调外部 `/rerank`（Qwen3-Reranker-8B）做真正的交叉注意力精排；
- 兜底：商业 reranker 调用失败或返回空，自动回退到轻量规则，回答链路不因外部服务挂掉而中断。

L14 方案 Y 的落点也在这一层：召回层不再硬过滤 historical 片段，
「用户没问历史才降权」这个软信号必须由 reranker 来给。
"""

from __future__ import annotations

import os

import httpx

from api.schemas import KnowledgeHit, RerankConfig, RerankOutcome
from config.settings import (
    RERANK_ENABLED_ENV,
    RERANK_INSTRUCTION_DEFAULT,
    RERANK_MODEL_DEFAULT,
    RERANK_TIMEOUT_SECONDS,
    api_key_is_missing,
    env_flag_enabled,
    load_course_env,
)
from rag.knowledge_base import query_asks_for_history
from rag.query_rewrite import normalize_query


# 轻量重排规则的加减分权重。
# 课程版是 current +0.22 / expired -0.25 / promotion +0.12 / 结算页 +0.08 / 非活动 -0.08；
# 本项目换成卡券业务的对应规则，权重沿用同一量级：单项不超过 0.25，
# 保证「重排是微调向量排序」，而不是把初召回的分完全推翻。
REASON_WEIGHTS: dict[str, float] = {
    "现行有效规则加权": 0.22,
    "历史卡券降权": -0.25,
    "用户明确问历史，历史片段放行不降权": 0.0,
    "投诉规则优先": 0.12,
    "退款条件匹配": 0.08,
    "非投诉规则轻微降权": -0.08,
}

# 投诉类检索词：投诉问题应该让投诉规则压过退款 / 使用规则。
COMPLAINT_TERMS = ["投诉", "举报", "曝光", "差评", "客服态度"]


def explain_rerank_reasons(query: str, hit: KnowledgeHit, asks_for_history: bool) -> list[str]:
    """解释候选片段为什么被加权或降权。

    这里读的是「归一化后」的查询，和检索用的是同一套归一化，
    否则「能退不 → 可以退款吗」这类口语改写就白做了（信号词对不上）。
    """

    query_text = normalize_query(query)
    reasons: list[str] = []

    # ① 现行规则信号：用户明确问「当前」时，active 片段该被顶上来。
    if hit.chunk.effective_status == "active" and "当前" in query_text:
        reasons.append("现行有效规则加权")

    # ② 历史片段信号（L14 方案 Y 的核心）：用户没问历史才降权，明确问了就放行不动。
    if hit.chunk.effective_status == "historical":
        if asks_for_history:
            reasons.append("用户明确问历史，历史片段放行不降权")
        else:
            reasons.append("历史卡券降权")

    # ③ 业务主题对齐：投诉问题里，topic=complaint 的片段优先。
    matched_complaint_term = any(term in query_text for term in COMPLAINT_TERMS)
    if hit.chunk.topic == "complaint" and matched_complaint_term:
        reasons.append("投诉规则优先")

    # ④ 精确片段加权：问「能不能退」时，退款条件 chunk 最该排第一。
    if hit.chunk.chunk_id == "coupon-refund-eligibility" and any(
        term in query_text for term in ["可以退款", "想退", "白买"]
    ):
        reasons.append("退款条件匹配")

    # ⑤ 反向信号：投诉问题里非投诉主题的片段轻微降权，给 topic 对齐的片段让位。
    if hit.chunk.topic != "complaint" and matched_complaint_term:
        reasons.append("非投诉规则轻微降权")

    return reasons


def rerank_candidates_lightweight(query: str, candidates: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """用本地轻量规则重排候选片段（默认模式）。"""

    asks_for_history = query_asks_for_history(query)
    reranked: list[KnowledgeHit] = []
    for hit in candidates:
        # L14 旧基准（注释保留，不删）：基准分显式用 vector_score，
        # 两者此时相等，但写清楚「重排是在向量分上加减」更不容易在后续改动里串味。
        # score = hit.vector_score
        # L15 新基准：hybrid 融合后 hit.score 已经是「双路取 max + 场景/时效小权重」的综合分，
        # 重排在这个新基准上微调；否则关键词通道捞进来的片段（vector_score 可能为 0）永远翻不了身。
        score = hit.score
        reasons = explain_rerank_reasons(query, hit, asks_for_history)
        for reason in reasons:
            score += REASON_WEIGHTS.get(reason, 0.0)

        # 加减分后可能越界，统一夹到 [0, 1]，避免和 KnowledgeHit 的字段约束打架。
        final_score = round(max(0.0, min(1.0, score)), 3)
        # L15：保留 merge 阶段写入的召回来源原因（vector召回/keyword召回/场景命中等），
        # rerank 自己的加减分原因追加在后，调试台能看到「进池 → 精排」的完整理由链。
        combined_reasons = [*hit.rerank_reasons, *reasons]
        reranked.append(
            hit.model_copy(
                update={
                    "score": final_score,
                    "rerank_score": final_score,
                    # 没命中任何规则时也要留一条原因，调试台才能区分「没重排」和「重排了但没信号」。
                    "rerank_reasons": combined_reasons or ["保留向量初始分"],
                }
            )
        )
    return sorted(reranked, key=lambda hit: hit.score, reverse=True)


def build_commercial_rerank_config() -> RerankConfig | None:
    """读取商业 reranker 配置；未启用或缺 Key 时返回 None（走轻量规则）。"""

    load_course_env()
    if not env_flag_enabled(RERANK_ENABLED_ENV, default=False):
        return None

    api_key = os.getenv("AGENT_RAG_RERANK_API_KEY") or os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        return None

    base_url = (
        os.getenv("AGENT_RAG_RERANK_BASE_URL")
        or os.getenv("AGENT_OPENAI_BASE_URL")
        or "https://api.siliconflow.cn/v1"
    ).rstrip("/")
    model = os.getenv("AGENT_RAG_RERANK_MODEL", RERANK_MODEL_DEFAULT)
    instruction = os.getenv("AGENT_RAG_RERANK_INSTRUCTION", RERANK_INSTRUCTION_DEFAULT)
    return RerankConfig(api_key=api_key or "", base_url=base_url, model=model, instruction=instruction)


def rerank_candidates_with_commercial_model(
    query: str,
    candidates: list[KnowledgeHit],
    config: RerankConfig,
) -> list[KnowledgeHit]:
    """调用商业 reranker 对候选片段精排。"""

    payload = {
        "model": config.model,
        "query": query,
        "documents": [hit.chunk.text for hit in candidates],
        "top_n": len(candidates),
        "return_documents": False,
    }
    if config.instruction:
        payload["instruction"] = config.instruction
    with httpx.Client(timeout=RERANK_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            f"{config.base_url}/rerank",
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    ranked: list[KnowledgeHit] = []
    for item in data.get("results", []):
        index = item.get("index")
        # 外部返回的 index 可能越界，先挡住，不能让脏数据直接下标到 candidates。
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            continue

        raw_score = item.get("relevance_score", item.get("score", candidates[index].score))
        final_score = round(max(0.0, min(1.0, float(raw_score))), 3)
        reasons = explain_rerank_reasons(query, candidates[index], query_asks_for_history(query))
        # L15：和轻量模式同一口径，保留 merge 阶段的召回来源原因再追加 rerank 原因。
        combined_reasons = [*candidates[index].rerank_reasons, *reasons]
        ranked.append(
            candidates[index].model_copy(
                update={
                    "score": final_score,
                    "rerank_score": final_score,
                    "rerank_reasons": combined_reasons or ["商业 reranker 精排"],
                }
            )
        )
    return ranked


def rerank_candidates(query: str, candidates: list[KnowledgeHit]) -> RerankOutcome:
    """根据配置选择商业 reranker 或轻量 fallback，并返回本次运行模式。"""

    config = build_commercial_rerank_config()
    if config is None:
        return RerankOutcome(hits=rerank_candidates_lightweight(query, candidates), mode="lightweight")

    try:
        ranked = rerank_candidates_with_commercial_model(query, candidates, config)
        if ranked:
            return RerankOutcome(hits=ranked, mode="commercial", model=config.model)
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        # 外部 reranker 失败不能让整轮问答 500：退回本地规则，同时把错误名带回 session_state 供排查。
        return RerankOutcome(
            hits=rerank_candidates_lightweight(query, candidates),
            mode="commercial_fallback",
            model=config.model,
            error=exc.__class__.__name__,
        )

    return RerankOutcome(
        hits=rerank_candidates_lightweight(query, candidates),
        mode="commercial_fallback",
        model=config.model,
        error="empty_rerank_result",
    )
