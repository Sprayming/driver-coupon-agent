"""知识路径的确定性结果构造（对标小哲 L41 `rag/knowledge.py`）。

职责边界：
- 本模块只做「不调模型也能给出的知识侧结果」：低置信兜底、寒暄、Citation 转换
- 检索链路由 `rag/hybrid_retrieval.py` 负责，回答生成由 `models/answer_client.py` 负责

把这些结果从 Agent 编排层搬出来，才能单独测试「无可靠依据时到底返回什么」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from api.schemas import Citation, KnowledgeHit


@dataclass(frozen=True)
class KnowledgePathResult:
    """稳定知识路径的确定性结果，供 Agent 编排层直接拼装响应。"""

    answer: str
    answer_path: str
    citations: list[Citation] = field(default_factory=list)


def citations_from_hits(hits: list[KnowledgeHit]) -> list[Citation]:
    """将检索命中的知识片段转换为公开 Citation。"""

    return [
        Citation(
            citation_id=f"C{index}",
            source_path=hit.chunk.source_path,
            source_title=hit.chunk.title,
            chunk_id=hit.chunk.chunk_id,
            score=hit.score,
            snippet=hit.chunk.text,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def low_confidence_result() -> KnowledgePathResult:
    """低置信或无可靠依据时的兜底回答，不调模型、不挂引用。"""

    return KnowledgePathResult(
        answer=(
            "我现在没有找到足够可靠的卡券规则依据，不能直接给出结论。"
            "请你补充卡券类型、购买时间或订单状态；如果涉及退款争议，我可以帮你转人工继续核验。"
        ),
        answer_path="clarify_or_handoff",
        citations=[],
    )


def general_chat_result() -> KnowledgePathResult:
    """普通寒暄不需要 RAG 依据，也不应挂引用。"""

    return KnowledgePathResult(
        answer=(
            "你好，我是网约车司机端卡券客服 Agent。你可以直接描述卡券购买、使用、退款或投诉问题；"
            "涉及具体规则时，我会优先依据当前知识库回答，找不到可靠依据时不会编造。"
        ),
        answer_path="general_chat_without_citations",
        citations=[],
    )
