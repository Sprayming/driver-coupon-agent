"""L13 查询改写。

纪律（课程反复强调）：
- original_query 保留用户原话，用于对话、Prompt 展示和审计；
- rewritten_query 只喂给向量检索，不直接展示给用户；
- 归一化解决「口语 vs 知识库措辞」不一致；
- 按意图补词解决「指代词缺上下文、旧规则抢第一名」。
"""

from __future__ import annotations

from api.schemas import Intent, QueryRewrite
from config.settings import NORMALIZATION_RULES


def normalize_query(text: str) -> str:
    """把用户口语按规则表归一到知识库更常见的检索词。"""
    normalized = text.lower()
    for source, target, _reason in NORMALIZATION_RULES:
        normalized = normalized.replace(source, target)
    return normalized


def describe_normalization(original_query: str) -> list[str]:
    """返回本轮归一化实际命中的规则原因（给调试台看）。"""
    query = original_query.lower()
    return [reason for source, _target, reason in NORMALIZATION_RULES if source in query]


def add_rewrite_terms(target: list[str], terms: list[str]) -> None:
    """追加不重复的检索补词。"""
    for term in terms:
        if term not in target:
            target.append(term)


def build_rewrite_reason(normalization_reasons: list[str], rewrite_reasons: list[str]) -> str:
    """把归一化原因和补词原因合并成调试台可读的一句话。"""
    reasons = [*normalization_reasons, *rewrite_reasons]
    if reasons:
        return "；".join(reasons) + "。"
    return "未命中归一化或补词规则，保留原问题直接检索。"


# 按意图补词：词全部对齐知识库 markdown 注释里的 keywords 和章节标题，
# 让改写后向量更靠近该意图下的现行规则 chunk。
INTENT_REWRITE_TERMS: dict[Intent, tuple[list[str], str]] = {
    "refund_request": (
        ["退款申请条件", "不支持退款情形", "退款申请流程", "退款到账时效", "已核销"],
        "退款意图补齐退款条件、不支持情形、申请流程、到账时效和已核销边界等规则检索词",
    ),
    "order_query": (
        ["卡券核销", "激活规则", "自动核销", "使用时段"],
        "核销意图补齐卡券核销与激活规则检索词",
    ),
    "promotion_consult": (
        ["当前有效", "现行规则"],
        "活动咨询补“当前有效、现行规则”，避免已下线的 2025 历史活动卡券抢到第一名",
    ),
    "complaint": (
        ["投诉受理", "投诉升级", "转人工"],
        "投诉意图补齐投诉受理与升级渠道检索词",
    ),
}


def rewrite_retrieval_query(user_message: str, intent: Intent) -> QueryRewrite:
    """根据归一化文本和粗意图，生成只用于检索的改写问题。"""
    normalized = normalize_query(user_message)
    added_terms: list[str] = []
    rewrite_reasons: list[str] = []

    terms_rule = INTENT_REWRITE_TERMS.get(intent)
    if terms_rule is not None:
        terms, reason = terms_rule
        add_rewrite_terms(added_terms, terms)
        rewrite_reasons.append(reason)

    rewritten_query = " ".join(part for part in [normalized, *added_terms] if part).strip() or user_message
    return QueryRewrite(
        original_query=user_message,
        rewritten_query=rewritten_query,
        applied=rewritten_query != user_message,
        added_terms=added_terms,
        reason=build_rewrite_reason(describe_normalization(user_message), rewrite_reasons),
    )
