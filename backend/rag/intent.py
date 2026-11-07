"""L13 粗意图分类。

从 customer_service_agent 抽出来是因为质量评测（rag/quality.py）也要走
「意图 -> 查询改写 -> 检索」同一条链路；放在 agent 里会造成循环导入。

L13 变化：意图在检索之前计算（改写依赖意图），不再用检索 hits 兜底，
所以 hits 参数变成可选的调试信息，默认不参与分类。
"""

from __future__ import annotations

from api.schemas import Intent, IntentResult, KnowledgeHit


def first_matched_keywords(message: str, keywords: list[str]) -> list[str]:
    """从消息中提取所有匹配的关键词。"""
    return [keyword for keyword in keywords if keyword in message]


def classify_intent(user_message: str, hits: list[KnowledgeHit] | None = None) -> IntentResult:
    """沿用粗意图规则，为查询改写和回答分支提供分类信号。"""
    message = user_message.strip().lower()
    intent_rules: list[tuple[Intent, list[str], str]] = [
        ("complaint", ["投诉", "举报", "赔偿", "曝光", "315", "别踢皮球"], "用户表达了投诉或强烈不满。"),
        ("refund_request", ["退款", "退卡", "取消卡券", "无法使用", "临时有事"], "用户在询问退款、退货或质量问题。"),
        ("order_query", ["卡券", "核销"], "用户在询问卡券核销状态。"),
        ("promotion_consult", ["优惠券", "活动", "打折", "券"], "用户在询问卡券优惠或活动。"),
        ("product_consult", ["日卡", "夜卡", "月卡", "时段卡", "推荐", "哪个好"], "用户在询问商品或推荐。"),
        ("general_chat", ["你好", "您好", "在吗", "谢谢"], "用户只是普通问候。"),
    ]
    for intent, keywords, explanation in intent_rules:
        matched = first_matched_keywords(message, keywords)
        if matched:
            return IntentResult(intent=intent, matched_keywords=matched, explanation=explanation)

    # 关键词一条都没命中（文本意图不明显）+ 但向量检索有 hits（知识命中）→ 兜底判为产品咨询（hits 仅作调试兜底，主链路不传）。
    # if hits:
    #     return IntentResult(
    #         intent="product_consult",
    #         matched_keywords=[],
    #         explanation="知识命中但文本意图不明显，默认归为产品咨询。",
    #     )
    return IntentResult(intent="unknown", matched_keywords=[], explanation="没有命中当前版本的粗意图规则。")
