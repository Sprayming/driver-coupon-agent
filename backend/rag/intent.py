"""L13 粗意图分类。

从 customer_service_agent 抽出来是因为质量评测（rag/quality.py）也要走
「意图 -> 查询改写 -> 检索」同一条链路；放在 agent 里会造成循环导入。

L13 变化：意图在检索之前计算（改写依赖意图），不再用检索 hits 兜底，
所以 hits 参数变成可选的调试信息，默认不参与分类。
"""

from __future__ import annotations

from api.schemas import Intent, IntentResult, KnowledgeHit
from safety.source_guard import inspect_source

# 规则表同时被 classify_intent（首命中）和 candidate_intents（冲突检测）使用，
# 避免两份关键词表漂移。
INTENT_RULES: list[tuple[Intent, list[str], str]] = [
    (
        "complaint",
        [
            "投诉", "举报", "赔偿", "曝光", "315", "别踢皮球", "踢皮球",
            # 规则挖掘发现的短板：complaint → unknown 是全库最高频错判（4 次），
            # 真实抱怨表达多是不带「投诉」二字的，这里补齐。
            "没人管", "没人处理", "推来推去", "态度差", "服务差",
            "差评", "太坑", "坑人", "要个说法", "给个说法", "退不了",
            "太差", "不满意", "什么破", "破系统",
        ],
        "用户表达了投诉或强烈不满。",
    ),
    ("refund_request", ["退款", "退卡", "取消卡券", "无法使用", "临时有事"], "用户在询问退款、退货或质量问题。"),
    ("order_query", ["卡券", "核销"], "用户在询问卡券核销状态。"),
    ("promotion_consult", ["优惠券", "活动", "打折", "券"], "用户在询问卡券优惠或活动。"),
    ("product_consult", ["日卡", "夜卡", "月卡", "时段卡", "推荐", "哪个好"], "用户在询问商品或推荐。"),
    ("general_chat", ["你好", "您好", "在吗", "谢谢"], "用户只是普通问候。"),
]


# L0 精确短语层：高频、确定性的问法，命中即返回。
# 不进升级判定，也不调模型——这类问法规则本就该 100% 判对，没有模糊空间。
EXACT_INTENT_PHRASES: list[tuple[Intent, list[str], str]] = [
    ("complaint", ["转人工", "人工客服", "人工服务", "投诉电话"], "用户明确要求人工介入。"),
]

# L41 对齐：司机询问自身身份/等级/权益的表达——答案只来自 Runtime Context，
# 不能走业务工具或知识检索。先排除退款/卡券等明确业务动作，
# 否则「我的卡券能退吗」会被「我的卡券」误判成身份问题。
RUNTIME_IDENTITY_KEYWORDS: list[str] = [
    "我是谁", "我的身份", "我的等级", "我几级", "我是什么等级",
    "我的账号", "我的用户", "我的权益", "我的资料", "我的信息",
]
RUNTIME_IDENTITY_EXCLUSIONS: list[str] = [
    "退款", "退货", "退券", "退卡", "申请", "卡券", "订单", "核销",
]

# L41 对齐（路由侧）：司机主动索要受保护信息，也归到 security_request。
# 这类问法本身不一定命中注入正则，所以单独列一份词表（与 L41 route_intent 同款口径）。
SECURITY_REQUEST_TERMS: list[str] = [
    "系统提示词", "隐藏推理", "hidden reasoning", "工具 schema", "工具schema", "内部策略",
]


def first_matched_keywords(message: str, keywords: list[str]) -> list[str]:
    """从消息中提取所有匹配的关键词。"""
    return [keyword for keyword in keywords if keyword in message]


def candidate_intents(user_message: str) -> list[str]:
    """返回文本命中关键词的全部意图（用于识别跨意图关键词冲突）。"""

    message = user_message.strip().lower()
    return [
        intent
        for intent, keywords, _ in INTENT_RULES
        if first_matched_keywords(message, keywords)
    ]


def classify_intent(user_message: str, hits: list[KnowledgeHit] | None = None) -> IntentResult:
    """沿用粗意图规则，为查询改写和回答分支提供分类信号。"""

    message = user_message.strip().lower()
    # L41 对齐（路由侧）：安全意图优先于一切业务意图。
    # 命中注入特征，或主动索要系统提示词/隐藏推理/工具 schema/内部策略，都走 security_request。
    input_safe, input_reason = inspect_source(user_message)
    if not input_safe:
        return IntentResult(
            intent="security_request",
            matched_keywords=[],
            explanation=f"输入命中 Prompt Injection 特征（{input_reason}），由响应侧统一拒答。",
        )
    security_matched = first_matched_keywords(message, SECURITY_REQUEST_TERMS)
    if security_matched:
        return IntentResult(
            intent="security_request",
            matched_keywords=security_matched,
            explanation="用户在索要受保护的系统信息，由响应侧统一拒答。",
        )
    # L0：精确短语优先，命中即返回，不参与后续规则和升级判定。
    for intent, phrases, explanation in EXACT_INTENT_PHRASES:
        matched = first_matched_keywords(message, phrases)
        if matched:
            return IntentResult(intent=intent, matched_keywords=matched, explanation=explanation)
    # L41 对齐：身份直答优先于普通业务意图，但必须先排除明确的退款/卡券动作。
    if not any(keyword in message for keyword in RUNTIME_IDENTITY_EXCLUSIONS):
        identity_matched = first_matched_keywords(message, RUNTIME_IDENTITY_KEYWORDS)
        if identity_matched:
            return IntentResult(
                intent="runtime_identity",
                matched_keywords=identity_matched,
                explanation="司机在询问自身身份、等级或权益，直接读 Runtime Context 作答。",
            )
    for intent, keywords, explanation in INTENT_RULES:
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
