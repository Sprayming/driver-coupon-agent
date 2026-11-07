"""L15 检索前计划（pre-retrieval plan）。

职责边界（和 L13 两个模块的分工）：
- rag/intent.py 回答「用户想干什么」——粗意图分类，本文件直接复用其结果；
- rag/query_rewrite.py 回答「检索词怎么写」——归一化 + 按意图补词，rewritten_query 直接取自它；
- 本文件回答「该去哪些主题里找、关键词通道带什么词」——场景路由 + 主题范围 + 关键词补词。

三个信号都只影响检索，不改用户原话，也不参与回答文案。
"""

from __future__ import annotations

from api.schemas import Intent, IntentResult, KnowledgeChunk, QueryRewrite, RetrievalPlan, RetrievalScene
from rag.query_rewrite import normalize_query


# 场景 -> 允许检索的知识主题。项目 chunk.topic 取值就是 intent 同款
# （refund_request / product_consult / complaint），映射非常直接；
# unknown 保留全主题，场景识别失败时退化为 L14 的全库召回行为。
ALLOWED_TOPICS_BY_SCENE: dict[RetrievalScene, list[str]] = {
    "refund": ["refund_request"],
    "usage": ["product_consult"],
    # 活动相关规则分布在两处：当前档位价格（product_consult）和已下线的历史活动
    # （coupon-old-promotion 标的是 refund_request topic），promotion 场景要同时放行两个主题。
    "promotion": ["product_consult", "refund_request"],
    "complaint": ["complaint"],
    "unknown": ["refund_request", "product_consult", "complaint"],
}

# intent -> 检索场景的基础映射。
SCENE_BY_INTENT: dict[Intent, RetrievalScene] = {
    "refund_request": "refund",
    "order_query": "usage",
    "product_consult": "usage",
    "promotion_consult": "promotion",
    "complaint": "complaint",
    "general_chat": "unknown",
    "unknown": "unknown",
}

# 各场景强信号词：query 归一化后命中时，以词为准修正 intent 映射的场景。
# 需要这层修正是因为 intent 规则表按顺序首命中即返回，可能漏掉句子里靠后的强场景词。
# dict 遍历顺序即优先级：投诉 > 退款 > 活动 > 使用（和 intent 规则表里投诉置顶一致）。
# 注意「到账」这种退款/使用双关词不能当强词，会互相误伤。
SCENE_STRONG_TERMS: dict[RetrievalScene, list[str]] = {
    "complaint": ["投诉", "举报", "曝光", "12315", "315"],
    "refund": ["退款", "退卡", "能退", "退钱"],
    "promotion": ["活动", "折扣", "优惠", "2025", "双11"],
    "usage": ["核销", "激活", "日卡", "夜卡", "月卡", "时段", "未到账", "没到账", "怎么买"],
}

# 关键词通道词表：全部对齐知识库 md 章节注释里的 keywords 和正文高频业务词。
# keyword_terms_for_scene 先收检索文本里已命中的词，命中本场景时该组词全补（和课程版同策略）。
SCENE_KEYWORD_TERMS: dict[RetrievalScene, list[str]] = {
    "refund": ["退款", "未使用", "已核销", "过期", "退款申请", "不支持退款", "到账时效", "原路退回", "申请退款", "有效期内"],
    "usage": ["日卡", "夜卡", "月卡", "时段卡", "核销", "激活", "有效期", "未到账", "购买", "钱包"],
    "promotion": ["当前", "活动", "折扣", "优惠", "2025", "双11", "历史", "价格", "赠券"],
    "complaint": ["投诉", "举报", "曝光", "12315", "差评", "转人工", "升级", "客服态度"],
}


def detect_scene(user_message: str, intent: Intent) -> RetrievalScene:
    """判断本轮检索场景：intent 映射打底，强场景词修正。"""

    text = normalize_query(user_message)
    for scene, terms in SCENE_STRONG_TERMS.items():
        if any(term in text for term in terms):
            return scene
    return SCENE_BY_INTENT.get(intent, "unknown")


def keyword_terms_for_scene(rewritten_query: str, scene: RetrievalScene) -> list[str]:
    """生成关键词通道用词：检索文本里已命中的业务词 + 场景核心补词。

    unknown 场景不补词（SCENE_KEYWORD_TERMS 没有 unknown 键），
    只靠问题里已命中的业务词，避免给无关问题硬造检索信号。
    """

    # rewritten_query 已经归一化过，再跑一遍 normalize 是幂等的，统一口径而已。
    text = normalize_query(rewritten_query)
    terms: list[str] = []
    for scene_name, scene_terms in SCENE_KEYWORD_TERMS.items():
        for term in scene_terms:
            # 和课程版同策略：词本身出现在检索文本里就带上；命中本场景时该组核心词全补。
            if term in text or scene_name == scene:
                if term not in terms:
                    terms.append(term)
    return terms


def build_plan_reason(scene: RetrievalScene, allowed_topics: list[str], keyword_terms: list[str]) -> str:
    """生成调试台可读的检索计划原因。"""

    scene_labels: dict[RetrievalScene, str] = {
        "refund": "退款",
        "usage": "使用/购买",
        "promotion": "活动",
        "complaint": "投诉",
        "unknown": "未知",
    }
    topic_text = "、".join(allowed_topics)
    if scene == "unknown":
        return f"未识别到稳定检索场景，保留全主题候选（{topic_text}），关键词通道只用问题里已命中的业务词。"
    return (
        f"识别为{scene_labels[scene]}检索场景，限定主题 {topic_text}，"
        f"关键词通道带 {len(keyword_terms)} 个业务词。"
    )


def pre_retrieval_plan(user_message: str, intent_result: IntentResult, rewrite: QueryRewrite) -> RetrievalPlan:
    """在真正检索前决定主题范围和关键词召回用词（复用 L13 的意图和改写结果）。"""

    scene = detect_scene(user_message, intent_result.intent)
    allowed_topics = ALLOWED_TOPICS_BY_SCENE[scene]
    keyword_terms = keyword_terms_for_scene(rewrite.rewritten_query, scene)
    return RetrievalPlan(
        original_query=user_message,
        rewritten_query=rewrite.rewritten_query,
        scene=scene,
        allowed_topics=allowed_topics,
        keyword_terms=keyword_terms,
        reason=build_plan_reason(scene, allowed_topics, keyword_terms),
    )


def topic_allowed(chunk: KnowledgeChunk, plan: RetrievalPlan) -> bool:
    """判断知识片段是否属于本轮允许主题。

    topic 为空的片段（md 章节漏标注释）保守放行：宁可让它进池被 reranker 处理，
    也不能因为标注缺失被场景过滤直接误杀。
    """

    if not chunk.topic:
        return True
    return chunk.topic in plan.allowed_topics


# L16：实时业务问题识别词表。全部是「查进度/查状态」句式，
# 刻意不收「多久到账」这类时效政策词——「退款多久能到账」是稳定知识（coupon-refund-timeline），
# 必须能走检索和缓存；「退款到账了吗」才是查个人工单进度，知识库答不了。
REALTIME_BUSINESS_TERMS = [
    "我的订单", "订单到哪", "物流到哪", "快递到哪", "发货了吗",
    "退款进度", "退款到账了吗", "到账了吗", "审核到哪", "处理进度",
    "我的退款", "券到账了吗", "钱包余额", "库存还有", "有没有库存",
]


def is_realtime_business_query(query: str) -> bool:
    """识别必须交给后续业务事实层的实时问题。

    第 16 课还不真正查询业务系统；这里只负责把实时订单、物流、退款进度、库存
    挡在 RAG 缓存和知识库回答之外，避免把个人当前状态缓存成稳定知识答案。
    """

    text = normalize_query(query)
    return any(term in text for term in REALTIME_BUSINESS_TERMS)
