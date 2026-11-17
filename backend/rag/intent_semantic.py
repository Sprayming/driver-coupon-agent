"""意图识别供应商：rule（纯规则）与 hybrid（规则优先 + 语义兜底）。

为什么需要这层：
- 规则零延迟、可解释，但遇到否定、口语改写、关键词跨意图冲突会误判；
- 全量走 LLM 准确率高，但每轮都多一次网络往返，还增加故障点；
- 折中方案：规则先判，只在「低置信」情形调一次语义分类，超时/失败自动回退规则。

默认 provider=rule，聊天链路行为与旧版完全一致；
把 AGENT_INTENT_PROVIDER 设为 hybrid 才会启用兜底。
"""

from __future__ import annotations

import json
import os
import re

import httpx

from api.schemas import IntentResult
from config import settings
from config.settings import api_key_is_missing
from models.llm_client import call_chat_model
from rag.intent import candidate_intents, classify_intent

INTENT_LABELS = (
    "refund_request",
    "order_query",
    "promotion_consult",
    "product_consult",
    "complaint",
    "general_chat",
    "unknown",
)

INTENT_CRITERIA: dict[str, str] = {
    "refund_request": "申请退款、退卡、取消购买，或询问退款进度与退款规则（不包含单纯查询订单/核销状态）",
    "order_query": "查询卡券或订单的状态、核销记录、到账情况（不包含要求发起退款）",
    "promotion_consult": "咨询优惠券、活动、打折、奖励、福利等营销信息",
    "product_consult": "咨询卡种、价格、时长、推荐、不同卡的区别",
    "complaint": "表达投诉、举报、索赔、强烈不满，或威胁曝光",
    "general_chat": "寒暄、问候、感谢、询问助手身份等无业务诉求的闲聊",
    "unknown": "与卡券退款业务无关，或信息不足无法判断",
}

SEMANTIC_PROMPT = (
    "你是卡券退款业务的意图分类器。只输出一个 JSON 对象，形如 {\"intent\": \"类别\"}，"
    "不要解释、不要多余文字。类别必须从下面选一个：\n"
    + "\n".join(f"- {label}: {desc}" for label, desc in INTENT_CRITERIA.items())
    + "\n补充规则：只有明确的问候、感谢、询问助手身份才算 general_chat；"
    "天气、写诗、股票、美食等与卡券业务无关的话题一律归 unknown。"
)

# 否定句：规则只看关键词，会把「不用退款了」判成退款。
NEGATION_PATTERN = re.compile(r"(不用|不想|不需要|别|不).{0,6}(退|退款|退卡)")


# 升级阈值：分数达到阈值才调语义模型。0.3 与改造前的「任一条件为真」等价，
# 后续可依据 evals 的跑分结果上调或下调，在准确率与延迟之间做权衡。
ESCALATE_THRESHOLD = 0.3


def escalation_score(user_message: str, rule_result: IntentResult) -> tuple[float, list[str]]:
    """给规则结果打「需要升级」的分，并返回命中理由。

    改造前是布尔判断，只能回答「升不升」；改成分数后能回答
    「为什么升」和「离阈值多远」，阈值也能按跑分结果调。

    分值对应实测失败类型的严重程度（见 evals/intent_cases.yml 的 tags）：
    - 完全没命中 0.6：最不可信
    - 否定表达 0.5：方向可能完全相反
    - 有「退」没判退款 0.4：大概率漏判
    - 多意图冲突 0.3：首命中可能选错
    """

    message = user_message.strip().lower()
    score = 0.0
    reasons: list[str] = []
    if rule_result.intent == "unknown":
        score += 0.6
        reasons.append("规则未命中")
    if NEGATION_PATTERN.search(message):
        score += 0.5
        reasons.append("否定表达")
    if "退" in message and rule_result.intent != "refund_request":
        score += 0.4
        reasons.append("有退字未判退款")
    if len(set(candidate_intents(message))) >= 2:
        score += 0.3
        reasons.append("多意图冲突")
    return min(score, 1.0), reasons


def should_escalate(user_message: str, rule_result: IntentResult) -> bool:
    """规则结果是否低置信到需要语义兜底。"""

    score, _ = escalation_score(user_message, rule_result)
    return score >= ESCALATE_THRESHOLD


def parse_intent_label(text: str) -> str | None:
    """解析模型输出；先按 JSON，再退化到类别名匹配。"""

    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            try:
                data = json.loads(text[start : end + 1])
                label = str(data.get("intent", "")).strip()
                if label in INTENT_LABELS:
                    return label
            except json.JSONDecodeError:
                pass
    for label in INTENT_LABELS:
        if label in text:
            return label
    return None


def semantic_classify(user_message: str) -> IntentResult | None:
    """调用一次模型做意图分类；任何异常都返回 None，由调用方回退规则。

    签名保持单参数：测试里用 lambda 替换本函数，新增关键字参数会破坏兼容。
    升级分的计算放在函数内部，纯内存开销可忽略。
    """

    api_key = os.getenv("AGENT_OPENAI_API_KEY", "")
    if api_key_is_missing(api_key):
        return None

    messages = [
        {"role": "system", "content": SEMANTIC_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        with httpx.Client(timeout=settings.INTENT_LLM_TIMEOUT_SECONDS) as client:
            result = call_chat_model(
                messages,
                http_client=client,
                model=os.getenv("AGENT_CLASSIFIER_MODEL") or None,
                temperature=0,
            )
    except Exception:  # noqa: BLE001 - 意图兜底失败必须回退，不能阻断聊天
        return None

    label = parse_intent_label(result.answer)
    if label is None:
        return None
    score, reasons = escalation_score(user_message, classify_intent(user_message))
    detail = f"（升级分={score:.1f}：{'、'.join(reasons)}）" if reasons else ""
    return IntentResult(
        intent=label,
        matched_keywords=[],
        explanation=f"规则低置信{detail}，语义分类兜底判定。",
    )


def classify_intent_with_provider(user_message: str) -> tuple[IntentResult, bool]:
    """按配置选择意图实现，返回 (意图结果, 是否走了语义兜底)。"""

    rule_result = classify_intent(user_message)
    if settings.INTENT_PROVIDER != "hybrid":
        return rule_result, False

    score, reasons = escalation_score(user_message, rule_result)
    if score < ESCALATE_THRESHOLD:
        return rule_result, False

    semantic_result = semantic_classify(user_message)
    if semantic_result is None:
        return rule_result, False
    return semantic_result, True
