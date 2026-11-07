"""L19 司机域：工具澄清与规划层。

本层全部是确定性代码，不调模型：
  - 校验 LLM 产出的 ClarificationPlan 草案（工具名白名单、参数正则）
  - 按 ToolSpec 重新计算缺失必填字段（不信模型自报的 missing_required）
  - 缺参数时生成 ClarificationRequest，让用户补参而不是硬调工具
  - 参数齐了才转成 ToolAction 交给 execute_tool_action

司机域只有 3 个工具带必填参数：
  get_activity_progress / get_activity_qualification → activity_id
  get_verification_record → coupon_id
其余 4 个工具无必填参数，永远不需要澄清。
"""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    ChatRequest,
    ClarificationCandidate,
    ClarificationPlan,
    ClarificationRequest,
    Intent,
    ToolAction,
    ToolObservation,
    ToolSpec,
)
from tools.contracts import TOOL_SPECS


_INTENTS: set[str] = {
    "general_chat", "promotion_consult", "product_consult",
    "order_query", "refund_request", "complaint", "unknown",
}

# 缺参时按字段给的不同追问话术
_FIELD_QUESTIONS: dict[str, str] = {
    "activity_id": "你问的是哪个活动？请补充活动编号（例如 ACT-12345），我才能查到你在这个活动里的实时进度。",
    "coupon_id": "你问的是哪张卡券？请补充卡券号（例如 CP-001），我才能查到这张券的核销记录。",
}


def normalize_intent(value: Any) -> Intent:
    """把模型输出的 intent 收敛到本项目的 Intent 字面量，非法值一律 unknown。"""
    return value if value in _INTENTS else "unknown"


def normalize_tool_name(value: Any, tool_specs: dict[str, ToolSpec]) -> str | None:
    """工具名必须在 TOOL_SPECS 里，模型编出来的工具名直接丢弃。"""
    if value in (None, "", "null", "none", "None"):
        return None
    text = str(value).strip()
    return text if text in tool_specs else None


def normalize_activity_id(value: Any) -> str | None:
    """校验活动编号：ACT-12345 / ACT12345 / 活动12345 三种写法都收。"""
    if not value:
        return None
    text = str(value).strip().upper()
    return text if re.fullmatch(r"(?:ACT[_-]?\d{3,}|活动\d{3,})", text) else None


def normalize_coupon_id(value: Any) -> str | None:
    """校验卡券号：CP-001 这类字母数字加连字符的编号。"""
    if not value:
        return None
    text = str(value).strip().upper()
    return text if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,}", text) else None


def normalize_order_id(value: Any) -> str | None:
    """校验订单号：至少 5 位字母数字，避免把「订单」两个字当编号。"""
    if not value:
        return None
    text = str(value).strip().upper()
    return text if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{4,}", text) else None


def normalize_time_range(value: Any) -> str | None:
    """把时间范围说法收敛成 今日/本周/本月 三档。"""
    if not value:
        return None
    text = str(value).strip()
    mapping = {"今天": "今日", "今日": "今日", "本周": "本周", "这周": "本周", "本月": "本月", "这个月": "本月"}
    return mapping.get(text)


def normalize_coupon_status(value: Any) -> str | None:
    """卡券状态只接受 unused / verified / expired。"""
    if not value:
        return None
    text = str(value).strip().lower()
    return text if text in {"unused", "verified", "expired"} else None


def normalize_known_arguments(
    tool_name: str | None,
    raw_arguments: Any,
    tool_specs: dict[str, ToolSpec],
) -> dict[str, Any]:
    """按 ToolSpec 白名单过滤模型给出的参数，并逐个做格式校验。

    模型即使输出 user_id / driver_id 这类越界字段，也会因为不在
    parameters_schema 里被直接丢掉。
    """
    if not tool_name or tool_name not in tool_specs or not isinstance(raw_arguments, dict):
        return {}

    allowed_fields = set(tool_specs[tool_name].parameters_schema)
    arguments: dict[str, Any] = {}

    validators = {
        "activity_id": normalize_activity_id,
        "coupon_id": normalize_coupon_id,
        "order_id": normalize_order_id,
        "time_range": normalize_time_range,
        "status": normalize_coupon_status,
    }
    for field, validator in validators.items():
        if field in allowed_fields and raw_arguments.get(field) is not None:
            normalized = validator(raw_arguments[field])
            if normalized:
                arguments[field] = normalized

    return arguments


def find_missing_required_fields(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """按 ToolSpec 的 required 计算缺失字段——后端唯一口径。"""
    return [field for field in spec.required if not arguments.get(field)]


def build_validated_clarification_plan(
    payload: dict[str, Any],
    tool_specs: dict[str, ToolSpec],
    model_name: str | None,
) -> ClarificationPlan:
    """把模型草案过一遍后端校验，生成可信的 ClarificationPlan。

    关键：missing_required 一律用 ToolSpec 重算，不用模型自报的值。
    """
    tool_name = normalize_tool_name(payload.get("tool_name"), tool_specs)
    arguments = normalize_known_arguments(tool_name, payload.get("known_arguments"), tool_specs)
    missing_required = (
        find_missing_required_fields(tool_specs[tool_name], arguments) if tool_name else []
    )

    return ClarificationPlan(
        intent=normalize_intent(payload.get("intent")),
        tool_name=tool_name,
        known_arguments=arguments,
        missing_required=missing_required,
        clarification_question=str(payload.get("clarification_question") or "").strip() or None,
        confidence=float(payload.get("confidence") or 0.7),
        reason=str(payload.get("reason") or "LLM 规划本轮工具与澄清字段。"),
        model_name=model_name,
    )


def _current_user_activities(request: ChatRequest) -> list[dict[str, Any]]:
    """读取前端随请求带入、已由业务后端校验过的当前司机活动上下文。"""
    context = request.runtime_context if isinstance(request.runtime_context, dict) else {}
    activities = context.get("currentUserActivities")
    if not isinstance(activities, list):
        return []
    return [item for item in activities if isinstance(item, dict)]


def activity_candidates(request: ChatRequest) -> list[ClarificationCandidate]:
    """把当前司机参加的活动转成澄清候选项（没有上下文时返回空列表）。"""
    candidates: list[ClarificationCandidate] = []
    for activity in _current_user_activities(request):
        activity_id = str(activity.get("activityId") or activity.get("activity_id") or "").strip()
        if not activity_id:
            continue
        name = str(activity.get("activityName") or activity.get("name") or activity_id).strip()
        status = str(activity.get("status") or "").strip()
        candidates.append(
            ClarificationCandidate(
                value=activity_id,
                label=f"{activity_id}｜{name}",
                hint=status or "当前司机已报名活动",
            )
        )
    return candidates


def _current_user_coupons(request: ChatRequest) -> list[dict[str, Any]]:
    """读取前端随请求带入、已由业务后端校验过的当前司机卡券上下文。"""
    context = request.runtime_context if isinstance(request.runtime_context, dict) else {}
    coupons = context.get("currentUserCoupons")
    if not isinstance(coupons, list):
        return []
    return [item for item in coupons if isinstance(item, dict)]


def coupon_candidates_from_coupons(coupons: list[dict[str, Any]]) -> list[ClarificationCandidate]:
    """把卡券字典列表转成澄清候选项（页面上下文和工具结果共用）。"""
    candidates: list[ClarificationCandidate] = []
    for coupon in coupons:
        coupon_id = str(coupon.get("couponId") or coupon.get("coupon_id") or "").strip()
        if not coupon_id:
            continue
        name = str(coupon.get("name") or coupon.get("couponName") or coupon_id).strip()
        status = str(coupon.get("statusText") or coupon.get("status") or "").strip()
        valid_to = str(coupon.get("validTo") or "").strip()
        hint = f"{status}，有效期至 {valid_to}" if valid_to else (status or "当前司机名下卡券")
        candidates.append(
            ClarificationCandidate(
                value=coupon_id,
                label=f"{coupon_id}｜{name}",
                hint=hint,
            )
        )
    return candidates


def coupon_candidates(request: ChatRequest) -> list[ClarificationCandidate]:
    """把当前司机名下的卡券（页面上下文）转成澄清候选项。"""
    return coupon_candidates_from_coupons(_current_user_coupons(request))


def enrich_plan_with_runtime_context(request: ChatRequest, plan: ClarificationPlan) -> ClarificationPlan:
    """用页面带入的活动上下文补全 activity_id，补完重算缺失字段。

    场景：司机在活动详情页点开客服，页面已经知道是哪个活动，不必再问。
    """
    if plan.tool_name not in {"get_activity_progress", "get_activity_qualification"}:
        return plan
    if plan.known_arguments.get("activity_id"):
        return plan

    context = request.runtime_context if isinstance(request.runtime_context, dict) else {}
    contextual_id = str(
        context.get("currentActivityId") or context.get("relatedActivityId") or ""
    ).strip()
    if not contextual_id:
        activities = _current_user_activities(request)
        if len(activities) == 1:
            contextual_id = str(
                activities[0].get("activityId") or activities[0].get("activity_id") or ""
            ).strip()

    activity_id = normalize_activity_id(contextual_id)
    if not activity_id:
        return plan

    known_arguments = {**plan.known_arguments, "activity_id": activity_id}
    return plan.model_copy(
        update={
            "known_arguments": known_arguments,
            "missing_required": find_missing_required_fields(
                TOOL_SPECS[plan.tool_name], known_arguments
            ),
            "reason": f"{plan.reason} 已用页面带入的活动上下文补全 activity_id。",
        }
    )


def pre_tool_clarification(
    request: ChatRequest,
    plan: ClarificationPlan,
) -> ClarificationRequest | None:
    """工具调用前澄清：缺必填参数就先反问，本轮不产生 ToolAction。

    返回 None 表示参数齐全，可以进入 plan_tool_action。
    """
    if not plan.tool_name or plan.tool_name not in TOOL_SPECS:
        return None

    spec = TOOL_SPECS[plan.tool_name]
    # 关键：即使模型已经给出 missing_required，后端仍按 ToolSpec 重算。
    missing = find_missing_required_fields(spec, plan.known_arguments)
    if not missing:
        return None

    clarification_field = missing[0]
    message = plan.clarification_question or _FIELD_QUESTIONS.get(
        clarification_field,
        f"这个查询还缺 {clarification_field}，请补充后我再查实时数据。",
    )
    if clarification_field == "activity_id":
        candidates = activity_candidates(request)
    elif clarification_field == "coupon_id":
        candidates = coupon_candidates(request)
    else:
        candidates = []
    return ClarificationRequest(
        clarification_field=clarification_field,
        message=message,
        candidates=candidates,
    )


def plan_tool_action(plan: ClarificationPlan) -> ToolAction | None:
    """把校验后的计划转成 ToolAction；还有缺参就返回 None。"""
    if not plan.tool_name or plan.tool_name not in TOOL_SPECS:
        return None
    spec = TOOL_SPECS[plan.tool_name]
    if find_missing_required_fields(spec, plan.known_arguments):
        return None
    return ToolAction(tool_name=plan.tool_name, arguments=plan.known_arguments, reason=plan.reason)


# 工具后澄清白名单（照搬课程 search_current_user_orders 的设计）：
# 只有「列候选」型工具查完才允许让用户再确认目标，其他工具查完直接出答案。
_POST_TOOL_CLARIFICATION_TOOLS: set[str] = {"get_coupon_status"}


def post_tool_clarification(
    action: ToolAction | None,
    observation: ToolObservation | None,
) -> ClarificationRequest | None:
    """工具调用后澄清：列候选型工具返回多个候选时让用户确认，模型不许代选。

    get_coupon_status 无必填参数，查回的是司机全部卡券；多张时结果本身
    就是候选列表，让用户指定要查/要操作的是哪一张。唯一结果直接放行。
    """
    if not action or not observation:
        return None
    if action.tool_name not in _POST_TOOL_CLARIFICATION_TOOLS:
        return None
    if observation.status != "success":
        return None

    coupons = observation.data.get("coupons")
    if not isinstance(coupons, list):
        return None
    candidates = coupon_candidates_from_coupons(coupons)
    if len(candidates) <= 1:
        return None

    return ClarificationRequest(
        clarification_field="coupon_id",
        message="你名下有多张卡券，请告诉我要查（或要操作）的是哪一张。",
        candidates=candidates,
    )
