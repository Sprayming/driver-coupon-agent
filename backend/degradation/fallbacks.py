"""L21：错误降级策略。工具超时、模型不可用和高风险动作都在这里给稳定兜底。

核心设计：
- 高风险操作（发起退款、申请补偿、修改订单）→ 直接拦截，转人工审批
- 工具超时 / 业务校验失败 → 返回安全模板口径，不编造业务事实
- 模型服务不可用 → 用工具链路拿到的 Observation 事实模板化回答
"""

from __future__ import annotations

from typing import Any


class ModelServiceError(Exception):
    """模型服务不可用时抛出。

    L21 的 compose_model_answer / Agent 编排层用 try/except 兜住：
    工具链路已完成核验的情况下，即使模型挂了也能用模板口径返回结果。
    """
    pass


def compose_model_answer(observation: Any, user_message: str) -> str:
    """执行 compose_model_answer 对应的课程逻辑。

    课程演示用故障注入：真实生产系统应通过模型客户端异常或健康状态判断服务不可用。
    """
    if "模型抽风" in user_message:
        raise ModelServiceError("模型服务暂时不可用。")
    if observation.status == "success":
        return f"我通过只读工具查到：{observation.summary}"
    return f"工具没有返回可用事实：{observation.summary}"


# ── 司机域降级模板 ──
# 对应 ErrorCategory 每个分类一个安全口径，不编造业务事实。
_DRIVER_FALLBACK_TEMPLATES: dict[str, str] = {
    "timeout": "运力后端查询暂时超时，我现在不能编造退款/卡券状态。你可以稍后重试，或转人工客服继续核验。",
    "validation_error": "当前请求缺少必要参数或格式不正确，请补充正确的订单号或卡券号后再试。",
    "not_found": "我没有查到对应退款记录或卡券状态，不能凭空补充业务事实。请核对订单号或卡券号后再发一次。",
    "forbidden": "当前账号无权查看这笔订单的退款记录，我不能透露其他司机的订单信息。",
    "business_error": "订单当前状态不满足退款/核销查询条件，我会按业务规则说明原因，必要时转人工继续处理。",
    "model_unavailable": "模型服务暂时不可用，但工具链路已完成核验。当前先用模板口径说明查询结果。",
    "system_error": "系统暂时异常，我不会编造结果。你可以稍后重试，或转人工客服继续核验。",
    "high_risk_write_blocked": "发起退款、申请补偿都属于高风险写操作，我不能自动执行，会建议转人工客服处理。",
}


def fallback_answer(category: str) -> str:
    """按错误分类返回安全兜底话术，不编造业务事实。"""
    return _DRIVER_FALLBACK_TEMPLATES.get(
        category,
        "系统暂时没有拿到可靠结果，我不会编造业务事实。",
    )


