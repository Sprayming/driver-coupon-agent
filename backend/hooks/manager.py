"""L23：Hooks 治理层。工具前、工具后、错误和完成事件在这里统一记录。

diver 司机域适配：
- redact 额外覆盖 driver_xxx / 银行卡号 等司机敏感字段
- 用 diver 的 ToolSpec / Observation / ChatRequest 类型
- 不改变治理职责：只管卡口、脱敏、降级信号，不做审批
"""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    HookEvent, HookCompletion, HookType, RiskLevel,
    ToolAction, ToolSpec, ChatRequest, Observation,
    ErrorCategory, NextAction, DegradationState,
)


class HookManager:
    """工具生命周期治理点。

    四个卡口：
    - pre_tool_call: 工具执行前统一校验参数、身份、脱敏摘要
    - post_tool_call: 工具执行后统一脱敏 Observation、检测提示注入
    - on_error: 把异常归一成可解释的降级事件
    - on_completion: 一轮结束生成公开治理摘要

    不做：批准退款、启动人工审批、保存完整 Trace。
    """

    # 脱敏正则（通用 + 司机域专属）
    _PHONE = re.compile(r"\b1[3-9]\d{9}\b")
    _EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
    _SECRET = re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[\w.\-]{6,}")
    _BANK_CARD = re.compile(r"\b622\d{13,16}\b")                        # 银行卡号
    _DRIVER_ID = re.compile(r"\bdriver[_-]?\w{3,}\b", re.IGNORECASE)   # 司机 ID
    _ORDER_ID = re.compile(r"\bO[-_]?\d{4,}\b")                        # 订单号（脱敏为 O-xxxx）
    _COUPON_ID = re.compile(r"\bCP[-_]?\d{6,}\b")                      # 卡券号

    # 提示注入标记
    _POLLUTION_MARKERS = (
        "忽略规则", "跳过审批", "直接退款",
        "ignore previous", "override system", "disregard all",
    )

    def __init__(self) -> None:
        self.events: list[HookEvent] = []
        self.touched_tools: set[str] = set()
        self.redacted_count = 0
        self.degraded_count = 0
        self.risk_hit_count = 0
        self.tool_count = 0

    # ── 卡口 1：工具前 ──

    def pre_tool_call(self, action: ToolAction, request: ChatRequest, spec: ToolSpec) -> HookEvent:
        """工具执行前统一校验必填参数 + 写动作拦截 + 脱敏摘要。

        L24：读工具定义自带的 read_only / risk_level / boundary，
        而不是在 Hooks 里硬编码某个工具名。
        """
        self.tool_count += 1
        self.touched_tools.add(action.tool_name)
        missing = [field for field in spec.required if not action.arguments.get(field)]
        # L24：写工具一律在轻路径拦截，必须走受控工作流。
        is_write_action = not spec.read_only
        safe_args, redacted_args = self.redact(action.arguments)
        safe_msg_preview, redacted_msg = self.redact(request.user_message[:240])
        redacted = redacted_args or redacted_msg
        if redacted:
            self.redacted_count += 1
        # L24：从工具定义读 resource_uris，把边界文档内容写进事件（供审计和调试台展示）。
        resource_contents = []
        for uri in spec.resource_uris:
            try:
                from tools.catalog import DRIVER_TOOL_CATALOG
                resource = DRIVER_TOOL_CATALOG.read_resource(uri)
                resource_contents.append({"uri": uri, "title": resource.title, "content": resource.content})
            except KeyError:
                resource_contents.append({"uri": uri, "title": "未找到", "content": ""})

        blocked = bool(missing) or is_write_action
        event = HookEvent(
            hook_type="pre_tool_call",
            target_name=action.tool_name,
            action="validate_arguments_and_capability_boundary",
            result="blocked" if blocked else "allowed",
            reason="工具调用前统一校验必填参数、只读边界和风险等级，并只写入可信运行时身份和脱敏参数摘要。",
            safe_summary={
                "tool": action.tool_name,
                "required": spec.required,
                "missing": missing,
                "arguments": safe_args,
                "read_only": spec.read_only,
                "risk_level": spec.risk_level,
                "boundary": spec.boundary,
                "resources": resource_contents,
                "runtime_user_id": request.runtime_user_id,
                "runtime_role": request.runtime_role,
                "message_preview": safe_msg_preview,
            },
            redacted=redacted,
        )
        self.events.append(event)
        return event

    # ── 卡口 2：工具后 ──

    def post_tool_call(self, observation: Observation) -> Observation:
        """工具执行后统一脱敏 Observation + 检测提示注入。"""
        safe_summary, redacted_summary = self.redact(observation.summary)
        safe_facts, redacted_facts = self.redact(observation.facts)
        pollution = self._detect_pollution(str(safe_summary))
        if pollution:
            safe_summary = self._neutralize_pollution(str(safe_summary))
        redacted = redacted_summary or redacted_facts
        if redacted:
            self.redacted_count += 1
        event = HookEvent(
            hook_type="post_tool_call",
            target_name=observation.tool_name,
            action="sanitize_observation",
            result="sanitized" if redacted or pollution else "passed",
            reason="工具结果回到 Agent 前统一脱敏 + 提示注入检测。",
            safe_summary={
                "tool": observation.tool_name,
                "status": observation.status,
                "observation_preview": str(safe_summary)[:180],
                "fact_keys": sorted(safe_facts) if isinstance(safe_facts, dict) else [],
                "omitted_fields": observation.omitted_fields,
                "pollution_detected": pollution,
            },
            redacted=redacted,
        )
        self.events.append(event)
        return observation.model_copy(update={"summary": str(safe_summary), "facts": safe_facts})

    # ── 卡口 3：错误 ──

    def on_error(self, tool_name: str, error_category: ErrorCategory, message: str, attempts: int = 1) -> HookEvent:
        """把异常归一成可解释的降级事件。"""
        self.degraded_count += 1
        safe_msg, redacted = self.redact(message)
        if redacted:
            self.redacted_count += 1
        event = HookEvent(
            hook_type="on_error",
            target_name=tool_name,
            action="normalize_error_for_degradation",
            result="degraded",
            reason="工具或模型异常被归一成公开可读的降级原因，不让客服链路跟着编事实。",
            safe_summary={"error_category": error_category, "attempts": attempts, "message": safe_msg},
            redacted=redacted,
            degraded=True,
        )
        self.events.append(event)
        return event

    # ── 卡口 4：完成 ──

    def on_completion(self, *, next_action: NextAction, risk_level: RiskLevel, degradation: DegradationState) -> HookCompletion:
        """一轮结束生成公开治理摘要，不暴露隐藏推理链。"""
        if risk_level == "high":
            self.risk_hit_count += 1
        safe_summary = {
            "next_action": next_action,
            "risk_level": risk_level,
            "degraded": degradation.degraded,
            "error_category": degradation.error_category,
            "touched_tools": sorted(self.touched_tools),
        }
        event = HookEvent(
            hook_type="on_completion",
            target_name="chat_request",
            action="summarize_tool_governance",
            result="completed",
            reason="本轮请求结束时输出公开治理摘要，不输出隐藏推理链。",
            safe_summary=safe_summary,
            degraded=degradation.degraded,
        )
        self.events.append(event)
        return HookCompletion(
            hook_count=len(self.events),
            tool_count=self.tool_count,
            touched_tools=sorted(self.touched_tools),
            redacted_count=self.redacted_count,
            degraded_count=self.degraded_count,
            risk_hit_count=self.risk_hit_count,
            safe_summary=safe_summary,
        )

    # ── 脱敏 ──

    def redact(self, value: Any) -> tuple[Any, bool]:
        """对外输出前统一脱敏，保护司机身份、银行卡、内部编号。"""
        if isinstance(value, dict):
            redacted = False
            safe: dict[str, Any] = {}
            for key, item in value.items():
                safe_item, item_redacted = self.redact(item)
                safe[key] = safe_item
                redacted = redacted or item_redacted
            return safe, redacted
        if isinstance(value, list):
            redacted = False
            safe_items = []
            for item in value:
                safe_item, item_redacted = self.redact(item)
                safe_items.append(safe_item)
                redacted = redacted or item_redacted
            return safe_items, redacted
        if not isinstance(value, str):
            return value, False
        safe = self._PHONE.sub("[phone-redacted]", value)
        safe = self._EMAIL.sub("[email-redacted]", safe)
        safe = self._SECRET.sub(lambda m: f"{m.group(1)}=[secret-redacted]", safe)
        safe = self._BANK_CARD.sub("[bank-redacted]", safe)
        safe = self._DRIVER_ID.sub("[driver-id-redacted]", safe)
        safe = self._ORDER_ID.sub("[order-redacted]", safe)
        safe = self._COUPON_ID.sub("[coupon-redacted]", safe)
        return safe, safe != value

    # ── 提示注入 ──

    def _detect_pollution(self, text: str) -> bool:
        lowered = text.lower()
        return any(m.lower() in lowered for m in self._POLLUTION_MARKERS)

    def _neutralize_pollution(self, text: str) -> str:
        cleaned = text
        for marker in self._POLLUTION_MARKERS:
            cleaned = re.sub(re.escape(marker), "[external-instruction-neutralized]", cleaned, flags=re.IGNORECASE)
        return cleaned
