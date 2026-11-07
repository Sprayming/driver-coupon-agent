"""L18 司机域：把工具契约包装成 LangChain StructuredTool。

LangChain 看到的是 StructuredTool（name/description/args schema），
每个工具内部仍回到后端 execute_tool_action 跑三道安全闸。
返回 JSON 是为了让调试台稳定解析 Observation，不是 LangChain 的强制写法。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from api.schemas import ChatRequest, ToolAction
from tools.contracts import TOOL_SPECS
from tools.tool_runtime import execute_tool_action


def tool_action_reason(tool_name: str) -> str:
    """给调试台展示本轮工具调用原因（教学文案，非生产必需）。"""
    reasons = {
        "get_activity_progress": "用户询问活动完成度，LangChain 选择活动进度工具。",
        "get_driver_order_count": "用户询问单量，LangChain 选择司机单量工具。",
        "get_activity_qualification": "用户询问活动资格，LangChain 选择资格判定工具。",
        "get_refund_status": "用户询问退款进度，LangChain 选择只读退款状态工具。",
        "get_coupon_status": "用户询问卡券状态，LangChain 选择卡券查询工具。",
        "get_verification_record": "用户询问核销记录，LangChain 选择核销查询工具。",
        "get_income_summary": "用户询问收入统计，LangChain 选择收入汇总工具。",
    }
    return reasons.get(tool_name, "LangChain 根据工具描述选择本轮工具。")


def build_langchain_tools(request: ChatRequest) -> list[StructuredTool]:
    """把 7 个司机域只读工具暴露给 LangChain，执行收口到 execute_tool_action。"""

    def _make_tool(name: str, spec):
        required = spec.required
        schema = spec.parameters_schema

        # 按工具的参数动态构造函数签名，让 LangChain 能正确生成 tool_calls
        if name == "get_activity_progress":
            def get_activity_progress(activity_id: str) -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_activity_progress",
                               arguments={"activity_id": activity_id},
                               reason=tool_action_reason("get_activity_progress")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_activity_progress, name=name, description=spec.description)

        if name == "get_driver_order_count":
            def get_driver_order_count(time_range: str = "今日") -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_driver_order_count",
                               arguments={"time_range": time_range},
                               reason=tool_action_reason("get_driver_order_count")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_driver_order_count, name=name, description=spec.description)

        if name == "get_activity_qualification":
            def get_activity_qualification(activity_id: str) -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_activity_qualification",
                               arguments={"activity_id": activity_id},
                               reason=tool_action_reason("get_activity_qualification")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_activity_qualification, name=name, description=spec.description)

        if name == "get_refund_status":
            def get_refund_status(order_id: str = "") -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_refund_status",
                               arguments={"order_id": order_id} if order_id else {},
                               reason=tool_action_reason("get_refund_status")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_refund_status, name=name, description=spec.description)

        if name == "get_coupon_status":
            def get_coupon_status(status: str = "") -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_coupon_status",
                               arguments={"status": status} if status else {},
                               reason=tool_action_reason("get_coupon_status")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_coupon_status, name=name, description=spec.description)

        if name == "get_verification_record":
            def get_verification_record(coupon_id: str) -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_verification_record",
                               arguments={"coupon_id": coupon_id},
                               reason=tool_action_reason("get_verification_record")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_verification_record, name=name, description=spec.description)

        if name == "get_income_summary":
            def get_income_summary(time_range: str = "本月") -> str:
                obs = execute_tool_action(
                    ToolAction(tool_name="get_income_summary",
                               arguments={"time_range": time_range},
                               reason=tool_action_reason("get_income_summary")),
                    request,
                )
                return obs.model_dump_json()
            return StructuredTool.from_function(func=get_income_summary, name=name, description=spec.description)

        # 兜底（TOOL_SPECS 里有但没写包装的工具）
        def fallback(**kwargs) -> str:
            obs = execute_tool_action(
                ToolAction(tool_name=name, arguments=kwargs, reason=tool_action_reason(name)),
                request,
            )
            return obs.model_dump_json()
        return StructuredTool.from_function(func=fallback, name=name, description=spec.description)

    return [_make_tool(name, spec) for name, spec in TOOL_SPECS.items()]
