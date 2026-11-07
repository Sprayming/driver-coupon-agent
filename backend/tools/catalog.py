"""L24 司机域 MCP-style 工具目录。

设计原则（对照小哲电商 L24 MCPCatalog）：
1. 工具定义自带 resource_uri 和 prompt_id，边界有标准位置，不散落在 description 里。
2. catalog 统一管理 tools / resources / prompts，Agent 和 Hooks 从目录读，不从 TOOL_SPECS 直接读。
3. binding_summary() 返回工具 + 资源 + Prompt 的绑定关系，让调试台看到"这个工具挂了哪些边界文档"。

司机域多团队场景：
- 退款团队：get_refund_status → resource://driver/tools/refund-status-boundary
- 营销运营团队：get_coupon_status / get_activity_progress / get_activity_qualification
- 风控/客服团队：get_verification_record / get_income_summary
- 订单/运力团队：get_driver_order_count
"""

from __future__ import annotations

from api.schemas import (
    MCPBindingSummary,
    MCPPrompt,
    MCPResource,
    ToolAction,
    ToolSpec,
    RiskLevel,
)
from tools.contracts import TOOL_SPECS


class DriverToolCatalog:
    """司机域 MCP-style 工具目录：多团队协作时的统一入口。

    每个工具挂 resource_uri 指向独立的边界文档，prompt_id 指向统一的口径模板。
    Agent 调用工具前可以先读边界，而不是靠 description 里的一句话。
    """

    def __init__(self, tool_specs: list[ToolSpec] | None = None) -> None:
        # 默认用本地静态注册表；接入 MCP 后由 load_tool_specs() 用工具池覆盖。
        if tool_specs is None:
            self._tools = dict(TOOL_SPECS)
        else:
            self._tools = {spec.name: spec for spec in tool_specs}
        self._resources = self._build_resources()
        self._prompts = self._build_prompts()

    def load_tool_specs(self, tool_specs: list[ToolSpec]) -> None:
        """L24：用 MCP 客户端拉回来的工具池重建目录，资源文档跟着重算。"""
        if not tool_specs:
            return
        self._tools = {spec.name: spec for spec in tool_specs}
        self._resources = self._build_resources()

    def _build_resources(self) -> dict[str, MCPResource]:
        """从当前工具池的 boundary 字段生成资源文档。"""
        resources: dict[str, MCPResource] = {}
        for name, spec in self._tools.items():
            for uri in spec.resource_uris:
                # 资源标题从 URI 推导：resource://driver/tools/refund-status-boundary → 退款进度工具边界
                title = self._resource_title(uri)
                resources[uri] = MCPResource(
                    uri=uri,
                    title=title,
                    content="\n".join(spec.boundary),
                )
        # 全局高风险边界资源
        resources["resource://driver/high-risk-boundary"] = MCPResource(
            uri="resource://driver/high-risk-boundary",
            title="高风险动作边界",
            content="退款、取消订单、补偿和改地址需要后续受控流程和人工确认，不能由普通 Tool Use 自动执行。",
        )
        return resources

    def _build_prompts(self) -> dict[str, MCPPrompt]:
        """统一工具返回的口径模板。"""
        return {
            "prompt://driver/tool-observation": MCPPrompt(
                prompt_id="prompt://driver/tool-observation",
                title="工具 Observation 口径",
                content=(
                    "把工具返回压缩成事实摘要，保留必要字段和 omitted_fields，"
                    "不把内部调试字段暴露给用户。司机域红线：driverId 和 bankAccount 必须过滤。"
                ),
            ),
            "prompt://driver/high-risk-handoff": MCPPrompt(
                prompt_id="prompt://driver/high-risk-handoff",
                title="高风险转人工口径",
                content=(
                    "遇到退款、取消订单、补偿等高风险动作时，说明不能自动执行，"
                    "并建议转人工处理。不要承诺退款到账时间，不要编造审批状态。"
                ),
            ),
        }

    @staticmethod
    def _resource_title(uri: str) -> str:
        """从 URI 推导资源标题。"""
        mapping = {
            "activity-progress-boundary": "活动完成度工具边界",
            "order-count-boundary": "单量统计工具边界",
            "activity-qualification-boundary": "活动资格工具边界",
            "refund-status-boundary": "退款进度工具边界",
            "coupon-status-boundary": "卡券状态工具边界",
            "verification-record-boundary": "核销记录工具边界",
            "income-summary-boundary": "收入统计工具边界",
        }
        for key, title in mapping.items():
            if key in uri:
                return title
        return "工具边界"

    def list_tools(self) -> list[ToolSpec]:
        """列出工具目录，供 Agent 选择前先看能力边界。"""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolSpec | None:
        """按名字查工具。"""
        return self._tools.get(name)

    def read_resource(self, uri: str) -> MCPResource:
        """读取工具能力附带的边界文档。"""
        return self._resources[uri]

    def get_prompt(self, prompt_id: str) -> MCPPrompt:
        """读取 Prompt 模板。"""
        return self._prompts[prompt_id]

    def binding_summary(
        self, action: ToolAction | None, risk_level: RiskLevel
    ) -> MCPBindingSummary:
        """汇总工具、资源和 Prompt 绑定关系，方便观察 MCP 不是单个函数列表。"""
        if risk_level == "high":
            return MCPBindingSummary(
                tool_source="driver_tool_catalog",
                selected_tool=None,
                available_tools=sorted(self._tools.keys()),
                resources=["resource://driver/high-risk-boundary"],
                prompts=["prompt://driver/high-risk-handoff"],
                boundary="高风险动作必须走受控工作流，不能由普通 Tool Use 自动执行。",
            )
        if action is None:
            return MCPBindingSummary(
                tool_source="driver_tool_catalog",
                selected_tool=None,
                available_tools=sorted(self._tools.keys()),
                resources=[],
                prompts=[],
                boundary="本轮没有选中工具；Tool Use 仍负责决定是否调用。",
            )
        definition = self._tools[action.tool_name]
        return MCPBindingSummary(
            tool_source="driver_tool_catalog",
            selected_tool=definition.name,
            available_tools=sorted(self._tools.keys()),
            resources=definition.resource_uris,
            prompts=definition.prompt_ids,
            boundary="MCP 负责提供标准化工具说明；Tool Use 仍负责规划、执行、Observation 和 Hooks 治理。",
        )


# 全局单例
DRIVER_TOOL_CATALOG = DriverToolCatalog()
