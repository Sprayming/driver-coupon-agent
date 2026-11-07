"""L18 司机域：工具契约注册表。

模型看到的是工具名、描述和参数 schema，真正执行仍由后端 execute_tool_action 控制。
形状定义（ToolSpec）在 api/schemas.py，这里放具体工具清单。

7 个司机域只读工具：
  - get_activity_progress: 活动完成度（达标/未达标/差几单）
  - get_driver_order_count: 今日/本周/本月单量
  - get_activity_qualification: 活动资格判定
  - get_refund_status: 退款进度/到账状态
  - get_coupon_status: 卡券列表/未使用张数
  - get_verification_record: 卡券核销记录
  - get_income_summary: 收入统计（卡券占比）
"""

from __future__ import annotations

from api.schemas import ToolSpec


TOOL_SPECS: dict[str, ToolSpec] = {
    "get_activity_progress": ToolSpec(
        name="get_activity_progress",
        description="查询当前司机在某个活动中的完成度（目标单量、已完成单量、是否达标）。只能查当前司机自己的活动。",
        required=["activity_id"],
        parameters_schema={"activity_id": "活动编号，例如 ACT-12345 或 活动12345"},
        domain="activity",
        read_only=True,
        risk_level="low",
        boundary=[
            "只能查当前司机自己的活动完成度，不能查其他司机。",
            "只返回统计事实，不修改活动、不发放奖励。",
        ],
        resource_uris=["resource://driver/tools/activity-progress-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_driver_order_count": ToolSpec(
        name="get_driver_order_count",
        description="查询当前司机今日/本周/本月完成的单量。只能查当前司机自己的数据。",
        required=[],
        parameters_schema={"time_range": "时间范围，可选值：今日/本周/本月，默认今日"},
        domain="order",
        read_only=True,
        risk_level="low",
        boundary=[
            "只能查当前司机自己的单量，不能按司机 ID 批量查询。",
            "只返回统计事实，不派单、不改订单状态。",
        ],
        resource_uris=["resource://driver/tools/order-count-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_activity_qualification": ToolSpec(
        name="get_activity_qualification",
        description="查询当前司机是否符合某个活动的参与资格及原因。只能查当前司机自己的资格。",
        required=["activity_id"],
        parameters_schema={"activity_id": "活动编号，例如 ACT-12345"},
        domain="activity",
        read_only=True,
        risk_level="low",
        boundary=[
            "只判断资格，不替司机报名、不授予资格。",
            "只能查当前司机自己的资格，不能查其他司机。",
        ],
        resource_uris=["resource://driver/tools/activity-qualification-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_refund_status": ToolSpec(
        name="get_refund_status",
        description="查询当前司机的退款进度或到账状态（只读，不创建退款申请）。可不传订单号，默认返回最近三条退款。",
        required=[],
        parameters_schema={"order_id": "订单号（可选），例如 订单88321 或 8 位以上数字"},
        domain="refund",
        read_only=True,
        risk_level="medium",
        boundary=[
            "只读查询，绝对不能创建退款、批准退款或修改退款状态。",
            "只能查当前司机自己的退款记录。",
        ],
        resource_uris=["resource://driver/tools/refund-status-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_coupon_status": ToolSpec(
        name="get_coupon_status",
        description="查询当前司机账户里的卡券列表（未使用/已核销/已过期）。只能查当前司机自己的卡券。",
        required=[],
        parameters_schema={"status": "卡券状态过滤（可选）：unused/verified/expired"},
        domain="coupon",
        read_only=True,
        risk_level="medium",
        boundary=[
            "只读查询，不能核销、作废、退款或转移卡券。",
            "只能查当前司机自己的卡券，不返回银行卡号和司机 ID。",
        ],
        resource_uris=["resource://driver/tools/coupon-status-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_verification_record": ToolSpec(
        name="get_verification_record",
        description="查询某张卡券的核销记录（核销时间、关联订单、金额）。只能查当前司机自己的卡券。",
        required=["coupon_id"],
        parameters_schema={"coupon_id": "卡券号，例如 CP-001"},
        domain="verification",
        read_only=True,
        risk_level="low",
        boundary=[
            "只能查当前司机自己卡券的核销记录。",
            "只读查询，不能补录核销、撤销核销或修改核销时间。",
        ],
        resource_uris=["resource://driver/tools/verification-record-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
    "get_income_summary": ToolSpec(
        name="get_income_summary",
        description="查询当前司机今日/本周/本月的收入统计及卡券收入占比。只能查当前司机自己的收入。",
        required=[],
        parameters_schema={"time_range": "时间范围，可选值：今日/本周/本月，默认本月"},
        domain="income",
        read_only=True,
        risk_level="low",
        boundary=[
            "只能查当前司机自己的收入统计，不能查询其他司机或团队汇总。",
            "只读查询，不能提现、转账或修改结算金额。",
        ],
        resource_uris=["resource://driver/tools/income-summary-boundary"],
        prompt_ids=["prompt://driver/tool-observation"],
    ),
}
