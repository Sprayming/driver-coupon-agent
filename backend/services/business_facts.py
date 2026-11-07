"""L17 司机域：实时业务事实服务。

三道安全闸（照搬小哲 L17 services/business_facts.py 的 fail-closed 结构）：
  1. 缺参数 → summary 解释缺什么，found=False
  2. 越权（user_matched=False）→ 事实不属于当前司机，found=False, user_matched=False
  3. driver_service 查不到 → found=False

lookup 内部逻辑暂时 mock（等 driver_service_client.py 接真实端点后替换）。
这故意保留空实现——小哲 L17 docstring 明写：「第 17 课先建立事实接入层，还没有开放 Tool Calling」，
当前阶段重点是把安全边界和双链路分叉结构搭起来，lookup 内部调用等下一步接完 driver_service_get 再填。
"""

from __future__ import annotations

from api.schemas import BusinessFactNeed, BusinessFactResult, ChatRequest
from integrations.driver_service_client import (
    driver_activity_fact_from_service,
    driver_daily_order_fact_from_service,
    driver_qualification_fact_from_service,
    # L17 新增四类
    driver_refund_fact_from_service,
    driver_coupon_fact_from_service,
    driver_verification_fact_from_service,
    driver_income_fact_from_service,
)
from tools.tool_runtime import (
    summarize_driver_activity,
    summarize_driver_order_count,
    # L17 新增四类 summary
    summarize_driver_refund,
    summarize_driver_refund_list,
    summarize_driver_coupons,
    summarize_driver_verification,
    summarize_driver_income,
)


class BusinessFactService:
    """司机域实时业务事实服务：只读运力后端校验过的司机活动/单量/资格事实。"""

    def lookup(self, need: BusinessFactNeed, request: ChatRequest) -> BusinessFactResult:
        """按当前司机身份读取实时业务事实。三道安全闸直接映射到 BusinessFactResult 字段。"""

        # ── 闸 1：缺参数 ──
        if need.kind == "driver_activity":
            if not need.activity_id:
                return BusinessFactResult(
                    source="business_fact_service",
                    found=False,
                    summary="需要活动编号才能查询活动完成度，你可以告诉我活动名称或编号（如 ACT-12345）。",
                )

        if need.kind == "driver_qualification":
            if not need.activity_id:
                return BusinessFactResult(
                    source="business_fact_service",
                    found=False,
                    summary="需要活动编号才能查询活动资格。",
                )

        if need.kind == "unknown" or not need.requires_realtime:
            return BusinessFactResult(
                source="business_fact_service",
                found=False,
                summary="本轮不需要实时业务事实，走知识库回答。",
            )

        driver_id = request.runtime_user_id

        # ── 闸 2 & 3：调 integration 层，越权/查不到统一回 None+user_matched ──
        if need.kind == "driver_activity":
            fact, user_matched = driver_activity_fact_from_service(need.activity_id, driver_id)
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    user_matched=False,
                    summary=f"活动 {need.activity_id} 的事实不属于当前司机，不能交给模型回答。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    summary=f"暂时没查到活动 {need.activity_id} 对应的完成度数据。",
                )
            return BusinessFactResult(
                source="driver_service",
                found=True,
                summary=summarize_driver_activity(fact),
                facts={"activity": fact},
            )

        if need.kind == "driver_order":
            fact, user_matched = driver_daily_order_fact_from_service(
                driver_id, date_range=need.time_range or "今日"
            )
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    user_matched=False,
                    summary="当前司机没有查询自身单量的权限。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    summary=f"暂时没查到 {need.time_range or '今日'} 的单量数据。",
                )
            return BusinessFactResult(
                source="driver_service",
                found=True,
                summary=summarize_driver_order_count(fact, need.time_range or "今日"),
                facts={"order_count": fact},
            )

        if need.kind == "driver_qualification":
            fact, user_matched = driver_qualification_fact_from_service(need.activity_id, driver_id)
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    user_matched=False,
                    summary=f"活动 {need.activity_id} 的资格事实不属于当前司机。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service",
                    found=False,
                    summary=f"暂时没查到活动 {need.activity_id} 的资格判定结果。",
                )
            qualified = bool(fact.get("qualified"))
            reason = fact.get("reason") or fact.get("rejectReason") or ""
            return BusinessFactResult(
                source="driver_service",
                found=True,
                summary=(
                    f"你{'符合' if qualified else '暂不符合'}活动 {need.activity_id} 的参与资格。"
                    + (f"原因：{reason}" if reason else "")
                ),
                facts={"qualification": fact},
            )

        # ── L17 新增四类：退款 / 卡券 / 核销 / 收入 ──

        if need.kind == "driver_refund":
            fact, user_matched = driver_refund_fact_from_service(
                driver_id, order_id=need.order_id
            )
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service", found=False, user_matched=False,
                    summary="当前司机没有查询退款记录的权限。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service", found=False,
                    summary=f"暂时没查到退款相关数据{'（订单号 ' + need.order_id + '）' if need.order_id else ''}。",
                )
            if isinstance(fact, list):
                return BusinessFactResult(
                    source="driver_service", found=True,
                    summary=summarize_driver_refund_list(fact),
                    facts={"refunds": fact},
                )
            return BusinessFactResult(
                source="driver_service", found=True,
                summary=summarize_driver_refund(fact),
                facts={"refund": fact},
            )

        if need.kind == "driver_coupon":
            fact, user_matched = driver_coupon_fact_from_service(driver_id)
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service", found=False, user_matched=False,
                    summary="当前司机没有查询卡券的权限。",
                )
            if fact is None or (isinstance(fact, list) and not fact):
                return BusinessFactResult(
                    source="driver_service", found=False,
                    summary="暂时没查到你的卡券数据。",
                )
            return BusinessFactResult(
                source="driver_service", found=True,
                summary=summarize_driver_coupons(fact),
                facts={"coupons": fact},
            )

        if need.kind == "driver_verification":
            if not need.coupon_id:
                return BusinessFactResult(
                    source="business_fact_service", found=False,
                    summary="需要卡券号才能查询核销记录。",
                )
            fact, user_matched = driver_verification_fact_from_service(need.coupon_id, driver_id)
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service", found=False, user_matched=False,
                    summary=f"卡券 {need.coupon_id} 不属于当前司机。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service", found=False,
                    summary=f"暂时没查到卡券 {need.coupon_id} 的核销记录。",
                )
            return BusinessFactResult(
                source="driver_service", found=True,
                summary=summarize_driver_verification(fact),
                facts={"verification": fact},
            )

        if need.kind == "driver_income":
            fact, user_matched = driver_income_fact_from_service(
                driver_id, date_range=need.time_range or "本月"
            )
            if not user_matched:
                return BusinessFactResult(
                    source="driver_service", found=False, user_matched=False,
                    summary="当前司机没有查询收入的权限。",
                )
            if fact is None:
                return BusinessFactResult(
                    source="driver_service", found=False,
                    summary=f"暂时没查到{need.time_range or '本月'}的收入数据。",
                )
            return BusinessFactResult(
                source="driver_service", found=True,
                summary=summarize_driver_income(fact),
                facts={"income": fact},
            )

        return BusinessFactResult(source="driver_service", found=False, summary="没有足够业务参数可查。")
