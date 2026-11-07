"""L20：Observation 压缩层。把内部 ToolObservation（含完整业务数据）
压缩成可进入模型回答上下文的安全 Observation。

设计原则（对齐小哲电商 L20 + 实际业务安全实践）：
- ToolObservation.data 是内部原始结果。
- Observation 只保留 summary + 关键 facts + omitted_fields + next_action。
- summary 必须基于安全 facts 重新构建，绝不能透传 tool_observation.summary
  （原始 summary 可能已把敏感字段写成自然语言，omitted_fields 管不了）。
- 三级过滤：
  1) 安全红线（永远过滤）：driverId、bankAccount
  2) 业务语义数据（司机自己有权知道）：时间戳、订单号、卡券名称、有效期 → 必须保留
  3) 冗余数据（防 token 爆炸）：列表超过阈值时压缩成统计数字
"""

from __future__ import annotations

from typing import Any

from api.schemas import Observation, ToolObservation


def _extract_business_fact(data: dict[str, Any]) -> tuple[str, Any]:
    """从 ToolObservation.data 中取出业务事实主体（去掉 found/user_matched/source 三个控制字段）。"""
    control_keys = {"found", "user_matched", "source"}
    for key, value in data.items():
        if key not in control_keys:
            return key, value
    return "", None


def build_observation(tool_observation: ToolObservation) -> Observation:
    """把内部 ToolObservation 压缩成可进入回答上下文的 Observation。"""
    data = tool_observation.data
    tool_name = tool_observation.tool_name

    # ── 错误态：映射成安全话术，不暴露原始错误 ──
    if tool_observation.status == "error":
        error_source = data.get("source", "unknown")
        safe_summary = {
            "driver_service": "当前还没有这个业务数据的查询权限。",
            "business_fact_service": "当前业务事实查询服务不可用。",
        }.get(error_source, "工具返回错误。")
        return Observation(
            tool_name=tool_name, status="error", summary=safe_summary,
            facts={"error_source": error_source}, omitted_fields=[],
            next_action="fallback_answer",
        )

    fact_key, fact = _extract_business_fact(data)
    if fact is None:
        return Observation(
            tool_name=tool_name, status="error",
            summary="工具没有返回可用事实。",
            facts={}, omitted_fields=[], next_action="fallback_answer",
        )

    # ── 按工具分支压缩 ──

    if tool_name == "get_activity_progress":
        activity = fact if isinstance(fact, dict) else {}
        name = str(activity.get("activityName") or activity.get("activityId") or "该活动")
        target = activity.get("targetOrders") or activity.get("target") or 0
        completed = activity.get("completedOrders") or activity.get("completed") or 0
        diff = max(target - completed, 0)
        safe_summary = f"{name}，目标 {target} 单，已完成 {completed} 单，还差 {diff} 单。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={"activity_name": name, "target_orders": target, "completed_orders": completed},
            # 安全红线 driverId 永远过滤；activityId/deadline 是内部标识，可过滤
            omitted_fields=["deadline", "endTime", "driverId", "activityId"],
            next_action="answer_user",
        )

    if tool_name == "get_driver_order_count":
        order = fact if isinstance(fact, dict) else {}
        total = order.get("totalOrders") or order.get("orderCount") or 0
        safe_summary = f"累计完成 {total} 单。" if total else "暂时没查到你的单量数据。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={"total_orders": total},
            omitted_fields=["peakOrders", "nightOrders", "driverId"],
            next_action="answer_user",
        )

    if tool_name == "get_activity_qualification":
        qual = fact if isinstance(fact, dict) else {}
        qualified = qual.get("qualified", False)
        reason = qual.get("reason") or qual.get("rejectReason") or ""
        safe_summary = "你符合报名条件。" if qualified else f"报名未通过：{reason}"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={"qualified": qualified, "reason": reason},
            omitted_fields=["driverId", "activityId"],
            next_action="answer_user",
        )

    if tool_name == "get_refund_status":
        if isinstance(fact, list):
            refunds = fact
            if not refunds:
                return Observation(
                    tool_name=tool_name, status="success",
                    summary="最近没有退款记录。",
                    facts={"refund_count": 0},
                    # 多条退款：压缩成统计，每条的完整字段（含 orderId/时间戳/bankAccount）全部过滤
                    omitted_fields=["orderId", "bankAccount", "appliedAt", "completedAt", "rejectReason", "driverId"],
                    next_action="answer_user",
                )
            total = len(refunds)
            statuses = [r.get("status", "") for r in refunds if isinstance(r, dict)]
            refunded = sum(1 for s in statuses if s in ("refunded", "已到账"))
            processing = sum(1 for s in statuses if s in ("processing", "处理中"))
            safe_summary = f"共 {total} 条退款记录：{refunded} 条已到账，{processing} 条处理中。"
            return Observation(
                tool_name=tool_name, status="success", summary=safe_summary,
                facts={"refund_count": total, "refunded_count": refunded, "processing_count": processing},
                omitted_fields=["orderId", "bankAccount", "appliedAt", "completedAt", "rejectReason", "driverId"],
                next_action="answer_user",
            )
        # 单条退款：司机自己的对账数据，orderId/时间戳需要知道；bankAccount 永远过滤
        refund = fact if isinstance(fact, dict) else {}
        status_text = refund.get("statusText") or refund.get("status") or "未知"
        amount = refund.get("amount", 0)
        order_id = refund.get("orderId", "")
        applied_at = refund.get("appliedAt", "")
        completed_at = refund.get("completedAt", "")
        safe_summary = f"退款 {amount} 元（订单 {order_id}），当前状态：{status_text}。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={
                "status": status_text, "amount": amount,
                "order_id": order_id, "applied_at": applied_at, "completed_at": completed_at,
            },
            omitted_fields=["bankAccount", "rejectReason", "driverId"],
            next_action="answer_user",
        )

    if tool_name == "get_coupon_status":
        coupons = fact if isinstance(fact, list) else [fact]
        unused = sum(1 for c in coupons if isinstance(c, dict) and c.get("status") == "unused")
        verified = sum(1 for c in coupons if isinstance(c, dict) and c.get("status") == "verified")
        expired = sum(1 for c in coupons if isinstance(c, dict) and c.get("status") == "expired")
        # 卡券 ≤ 5 张：保留每张券的安全详情（名称、状态、有效期）；> 5 张：压缩成统计数字防 token 爆炸
        COUPON_DETAIL_THRESHOLD = 5
        if len(coupons) <= COUPON_DETAIL_THRESHOLD:
            coupon_details = []
            for c in coupons:
                if not isinstance(c, dict):
                    continue
                coupon_details.append({
                    "name": c.get("name", "卡券"),
                    "status": c.get("status", ""),
                    "valid_from": c.get("validFrom", ""),
                    "valid_to": c.get("validTo", ""),
                })
            # 构建包含每张券详情的 summary
            detail_parts = []
            for d in coupon_details:
                status_label = {"unused": "未使用", "verified": "已核销", "expired": "已过期"}.get(d["status"], d["status"])
                detail_parts.append(f"{d['name']}（{status_label}，有效期 {d['valid_from']} ~ {d['valid_to']}）")
            safe_summary = f"你共有 {len(coupons)} 张卡券：" + "；".join(detail_parts)
            return Observation(
                tool_name=tool_name, status="success", summary=safe_summary,
                facts={
                    "total_count": len(coupons),
                    "unused_count": unused,
                    "verified_count": verified,
                    "expired_count": expired,
                    "coupon_details": coupon_details,
                },
                # 保留 name/validFrom/validTo；安全红线 driverId + 内部标识 couponId 过滤
                omitted_fields=["couponId", "driverId"],
                next_action="answer_user",
            )
        # 卡券 > 5 张：压缩成统计数字
        safe_summary = f"你共有 {len(coupons)} 张卡券：{unused} 张未使用、{verified} 张已核销、{expired} 张已过期。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={
                "total_count": len(coupons),
                "unused_count": unused,
                "verified_count": verified,
                "expired_count": expired,
            },
            omitted_fields=["name", "couponId", "validFrom", "validTo", "driverId"],
            next_action="answer_user",
        )

    if tool_name == "get_verification_record":
        ver = fact if isinstance(fact, dict) else {}
        # 关键：coupon_name 绝不 fallback 到 couponId！
        coupon_name = ver.get("couponName") or "该卡券"
        verified = ver.get("verified", False)
        amount = ver.get("amount", 0)
        verified_at = ver.get("verifiedAt", "")
        order_id = ver.get("orderId", "")
        if verified:
            safe_summary = f"{coupon_name} 已核销（{verified_at}，订单 {order_id}，核销金额 {amount} 元）。"
        else:
            safe_summary = f"{coupon_name} 还没有被核销过。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={
                "coupon_name": coupon_name, "verified": verified, "amount": amount,
                # 司机自己的对账数据，核销时间和订单号需要知道
                "verified_at": verified_at, "order_id": order_id,
            },
            # couponId 是内部标识（司机习惯用名称）；driverId 安全红线
            omitted_fields=["couponId", "driverId"],
            next_action="answer_user",
        )

    if tool_name == "get_income_summary":
        income = fact if isinstance(fact, dict) else {}
        time_range = income.get("timeRange", "本月")
        total = income.get("totalIncome", 0)
        orders = income.get("orders", 0)
        percent = income.get("couponIncomePercent", 0)
        safe_summary = f"{time_range}你总收入 {total} 元，完成 {orders} 单；其中卡券贡献约占 {percent}%。"
        return Observation(
            tool_name=tool_name, status="success", summary=safe_summary,
            facts={
                "time_range": time_range, "total_income": total,
                "orders": orders, "coupon_income_percent": percent,
            },
            omitted_fields=["couponIncome", "driverId"],
            next_action="answer_user",
        )

    # ── 未识别工具：保守返回安全占位，不暴露 data ──
    return Observation(
        tool_name=tool_name, status="success",
        summary="工具返回了业务数据，但当前课程版本未实现该工具的压缩策略。",
        facts={}, omitted_fields=list(data.keys()),
        next_action="answer_user",
    )
