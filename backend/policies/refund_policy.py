"""L26 司机域退款政策判断层。

设计原则（对照小哲电商 L26 after_sale_policy.py）：
1. 高风险动作（退款/补偿）只做"资格判断"，不执行写动作。
2. 资格判断必须同时查看：卡券实时状态（工具返回）+ 退款政策依据（RAG/政策常量）。
3. 输出 HighRiskAssessment，明确列出 blocked_write_actions，让调试台和审计直接可见。
4. 任何分支都 needs_human_approval=True，因为司机域退款必须走受控审批流程。

司机域退款政策（基于 knowledge/coupon-refund.md）：
- 未核销卡券：可发起退款申请（eligible_for_application）
- 已核销卡券：核销后 48h 内可申请退款；超 48h 不可退（not_eligible）
- 已过期卡券：不可退（not_eligible）
- 无卡券号/订单号：需要澄清（needs_clarification）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from api.schemas import Citation, HighRiskAssessment


# 司机域退款政策依据（调试台可直接展示的政策片段）。
REFUND_POLICIES: dict[str, Citation] = {
    "POLICY-COUPON-REFUND-UNUSED": Citation(
        citation_id="c-coupon-refund-unused",
        source_path="knowledge/coupon-refund.md",
        source_title="司机卡券未使用退款政策",
        chunk_id="coupon-refund-unused",
        score=1.0,
        snippet="未核销的卡券可以发起退款申请；退款必须进入审批流程，客服不得直接批准。",
    ),
    "POLICY-COUPON-REFUND-VERIFIED": Citation(
        citation_id="c-coupon-refund-verified",
        source_path="knowledge/coupon-refund.md",
        source_title="司机卡券核销后退款政策",
        chunk_id="coupon-refund-verified",
        score=1.0,
        snippet="已核销的卡券，核销后 48 小时内可以发起退款申请；超过 48 小时不可退。",
    ),
    "POLICY-COUPON-REFUND-EXPIRED": Citation(
        citation_id="c-coupon-refund-expired",
        source_path="knowledge/coupon-refund.md",
        source_title="司机卡券过期退款政策",
        chunk_id="coupon-refund-expired",
        score=1.0,
        snippet="已过期的卡券不可申请退款。",
    ),
}

# 被禁止的写动作（任何高风险分支都不允许 Agent 直接执行）。
BLOCKED_WRITE_ACTIONS: list[str] = [
    "create_refund",
    "approve_refund",
    "cancel_coupon",
    "compensate",
]


def retrieve_refund_policy(coupon_status: str) -> tuple[list[Citation], str]:
    """根据卡券状态读取对应的退款政策依据。"""
    if coupon_status == "verified":
        return [REFUND_POLICIES["POLICY-COUPON-REFUND-VERIFIED"]], "已核销卡券退款政策"
    if coupon_status == "expired":
        return [REFUND_POLICIES["POLICY-COUPON-REFUND-EXPIRED"]], "已过期卡券退款政策"
    return [REFUND_POLICIES["POLICY-COUPON-REFUND-UNUSED"]], "未使用卡券退款政策"


def assess_refund_boundary(
    coupon_status: str | None,
    verified_at: str | None,
    coupon_id: str | None = None,
    order_id: str | None = None,
) -> HighRiskAssessment:
    """判断司机卡券退款的资格和边界。

    Args:
        coupon_status: 卡券状态（unused/verified/expired），None 表示没查到。
        verified_at: 核销时间 ISO 字符串（coupon_status=verified 时用）。
        coupon_id: 卡券号。
        order_id: 退款关联的订单号。

    Returns:
        HighRiskAssessment：资格判断结果，永远带 blocked_write_actions。
    """
    action_type: Literal["refund", "compensation", "unknown"] = "refund"

    # 没卡券号也没订单号 → 需要澄清
    if not coupon_id and not order_id:
        return HighRiskAssessment(
            action_type=action_type,
            coupon_id=coupon_id,
            order_id=order_id,
            eligibility_status="needs_clarification",
            risk_level="high",
            needs_human_approval=True,
            evidence_checklist=[],
            policy_basis=[],
            reasons=["高风险退款必须先提供卡券号或订单号，不能只凭一句自然语言执行。"],
            blocked_write_actions=BLOCKED_WRITE_ACTIONS,
        )

    # 没查到卡券状态 → blocked
    if coupon_status is None:
        return HighRiskAssessment(
            action_type=action_type,
            coupon_id=coupon_id,
            order_id=order_id,
            eligibility_status="blocked",
            risk_level="high",
            needs_human_approval=True,
            evidence_checklist=["卡券状态未获取"],
            policy_basis=[],
            reasons=["当前没有拿到可信的卡券状态事实，不能执行退款。"],
            blocked_write_actions=BLOCKED_WRITE_ACTIONS,
        )

    citations, _policy_name = retrieve_refund_policy(coupon_status)
    evidence = ["卡券状态", "退款政策依据"]
    reasons: list[str] = []
    eligible = False

    if coupon_status == "unused":
        eligible = True
        reasons.append("卡券未核销，可以发起退款申请；但必须进入审批流程，客服不会直接批准。")
    elif coupon_status == "verified":
        if verified_at:
            try:
                verified_time = datetime.fromisoformat(verified_at)
                deadline = verified_time + timedelta(hours=48)
                if datetime.now() <= deadline:
                    eligible = True
                    reasons.append(
                        f"卡券已于 {verified_at} 核销，在 48 小时退款期内，可以发起退款申请。"
                    )
                    evidence.append("核销时间")
                else:
                    reasons.append(
                        f"卡券已于 {verified_at} 核销，已超过 48 小时退款期，不可退。"
                    )
            except (ValueError, TypeError):
                reasons.append("卡券已核销，但核销时间无法解析，建议转人工核实。")
        else:
            reasons.append("卡券已核销，但未获取到核销时间，建议转人工核实。")
    elif coupon_status == "expired":
        reasons.append("卡券已过期，不可申请退款。")
    else:
        reasons.append(f"卡券状态 {coupon_status} 不支持退款。")

    return HighRiskAssessment(
        action_type=action_type,
        coupon_id=coupon_id,
        order_id=order_id,
        eligibility_status="eligible_for_application" if eligible else "not_eligible",
        risk_level="high",
        needs_human_approval=True,
        evidence_checklist=evidence,
        policy_basis=citations,
        reasons=reasons,
        blocked_write_actions=BLOCKED_WRITE_ACTIONS,
    )
