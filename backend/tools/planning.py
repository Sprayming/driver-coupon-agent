r"""L17 司机域：实时业务事实识别。

和 rag/planning.py（L15 检索前计划）是不同职责：
- rag.planning → 产出 RetrievalPlan，决定 RAG 链走哪些主题
- tools.planning → 产出 BusinessFactNeed，决定要不要分叉到实时链

司机域关键词映射 + 参数抽取：
  - 活动完成度类 → kind=driver_activity
  - 今日/本周单量类 → kind=driver_order
  - 活动资格类 → kind=driver_qualification
  - 退款进度/状态 → kind=driver_refund
  - 卡券状态 → kind=driver_coupon
  - 核销记录 → kind=driver_verification
  - 收入统计 → kind=driver_income
  - 抽 activity_id（ACT-\d+ 或 "活动\d+"），time_range（今日/本周/本月）
  - 抽 order_id（订单\d+ / 8位以上数字），coupon_id（卡券号）
"""

from __future__ import annotations

import re

from api.schemas import BusinessFactNeed, ChatRequest


# ── 关键词表：按业务场景分组，命中即走实时链 ──
_ACTIVITY_COMPLETION_TERMS = [
    "活动", "达标", "单量", "还差", "差多少", "目标", "进度", "完成度",
    "跑了多少", "跑到多少", "做了多少", "完成多少",
]
_ORDER_COUNT_TERMS = [
    "今日", "今天", "本周", "本月", "跑了多少单", "今天几单",
    "今天做了多少", "这个月跑了", "这周几单",
]
_QUALIFICATION_TERMS = [
    "资格", "能参加", "能不能进", "报名条件", "报名门槛",
    "能不能报", "能报名", "能报", "符合条件", "我符合",
]

# 纯时间单量查询词（不含"活动"的单量问法，优先级高于活动完成度）
_TIME_ORDER_TERMS = [
    "今日", "今天", "本周", "本月", "跑了多少单", "今天几单",
    "今天做了多少", "这个月跑了", "这周几单",
]

# L19：强实时完成度信号。只有"活动"两个字命中时可能是规则问题（如"活动规则是什么"），
# 交给 RAG；命中这些词说明司机在问自己的进度，缺 activity_id 也该进实时链做澄清。
_STRONG_COMPLETION_TERMS = [
    "达标", "还差", "差多少", "进度", "完成度", "目标",
]

# ── 新四类：退款/卡券/核销/收入 ──
_REFUND_TERMS = [
    "退款", "退款到账", "到账了吗", "退款进度", "订单退款", "退款状态",
    "退款申请", "退款处理", "审核到哪", "处理进度", "到账没", "退款了吗",
]
_COUPON_TERMS = [
    "卡券", "优惠券", "券", "未使用", "没用", "还剩几张", "还有几张",
    "卡券状态", "卡券余额", "券到账了吗",
]
_VERIFICATION_TERMS = [
    "核销", "核销记录", "核销了吗", "被核销", "核销没",
]
_INCOME_TERMS = [
    "收入", "卡券占", "卡券收入", "本月收入", "我的收入", "卡券贡献",
]


def extract_activity_id(user_message: str) -> str | None:
    """从用户消息抽活动 ID。支持 ACT-12345 或 "活动12345" 两种格式。"""
    match = re.search(r"(?:ACT[_\-]?\d{3,}|活动\d{3,})", user_message, re.IGNORECASE)
    return match.group(0) if match else None


def extract_time_range(user_message: str) -> str | None:
    """从用户消息抽时间范围（今日/本周/本月）。"""
    for label in ["本月", "这周", "本周", "今日", "今天", "昨天", "最近"]:
        if label in user_message:
            return label
    return None


def extract_order_id(user_message: str) -> str | None:
    """从用户消息抽订单号。支持 "订单88321" 或 8 位以上纯数字。"""
    m = re.search(r"订单\s*(\d{5,})", user_message)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{8,})\b", user_message)
    return m.group(1) if m else None


def extract_coupon_id(user_message: str) -> str | None:
    """从用户消息抽卡券号。支持 "卡券号"/"券号"/"卡券" 后面跟的编号（允许含连字符）。"""
    # 先试 "卡券号 CP-xxx" / "券号: CP-xxx"
    m = re.search(r"(?:卡券号|券号|卡券)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9\-]{3,})", user_message)
    if m:
        return m.group(1)
    # 再试纯编号模式 CP-xxx（独立出现）
    m = re.search(r"\b([A-Z]{2,}-\d{4,})\b", user_message)
    return m.group(1) if m else None


def detect_business_fact_need(request: ChatRequest) -> BusinessFactNeed:
    """判断本轮是否需要实时业务事实（司机域）。

    规则优先级（从高到低）：
    1. 资格类（带/不带 activity_id） → driver_qualification
    2. 活动完成度（且有 activity_id） → driver_activity
    3. 核销记录 → driver_verification（先于卡券，更精确）
    4. 退款类 → driver_refund（planning.py 新增，覆盖之前 realtime_gap 的退款问题）
    5. 收入统计 → driver_income
    6. 卡券状态 → driver_coupon
    7. 纯时间单量查询（有时间词）→ driver_order
    8. 活动完成度关键词命中但没 activity_id → 降级 driver_order 或 unknown
    9. 其他 → unknown，走 RAG 链
    """
    message = request.user_message

    activity_id = extract_activity_id(message)
    time_range = extract_time_range(message)
    order_id = extract_order_id(message)
    coupon_id = extract_coupon_id(message)

    # ── 最高优先：资格类（带或不带 activity_id 都可判定） ──
    if any(term in message for term in _QUALIFICATION_TERMS):
        return BusinessFactNeed(
            kind="driver_qualification",
            requires_realtime=True,
            activity_id=activity_id,
            reason="活动资格判定依赖司机最近单量/违规记录等实时业务数据。",
        )

    # ── 活动完成度（必须有 activity_id 才走这个分支） ──
    if any(term in message for term in _ACTIVITY_COMPLETION_TERMS):
        if activity_id:
            return BusinessFactNeed(
                kind="driver_activity",
                requires_realtime=True,
                activity_id=activity_id,
                time_range=time_range,
                reason="活动完成度/单量进度是实时业务事实，不能从卡券知识库推断。",
            )
        # 没 activity_id → 降级：如果有时间词，当纯单量查询处理
        if time_range or any(term in message for term in _TIME_ORDER_TERMS):
            return BusinessFactNeed(
                kind="driver_order",
                requires_realtime=True,
                time_range=time_range or "今日",
                reason="纯单量查询，无活动编号，按时间维度查运力系统。",
            )
        # L19：司机在问自己的达标/进度但没给活动编号 → 仍然进实时链，
        # 由澄清规划器追问 activity_id，而不是悄悄降级成 RAG 规则回答。
        if any(term in message for term in _STRONG_COMPLETION_TERMS):
            return BusinessFactNeed(
                kind="driver_activity",
                requires_realtime=True,
                activity_id=None,
                time_range=time_range,
                reason="司机在问自己的活动进度，但没给活动编号，需要先澄清 activity_id。",
            )

    # ── 核销记录（先于退款/卡券，更精确的意图） ──
    if any(term in message for term in _VERIFICATION_TERMS):
        return BusinessFactNeed(
            kind="driver_verification",
            requires_realtime=True,
            coupon_id=coupon_id,
            reason="核销记录是实时业务事实，必须查卡券系统。",
        )

    # ── 退款类（覆盖原 realtime_gap 的退款/到账问题） ──
    if any(term in message for term in _REFUND_TERMS):
        return BusinessFactNeed(
            kind="driver_refund",
            requires_realtime=True,
            order_id=order_id,
            time_range=time_range,
            reason="退款进度/到账状态是个人当前状态，知识库答不了，必须查订单系统。",
        )

    # ── 收入统计 ──
    if any(term in message for term in _INCOME_TERMS):
        return BusinessFactNeed(
            kind="driver_income",
            requires_realtime=True,
            time_range=time_range or "本月",
            reason="司机收入统计是实时业务事实，必须查财务系统。",
        )

    # ── 卡券状态 ──
    if any(term in message for term in _COUPON_TERMS):
        return BusinessFactNeed(
            kind="driver_coupon",
            requires_realtime=True,
            coupon_id=coupon_id,
            reason="卡券余额/状态是实时业务事实，必须查卡券系统。",
        )

    # ── 纯时间单量查询 ──
    if any(term in message for term in _TIME_ORDER_TERMS):
        return BusinessFactNeed(
            kind="driver_order",
            requires_realtime=True,
            time_range=time_range or "今日",
            reason="司机今日/本周/本月单量属于实时业务事实，必须查运力系统。",
        )

    return BusinessFactNeed(
        kind="unknown",
        requires_realtime=False,
        reason="本轮没有识别到需要实时业务事实的司机域问题。",
    )
