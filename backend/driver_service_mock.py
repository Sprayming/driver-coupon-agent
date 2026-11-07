"""L17 司机域：运力业务后端 Mock 服务（FastAPI，端口 8082）。

复刻小哲 L17 电商后端的安全边界：
  - Dependency 验 X-Agent-Service-Token，缺失/错配 → 403
  - 每个端点从 Header 读 X-Agent-Driver-Id（身份委托），必须和 URL 里的 driver_id 匹配
  - 身份委托下查不到 → 404（和越权统一口径，Agent 侧保守判 user_matched=False）

Mock 数据覆盖 6 个场景：
  1. driver_001 在 ACT-12345 差 2 单未达标（target=30, completed=28）
  2. driver_001 在 ACT-67890 已达标（target=100, completed=105）
  3. driver_001 查 driver_002 的活动 → 404（越权）
  4. driver_002 查自己没参与的活动 → 404（资源不存在）
  5. driver_001 查活动资格 → qualified=True
  6. driver_999 查活动资格 → qualified=False（近30天违规≥3次）

启动：  uvicorn driver_service_mock:app --host 0.0.0.0 --port 8082
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field


# ── 配置 ──
EXPECTED_SERVICE_TOKEN = os.getenv(
    "AGENT_DRIVER_SERVICE_TOKEN",
    os.getenv("AGENT_SERVICE_AUTH_TOKEN", "driver-coupon-debug-agent"),
)


# ── 统一响应 ──
class ApiResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None


# ── Mock 数据（内存 dict，进程内有效） ──

# 活动完成度：key = (activity_id, driver_id) → fact dict
ACTIVITY_PROGRESS: dict[tuple[str, str], dict[str, Any]] = {
    ("ACT-12345", "driver_001"): {
        "driverId": "driver_001",
        "activityId": "ACT-12345",
        "activityName": "周末冲单赛",
        "targetOrders": 30,
        "completedOrders": 28,
        "deadline": "2026-10-25 23:59",
    },
    ("ACT-67890", "driver_001"): {
        "driverId": "driver_001",
        "activityId": "ACT-67890",
        "activityName": "全月稳行奖",
        "targetOrders": 100,
        "completedOrders": 105,
        "deadline": "2026-10-31 23:59",
    },
}

# 司机单量：key = (driver_id, date_range) → fact dict
DRIVER_ORDERS: dict[tuple[str, str], dict[str, Any]] = {
    ("driver_001", "今日"): {
        "driverId": "driver_001",
        "dateRange": "今日",
        "totalOrders": 15,
        "peakOrders": 10,
        "nightOrders": 5,
    },
    ("driver_002", "今日"): {
        "driverId": "driver_002",
        "dateRange": "今日",
        "totalOrders": 5,
        "peakOrders": 3,
        "nightOrders": 2,
    },
}

# 活动资格：key = (activity_id, driver_id) → fact dict
ACTIVITY_QUALIFICATION: dict[tuple[str, str], dict[str, Any]] = {
    ("ACT-12345", "driver_001"): {
        "driverId": "driver_001",
        "activityId": "ACT-12345",
        "activityName": "周末完单奖",
        "qualified": True,
        "reason": "",
    },
    ("ACT-12345", "driver_999"): {
        "driverId": "driver_999",
        "activityId": "ACT-12345",
        "activityName": "周末完单奖",
        "qualified": False,
        "reason": "近30天违规≥3次，暂不符合报名条件",
    },
}


# ── 安全依赖 ──
def verify_service_token(
    x_agent_service_token: str | None = Header(default=None, alias="X-Agent-Service-Token"),
) -> None:
    """全局验 Agent 服务身份头。缺失或错配 → 403。"""
    if not x_agent_service_token or x_agent_service_token.strip() != EXPECTED_SERVICE_TOKEN:
        raise HTTPException(status_code=403, detail="FORBIDDEN: invalid or missing X-Agent-Service-Token")


def resolve_delegated_driver(
    request: Request,
    header_driver_id: str | None = Header(default=None, alias="X-Agent-Driver-Id"),
) -> str:
    """身份委托校验：Header 里的 driver_id 必须和 URL 里的 driver_id 一致（如果 URL 里有的话）。

    注意：不用 Path() 声明参数（会和端点函数自己的 Path 参数冲突），
    而是从 request.path_params 手动拿 driver_id。

    对于 URL 里没有 driver_id 的端点（如 /api/coupons/{coupon_id}/verification），
    只要求 header 里有 driver_id 即可，不再强制比对。
    """
    if not header_driver_id:
        raise HTTPException(status_code=404, detail="NOT_FOUND: missing X-Agent-Driver-Id")
    url_driver_id = request.path_params.get("driver_id")
    if url_driver_id and header_driver_id.strip() != url_driver_id:
        raise HTTPException(status_code=404, detail="NOT_FOUND: driver not found or not delegated")
    return header_driver_id.strip()


# ── App ──
app = FastAPI(title="Driver Service Mock", version="0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "driver-service-mock"}


# 1. 活动完成度
@app.get(
    "/api/activities/{activity_id}/driver/{driver_id}",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_activity_progress(
    activity_id: str = Path(...),
    driver_id: str = Path(...),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    key = (activity_id, driver_id)
    fact = ACTIVITY_PROGRESS.get(key)
    if fact is None:
        # 这个司机没参与这个活动 / 活动不存在 → 404
        raise HTTPException(status_code=404, detail=f"Activity {activity_id} not found for driver {driver_id}")
    return ApiResponse(success=True, data=fact)


# 2. 司机单量
@app.get(
    "/api/drivers/{driver_id}/orders",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_driver_orders(
    driver_id: str = Path(...),
    range: str = Query(default="今日", description="今日 / 本周 / 本月"),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    fact = DRIVER_ORDERS.get((driver_id, range))
    if fact is None:
        raise HTTPException(status_code=404, detail=f"No order data for driver {driver_id} ({range})")
    return ApiResponse(success=True, data=fact)


# 3. 活动资格
@app.get(
    "/api/activities/{activity_id}/qualification/{driver_id}",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_activity_qualification(
    activity_id: str = Path(...),
    driver_id: str = Path(...),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    fact = ACTIVITY_QUALIFICATION.get((activity_id, driver_id))
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Qualification not found for {activity_id}/{driver_id}")
    return ApiResponse(success=True, data=fact)


# ── L17 新增四类：退款 / 卡券 / 核销 / 收入 ──

# 退款记录：key = (driver_id, order_id?) → 单条或列表
REFUND_RECORDS: dict[tuple[str, str | None], dict[str, Any] | list[dict[str, Any]]] = {
    ("driver_001", None): [  # 查"最近退款"（不限 order_id）
        {
            "driverId": "driver_001",
            "orderId": "88321",
            "amount": 15.0,
            "status": "refunded",
            "statusText": "已到账",
            "appliedAt": "2026-10-18 14:20",
            "completedAt": "2026-10-19 09:30",
            "bankAccount": "尾号 8821",
        },
        {
            "driverId": "driver_001",
            "orderId": "88102",
            "amount": 8.5,
            "status": "processing",
            "statusText": "处理中",
            "appliedAt": "2026-10-19 16:45",
            "completedAt": None,
            "bankAccount": "尾号 8821",
        },
        {
            "driverId": "driver_001",
            "orderId": "87990",
            "amount": 20.0,
            "status": "rejected",
            "statusText": "已驳回",
            "appliedAt": "2026-10-17 10:00",
            "completedAt": "2026-10-17 11:30",
            "bankAccount": "尾号 8821",
            "rejectReason": "已过退款时效（购买超过 7 天）",
        },
    ],
    ("driver_001", "88321"): {
        "driverId": "driver_001",
        "orderId": "88321",
        "amount": 15.0,
        "status": "refunded",
        "statusText": "已到账",
        "appliedAt": "2026-10-18 14:20",
        "completedAt": "2026-10-19 09:30",
        "bankAccount": "尾号 8821",
    },
    ("driver_001", "88102"): {
        "driverId": "driver_001",
        "orderId": "88102",
        "amount": 8.5,
        "status": "processing",
        "statusText": "处理中",
        "appliedAt": "2026-10-19 16:45",
        "completedAt": None,
        "bankAccount": "尾号 8821",
    },
    ("driver_001", "87990"): {
        "driverId": "driver_001",
        "orderId": "87990",
        "amount": 20.0,
        "status": "rejected",
        "statusText": "已驳回",
        "appliedAt": "2026-10-17 10:00",
        "completedAt": "2026-10-17 11:30",
        "bankAccount": "尾号 8821",
        "rejectReason": "已过退款时效（购买超过 7 天）",
    },
}

# 司机卡券列表
DRIVER_COUPONS: dict[tuple[str, str | None], list[dict[str, Any]]] = {
    ("driver_001", None): [
        {
            "driverId": "driver_001",
            "couponId": "CP-20261015-001",
            "name": "周末特惠月卡",
            "status": "unused",
            "statusText": "未使用",
            "price": 39.9,
            "validFrom": "2026-10-15",
            "validTo": "2026-11-14",
        },
        {
            "driverId": "driver_001",
            "couponId": "CP-20261010-003",
            "name": "全月稳行周卡",
            "status": "unused",
            "statusText": "未使用",
            "price": 19.9,
            "validFrom": "2026-10-10",
            "validTo": "2026-10-24",
        },
        {
            "driverId": "driver_001",
            "couponId": "CP-20260920-007",
            "name": "中秋限时券",
            "status": "verified",
            "statusText": "已核销",
            "price": 9.9,
            "validFrom": "2026-09-20",
            "validTo": "2026-09-27",
            "verifiedAt": "2026-09-25 18:30",
        },
    ],
}

# 核销记录：key = (coupon_id, driver_id)
VERIFICATION_RECORDS: dict[tuple[str, str], dict[str, Any]] = {
    ("CP-20260920-007", "driver_001"): {
        "couponId": "CP-20260920-007",
        "couponName": "中秋限时券",
        "driverId": "driver_001",
        "verified": True,
        "verifiedAt": "2026-09-25 18:30",
        "orderId": "O-99283",
        "amount": 9.9,
    },
}

# 收入统计：key = (driver_id, time_range)
DRIVER_INCOME: dict[tuple[str, str], dict[str, Any]] = {
    ("driver_001", "本月"): {
        "driverId": "driver_001",
        "timeRange": "本月",
        "totalIncome": 3280.0,
        "couponIncome": 420.0,
        "couponIncomePercent": 12.8,
        "orders": 245,
    },
    ("driver_001", "今日"): {
        "driverId": "driver_001",
        "timeRange": "今日",
        "totalIncome": 135.0,
        "couponIncome": 18.0,
        "couponIncomePercent": 13.3,
        "orders": 15,
    },
}


# ── 新端点 ──

@app.get(
    "/api/drivers/{driver_id}/refunds",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_driver_refunds(
    driver_id: str = Path(...),
    order_id: str | None = Query(default=None, description="指定订单号时返回单条，不传返回最近三条"),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    key = (driver_id, order_id)
    fact = REFUND_RECORDS.get(key)
    if fact is None:
        if order_id is None:
            raise HTTPException(status_code=404, detail=f"No refund data for driver {driver_id}")
        raise HTTPException(status_code=404, detail=f"Refund not found for order {order_id}")
    return ApiResponse(success=True, data=fact)


@app.get(
    "/api/drivers/{driver_id}/coupons",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_driver_coupons(
    driver_id: str = Path(...),
    status: str | None = Query(default=None, description="unused / verified / expired"),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    coupons = DRIVER_COUPONS.get((driver_id, None))
    if coupons is None:
        raise HTTPException(status_code=404, detail=f"No coupon data for driver {driver_id}")
    if status:
        coupons = [c for c in coupons if c["status"] == status]
    return ApiResponse(success=True, data=coupons)


@app.get(
    "/api/coupons/{coupon_id}/verification",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_coupon_verification(
    coupon_id: str = Path(...),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    fact = VERIFICATION_RECORDS.get((coupon_id, delegated))
    if fact is None:
        # 核销记录不存在 → 保守返回"未核销"
        return ApiResponse(success=True, data={
            "couponId": coupon_id,
            "driverId": delegated,
            "verified": False,
        })
    return ApiResponse(success=True, data=fact)


@app.get(
    "/api/drivers/{driver_id}/income",
    dependencies=[Depends(verify_service_token)],
    response_model=ApiResponse,
)
def get_driver_income(
    driver_id: str = Path(...),
    range: str = Query(default="本月", description="今日 / 本周 / 本月"),
    delegated: str = Depends(resolve_delegated_driver),
) -> ApiResponse:
    fact = DRIVER_INCOME.get((driver_id, range))
    if fact is None:
        raise HTTPException(status_code=404, detail=f"No income data for driver {driver_id} ({range})")
    return ApiResponse(success=True, data=fact)
