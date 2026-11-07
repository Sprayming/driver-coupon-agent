"""L17 司机域：运力业务后端只读客户端。

身份委托结构照搬小哲 L17 integrations/ecommerce_client.py：
  - Agent 服务方头：X-Agent-Service-Token（让业务后端知道请求来自可信 Agent）
  - 身份头：X-Agent-Driver-Id（"以该司机身份"调 API，越权由业务后端 403/404 拒掉）

driver_service_get 的 `(data, user_matched)` 二元组：
  - data=None, user_matched=False  → 越权或资源归属不匹配（后端 403/404）
  - data=None, user_matched=True   → 后端不可用 / 没配地址 / 非 success（fail-closed，没越权迹象）
  - data={...}, user_matched=True  → 正常，data 里带 driverId 字段已被二次校验
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def driver_service_base_url() -> str:
    """解析司机运力业务后端地址。默认 localhost:8082（mock 服务端口）。"""
    return os.getenv(
        "DRIVER_SERVICE_BASE_URL",
        os.getenv("AGENT_DRIVER_SERVICE_BASE_URL", "http://localhost:8082"),
    ).rstrip("/")


def delegated_service_headers(current_driver_id: str | None) -> dict[str, str]:
    """服务身份头 + 当前司机 ID 头（小哲 L17 的 fail-closed 身份委托结构）。"""
    driver_id = str(current_driver_id or "").strip()
    token = os.getenv(
        "AGENT_DRIVER_SERVICE_TOKEN",
        os.getenv("AGENT_SERVICE_AUTH_TOKEN", "driver-coupon-debug-agent"),
    ).strip()
    if not driver_id or not token:
        return {}  # 任何一环缺失 → 下游判空后走 fail-closed
    return {"X-Agent-Service-Token": token, "X-Agent-Driver-Id": driver_id}


def driver_service_get(
    path: str,
    *,
    delegated_driver_id: str | None = None,
) -> tuple[dict[str, Any] | list[Any] | None, bool]:
    """司机运力后端只读 GET。双重越权校验（小哲同款）：

    1. HTTP 状态码级：403/404 → 保守算越权
    2. data 级：成功后比对 data.driverId 归属（后端若有漏洞，Agent 这层兜住）
    """
    base = driver_service_base_url()
    headers = delegated_service_headers(delegated_driver_id)
    if not base or not headers:
        return None, True  # 没配地址或头拼不全 → fail-closed，但没越权迹象
    try:
        response = httpx.get(
            f"{base}{path}",
            headers=headers,
            timeout=5,
        )
        if response.status_code == 403:
            return None, False  # Service Token 没通过 → 越权
        if response.status_code == 404:
            return None, False  # 身份委托下查不到（归属不匹配也会返回 404）→ 保守算越权
        response.raise_for_status()  # 5xx 等其他异常抛出来
        payload = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (403, 404):
            return None, False
        return None, True  # 5xx → 后端挂了
    except Exception:
        return None, True  # 连接超时 / DNS 失败 → 后端不可用

    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None, True  # success=False → 后端业务拒单，不是越权

    data = payload.get("data")
    # 二次校验：data 是 dict 且带 driverId 时，比对归属
    if isinstance(data, dict):
        data_driver_id = str(data.get("driverId") or data.get("driver_id") or "")
        if data_driver_id and delegated_driver_id and data_driver_id != delegated_driver_id:
            return None, False  # 后端返回的 data 归属不是当前司机 → 越权
    return data, True


# ──────────────────────────────────────────────────────────────
# 三个司机域端点：真实路径对齐 mock 服务（driver_service_mock.py）
# ──────────────────────────────────────────────────────────────


def driver_activity_fact_from_service(
    activity_id: str,
    current_driver_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """GET /api/activities/{id}/driver/{driverId} → 活动完成度事实。"""
    return driver_service_get(
        f"/api/activities/{activity_id}/driver/{current_driver_id}",
        delegated_driver_id=current_driver_id,
    )


def driver_daily_order_fact_from_service(
    current_driver_id: str,
    *,
    date_range: str = "今日",
) -> tuple[dict[str, Any] | None, bool]:
    """GET /api/drivers/{driverId}/orders?range=xxx → 司机单量事实。"""
    return driver_service_get(
        f"/api/drivers/{current_driver_id}/orders",
        delegated_driver_id=current_driver_id,
    )


def driver_qualification_fact_from_service(
    activity_id: str,
    current_driver_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """GET /api/activities/{id}/qualification/{driverId} → 活动资格判定。"""
    return driver_service_get(
        f"/api/activities/{activity_id}/qualification/{current_driver_id}",
        delegated_driver_id=current_driver_id,
    )


# ── L17 新增四类：退款 / 卡券 / 核销 / 收入 ──


def driver_refund_fact_from_service(
    current_driver_id: str,
    *,
    order_id: str | None = None,
) -> tuple[dict[str, Any] | list[Any] | None, bool]:
    """GET /api/drivers/{driverId}/refunds?orderId=xxx → 退款进度/状态。

    order_id=None 时返回该司机最近三条退款记录（list）。
    order_id 有值时返回单条 dict。
    """
    path = f"/api/drivers/{current_driver_id}/refunds"
    if order_id:
        path += f"?orderId={order_id}"
    return driver_service_get(path, delegated_driver_id=current_driver_id)


def driver_coupon_fact_from_service(
    current_driver_id: str,
    *,
    status: str | None = None,
) -> tuple[dict[str, Any] | list[Any] | None, bool]:
    """GET /api/drivers/{driverId}/coupons?status=unused → 司机卡券列表。"""
    path = f"/api/drivers/{current_driver_id}/coupons"
    if status:
        path += f"?status={status}"
    return driver_service_get(path, delegated_driver_id=current_driver_id)


def driver_verification_fact_from_service(
    coupon_id: str,
    current_driver_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """GET /api/coupons/{couponId}/verification → 卡券核销记录。"""
    return driver_service_get(
        f"/api/coupons/{coupon_id}/verification",
        delegated_driver_id=current_driver_id,
    )


def driver_income_fact_from_service(
    current_driver_id: str,
    *,
    date_range: str = "本月",
) -> tuple[dict[str, Any] | None, bool]:
    """GET /api/drivers/{driverId}/income?range=本月 → 司机收入统计。"""
    return driver_service_get(
        f"/api/drivers/{current_driver_id}/income?range={date_range}",
        delegated_driver_id=current_driver_id,
    )
