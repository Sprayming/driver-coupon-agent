"""Runtime Context：读取请求自带的已知事实 + 归属边界 + 脱敏。

对标小哲 L41 `tools/runtime_context.py`。和 `integrations/driver_service_client.py` 的分工：

- 下游客户端：调后端拿**实时业务事实**（可能失败 → 降级）
- 本模块：只读请求里用户**已知**的信息，且只允许当前用户自己的；进入模型前统一脱敏

归属边界在这里收口，避免每个调用点各写一遍。
"""

from __future__ import annotations

import re
from typing import Any

# 只打码中间段，保留首尾便于人工核对；宁可少脱敏也不要把整段抹掉导致客服无法定位。
PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)")
BANK_CARD_PATTERN = re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)")


def mask_sensitive_text(text: str) -> str:
    """把手机号、身份证、银行卡的中间段打码。"""

    if not text:
        return text
    masked = PHONE_PATTERN.sub(r"\1****\2", text)
    masked = ID_CARD_PATTERN.sub(r"\1********\2", masked)
    masked = BANK_CARD_PATTERN.sub(r"\1********\2", masked)
    return masked


def mask_payload(value: Any) -> Any:
    """递归脱敏 dict / list / str，其余类型原样返回。"""

    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, dict):
        return {key: mask_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_payload(item) for item in value)
    return value


def known_facts_from_request(request: Any) -> dict[str, Any]:
    """读取请求里随上下文带进来的已知事实（前端透传，可能为空）。"""

    context = getattr(request, "runtime_context", None)
    return context if isinstance(context, dict) else {}


def known_coupons_from_request(request: Any) -> list[dict[str, Any]]:
    """从请求上下文里取用户已知卡券；只返回归属当前用户的条目。"""

    driver_id = str(getattr(request, "runtime_user_id", "") or "").strip()
    coupons = known_facts_from_request(request).get("coupons") or []
    if not isinstance(coupons, list):
        return []
    return [item for item in coupons if isinstance(item, dict) and ensure_coupon_owned(item, driver_id)]


def ensure_coupon_owned(coupon: dict[str, Any], driver_id: str) -> bool:
    """归属校验：卡券上带 driverId 时必须等于当前司机。

    没带归属信息时放行——和 `driver_service_client` 的 data 级校验同口径：
    后端没给归属就无法判断，交回给下游的 404 语义兜底。
    """

    owner = str(coupon.get("driverId") or coupon.get("driver_id") or "").strip()
    if not owner:
        return True
    return owner == driver_id.strip()
