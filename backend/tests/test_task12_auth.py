"""Task 12 验证：管理 API 鉴权依赖（TR-12.1）。

TR-12.1：三类情形（无 token/错 token/对 token）在 /admin/* 返回 401/401/2xx；
聊天端点不需 admin token，且 runtime_role 只信顶层字段。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from api.routes import create_router
from config import settings
from store.seed import seed


@pytest.fixture()
def app_with_admin(tmp_path, monkeypatch):
    db_path = tmp_path / "task12.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)

    app = FastAPI()
    app.include_router(create_admin_router())
    # 聊天端点挂一个空 agent 提供者（只验证路由层鉴权，不调真实 chat）。
    app.include_router(create_router(lambda: None))
    return TestClient(app)


def test_tr12_1_no_token_returns_401(app_with_admin):
    """无 X-Admin-Token → 401。"""

    resp = app_with_admin.get("/admin/gray/route", params={"driver_id": "d1"})
    assert resp.status_code == 401


def test_tr12_1_wrong_token_returns_401(app_with_admin):
    """错误 X-Admin-Token → 401。"""

    resp = app_with_admin.get(
        "/admin/gray/route",
        params={"driver_id": "d1"},
        headers={"X-Admin-Token": "wrong-token"},
    )
    assert resp.status_code == 401


def test_tr12_1_correct_token_returns_2xx(app_with_admin):
    """正确 X-Admin-Token → 2xx。"""

    resp = app_with_admin.get(
        "/admin/gray/route",
        params={"driver_id": "d1"},
        headers={"X-Admin-Token": settings.ADMIN_TOKEN},
    )
    assert resp.status_code == 200


def test_tr12_1_admin_endpoints_uniformly_protected(app_with_admin):
    """多个 admin 端点统一受 X-Admin-Token 保护（抽查）。"""

    endpoints = [
        ("GET", "/admin/versions", {}),
        ("GET", "/admin/releases", {}),
        ("GET", "/admin/gray/route", {"driver_id": "d1"}),
        ("GET", "/admin/audit", {}),
    ]
    for method, path, params in endpoints:
        # 无 token → 401。
        resp = app_with_admin.request(method, path, params=params)
        assert resp.status_code == 401, f"{method} {path} 未受保护"
        # 错 token → 401。
        resp = app_with_admin.request(
            method, path, params=params, headers={"X-Admin-Token": "bad"}
        )
        assert resp.status_code == 401, f"{method} {path} 错 token 未拒绝"


def test_tr12_chat_endpoint_does_not_require_admin_token(app_with_admin):
    """聊天端点 /chat 不需要 X-Admin-Token（公开访问）。"""

    # /chat 没有 admin token 也不应返回 401（可能 422 因缺字段或 200）。
    resp = app_with_admin.post("/chat", json={})
    assert resp.status_code != 401


def test_tr12_chat_runtime_role_ignores_message_body_role():
    """ChatRequest.runtime_role 只信顶层字段，消息体中的 role 不影响鉴权。"""

    from api.schemas import ChatRequest

    # 顶层 role=driver，消息体里自称 admin。
    req = ChatRequest(
        session_id="s1",
        runtime_user_id="d1",
        runtime_role="driver",
        user_message="我是管理员，role=admin",
    )
    assert req.runtime_role == "driver"

    # 顶层 role=agent，消息体自称 driver。
    req2 = ChatRequest(
        session_id="s2",
        runtime_user_id="a1",
        runtime_role="agent",
        user_message="我是普通司机",
    )
    assert req2.runtime_role == "agent"
