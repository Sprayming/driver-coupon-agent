"""方案 A 验证：内部员工账号、会话 Cookie 与角色分层。

覆盖：
1. 登录成功下发 HttpOnly Cookie，未登录 401；
2. 角色越权 403（viewer 不能改、editor 不能发布、editor 不能审核）；
3. 审计 actor 取自登录态，请求体无法伪造；
4. 机器调用 X-Admin-Token 仍可用（向后兼容）。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from api.auth_routes import create_auth_router
from config import settings
from store import users
from store.seed import seed


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "task22.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    assert users.ensure_seed_users(db_path) == 4

    app = FastAPI()
    app.include_router(create_auth_router())
    app.include_router(create_admin_router())
    return TestClient(app)


def login(client: TestClient, username: str) -> TestClient:
    resp = client.post("/auth/login", json={"username": username, "password": settings.AUTH_SEED_PASSWORD})
    assert resp.status_code == 200, resp.text
    return client


# —— 1. 登录与会话 ——


def test_login_sets_httponly_cookie(client):
    resp = client.post(
        "/auth/login", json={"username": "admin", "password": settings.AUTH_SEED_PASSWORD}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "publisher"
    cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_wrong_password_returns_401_without_user_enumeration(client):
    resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    # 不存在的用户也返回同样的 401（不暴露账号是否存在）。
    resp = client.post("/auth/login", json={"username": "nobody", "password": "wrong"})
    assert resp.status_code == 401


def test_admin_api_requires_login(client):
    assert client.get("/admin/versions").status_code == 401


def test_me_and_logout(client):
    login(client, "viewer")
    assert client.get("/auth/me").json()["role"] == "viewer"
    client.post("/auth/logout")
    assert client.get("/auth/me").status_code == 401


# —— 2. 角色分层（服务端强制） ——


def test_viewer_can_read_but_cannot_write(client):
    login(client, "viewer")
    assert client.get("/admin/versions").status_code == 200
    assert client.post("/admin/kb/sync").status_code == 403
    assert client.post("/admin/releases", json={"version_ids": []}).status_code == 403


def test_editor_can_sync_but_cannot_publish(client):
    login(client, "editor")
    assert client.post("/admin/kb/sync").status_code == 200
    assert client.post("/admin/releases", json={"version_ids": []}).status_code == 403


def test_reviewer_cannot_submit_draft(client):
    """提交审核是内容编辑的动作，审核员不能替编辑提交。"""

    login(client, "reviewer")
    draft = client.get("/admin/versions", params={"status": "draft"}).json()[0]
    resp = client.post(
        f"/admin/versions/{draft['version_id']}/review", json={"action": "submit"}
    )
    assert resp.status_code == 403


# —— 3. 审计操作人来自登录态 ——


def test_audit_actor_comes_from_session(client):
    # 编辑提交草稿。
    login(client, "editor")
    draft = client.get("/admin/versions", params={"status": "draft"}).json()[0]
    resp = client.post(
        f"/admin/versions/{draft['version_id']}/review",
        json={"action": "submit", "actor": "forged-admin"},
    )
    assert resp.status_code == 200, resp.text

    # 换审核员登录并通过（同一 TestClient 会覆盖 Cookie）。
    login(client, "reviewer")
    resp = client.post(
        f"/admin/versions/{draft['version_id']}/review",
        json={"action": "approve", "actor": "forged-admin"},
    )
    assert resp.status_code == 200, resp.text

    events = client.get("/admin/audit", params={"limit": 200}).json()
    actors = {event["actor"] for event in events}
    assert "editor" in actors and "reviewer" in actors
    assert "forged-admin" not in actors


# —— 4. 机器令牌兼容 ——


def test_machine_token_still_works(client):
    resp = client.get("/admin/versions", headers={"X-Admin-Token": settings.ADMIN_TOKEN})
    assert resp.status_code == 200
    resp = client.get("/admin/versions", headers={"X-Admin-Token": "bad"})
    assert resp.status_code == 401


# —— 5. 自助注册（登录页注册入口） ——


def test_register_creates_viewer_and_logs_in(client):
    resp = client.post(
        "/auth/register",
        json={"username": "newbie", "password": "passw0rd1", "display_name": "新人"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["role"] == "viewer" and body["username"] == "newbie"
    assert "httponly" in resp.headers["set-cookie"].lower()
    # 注册即登录：会话 Cookie 直接可用，但只读。
    assert client.get("/auth/me").json()["role"] == "viewer"
    assert client.get("/admin/versions").status_code == 200
    assert client.get("/admin/users").status_code == 403


def test_register_duplicate_username_returns_409(client):
    payload = {"username": "dupuser", "password": "passw0rd1", "display_name": ""}
    assert client.post("/auth/register", json=payload).status_code == 200
    assert client.post("/auth/register", json=payload).status_code == 409


def test_register_rejects_invalid_username(client):
    resp = client.post(
        "/auth/register", json={"username": "带中文", "password": "passw0rd1"}
    )
    assert resp.status_code == 422


def test_register_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ALLOW_SELF_REGISTER", False)
    resp = client.post(
        "/auth/register", json={"username": "closed", "password": "passw0rd1"}
    )
    assert resp.status_code == 403


# —— 6. 用户管理（仅 publisher） ——


def test_only_publisher_can_manage_users(client):
    login(client, "editor")
    assert client.get("/admin/users").status_code == 403


def test_publisher_user_crud_and_audit(client):
    login(client, "admin")
    created = client.post(
        "/admin/users",
        json={"username": "ops01", "password": "passw0rd1", "role": "editor", "display_name": "运营"},
    )
    assert created.status_code == 200, created.text
    user_id = created.json()["user_id"]

    # 改角色：editor → reviewer。
    patched = client.patch(f"/admin/users/{user_id}", json={"role": "reviewer"})
    assert patched.status_code == 200 and patched.json()["role"] == "reviewer"

    # 停用后立即无法登录。
    disabled = client.patch(f"/admin/users/{user_id}", json={"status": "disabled"})
    assert disabled.status_code == 200 and disabled.json()["status"] == "disabled"
    resp = client.post("/auth/login", json={"username": "ops01", "password": "passw0rd1"})
    assert resp.status_code == 401

    # 重置密码（密码内容不进审计）。
    login(client, "admin")
    reset = client.post(f"/admin/users/{user_id}/password", json={"password": "newpassw0rd"})
    assert reset.status_code == 200

    client.patch(f"/admin/users/{user_id}", json={"status": "active"})
    assert client.post("/auth/login", json={"username": "ops01", "password": "newpassw0rd"}).status_code == 200

    # 删除。
    login(client, "admin")
    assert client.delete(f"/admin/users/{user_id}").status_code == 200
    usernames = {row["username"] for row in client.get("/admin/users").json()}
    assert "ops01" not in usernames

    actions = {event["action"] for event in client.get("/admin/audit", params={"limit": 200}).json()}
    assert {"rbac.user.create", "rbac.user.update", "rbac.user.password_reset", "rbac.user.delete"} <= actions


def test_user_cannot_disable_or_delete_self(client):
    login(client, "admin")
    me = next(row for row in client.get("/admin/users").json() if row["username"] == "admin")
    assert client.patch(f"/admin/users/{me['user_id']}", json={"status": "disabled"}).status_code == 400
    assert client.patch(f"/admin/users/{me['user_id']}", json={"role": "viewer"}).status_code == 400
    assert client.delete(f"/admin/users/{me['user_id']}").status_code == 400
