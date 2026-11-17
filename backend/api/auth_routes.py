"""内部员工登录 API（方案 A）：/auth/login、/auth/logout、/auth/me。

- 登录成功后下发 HttpOnly + SameSite=Lax 会话 Cookie，前端不接触 token；
- 会话只存 sha256 哈希，过期 8 小时（可配）；
- 账号不存在与密码错误统一返回 401，避免用户名枚举。
"""

from __future__ import annotations

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, Field

from config import settings
from store import users


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=64)


class AdminUserView(BaseModel):
    username: str
    display_name: str
    role: str


def _to_view(user: dict[str, str]) -> AdminUserView:
    return AdminUserView(
        username=user["username"], display_name=user["display_name"], role=user["role"]
    )


def create_auth_router() -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/login", response_model=AdminUserView)
    def login(request: LoginRequest, response: Response) -> AdminUserView:
        user = users.authenticate(settings.DB_PATH, request.username, request.password)
        if user is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        token = users.create_session(settings.DB_PATH, user["user_id"])
        response.set_cookie(
            key=settings.AUTH_COOKIE_NAME,
            value=token,
            max_age=settings.AUTH_SESSION_TTL_HOURS * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return _to_view(user)

    @router.post("/register", response_model=AdminUserView)
    def register(request: RegisterRequest, response: Response) -> AdminUserView:
        """自助注册：固定创建 viewer（只读），提权必须由 publisher 在用户管理里做。"""

        if not settings.AUTH_ALLOW_SELF_REGISTER:
            raise HTTPException(status_code=403, detail="当前环境未开放自助注册，请联系管理员")

        import sqlite3

        try:
            created = users.create_user(
                settings.DB_PATH,
                username=request.username,
                role="viewer",
                password=request.password,
                display_name=request.display_name,
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="用户名已存在")

        token = users.create_session(settings.DB_PATH, created["user_id"])
        response.set_cookie(
            key=settings.AUTH_COOKIE_NAME,
            value=token,
            max_age=settings.AUTH_SESSION_TTL_HOURS * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return _to_view(created)

    @router.post("/logout")
    def logout(
        response: Response,
        session_token: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    ) -> dict[str, bool]:
        users.revoke_session(settings.DB_PATH, session_token)
        response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")
        return {"ok": True}

    @router.get("/me", response_model=AdminUserView)
    def me(
        session_token: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    ) -> AdminUserView:
        user = users.resolve_session(settings.DB_PATH, session_token)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录或会话已过期")
        return _to_view(user)

    return router
