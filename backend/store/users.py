"""内部员工账号与会话（方案 A：最小可用账号体系）。

设计约定：
- 口令只存 pbkdf2_sha256 哈希，不存明文；
- 会话只存 token 的 sha256，DB 泄露也拿不到可用会话；
- 角色四档：viewer（只读）/ editor（改草稿、提交审核）/ reviewer（审核）/
  publisher（发布、灰度、回滚），权限在服务端校验；
- 首次启动播种演示账号，仅在 users 表为空时执行（幂等）。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings
from store.db import transaction, utc_now_iso

ROLES: tuple[str, ...] = ("viewer", "editor", "reviewer", "publisher")

_PBKDF2_ITERATIONS = 200_000

# 独立于 schema.sql 再声明一次：老库（已建表但缺 users）启动时也能补齐，无需重建数据。
AUTH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    username     TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL CHECK (role IN ('viewer','editor','reviewer','publisher')),
    password_hash TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS admin_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_user ON admin_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_expires ON admin_sessions(expires_at);
"""


def hash_password(password: str, *, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """生成 pbkdf2_sha256$迭代次数$盐$摘要 格式的哈希。"""

    salt = secrets.token_bytes(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """常量时间比较，避免时序侧信道。"""

    try:
        algorithm, iterations, salt, expected = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
    ).hex()
    return hmac.compare_digest(actual, expected)


def init_auth_schema(db_path: Path | str | None = None) -> None:
    """幂等补齐账号相关表（老库无需重建即可升级）。"""

    with transaction(db_path) as connection:
        connection.executescript(AUTH_SCHEMA_SQL)


def create_user(
    db_path: Path | str | None,
    *,
    username: str,
    role: str,
    password: str,
    display_name: str = "",
) -> dict[str, str]:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")

    user_id = f"u-{uuid.uuid4().hex[:12]}"
    with transaction(db_path) as connection:
        connection.execute(
            """
            INSERT INTO users (user_id, username, display_name, role, password_hash, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (user_id, username, display_name or username, role, hash_password(password), utc_now_iso()),
        )
    return {"user_id": user_id, "username": username, "display_name": display_name or username, "role": role}


def get_user_by_username(db_path: Path | str | None, username: str) -> dict[str, str] | None:
    with transaction(db_path) as connection:
        row = connection.execute(
            "SELECT user_id, username, display_name, role, password_hash, status FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def authenticate(db_path: Path | str | None, username: str, password: str) -> dict[str, str] | None:
    """校验账号密码；账号不存在或密码错误一律返回 None（不区分，避免枚举）。"""

    user = get_user_by_username(db_path, username)
    if not user or user["status"] != "active":
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {key: user[key] for key in ("user_id", "username", "display_name", "role")}


def create_session(
    db_path: Path | str | None,
    user_id: str,
    *,
    ttl_hours: int | None = None,
) -> str:
    """创建会话，返回明文 token（只在这里出现一次，落库的是哈希）。"""

    hours = ttl_hours if ttl_hours is not None else settings.AUTH_SESSION_TTL_HOURS
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=hours)).replace(microsecond=0).isoformat()
    with transaction(db_path) as connection:
        # 顺手清理过期会话，避免表无限增长。
        connection.execute("DELETE FROM admin_sessions WHERE expires_at <= ?", (utc_now_iso(),))
        connection.execute(
            "INSERT INTO admin_sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token_hash, user_id, expires_at, utc_now_iso()),
        )
    return token


def resolve_session(db_path: Path | str | None, token: str | None) -> dict[str, str] | None:
    """按明文 token 解析当前登录用户；过期/停用/不存在都返回 None。"""

    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with transaction(db_path) as connection:
        row = connection.execute(
            """
            SELECT u.user_id, u.username, u.display_name, u.role
            FROM admin_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ? AND u.status = 'active'
            """,
            (token_hash, utc_now_iso()),
        ).fetchone()
    return dict(row) if row else None


def revoke_session(db_path: Path | str | None, token: str | None) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM admin_sessions WHERE token_hash = ?", (token_hash,))


# 演示账号：用户名 → (角色, 展示名)
SEED_USERS: tuple[tuple[str, str, str], ...] = (
    ("admin", "publisher", "发布管理员"),
    ("reviewer", "reviewer", "审核员"),
    ("editor", "editor", "内容编辑"),
    ("viewer", "viewer", "只读观察员"),
)


def ensure_seed_users(db_path: Path | str | None = None) -> int:
    """users 表为空时播种演示账号，返回新建数量（幂等）。"""

    init_auth_schema(db_path)
    with transaction(db_path) as connection:
        existing = connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if existing:
        return 0

    created = 0
    for username, role, display_name in SEED_USERS:
        create_user(
            db_path,
            username=username,
            role=role,
            password=settings.AUTH_SEED_PASSWORD,
            display_name=display_name,
        )
        created += 1
    return created


# —— 用户管理（publisher 专用） ——


def list_users(db_path: Path | str | None = None) -> list[dict[str, str]]:
    """列出全部账号（不含 password_hash）。"""

    with transaction(db_path) as connection:
        rows = connection.execute(
            "SELECT user_id, username, display_name, role, status, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_user(db_path: Path | str | None, user_id: str) -> dict[str, str] | None:
    with transaction(db_path) as connection:
        row = connection.execute(
            "SELECT user_id, username, display_name, role, status, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def count_active_publishers(db_path: Path | str | None, *, exclude_user_id: str | None = None) -> int:
    """统计可发布的在职账号，用于防止把最后一个 publisher 降级/停用/删除导致锁死后台。"""

    with transaction(db_path) as connection:
        if exclude_user_id:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'publisher' AND status = 'active' AND user_id != ?",
                (exclude_user_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'publisher' AND status = 'active'"
            ).fetchone()
    return int(row["n"])


def update_user(
    db_path: Path | str | None,
    user_id: str,
    *,
    role: str | None = None,
    display_name: str | None = None,
    status: str | None = None,
) -> dict[str, str] | None:
    """更新角色/展示名/状态；被停用的账号立即失效（会话一并清理）。"""

    fields: list[str] = []
    values: list[str] = []
    if role is not None:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        fields.append("role = ?")
        values.append(role)
    if display_name is not None:
        fields.append("display_name = ?")
        values.append(display_name)
    if status is not None:
        if status not in ("active", "disabled"):
            raise ValueError(f"unknown status: {status}")
        fields.append("status = ?")
        values.append(status)

    if not fields:
        return get_user(db_path, user_id)

    with transaction(db_path) as connection:
        connection.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?", (*values, user_id)
        )
        if status == "disabled":
            connection.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
    return get_user(db_path, user_id)


def reset_password(db_path: Path | str | None, user_id: str, password: str) -> None:
    """重置口令并踢掉该账号的全部会话（防止旧会话继续有效）。"""

    with transaction(db_path) as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (hash_password(password), user_id),
        )
        connection.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))


def delete_user(db_path: Path | str | None, user_id: str) -> None:
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

