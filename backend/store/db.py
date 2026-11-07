"""RAG 平台 SQLite 连接管理与幂等初始化（Task 1）。

设计要点：
- 连接一律开启外键约束（PRAGMA foreign_keys=ON），SQLite 默认是关的；
- schema.sql 全部用 CREATE TABLE IF NOT EXISTS，init_db 可重复执行（TR-1.1）；
- 时间戳统一走 utc_now_iso()，全库口径一致（schema 约定 2）。

说明：DB_PATH 在 Task 2 接入 settings（AGENT_DB_PATH）；在此之前用模块级默认值，
保证 Task 1 可独立验证。旧路径常量保留，不删。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# store/db.py -> backend/store/db.py，parents[1] 即 backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_DB_PATH = BACKEND_DIR / "data" / "rag_platform.db"

# Task 2 将用 settings.DB_PATH 覆盖此默认值（env: AGENT_DB_PATH）。
DB_PATH: Path = DEFAULT_DB_PATH


def utc_now_iso() -> str:
    """全库统一时间戳：UTC、秒精度、带时区标记的 ISO-8601。"""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """打开一个带外键约束、Row 工厂的连接。调用方负责关闭（或用 transaction()）。"""

    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    # SQLite 外键约束默认关闭，必须逐连接打开，否则 REFERENCES / 悬空插入不会报错。
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def transaction(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """事务上下文：正常退出 commit，异常 rollback，始终关闭连接。"""

    connection = get_connection(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: Path | str | None = None) -> Path:
    """幂等初始化：执行 schema.sql 建表（IF NOT EXISTS），返回数据库路径（TR-1.1）。"""

    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    connection = get_connection(path)
    try:
        connection.executescript(ddl)
        connection.commit()
    finally:
        connection.close()
    return path


def list_tables(db_path: Path | str | None = None) -> list[str]:
    """列出当前库中全部业务表名（验证/调试用）。"""

    with transaction(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [str(row["name"]) for row in rows]
