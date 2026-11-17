"""HITL 审批结果持久化：跨进程、重启后仍然幂等。

为什么不用内存字典：
- 原实现是模块级 dict，进程重启即丢，多 worker 部署时各进程各一份，
  同一条审批回调会被重复处理；
- 审批结果落库后，靠主键 (thread_id, approval_id) 兜住并发，
  第二个请求 `INSERT OR IGNORE` 影响 0 行即判定为重复。

状态流转：processing（抢占）→ done（完成）/ 失败时删除记录允许重试。
"""

from __future__ import annotations

from pathlib import Path

import sqlite3

from config import settings
from store.db import transaction

CLAIMED = "processing"
DONE = "done"

# 抢占租约：超过这个时长仍未完成，视为上个进程已崩，允许其他请求接管。
# release 只覆盖 Python 异常，覆盖不了 kill -9 / 容器驱逐，没有租约这条审批会永久卡死。
DEFAULT_LEASE_SECONDS = 300


def _path(db_path: Path | str | None) -> Path | str:
    """统一解析库路径：默认跟 settings.DB_PATH 走，测试切库才能生效。"""

    return db_path if db_path is not None else settings.DB_PATH


def claim(
    thread_id: str,
    approval_id: str,
    db_path: Path | str | None = None,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> bool:
    """抢占一条审批结果，返回是否拿到执行权。

    两种情况算抢到：
    1. 首次插入成功（INSERT OR IGNORE 影响 1 行）；
    2. 记录停在 processing 但已超过租约（上个进程没来得及 finish/release），
       用条件 UPDATE 接管并把 created_at 重置为当前时间。
    """

    if lease_seconds < 1:
        raise ValueError("lease_seconds 必须 >= 1")

    with transaction(_path(db_path)) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO approval_results
                (thread_id, approval_id, status, created_at, updated_at)
            VALUES (?, ?, ?, datetime('now'), datetime('now'))
            """,
            (thread_id, approval_id, CLAIMED),
        )
        if cursor.rowcount == 1:
            return True

        # 主键冲突：要么已经在处理，要么已经处理完。只有「processing 且已过期」能接管。
        takeover = connection.execute(
            """
            UPDATE approval_results
               SET created_at = datetime('now'), updated_at = datetime('now')
             WHERE thread_id = ? AND approval_id = ? AND status = ?
               AND created_at < datetime('now', ?)
            """,
            (thread_id, approval_id, CLAIMED, f"-{lease_seconds} seconds"),
        )
        return takeover.rowcount == 1


def finish(
    thread_id: str,
    approval_id: str,
    *,
    approved: bool,
    approver: str = "",
    comment: str = "",
    db_path: Path | str | None = None,
) -> None:
    """标记审批已处理，并保留结论供重复请求返回。"""

    with transaction(_path(db_path)) as connection:
        connection.execute(
            """
            UPDATE approval_results
               SET status = ?, approved = ?, approver = ?, comment = ?,
                   updated_at = datetime('now')
             WHERE thread_id = ? AND approval_id = ?
            """,
            (DONE, 1 if approved else 0, approver, comment, thread_id, approval_id),
        )


def release(
    thread_id: str,
    approval_id: str,
    db_path: Path | str | None = None,
) -> None:
    """恢复失败时释放占位，允许人工重试。"""

    with transaction(_path(db_path)) as connection:
        connection.execute(
            "DELETE FROM approval_results WHERE thread_id = ? AND approval_id = ? AND status = ?",
            (thread_id, approval_id, CLAIMED),
        )


def get(
    thread_id: str,
    approval_id: str,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """查询已有审批记录；没有则返回 None。"""

    with transaction(_path(db_path)) as connection:
        return connection.execute(
            "SELECT * FROM approval_results WHERE thread_id = ? AND approval_id = ?",
            (thread_id, approval_id),
        ).fetchone()


def list_for_thread(
    thread_id: str,
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """列出某个会话的全部审批记录（新到旧）。"""

    with transaction(_path(db_path)) as connection:
        return connection.execute(
            "SELECT * FROM approval_results WHERE thread_id = ? ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
