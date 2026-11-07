"""RAG 平台存储访问层（Task 3 起）。

Task 3 只放种子/生命周期需要的基础写方法与少量查询；
Task 4–8 的同步、审核流、发布、灰度、回滚查询继续在本文件扩展。
所有写操作必须经过 write_audit 留痕（NFR-2）。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from store.db import utc_now_iso


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_audit(
    connection: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    target_type: str = "",
    target_id: str = "",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    comment: str = "",
) -> int:
    """写一条统一审计日志，返回自增 id。"""

    cursor = connection.execute(
        "INSERT INTO audit_log(actor, action, target_type, target_id, before_state, after_state, "
        "comment, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            actor,
            action,
            target_type,
            target_id,
            _json_dumps(before) if before is not None else None,
            _json_dumps(after) if after is not None else None,
            comment,
            utc_now_iso(),
        ),
    )
    return int(cursor.lastrowid)


def insert_knowledge_base(
    connection: sqlite3.Connection,
    *,
    kb_id: str,
    name: str,
    domain: str,
    allowed_roles: list[str],
) -> None:
    connection.execute(
        "INSERT INTO knowledge_bases(kb_id, name, domain, allowed_roles, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (kb_id, name, domain, _json_dumps(allowed_roles), utc_now_iso()),
    )


def insert_document(
    connection: sqlite3.Connection,
    *,
    doc_id: str,
    kb_id: str,
    source_path: str,
    title: str,
) -> None:
    now = utc_now_iso()
    connection.execute(
        "INSERT INTO documents(doc_id, kb_id, source_path, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, kb_id, source_path, title, now, now),
    )


def insert_version(
    connection: sqlite3.Connection,
    *,
    version_id: str,
    doc_id: str,
    version_no: int,
    content_md: str,
    content_hash: str,
    status: str = "draft",
    audience: str = "driver",
    change_note: str = "",
    created_by: str = "system",
    effective_from: str | None = None,
    effective_to: str | None = None,
) -> None:
    now = utc_now_iso()
    connection.execute(
        "INSERT INTO doc_versions(version_id, doc_id, version_no, content_md, content_hash, "
        "status, audience, effective_from, effective_to, change_note, created_by, created_at, "
        "status_updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            version_id,
            doc_id,
            version_no,
            content_md,
            content_hash,
            status,
            audience,
            effective_from,
            effective_to,
            change_note,
            created_by,
            now,
            now,
        ),
    )


def update_version_status(
    connection: sqlite3.Connection,
    *,
    version_id: str,
    status: str,
) -> None:
    """更新版本生命周期状态（状态机合法性由服务层 lifecycle 保证）。"""

    connection.execute(
        "UPDATE doc_versions SET status = ?, status_updated_at = ? WHERE version_id = ?",
        (status, utc_now_iso(), version_id),
    )


def insert_review_event(
    connection: sqlite3.Connection,
    *,
    version_id: str,
    action: str,
    actor: str,
    comment: str = "",
) -> None:
    connection.execute(
        "INSERT INTO review_events(version_id, action, actor, comment, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (version_id, action, actor, comment, utc_now_iso()),
    )


def insert_release(
    connection: sqlite3.Connection,
    *,
    release_id: str,
    release_no: int,
    label: str = "",
    status: str = "candidate",
    gray_whitelist: list[str] | None = None,
    gray_percent: int = 0,
    created_by: str = "admin",
    activated_at: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO releases(release_id, release_no, label, status, gray_whitelist, gray_percent, "
        "created_by, created_at, activated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            release_id,
            release_no,
            label,
            status,
            _json_dumps(gray_whitelist or []),
            gray_percent,
            created_by,
            utc_now_iso(),
            activated_at or utc_now_iso(),
        ),
    )


def attach_release_item(
    connection: sqlite3.Connection,
    *,
    release_id: str,
    version_id: str,
    content_hash: str,
) -> None:
    connection.execute(
        "INSERT INTO release_items(release_id, version_id, content_hash) VALUES (?, ?, ?)",
        (release_id, version_id, content_hash),
    )


def count_rows(connection: sqlite3.Connection, table: str) -> int:
    """表层计数（仅允许白名单表名，防止 SQL 注入）。"""

    allowed = {
        "knowledge_bases",
        "documents",
        "doc_versions",
        "review_events",
        "releases",
        "release_items",
        "audit_log",
    }
    if table not in allowed:
        raise ValueError(f"unsupported table: {table}")
    row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608
    return int(row["n"])


# —— Task 4 起：生命周期查询 ——

_VERSION_SELECT = (
    "SELECT v.version_id, v.doc_id, v.version_no, v.content_md, v.content_hash, v.status, "
    "v.audience, v.effective_from, v.effective_to, v.change_note, v.created_by, "
    "v.created_at, v.status_updated_at, d.title, d.source_path, d.kb_id "
    "FROM doc_versions v JOIN documents d ON d.doc_id = v.doc_id"
)


def get_knowledge_base_by_code(
    connection: sqlite3.Connection, kb_code: str
) -> sqlite3.Row | None:
    """按知识库业务码（domain/kb 字段，如 refund）或 kb_id 查知识库。"""

    return connection.execute(
        "SELECT * FROM knowledge_bases WHERE kb_id = ? OR domain = ? LIMIT 1",
        (f"kb-{kb_code}", kb_code),
    ).fetchone()


def get_document_by_source_path(
    connection: sqlite3.Connection, source_path: str
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM documents WHERE source_path = ?", (source_path,)
    ).fetchone()


def get_latest_version(connection: sqlite3.Connection, doc_id: str) -> sqlite3.Row | None:
    """取文档 version_no 最大的版本（无论状态）。"""

    return connection.execute(
        "SELECT * FROM doc_versions WHERE doc_id = ? ORDER BY version_no DESC LIMIT 1",
        (doc_id,),
    ).fetchone()


def get_version_by_hash(
    connection: sqlite3.Connection, doc_id: str, content_hash: str
) -> sqlite3.Row | None:
    """按内容哈希查该文档是否已存在相同版本（sync 去重判据，Task 4）。

    注意不能只比 latest：最新版本可能是 draft/灰度变体，
    而磁盘 md 与线上 stable 原文一致时应判 unchanged，而不是再建草稿。
    """

    return connection.execute(
        "SELECT * FROM doc_versions WHERE doc_id = ? AND content_hash = ? LIMIT 1",
        (doc_id, content_hash),
    ).fetchone()


def get_version(connection: sqlite3.Connection, version_id: str) -> sqlite3.Row | None:
    return connection.execute(
        f"{_VERSION_SELECT} WHERE v.version_id = ?", (version_id,)
    ).fetchone()


def list_versions(
    connection: sqlite3.Connection, *, status: str | None = None
) -> list[sqlite3.Row]:
    """列出版本（可按状态过滤），按文档、版本号排序。"""

    if status:
        rows = connection.execute(
            f"{_VERSION_SELECT} WHERE v.status = ? ORDER BY d.source_path, v.version_no",
            (status,),
        ).fetchall()
    else:
        rows = connection.execute(
            f"{_VERSION_SELECT} ORDER BY d.source_path, v.version_no"
        ).fetchall()
    return list(rows)


def get_live_version_for_doc(
    connection: sqlite3.Connection, doc_id: str
) -> sqlite3.Row | None:
    """取当前 stable release 中该文档的版本（diff 的线上基线）。

    candidate 灰度版不算「线上基线」：草稿对比始终面向全量稳定版。
    """

    return connection.execute(
        "SELECT v.* FROM release_items ri "
        "JOIN releases r ON r.release_id = ri.release_id "
        "JOIN doc_versions v ON v.version_id = ri.version_id "
        "WHERE r.status = 'stable' AND v.doc_id = ? "
        "ORDER BY r.release_no DESC LIMIT 1",
        (doc_id,),
    ).fetchone()


def list_audit_events(
    connection: sqlite3.Connection, *, limit: int = 100
) -> list[sqlite3.Row]:
    rows = connection.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return list(rows)


# —— Task 6 起：发布/灰度/回滚 ——

def max_release_no(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT MAX(release_no) AS m FROM releases").fetchone()
    return int(row["m"] or 0)


def list_releases(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute("SELECT * FROM releases ORDER BY release_no DESC").fetchall()
    )


def get_release(connection: sqlite3.Connection, release_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM releases WHERE release_id = ?", (release_id,)
    ).fetchone()


def get_release_by_no(connection: sqlite3.Connection, release_no: int) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM releases WHERE release_no = ?", (release_no,)
    ).fetchone()


def list_release_version_ids(connection: sqlite3.Connection, release_id: str) -> list[str]:
    rows = connection.execute(
        "SELECT version_id FROM release_items WHERE release_id = ? ORDER BY version_id",
        (release_id,),
    ).fetchall()
    return [row["version_id"] for row in rows]


def list_active_releases(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """当前仍有效的 stable/candidate 发布（发布新 stable 时需要作废旧版）。"""

    return list(
        connection.execute(
            "SELECT * FROM releases WHERE status IN ('stable','candidate') ORDER BY release_no"
        ).fetchall()
    )


def mark_release_rolled_back(
    connection: sqlite3.Connection,
    *,
    release_id: str,
    reason: str,
) -> None:
    """把一个发布置为 rolled_back（回滚或被新 stable 取代时使用）。"""

    connection.execute(
        "UPDATE releases SET status = 'rolled_back', rolled_back_at = ?, "
        "rollback_reason = ?, gray_whitelist = '[]', gray_percent = 0 WHERE release_id = ?",
        (utc_now_iso(), reason, release_id),
    )
