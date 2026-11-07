"""Task 1 验证：SQLite 存储层幂等建表（TR-1.1）与外键约束（TR-1.2）。"""

from __future__ import annotations

import sqlite3

import pytest

from store.db import init_db, list_tables, transaction, utc_now_iso

EXPECTED_TABLES = {
    "knowledge_bases",
    "documents",
    "doc_versions",
    "review_events",
    "releases",
    "release_items",
    "audit_log",
}


def test_tr1_1_init_idempotent_and_seven_tables(tmp_path):
    db_path = tmp_path / "task1.db"

    returned_path_1 = init_db(db_path)
    # 全新库第一次初始化后 7 张表必须齐全。
    assert set(list_tables(returned_path_1)) == EXPECTED_TABLES

    # 重复执行不得报错、不得丢表（幂等）。
    returned_path_2 = init_db(db_path)
    assert set(list_tables(returned_path_2)) == EXPECTED_TABLES


def _seed_minimal_chain(connection: sqlite3.Connection) -> None:
    """插入 知识库→文档→版本→审核事件→发布→发布明细 的最小完整链路。"""

    now = utc_now_iso()
    connection.execute(
        "INSERT INTO knowledge_bases(kb_id, name, domain, allowed_roles, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("kb-refund", "退款知识库", "refund", '["driver","agent","admin"]', now),
    )
    connection.execute(
        "INSERT INTO documents(doc_id, kb_id, source_path, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("doc-1", "kb-refund", "coupon-refund.md", "卡券退款规则", now, now),
    )
    connection.execute(
        "INSERT INTO doc_versions(version_id, doc_id, version_no, content_md, content_hash, "
        "status, audience, change_note, created_by, created_at, status_updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ver-1", "doc-1", 1, "# x", "hash-aaa", "published", "driver", "", "system", now, now),
    )
    connection.execute(
        "INSERT INTO review_events(version_id, action, actor, comment, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ver-1", "approve", "admin", "ok", now),
    )
    connection.execute(
        "INSERT INTO releases(release_id, release_no, label, status, gray_whitelist, "
        "gray_percent, created_by, created_at, activated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("rel-1", 1, "v1", "stable", "[]", 0, "admin", now, now),
    )
    connection.execute(
        "INSERT INTO release_items(release_id, version_id, content_hash) VALUES (?, ?, ?)",
        ("rel-1", "ver-1", "hash-aaa"),
    )


def test_tr1_2_full_chain_and_dangling_release_item_rejected(tmp_path):
    db_path = tmp_path / "task1_fk.db"
    init_db(db_path)

    # 完整链路插入成功。
    with transaction(db_path) as connection:
        _seed_minimal_chain(connection)
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM release_items WHERE release_id='rel-1' AND version_id='ver-1'"
        ).fetchone()
        assert row["n"] == 1

    # 悬空 release_item（version_id 不存在）必须被外键拒绝。
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db_path) as connection:
            connection.execute(
                "INSERT INTO release_items(release_id, version_id, content_hash) VALUES (?, ?, ?)",
                ("rel-1", "ver-not-exist", "hash-x"),
            )

    # 悬空 document（kb_id 不存在）同样必须被拒绝。
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db_path) as connection:
            now = utc_now_iso()
            connection.execute(
                "INSERT INTO documents(doc_id, kb_id, source_path, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("doc-bad", "kb-not-exist", "x.md", "x", now, now),
            )
