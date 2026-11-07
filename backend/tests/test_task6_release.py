"""Task 6 验证：发布门禁/快照与 stable 切换（TR-6.1）、旧快照不可变（TR-6.2）。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from config import settings
from store import repository as repo
from store.db import transaction
from store.seed import seed


@pytest.fixture()
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "task6.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app), db_path


def _release_map(db_path):
    with transaction(db_path) as connection:
        return {row["release_id"]: dict(row) for row in repo.list_releases(connection)}


def _review(client, headers, version_id, action, **kwargs):
    return client.post(
        f"/admin/versions/{version_id}/review",
        headers=headers,
        json={"action": action, **kwargs},
    )


def test_tr6_1_publish_guard_and_stable_cutover(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # draft 未审核直接发布 → 409。
    draft_publish = client.post(
        "/admin/releases",
        headers=headers,
        json={"version_ids": ["doc-coupon-refund-v2"], "as_candidate": False},
    )
    assert draft_publish.status_code == 409

    # 空版本集合 → 409。
    assert client.post(
        "/admin/releases", headers=headers, json={"version_ids": [], "as_candidate": True}
    ).status_code == 409

    # 走完整审核：draft → submitted → approved。
    assert _review(client, headers, "doc-coupon-refund-v2", "submit").status_code == 200
    assert _review(
        client, headers, "doc-coupon-refund-v2", "approve", comment="审核通过"
    ).status_code == 200

    # 种子已带 candidate rel-0002：此时再发 candidate 必须 409。
    conflict = client.post(
        "/admin/releases",
        headers=headers,
        json={"version_ids": ["doc-coupon-refund-v2"], "as_candidate": True},
    )
    assert conflict.status_code == 409
    assert "灰度" in conflict.json()["detail"]

    # 已有 candidate 时直接发 stable：允许（全量切版会同时终止灰度）。
    response = client.post(
        "/admin/releases",
        headers=headers,
        json={
            "version_ids": ["doc-coupon-refund-v2"],
            "label": "冷静期全量版",
            "as_candidate": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["release"]["release_no"] == 3
    assert body["release"]["status"] == "stable"
    # Task 6 门禁为 placeholder，Task 16 替换为 quality_eval。
    assert body["gate"]["passed"] is True
    assert body["gate"]["mode"] == "placeholder"

    releases_map = _release_map(db_path)
    assert releases_map["rel-0003"]["status"] == "stable"
    # 旧 stable 与旧 candidate 都被 superseded 作废。
    assert releases_map["rel-0001"]["status"] == "rolled_back"
    assert "superseded by rel-0003" in releases_map["rel-0001"]["rollback_reason"]
    assert releases_map["rel-0002"]["status"] == "rolled_back"

    # 已发布版本再次发布 → 409（状态已是 published）。
    again = client.post(
        "/admin/releases",
        headers=headers,
        json={"version_ids": ["doc-coupon-refund-v2"], "as_candidate": False},
    )
    assert again.status_code == 409

    # 审计含发布事件。
    with transaction(db_path) as connection:
        actions = {row["action"] for row in repo.list_audit_events(connection, limit=50)}
        assert "release.published" in actions


def test_tr6_2_old_snapshot_hashes_immutable(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 记录发布前 rel-0001 全部明细哈希。
    with transaction(db_path) as connection:
        before_items = {
            row["version_id"]: row["content_hash"]
            for row in connection.execute(
                "SELECT version_id, content_hash FROM release_items WHERE release_id='rel-0001'"
            ).fetchall()
        }
        assert len(before_items) == 5

    # 审批并发新 stable（同 test_tr6_1 的流程）。
    _review(client, headers, "doc-coupon-refund-v2", "submit")
    _review(client, headers, "doc-coupon-refund-v2", "approve", comment="ok")
    assert client.post(
        "/admin/releases",
        headers=headers,
        json={"version_ids": ["doc-coupon-refund-v2"], "as_candidate": False},
    ).status_code == 200

    with transaction(db_path) as connection:
        after_items = {
            row["version_id"]: row["content_hash"]
            for row in connection.execute(
                "SELECT version_id, content_hash FROM release_items WHERE release_id='rel-0001'"
            ).fetchall()
        }
        # 旧快照明细哈希逐项不变。
        assert before_items == after_items
        # 冗余哈希与版本行内容哈希一致（回滚校验的基础）。
        for version_id, item_hash in after_items.items():
            version_hash = connection.execute(
                "SELECT content_hash FROM doc_versions WHERE version_id=?", (version_id,)
            ).fetchone()["content_hash"]
            assert item_hash == version_hash
