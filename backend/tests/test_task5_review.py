"""Task 5 验证：审核状态机、422/409/401 与双写留痕（TR-5.1/TR-5.2）。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from config import settings
from store import repository as repo
from store.db import transaction
from store.seed import seed

DRAFT_VERSION = "doc-coupon-refund-v2"  # 种子中自带的草稿


@pytest.fixture()
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "task5.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app), db_path


def _status(db_path, version_id: str) -> str:
    with transaction(db_path) as connection:
        return repo.get_version(connection, version_id)["status"]


def test_tr5_1_reject_without_comment_422_and_state_unchanged(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 先送审。
    resp = client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers,
        json={"action": "submit", "actor": "admin-a"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "submitted"

    # 驳回不填理由 → 422，状态必须仍是 submitted。
    bad = client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers,
        json={"action": "reject", "comment": "  ", "actor": "admin-a"},
    )
    assert bad.status_code == 422
    assert _status(db_path, DRAFT_VERSION) == "submitted"


def test_tr5_1_illegal_transitions_409(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # draft 状态直接 approve → 409。
    resp = client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers,
        json={"action": "approve", "comment": "ok"},
    )
    assert resp.status_code == 409
    assert _status(db_path, DRAFT_VERSION) == "draft"

    # 已发布版本 approve → 409。
    published = client.post(
        "/admin/versions/doc-coupon-refund-v1/review",
        headers=headers,
        json={"action": "approve", "comment": "ok"},
    )
    assert published.status_code == 409

    # 不存在的版本 → 404。
    missing = client.post(
        "/admin/versions/no-such-version/review",
        headers=headers,
        json={"action": "submit"},
    )
    assert missing.status_code == 404


def test_tr5_1_review_requires_token(client_db):
    client, _ = client_db
    resp = client.post(
        f"/admin/versions/{DRAFT_VERSION}/review", json={"action": "submit"}
    )
    assert resp.status_code == 401


def test_tr5_2_happy_path_events_and_audit(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 完整路径：draft → submit → reject(带理由) → draft → submit → approve。
    assert client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers, json={"action": "submit", "actor": "admin-a"},
    ).status_code == 200
    assert client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers, json={"action": "reject", "actor": "admin-a", "comment": "缺依据，补退款条款"},
    ).status_code == 200
    assert _status(db_path, DRAFT_VERSION) == "draft"
    assert client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers, json={"action": "submit", "actor": "admin-a"},
    ).status_code == 200
    approved = client.post(
        f"/admin/versions/{DRAFT_VERSION}/review",
        headers=headers, json={"action": "approve", "actor": "admin-b", "comment": "依据齐全"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    with transaction(db_path) as connection:
        events = connection.execute(
            "SELECT action, actor, comment, created_at FROM review_events "
            "WHERE version_id = ? ORDER BY id",
            (DRAFT_VERSION,),
        ).fetchall()
        actions = [row["action"] for row in events]
        assert actions == ["submit", "reject", "submit", "approve"]
        # actor/comment/时间戳均可查。
        reject_row = next(row for row in events if row["action"] == "reject")
        assert reject_row["actor"] == "admin-a"
        assert reject_row["comment"] == "缺依据，补退款条款"
        assert reject_row["created_at"]
        approve_row = next(row for row in events if row["action"] == "approve")
        assert approve_row["comment"] == "依据齐全"

        # audit_log 双写：review.submit/reject/approve 三类都在。
        audit_actions = {
            row["action"]
            for row in connection.execute(
                "SELECT action FROM audit_log WHERE target_id = ?", (DRAFT_VERSION,)
            ).fetchall()
        }
        assert {"review.submit", "review.reject", "review.approve"} <= audit_actions
