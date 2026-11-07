"""Task 4 验证：md 同步草稿（TR-4.1）与差异对比/API 鉴权（TR-4.2）。"""

from __future__ import annotations

import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from api.schemas import DocVersionView
from config import settings
from store import lifecycle, repository as repo
from store.db import transaction
from store.seed import seed


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    """临时 db + 临时知识目录（复制真实 md），不碰真实库。"""

    db_path = tmp_path / "task4.db"
    seed(db_path)
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    for md in settings.KNOWLEDGE_DIR.glob("*.md"):
        shutil.copy(md, kb_dir / md.name)
    # 服务层动态读 settings（见 lifecycle.py 顶部注释），只需 patch settings 一处。
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "KNOWLEDGE_DIR", kb_dir)
    return db_path, kb_dir


def test_tr4_1_sync_no_change_creates_nothing(isolated_env):
    db_path, _ = isolated_env
    result = lifecycle.sync_knowledge_dir(db_path=db_path)
    assert result.scanned == 5
    assert result.changed == 0
    assert result.unchanged == 5
    assert result.created_version_ids == []


def test_tr4_1_sync_changed_md_creates_draft_only(isolated_env):
    db_path, kb_dir = isolated_env

    # 模拟运营改了 usage md 的一行（追加新条款）。
    usage_path = kb_dir / "coupon-usage.md"
    original = usage_path.read_text(encoding="utf-8")
    usage_path.write_text(
        original.rstrip() + "\n- 新增条款 TRIAL-T4：时段卡跨月规则试运行。\n",
        encoding="utf-8",
    )

    result = lifecycle.sync_knowledge_dir(actor="admin-t4", db_path=db_path)
    assert result.changed == 1
    assert result.unchanged == 4
    draft_id = result.created_version_ids[0]
    assert draft_id == "doc-coupon-usage-v2"

    # 线上 stable 基线仍是旧快照：draft 不影响线上。
    with transaction(db_path) as connection:
        draft = repo.get_version(connection, draft_id)
        live = repo.get_live_version_for_doc(connection, "doc-coupon-usage")
        assert draft["status"] == "draft"
        assert draft["content_hash"] != live["content_hash"]
        assert "TRIAL-T4" in draft["content_md"]
        assert "TRIAL-T4" not in live["content_md"]
        # 变更写了审计。
        actions = {row["action"] for row in repo.list_audit_events(connection, limit=50)}
        assert "draft.created" in actions


def test_tr4_2_diff_reports_added_line(isolated_env):
    db_path, kb_dir = isolated_env
    purchase_path = kb_dir / "coupon-purchase.md"
    purchase_path.write_text(
        purchase_path.read_text(encoding="utf-8").rstrip()
        + "\n- 差异测试行 DIFF-T4：仅存在于草稿。\n",
        encoding="utf-8",
    )
    result = lifecycle.sync_knowledge_dir(db_path=db_path)
    diff = lifecycle.diff_against_live(result.created_version_ids[0], db_path=db_path)

    assert diff["baseline_version_id"] == "doc-coupon-purchase-v1"
    assert any("DIFF-T4" in line for line in diff["added_lines"])
    assert diff["removed_lines"] == []


def test_tr4_admin_api_401_and_happy_path(isolated_env):
    app = FastAPI()
    app.include_router(create_admin_router())
    client = TestClient(app)

    # 无 token → 401；错 token → 401。
    assert client.post("/admin/kb/sync").status_code == 401
    assert client.post(
        "/admin/kb/sync", headers={"X-Admin-Token": "wrong-token"}
    ).status_code == 401

    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}
    # 无变更同步：changed=0。
    response = client.post("/admin/kb/sync", headers=headers)
    assert response.status_code == 200
    assert response.json()["changed"] == 0

    # 种子库自带 1 个 draft（refund v2）。
    drafts = client.get("/admin/versions?status=draft", headers=headers)
    assert drafts.status_code == 200
    draft_ids = {item["version_id"] for item in drafts.json()}
    assert "doc-coupon-refund-v2" in draft_ids

    # diff 端点能返回结构。
    diff_resp = client.get("/admin/versions/doc-coupon-refund-v2/diff", headers=headers)
    assert diff_resp.status_code == 200
    assert "added_lines" in diff_resp.json()
    assert "DRAFT-X9" in "\n".join(diff_resp.json()["added_lines"])

    # 不存在的版本 → 404。
    assert client.get("/admin/versions/nope/diff", headers=headers).status_code == 404
