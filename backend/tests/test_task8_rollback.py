"""Task 8 验证：回滚 API、缓存联动与审计留痕（TR-8.1 / TR-8.2）。

TR-8.1：回滚后 candidate 失效、全体司机走 stable、回滚触发索引/缓存清空；
TR-8.2：reason 为空返回 422；审计含回滚事件与原因。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from api.schemas import RollbackRequest
from config import settings
from rag import index_cache
from store import releases
from store import repository as repo
from store.db import transaction
from store.seed import seed


@pytest.fixture()
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "task8.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    # 测试用独立 app：手动注册缓存失效监听器（main.create_app 的等价动作）。
    releases._release_change_listeners.clear()
    releases.add_release_change_listener(index_cache.on_release_change)
    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app), db_path


def _review(client, headers, version_id, action, **kwargs):
    return client.post(
        f"/admin/versions/{version_id}/review",
        headers=headers,
        json={"action": action, **kwargs},
    )


def test_tr8_1_rollback_candidate_and_cache_invalidation(client_db):
    """回滚 candidate：灰度失效、全体司机走 stable、索引与缓存清空。"""

    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 回滚前：白名单司机命中 candidate。
    before = client.get(
        "/admin/gray/route", headers=headers, params={"driver_id": "driver-gray-whitelist"}
    )
    assert before.json()["channel"] == "candidate"
    assert before.json()["release_id"] == "rel-0002"

    # 预置一个缓存条目 + 索引，验证回滚后被清空。
    index_cache.KNOWLEDGE_INDEX = index_cache.build_knowledge_index([])
    index_cache.RAG_RETRIEVAL_CACHE["stale-key"] = "stale-entry"  # type: ignore[assignment]

    # 回滚 candidate（rel-0002）。
    resp = client.post(
        "/admin/releases/rel-0002/rollback",
        headers=headers,
        json={"reason": "灰度效果不达预期，撤回"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["release_id"] == "rel-0002"
    assert body["status"] == "rolled_back"

    # candidate 失效：白名单司机也走 stable。
    after = client.get(
        "/admin/gray/route", headers=headers, params={"driver_id": "driver-gray-whitelist"}
    )
    assert after.json()["channel"] == "stable"
    assert after.json()["release_id"] == "rel-0001"
    assert after.json()["reason"] == "none"

    # 缓存联动：索引与检索缓存被清空（回滚后首请求 cache_hit=False 的基础）。
    assert index_cache.KNOWLEDGE_INDEX is None
    assert len(index_cache.RAG_RETRIEVAL_CACHE) == 0


def test_tr8_1_rollback_to_historical_snapshot(client_db):
    """回滚到历史 rolled_back 快照：完整性校验通过后恢复为 stable，旧 stable 作废。"""

    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 先发布一个新 stable（rel-0003），让 rel-0001 变成 rolled_back。
    _review(client, headers, "doc-coupon-refund-v2", "submit")
    _review(client, headers, "doc-coupon-refund-v2", "approve", comment="ok")
    publish = client.post(
        "/admin/releases",
        headers=headers,
        json={"version_ids": ["doc-coupon-refund-v2"], "as_candidate": False},
    )
    assert publish.status_code == 200
    with transaction(db_path) as connection:
        rel1 = repo.get_release(connection, "rel-0001")
        rel3 = repo.get_release(connection, "rel-0003")
    assert rel1["status"] == "rolled_back"
    assert rel3["status"] == "stable"

    # 回滚 rel-0001（历史快照）：恢复为 stable，rel-0003 被作废。
    resp = client.post(
        "/admin/releases/rel-0001/rollback",
        headers=headers,
        json={"reason": "冷静期引发客诉，回滚到初始口径"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["release_id"] == "rel-0001"
    assert body["status"] == "stable"

    with transaction(db_path) as connection:
        rel1 = repo.get_release(connection, "rel-0001")
        rel3 = repo.get_release(connection, "rel-0003")
    assert rel1["status"] == "stable"
    assert rel3["status"] == "rolled_back"
    # 灰度配置随回滚清零。
    assert rel1["gray_percent"] == 0
    assert rel1["gray_whitelist"] == "[]"


def test_tr8_1_reject_rollback_current_stable_and_missing(client_db):
    """回滚当前 stable → 409；不存在的 release → 404。"""

    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # 当前 stable 不能回滚（需明确指定历史快照）。
    stable_resp = client.post(
        "/admin/releases/rel-0001/rollback",
        headers=headers,
        json={"reason": "尝试回滚当前 stable"},
    )
    assert stable_resp.status_code == 409

    # 不存在的 release → 404。
    missing_resp = client.post(
        "/admin/releases/rel-9999/rollback",
        headers=headers,
        json={"reason": "不存在"},
    )
    assert missing_resp.status_code == 404


def test_tr8_2_reason_required_422_and_audit_trail(client_db):
    """reason 为空/缺失 → 422；回滚成功后审计含事件与原因。"""

    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    # reason 缺失 → 422（pydantic 必填校验）。
    assert client.post(
        "/admin/releases/rel-0002/rollback", headers=headers, json={}
    ).status_code == 422

    # reason 空白字符串 → 422（RollbackRequest model_validator）。
    assert client.post(
        "/admin/releases/rel-0002/rollback", headers=headers, json={"reason": "   "}
    ).status_code == 422

    # 成功回滚 candidate。
    ok = client.post(
        "/admin/releases/rel-0002/rollback",
        headers=headers,
        json={"reason": "灰度效果不达预期，撤回"},
    )
    assert ok.status_code == 200

    # 审计：release.rolled_back 事件含目标 release 与回滚原因。
    with transaction(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM audit_log WHERE action='release.rolled_back' ORDER BY id"
        ).fetchall()
        assert len(rows) >= 1
        latest = rows[-1]
        assert latest["target_id"] == "rel-0002"
        assert latest["comment"] == "灰度效果不达预期，撤回"
        after = __import__("json").loads(latest["after_state"])
        assert after["status"] == "rolled_back"
        assert after["kind"] == "candidate_discarded"


def test_tr8_2_service_layer_rollback_request_validation():
    """服务层 RollbackRequest 校验：空白 reason 抛 ValueError。"""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RollbackRequest(reason="")
    with pytest.raises(ValidationError):
        RollbackRequest(reason="   ")
    # 正常构造不报错。
    req = RollbackRequest(reason="合法回滚理由")
    assert req.reason == "合法回滚理由"
