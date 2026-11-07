"""Task 7 验证：灰度白名单/哈希桶路由与恒定性（TR-7.1）、配置即时生效与审计（TR-7.2）。"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from api.schemas import GrayConfigRequest
from config import settings
from store import releases
from store import repository as repo
from store.db import transaction
from store.seed import seed


@pytest.fixture()
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "task7.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app), db_path


def _find_bucket_ids(db_path, target: str) -> str:
    """在探测 ID 空间里找一个桶内(<20)/桶外(>=20)的 driver_id。"""

    for i in range(500):
        candidate = f"driver-probe-{i:03d}"
        bucket = releases.gray_bucket(candidate)
        if target == "in" and bucket < 20:
            return candidate
        if target == "out" and bucket >= 20:
            return candidate
    raise AssertionError(f"no {target}-bucket probe id found")


def test_tr7_1_whitelist_bucket_and_stable_routing(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}

    in_id = _find_bucket_ids(db_path, "in")
    out_id = _find_bucket_ids(db_path, "out")

    # 白名单：直接命中 candidate。
    resp_wl = client.get("/admin/gray/route", headers=headers, params={"driver_id": "driver-gray-whitelist"})
    assert resp_wl.json()["channel"] == "candidate"
    assert resp_wl.json()["reason"] == "whitelist"
    assert resp_wl.json()["release_id"] == "rel-0002"

    # 哈希桶内：candidate/percent。
    resp_in = client.get("/admin/gray/route", headers=headers, params={"driver_id": in_id})
    assert resp_in.json()["channel"] == "candidate"
    assert resp_in.json()["reason"] == "percent"
    assert resp_in.json()["gray_percent"] == 20

    # 桶外：stable。
    resp_out = client.get("/admin/gray/route", headers=headers, params={"driver_id": out_id})
    assert resp_out.json()["channel"] == "stable"
    assert resp_out.json()["reason"] == "stable"
    assert resp_out.json()["release_id"] == "rel-0001"

    # 恒定性：同一桶内 ID 服务层解析 20 次全部一致。
    routes = {releases.resolve_gray_route(in_id, db_path).channel for _ in range(20)}
    assert routes == {"candidate"}


def test_tr7_2_config_change_takes_effect_and_audited(client_db):
    client, db_path = client_db
    headers = {"X-Admin-Token": settings.ADMIN_TOKEN}
    in_id = _find_bucket_ids(db_path, "in")
    out_id = _find_bucket_ids(db_path, "out")

    # 灰度百分比调到 0：桶内司机立刻回到 stable。
    resp = client.post(
        "/admin/releases/rel-0002/gray-config",
        headers=headers,
        json={"gray_percent": 0},
    )
    assert resp.status_code == 200
    assert releases.resolve_gray_route(in_id, db_path).channel == "stable"

    # 把桶外司机加进白名单：立即命中 candidate。
    resp2 = client.post(
        "/admin/releases/rel-0002/gray-config",
        headers=headers,
        json={"gray_whitelist": ["driver-gray-whitelist", out_id], "gray_percent": 20},
    )
    assert resp2.status_code == 200
    route_out = releases.resolve_gray_route(out_id, db_path)
    assert route_out.channel == "candidate"
    assert route_out.reason == "whitelist"

    # stable 发布不允许配灰度 → 409；不存在的 release → 404。
    assert client.post(
        "/admin/releases/rel-0001/gray-config", headers=headers, json={"gray_percent": 10}
    ).status_code == 409
    assert client.post(
        "/admin/releases/rel-9999/gray-config", headers=headers, json={"gray_percent": 10}
    ).status_code == 404

    # 审计含灰度配置事件且带 before/after。
    with transaction(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM audit_log WHERE action='release.gray_configured' ORDER BY id"
        ).fetchall()
        assert len(rows) >= 2
        latest = rows[-1]
        assert latest["target_id"] == "rel-0002"
        assert out_id in json.loads(latest["after_state"])["gray_whitelist"]

    # 服务层入参等价校验（GrayConfigRequest None 表示不改该项）。
    view = releases.update_gray_config(
        "rel-0002", GrayConfigRequest(gray_percent=50), db_path=db_path
    )
    assert view.gray_percent == 50
    assert out_id in view.gray_whitelist  # 未传白名单，保持上一次配置
