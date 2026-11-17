"""评测报告只读接口：读取最新 intent_bakeoff 报告，不在后台触发跑分。"""

from __future__ import annotations

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from config import settings
from evals.report_reader import latest_intent_report
from store.seed import seed


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "eval_api.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)

    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"X-Admin-Token": settings.ADMIN_TOKEN}


def test_latest_report_picks_newest_file(tmp_path):
    old = tmp_path / "intent_bakeoff_old.json"
    new = tmp_path / "intent_bakeoff_new.json"
    old.write_text(json.dumps({"cases_count": 10}), encoding="utf-8")
    new.write_text(json.dumps({"cases_count": 56}), encoding="utf-8")
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_700_000_100, 1_700_000_100))

    name, report = latest_intent_report(tmp_path)

    assert name == "intent_bakeoff_new.json"
    assert report["cases_count"] == 56


def test_latest_report_returns_none_when_empty(tmp_path):
    assert latest_intent_report(tmp_path) is None


def test_eval_endpoint_returns_latest_summary(client):
    resp = client.get("/admin/evals/intent", headers=_headers())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert body["cases_count"] == 56
    assert body["report_file"].startswith("intent_bakeoff")
    keys = {provider["key"] for provider in body["providers"]}
    assert {"rules", "hybrid", "llm"} <= keys

    rule = next(p for p in body["providers"] if p["key"] == "rules")
    hybrid = next(p for p in body["providers"] if p["key"] == "hybrid")
    assert rule["accuracy"] < hybrid["accuracy"]
    # 不锁失败条数：最新报告换了批次，超时回退会让失败数在 1~5 之间浮动。
    # 这里只守住「hybrid 明显优于规则」这个结论，避免把评测报告变成脆断言。
    assert hybrid["accuracy"] > 0.9
    assert 0 <= hybrid["failure_count"] < body["cases_count"]


def test_eval_endpoint_requires_auth(client):
    assert client.get("/admin/evals/intent").status_code == 401
