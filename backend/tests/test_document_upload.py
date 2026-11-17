"""后台文档更新（最小版）：在线编辑 / .md / .docx 上传 → 生成 draft。

重点验证：只生成草稿、可追加版本、重复内容拒绝、非法格式拒绝、内容解析正确。
"""

from __future__ import annotations

import io

import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import create_admin_router
from config import settings
from store.seed import seed

ARTICLE = "# 退款时效\n\n未核销卡券可在购买后 7 天内申请退款，已核销卡券需人工审核。"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "kb_upload.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)

    app = FastAPI()
    app.include_router(create_admin_router())
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"X-Admin-Token": settings.ADMIN_TOKEN}


def _create(client: TestClient, **overrides):
    payload = {
        "title": "退款时效说明",
        "kb_code": "refund",
        "content": ARTICLE,
        "audience": "driver",
        "change_note": "补充退款时效",
        **overrides,
    }
    return client.post("/admin/kb/documents", json=payload, headers=_headers())


def test_create_draft_from_editor(client):
    resp = _create(client)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "draft"
    assert body["version_no"] == 1
    assert "未核销卡券" in body["content_md"]
    assert body["created_by"] == "machine"


def test_update_existing_document_appends_next_version(client):
    first = _create(client).json()

    second = _create(client, doc_id=first["doc_id"], content=ARTICLE + "\n\n补充：节假日顺延。")

    assert second.status_code == 200, second.text
    body = second.json()
    assert body["version_no"] == 2
    assert body["doc_id"] == first["doc_id"]


def test_duplicate_content_is_rejected(client):
    first = _create(client).json()

    resp = _create(client, doc_id=first["doc_id"])

    assert resp.status_code == 409
    assert "完全相同" in resp.json()["detail"]


def test_upload_markdown_with_front_matter(client):
    markdown = (
        "---\n"
        "title: 卡券核销说明\n"
        "kb: usage\n"
        "audience: internal\n"
        "---\n\n"
        "核销失败时先检查卡券状态，再提交人工复核。\n"
    )
    resp = client.post(
        "/admin/kb/upload",
        files={"file": ("usage.md", markdown.encode("utf-8"), "text/markdown")},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["title"] == "卡券核销说明"
    assert body["audience"] == "internal"
    assert "核销失败" in body["content_md"]


def test_upload_docx_is_converted_to_markdown(client):
    document = Document()
    document.add_heading("退款时效", level=1)
    document.add_paragraph("未核销卡券可在购买后 7 天内申请退款。")
    buffer = io.BytesIO()
    document.save(buffer)

    resp = client.post(
        "/admin/kb/upload",
        files={
            "file": (
                "refund.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"title": "退款时效说明", "kb_code": "refund"},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    content = resp.json()["content_md"]
    assert "# 退款时效" in content
    assert "未核销卡券可在购买后 7 天内申请退款" in content


def test_pdf_upload_is_rejected(client):
    resp = client.post(
        "/admin/kb/upload",
        files={"file": ("policy.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"title": "PDF 文档", "kb_code": "refund"},
        headers=_headers(),
    )

    assert resp.status_code == 400
    assert "仅支持" in resp.json()["detail"]


def test_upload_without_title_and_kb_is_rejected(client):
    resp = client.post(
        "/admin/kb/upload",
        files={"file": ("plain.md", "退款规则正文，足够长度的内容。".encode("utf-8"), "text/markdown")},
        headers=_headers(),
    )

    assert resp.status_code == 400
