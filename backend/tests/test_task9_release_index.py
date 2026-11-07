"""Task 9 验证：按 release 构建生产索引与聊天链路灰度路由（TR-9.1 / TR-9.2）。

TR-9.1：candidate/stable 两类司机同问得到不同知识版本，索引 version 各自带 release 前缀；
TR-9.2：lifecycle 开关关闭时走 L16 全局单索引，release 路由信息为 None。
"""

from __future__ import annotations

import pytest

from api.schemas import ChatRequest
from config import settings
from rag import index_cache
from store.seed import seed


@pytest.fixture()
def db_with_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "task9.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    # 清空模块级索引/缓存，避免测试间污染。
    index_cache.reset_index_and_cache()
    yield db_path
    index_cache.reset_index_and_cache()


def _patch_embeddings(monkeypatch):
    """mock embedding 调用，返回固定 1024 维向量，避免真实模型加载。"""

    fake_vec = [0.1] * 1024
    monkeypatch.setattr("rag.vector_store.embed_text", lambda *a, **kw: fake_vec)
    monkeypatch.setattr(
        "rag.vector_store.embed_texts",
        lambda texts, *a, **kw: [fake_vec for _ in texts],
    )
    monkeypatch.setattr("rag.quality.peek_quality_summary", lambda: None)


def test_tr9_1_release_indexes_differ_by_content(db_with_seed):
    """candidate/stable 两个 release 的索引内容不同（灰度独句 GRAY-X2 只在 candidate）。"""

    stable_idx = index_cache.get_release_index("rel-0001", db_path=db_with_seed)
    candidate_idx = index_cache.get_release_index("rel-0002", db_path=db_with_seed)

    # 索引 version 带 release_id 前缀，检索缓存 key 自然按 release 隔离。
    assert stable_idx.version.startswith("rel-0001:")
    assert candidate_idx.version.startswith("rel-0002:")
    assert stable_idx.version != candidate_idx.version

    # candidate 含 GRAY-X2 灰度独句，stable 不含。
    stable_texts = {chunk.text for chunk in stable_idx.chunks_by_id.values()}
    candidate_texts = {chunk.text for chunk in candidate_idx.chunks_by_id.values()}
    assert any("GRAY-X2" in t for t in candidate_texts)
    assert not any("GRAY-X2" in t for t in stable_texts)


def test_tr9_1_chat_route_differs_by_driver(db_with_seed, monkeypatch):
    """白名单司机走 candidate，普通司机走 stable，session_state 中 release 信息正确。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    _patch_embeddings(monkeypatch)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()

    # 白名单司机 → candidate（rel-0002）。
    req_wl = ChatRequest(
        session_id="s1",
        runtime_user_id="driver-gray-whitelist",
        user_message="周末日卡怎么买",
    )
    resp_wl = agent.chat(req_wl)
    release_wl = resp_wl.session_state["rag"]["index"]["release"]
    assert release_wl["channel"] == "candidate"
    assert release_wl["release_id"] == "rel-0002"
    assert resp_wl.session_state["rag"]["index"]["version"].startswith("rel-0002:")

    # 普通司机 → stable（rel-0001）。
    req_normal = ChatRequest(
        session_id="s2",
        runtime_user_id="driver-normal-001",
        user_message="周末日卡怎么买",
    )
    resp_normal = agent.chat(req_normal)
    release_normal = resp_normal.session_state["rag"]["index"]["release"]
    assert release_normal["channel"] == "stable"
    assert release_normal["release_id"] == "rel-0001"
    assert resp_normal.session_state["rag"]["index"]["version"].startswith("rel-0001:")

    # 两类司机索引 version 不同（灰度内容隔离）。
    assert resp_wl.session_state["rag"]["index"]["version"] != resp_normal.session_state["rag"]["index"]["version"]


def test_tr9_2_lifecycle_off_uses_global_index(db_with_seed, monkeypatch):
    """开关关闭时走 L16 全局单索引：version 不带 release 前缀，release 路由为 None。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", False)
    _patch_embeddings(monkeypatch)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()

    # 即使是白名单司机，开关关闭时也走全局单索引。
    req = ChatRequest(
        session_id="s3",
        runtime_user_id="driver-gray-whitelist",
        user_message="周末日卡怎么买",
    )
    resp = agent.chat(req)
    idx_info = resp.session_state["rag"]["index"]
    assert not idx_info["version"].startswith("rel-")
    assert idx_info["release"] is None
