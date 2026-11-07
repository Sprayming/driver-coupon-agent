"""Task 2 验证：契约向后兼容（TR-2.1）与平台开关安全默认（TR-2.2）。"""

from __future__ import annotations

from api.schemas import (
    ChatRequest,
    DocVersionView,
    GrayConfigRequest,
    GrayRoute,
    KnowledgeChunk,
    ReleaseCreateRequest,
    ReviewActionRequest,
    RollbackRequest,
)
from config import settings
from config.settings import env_flag_enabled


def test_tr2_1_legacy_request_defaults_to_driver():
    # 旧前端请求体不带 runtime_role：必须构造成功（不 422）且按 driver 处理。
    request = ChatRequest(
        session_id="s-1",
        runtime_user_id="driver-1001",
        user_message="卡券还没用能退吗",
    )
    assert request.runtime_role == "driver"
    assert request.debug is True


def test_tr2_1_knowledge_chunk_new_fields_have_defaults():
    # L10–L16 旧构造方式（只传老字段）必须继续可用。
    chunk = KnowledgeChunk(
        chunk_id="x-1",
        title="t",
        source_path="x.md",
        section="s",
        keywords=["k"],
        text="body",
    )
    assert chunk.kb_id == ""
    assert chunk.audience == "driver"
    assert chunk.policy_codes == []
    assert chunk.product_names == []
    assert chunk.effective_from is None
    assert chunk.effective_to is None
    # 老字段原样保留。
    assert chunk.effective_status == "active"


def test_tr2_2_platform_switches_default_off():
    # pytest 不加载 course.env；平台开关缺省必须全部为 False（行为等同 L16）。
    assert settings.KNOWLEDGE_LIFECYCLE_ENABLED is False
    assert settings.RBAC_ENABLED is False
    assert settings.BM25_ENABLED is False
    assert settings.RAG_FUSION_ENABLED is False
    assert settings.ADMIN_TOKEN == "dev-admin-token"
    assert settings.FUSION_QUERY_COUNT == 3
    assert settings.RRF_K == 60


def test_tr2_2_env_flag_semantics(monkeypatch):
    # env_flag_enabled 三分支：显式 true / 显式 false / 缺省取默认。
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TMP_FLAG_X", truthy)
        assert env_flag_enabled("TMP_FLAG_X", default=False) is True
    monkeypatch.setenv("TMP_FLAG_X", "anything-else")
    assert env_flag_enabled("TMP_FLAG_X", default=True) is False
    monkeypatch.delenv("TMP_FLAG_X", raising=False)
    assert env_flag_enabled("TMP_FLAG_X", default=True) is True
    assert env_flag_enabled("TMP_FLAG_X", default=False) is False


def test_tr2_2_admin_contract_models_smoke():
    # 新增管理契约的基本构造与校验。
    assert ReviewActionRequest(action="reject", comment="缺依据", actor="admin").action == "reject"
    assert ReleaseCreateRequest(version_ids=["v1"], as_candidate=True, gray_percent=20).gray_percent == 20
    assert RollbackRequest(reason="口径错误").reason == "口径错误"
    route = GrayRoute(release_id="r2", release_no=2, channel="candidate", reason="whitelist")
    assert route.channel == "candidate"
    view = DocVersionView(
        version_id="v1", doc_id="d1", version_no=1, title="t", source_path="x.md",
        kb_id="kb", status="draft", audience="driver", content_hash="h",
        created_at="now", status_updated_at="now",
    )
    assert view.content_md is None
    assert GrayConfigRequest(gray_percent=0).gray_whitelist is None
