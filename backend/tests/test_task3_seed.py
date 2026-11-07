"""Task 3 验证：种子幂等与计数（TR-3.1）、md 新字段解析到 chunk（TR-3.2）。"""

from __future__ import annotations

import pytest

from rag.knowledge_base import build_knowledge_chunks
from store.seed import reset, seed

EXPECTED_COUNTS = {
    "knowledge_bases": 5,
    "documents": 5,
    "doc_versions": 7,  # 5 个 v1 + refund v2(draft) + purchase v2(灰度)
    "review_events": 5,
    "releases": 2,
    "release_items": 6,  # stable 5 + candidate 1
    "audit_log": 1,
}


def test_tr3_1_seed_reset_idempotent(tmp_path):
    db_path = tmp_path / "task3.db"

    counts_first = seed(db_path)
    assert counts_first == EXPECTED_COUNTS

    # reset：删库重建后计数必须完全一致。
    counts_reset = reset(db_path)
    assert counts_reset == EXPECTED_COUNTS

    # 非空库直接 seed（不 force）必须报错保护，防止重复播种污染。
    with pytest.raises(RuntimeError):
        seed(db_path)


def _chunks_by_id() -> dict[str, object]:
    return {chunk.chunk_id: chunk for chunk in build_knowledge_chunks()}


def test_tr3_2_policy_codes_and_kb_parsed():
    chunks = _chunks_by_id()

    eligibility = chunks["coupon-refund-eligibility"]
    assert eligibility.kb_id == "refund"
    assert eligibility.policy_codes == ["P-REF-001"]
    assert eligibility.audience == "driver"

    purchase = chunks["coupon-purchase-eligibility"]
    assert purchase.kb_id == "purchase"
    assert purchase.policy_codes == ["P-PUR-001"]
    assert "周末日卡" in purchase.product_names
    # 正文里也要真实出现周末日卡（AC-9 精确商品名锚点）。
    assert "周末日卡" in purchase.text


def test_tr3_2_internal_doc_isolated_metadata():
    chunks = _chunks_by_id()

    internal = chunks["internal-refund-exception"]
    assert internal.kb_id == "internal_playbook"
    assert internal.audience == "internal"
    assert internal.policy_codes == ["P-IN-001"]
    # 对抗测试锚点句真实存在于 internal chunk 文本中（Task 20 会尝试逐字套取）。
    assert "TB-" in internal.text

    risk = chunks["internal-risk-driver"]
    assert risk.audience == "internal"
    assert risk.policy_codes == ["P-IN-002"]


def test_tr3_2_historical_chunk_keeps_status_and_codes():
    chunks = _chunks_by_id()

    old = chunks["coupon-old-promotion"]
    # 方案Y 不改：historical 状态照常解析出来，由召回/重排层决定软降权。
    assert old.effective_status == "historical"
    assert old.policy_codes == ["P-REF-2025-01"]
    assert old.product_names == ["月卡折扣券"]
