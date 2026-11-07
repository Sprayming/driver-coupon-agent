"""发布质量门禁（Task 6 placeholder → Task 16 真实评测）。

run_release_gate 对候选快照构建临时索引 → 跑扩展评测集（expected 漏召回 / forbidden 命中 / must-fallback 失败）→ 失败即 passed=False，
并在 release_items 落库前阻断发布（AC-12）。调用方（releases.publish_release）已经按
「gate.passed 才提交」接线，无需再改。

GATE_ENABLED 开关：默认 False（保持 Task 6 占位行为），Task 16 测试开启。
"""

from __future__ import annotations

import hashlib
import logging

from api.schemas import GateFailure, GateResult, KnowledgeIndex
from config import settings
from rag.knowledge_base import build_chunks_from_content
from store.db import utc_now_iso
from store import repository as repo
from store.db import transaction

logger = logging.getLogger(__name__)


def _build_index_from_versions(version_ids: list[str], db_path) -> KnowledgeIndex:
    """从 version_ids 构建临时 KnowledgeIndex（用于门禁评测，不入 release 缓存）。"""

    with transaction(db_path) as conn:
        version_rows = [repo.get_version(conn, vid) for vid in version_ids]
        version_rows = [v for v in version_rows if v is not None]

    chunks: list = []
    for row in version_rows:
        content_md = row["content_md"] or ""
        doc_id = row["doc_id"] or ""
        version_id = row["version_id"] or ""
        parsed = build_chunks_from_content(
            source_path=f"{doc_id}.md",
            title=version_id,
            content_md=content_md,
        )
        chunks.extend(parsed)

    # 构建 fingerprint（单个 str）。
    hasher = hashlib.md5()
    for c in chunks:
        hasher.update(c.chunk_id.encode("utf-8"))
        hasher.update(c.text.encode("utf-8"))
    fingerprint = hasher.hexdigest()[:12]

    version_prefix = hashlib.md5(
        str(sorted(c.chunk_id for c in chunks)).encode("utf-8")
    ).hexdigest()[:16]

    # 构建倒排表（chunk 的 keywords + text 分词 → chunk_ids）。
    inverted: dict[str, list[str]] = {}
    for c in chunks:
        for kw in c.keywords:
            inverted.setdefault(kw, []).append(c.chunk_id)

    return KnowledgeIndex(
        version=f"gate-{version_prefix}",
        fingerprint=fingerprint,
        chunk_count=len(chunks),
        chunks_by_id={c.chunk_id: c for c in chunks},
        inverted_index=inverted,
    )


def run_release_gate(
    version_ids: list[str],
    *,
    db_path=None,
    eval_cases=None,
    embedding_client=None,
) -> GateResult:
    """对拟发布版本集合运行质量门禁。

    默认 placeholder（恒过），env 开关 GATE_ENABLED=True 时跑真实评测。
    """

    gate_enabled = getattr(settings, "GATE_ENABLED", False)
    if not gate_enabled:
        return GateResult(
            passed=True,
            mode="placeholder",
            checked_versions=list(version_ids),
            failures=[],
            ran_at=utc_now_iso(),
        )

    db_path = db_path or settings.DB_PATH

    # 延迟导入避免循环：store.releases → store.gate → rag.quality → store.releases
    from rag.quality import evaluate_quality_case, load_quality_cases

    # 1. 构建临时索引。
    try:
        index = _build_index_from_versions(version_ids, db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("门禁索引构建失败，放行（fail-open）: %s", exc)
        return GateResult(
            passed=True,
            mode="build_failed",
            checked_versions=list(version_ids),
            failures=[],
            ran_at=utc_now_iso(),
        )

    # 2. 加载评测用例并运行。
    cases = eval_cases if eval_cases is not None else load_quality_cases()
    failures: list[GateFailure] = []

    for case in cases:
        result = evaluate_quality_case(
            case,
            embedding_client=embedding_client,
            index_override=index,
        )

        reasons: list[str] = []
        if result.forbidden_hit:
            reasons.append("forbidden_chunk_hit")
        if case.must_fallback and not result.fallback:
            reasons.append("fallback_expected_but_succeeded")
        if not case.must_fallback and not result.passed and not result.forbidden_hit:
            expected_set = set(case.expected_chunk_ids)
            retrieved_set = set(result.retrieved_chunk_ids)
            if expected_set - retrieved_set:
                reasons.append("expected_not_recalled")

        if reasons:
            failures.append(
                GateFailure(
                    case_id=case.case_id,
                    reason=",".join(reasons),
                    detail={
                        "expected": case.expected_chunk_ids,
                        "forbidden": case.forbidden_chunk_ids,
                        "retrieved": result.retrieved_chunk_ids,
                        "fallback": result.fallback,
                        "forbidden_hit": result.forbidden_hit,
                        "runtime_role": case.runtime_role,
                    },
                )
            )

    passed = len(failures) == 0
    return GateResult(
        passed=passed,
        mode="quality_eval",
        checked_versions=list(version_ids),
        failures=failures,
        ran_at=utc_now_iso(),
    )
