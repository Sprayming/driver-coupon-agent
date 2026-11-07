"""知识生命周期服务层（Task 4 起）：md 同步、版本查询、差异对比。

审核流（Task 5）、发布（Task 6）、灰度（Task 7）、回滚（Task 8）
继续在本模块追加；所有写操作统一走 repository.write_audit 留痕。
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any

from api.schemas import DocVersionView, SyncResult
# 注意：必须通过 settings.DB_PATH / settings.KNOWLEDGE_DIR 动态读取，
# 不能在顶部 from config.settings import DB_PATH 绑定常量——
# 测试 patch(settings.DB_PATH) 改不到已绑定的名字，会让写操作误打到真实库（Task 5 踩过的坑）。
from config import settings
from rag.knowledge_base import parse_front_matter
from store import repository as repo
from store.db import transaction


def content_hash_of(content_md: str) -> str:
    return hashlib.sha256(content_md.encode("utf-8")).hexdigest()


def _version_view(row: Any, *, include_content: bool = False) -> DocVersionView:
    return DocVersionView(
        version_id=row["version_id"],
        doc_id=row["doc_id"],
        version_no=row["version_no"],
        title=row["title"],
        source_path=row["source_path"],
        kb_id=row["kb_id"],
        status=row["status"],
        audience=row["audience"],
        content_hash=row["content_hash"],
        change_note=row["change_note"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        status_updated_at=row["status_updated_at"],
        effective_from=row["effective_from"],
        effective_to=row["effective_to"],
        content_md=row["content_md"] if include_content else None,
    )


def list_versions(
    *, status: str | None = None, db_path: Path | str | None = None
) -> list[DocVersionView]:
    with transaction(db_path or settings.DB_PATH) as connection:
        rows = repo.list_versions(connection, status=status)
        return [_version_view(row) for row in rows]


def list_drafts(db_path: Path | str | None = None) -> list[DocVersionView]:
    return list_versions(status="draft", db_path=db_path)


def get_version_detail(
    version_id: str, db_path: Path | str | None = None
) -> DocVersionView | None:
    with transaction(db_path or settings.DB_PATH) as connection:
        row = repo.get_version(connection, version_id)
        return _version_view(row, include_content=True) if row else None


def diff_against_live(
    version_id: str, db_path: Path | str | None = None
) -> dict[str, Any] | None:
    """草稿/任意版本相对当前 stable 线上版本的行级差异。"""

    with transaction(db_path or settings.DB_PATH) as connection:
        row = repo.get_version(connection, version_id)
        if row is None:
            return None
        live = repo.get_live_version_for_doc(connection, row["doc_id"])
        base_text = live["content_md"] if live is not None else ""
        base_id = live["version_id"] if live is not None else None

    added: list[str] = []
    removed: list[str] = []
    for line in difflib.ndiff(base_text.splitlines(), row["content_md"].splitlines()):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    return {
        "version_id": version_id,
        "baseline_version_id": base_id,
        "added_lines": added,
        "removed_lines": removed,
    }


class ReviewStateError(ValueError):
    """审核状态机非法跃迁（API 层映射为 409）。"""


# 审核状态机：action → (要求的当前状态, 目标状态)。
# submit: draft → submitted；approve: submitted → approved；reject: submitted → draft。
REVIEW_TRANSITIONS: dict[str, tuple[str, str]] = {
    "submit": ("draft", "submitted"),
    "approve": ("submitted", "approved"),
    "reject": ("submitted", "draft"),
}


def review_version(
    version_id: str,
    *,
    action: str,
    actor: str = "admin",
    comment: str = "",
    db_path: Path | str | None = None,
) -> DocVersionView:
    """执行一次审核动作（submit/approve/reject），review_events + audit_log 双写。

    - 版本不存在：由调用方映射 404；
    - 当前状态不满足跃迁要求：抛 ReviewStateError（映射 409）；
    - reject 的 comment 必填由 pydantic ReviewActionRequest 在 API 层保证（422）。
    """

    if action not in REVIEW_TRANSITIONS:
        raise ReviewStateError(f"unsupported review action: {action}")
    required_from, target_status = REVIEW_TRANSITIONS[action]

    with transaction(db_path or settings.DB_PATH) as connection:
        row = repo.get_version(connection, version_id)
        if row is None:
            raise LookupError(version_id)
        before_status = row["status"]
        if before_status != required_from:
            raise ReviewStateError(
                f"版本 {version_id} 当前状态为 {before_status}，不允许 {action}"
                f"（要求状态 {required_from}）"
            )

        repo.update_version_status(connection, version_id=version_id, status=target_status)
        repo.insert_review_event(
            connection,
            version_id=version_id,
            action=action,
            actor=actor,
            comment=comment,
        )
        repo.write_audit(
            connection,
            actor=actor,
            action=f"review.{action}",
            target_type="version",
            target_id=version_id,
            before={"status": before_status},
            after={"status": target_status},
            comment=comment,
        )
        updated = repo.get_version(connection, version_id)
        return _version_view(updated)


def sync_knowledge_dir(
    *, actor: str = "admin", db_path: Path | str | None = None
) -> SyncResult:
    """扫描 knowledge/ 目录，与库中最新版本比对内容哈希；有变更则落 draft（不动线上）。

    - 新 md（库中无 document）：要求其 frontmatter kb 对应已存在的知识库，建文档 + v1 draft；
    - 已有 md：与 version_no 最大版本比哈希，不同才建 next-version draft；
    - 每个变更写 draft.created 审计。
    """

    path = db_path or settings.DB_PATH
    details: list[dict[str, Any]] = []
    created_version_ids: list[str] = []
    scanned = unchanged = changed = 0

    md_files = sorted(settings.KNOWLEDGE_DIR.glob("*.md"))

    with transaction(path) as connection:
        for md_path in md_files:
            scanned += 1
            content_md = md_path.read_text(encoding="utf-8")
            metadata, _ = parse_front_matter(content_md)
            title = str(metadata.get("title") or md_path.stem)
            audience = str(metadata.get("audience") or "driver")
            kb_code = str(metadata.get("kb") or "").strip()
            digest = content_hash_of(content_md)

            document = repo.get_document_by_source_path(connection, md_path.name)
            if document is None:
                # 新文档：必须归属一个已播种的知识库，避免孤儿文档。
                kb_row = repo.get_knowledge_base_by_code(connection, kb_code) if kb_code else None
                if kb_row is None:
                    raise ValueError(
                        f"{md_path.name} 声明 kb={kb_code!r}，但知识库不存在；请先 seed 或检查 frontmatter"
                    )
                doc_id = f"doc-{md_path.stem}"
                repo.insert_document(
                    connection,
                    doc_id=doc_id,
                    kb_id=kb_row["kb_id"],
                    source_path=md_path.name,
                    title=title,
                )
                version_id = f"{doc_id}-v1"
                repo.insert_version(
                    connection,
                    version_id=version_id,
                    doc_id=doc_id,
                    version_no=1,
                    content_md=content_md,
                    content_hash=digest,
                    status="draft",
                    audience=audience,
                    change_note=f"sync 导入新文档 {md_path.name}",
                    created_by=actor,
                )
                repo.write_audit(
                    connection,
                    actor=actor,
                    action="document.created",
                    target_type="document",
                    target_id=doc_id,
                    after={"source_path": md_path.name, "kb_id": kb_row["kb_id"]},
                )
                repo.write_audit(
                    connection,
                    actor=actor,
                    action="draft.created",
                    target_type="version",
                    target_id=version_id,
                    after={"version_no": 1, "source_path": md_path.name},
                    comment="sync 发现新文档",
                )
                created_version_ids.append(version_id)
                changed += 1
                details.append({"source_path": md_path.name, "outcome": "new_document", "version_id": version_id})
                continue

            # 去重判据：库中「任意版本」（stable/draft/灰度变体）与磁盘 md 哈希相同即 unchanged。
            # 只比 latest 会被 seed 的注入变体（draft/gray v2）误伤，重复制造 v3 草稿。
            same_content = repo.get_version_by_hash(connection, document["doc_id"], digest)
            if same_content is not None:
                unchanged += 1
                details.append({"source_path": md_path.name, "outcome": "unchanged"})
                continue

            latest = repo.get_latest_version(connection, document["doc_id"])
            next_no = int(latest["version_no"]) + 1 if latest else 1
            version_id = f"{document['doc_id']}-v{next_no}"
            repo.insert_version(
                connection,
                version_id=version_id,
                doc_id=document["doc_id"],
                version_no=next_no,
                content_md=content_md,
                content_hash=digest,
                status="draft",
                audience=audience,
                change_note=f"sync 检测到 {md_path.name} 内容变更",
                created_by=actor,
            )
            repo.write_audit(
                connection,
                actor=actor,
                action="draft.created",
                target_type="version",
                target_id=version_id,
                before={"content_hash": latest["content_hash"] if latest else None},
                after={"content_hash": digest, "version_no": next_no},
                comment=f"sync 导入变更草稿（{md_path.name}）",
            )
            created_version_ids.append(version_id)
            changed += 1
            details.append({"source_path": md_path.name, "outcome": "changed", "version_id": version_id})

    return SyncResult(
        scanned=scanned,
        unchanged=unchanged,
        changed=changed,
        created_version_ids=created_version_ids,
        details=details,
    )
