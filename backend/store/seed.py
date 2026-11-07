"""RAG 平台演示数据种子（Task 3 / NFR-7）。

用法（backend 目录下）：
    python -m store.seed reset   # 删除 db 文件重建并播种
    python -m store.seed seed    # 在已初始化的空库上播种（非空则报错保护）

播种内容：
- 5 个知识库（4 业务库全角色 + internal_playbook 仅 agent/admin）；
- 5 个已发布 v1 文档（内容快照 = knowledge/ 下 md 全文），组成 rel-0001 stable；
- coupon-refund.md 的 v2 草稿（含 DRAFT-X9 独有条款，供 AC-1 草稿不可见测试）；
- coupon-purchase.md 的 v2 版本挂 rel-0002 candidate（白名单 driver-gray-whitelist + 20%，
  含 GRAY-X2 独有条款，供 AC-4 灰度路由测试）；
- historical chunk（2025 月卡折扣券）天然存在于 coupon-refund.md v1 中，走方案Y 软降权。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from config import settings
from rag.knowledge_base import parse_front_matter
from store import repository as repo
from store.db import init_db, transaction

# 知识库定义：kb_id → (文件名, 名称, domain, 允许角色)
KNOWLEDGE_BASES: list[tuple[str, str, str, str, list[str]]] = [
    ("kb-refund", "coupon-refund.md", "卡券退款知识库", "refund", ["driver", "agent", "admin"]),
    ("kb-purchase", "coupon-purchase.md", "卡券购买知识库", "purchase", ["driver", "agent", "admin"]),
    ("kb-usage", "coupon-usage.md", "卡券使用知识库", "usage", ["driver", "agent", "admin"]),
    ("kb-complaint", "coupon-complaint.md", "卡券投诉知识库", "complaint", ["driver", "agent", "admin"]),
    (
        "kb-internal-playbook",
        "internal-refund-playbook.md",
        "内部作业口径库",
        "internal_playbook",
        ["agent", "admin"],  # 司机不可检索该库（FR-8/9）
    ),
]

# 草稿独句（AC-1 锚点）：只存在于 draft v2，任何已发布索引都读不到。
DRAFT_CLAUSE = (
    "- 【草稿试验条款 DRAFT-X9】未核销卡券退款新增 24 小时冷静期，"
    "冷静期内申请退款免审核即时到账。"
)
# 灰度新版独句（AC-4 锚点）：只存在于 candidate rel-0002 的 purchase v2。
GRAY_CLAUSE = (
    "- 【灰度试运行 GRAY-X2】周末日卡在灰度期间买一赠一，"
    "该优惠规则仅灰度司机可见。"
)

GRAY_WHITELIST = ["driver-gray-whitelist"]
GRAY_PERCENT = 20


def _content_hash(content_md: str) -> str:
    return hashlib.sha256(content_md.encode("utf-8")).hexdigest()


def _inject_before_section_end(content_md: str, marker_section: str, clause: str) -> str:
    """在指定 ## 章节的下一个 ## 标题（或文末）前插入一行条款。"""

    marker = f"## {marker_section}"
    start = content_md.find(marker)
    if start == -1:
        raise ValueError(f"marker section not found: {marker_section}")
    next_section = content_md.find("\n## ", start + len(marker))
    if next_section == -1:
        return content_md.rstrip() + "\n" + clause + "\n"
    return content_md[:next_section].rstrip() + "\n" + clause + "\n" + content_md[next_section:]


def seed(db_path: Path | str | None = None, *, force: bool = False) -> dict[str, int]:
    """执行播种，返回各类别计数。"""

    path = Path(db_path) if db_path is not None else settings.DB_PATH
    init_db(path)

    with transaction(path) as connection:
        existing = repo.count_rows(connection, "knowledge_bases")
        if existing > 0 and not force:
            raise RuntimeError(f"数据库非空（{existing} 个知识库），如需重建请用 reset")

        stable_version_ids: list[str] = []
        doc_meta: dict[str, dict[str, str]] = {}

        for kb_id, filename, kb_name, domain, roles in KNOWLEDGE_BASES:
            repo.insert_knowledge_base(
                connection, kb_id=kb_id, name=kb_name, domain=domain, allowed_roles=roles
            )
            content_md = (settings.KNOWLEDGE_DIR / filename).read_text(encoding="utf-8")
            metadata, _ = parse_front_matter(content_md)
            title = str(metadata.get("title") or Path(filename).stem)
            audience = str(metadata.get("audience") or "driver")
            doc_id = f"doc-{Path(filename).stem}"
            repo.insert_document(
                connection, doc_id=doc_id, kb_id=kb_id, source_path=filename, title=title
            )
            version_id = f"{doc_id}-v1"
            repo.insert_version(
                connection,
                version_id=version_id,
                doc_id=doc_id,
                version_no=1,
                content_md=content_md,
                content_hash=_content_hash(content_md),
                status="published",
                audience=audience,
                change_note="种子数据：初始发布版本",
            )
            # 补一条 approve 审核事件，让审核时间线在初次播种后就有数据可看。
            repo.insert_review_event(
                connection, version_id=version_id, action="approve", actor="admin", comment="初始口径审核通过"
            )
            stable_version_ids.append(version_id)
            doc_meta[doc_id] = {"audience": audience, "filename": filename}

        # rel-0001：全量稳定版。
        repo.insert_release(
            connection,
            release_id="rel-0001",
            release_no=1,
            label="初始稳定版",
            status="stable",
            created_by="seed",
        )
        for version_id in stable_version_ids:
            row = connection.execute(
                "SELECT content_hash FROM doc_versions WHERE version_id=?", (version_id,)
            ).fetchone()
            repo.attach_release_item(
                connection, release_id="rel-0001", version_id=version_id, content_hash=row["content_hash"]
            )

        # coupon-refund v2：草稿（AC-1 锚点）。
        refund_md = (settings.KNOWLEDGE_DIR / "coupon-refund.md").read_text(encoding="utf-8")
        draft_md = _inject_before_section_end(refund_md, "退款申请条件", DRAFT_CLAUSE)
        repo.insert_version(
            connection,
            version_id="doc-coupon-refund-v2",
            doc_id="doc-coupon-refund",
            version_no=2,
            content_md=draft_md,
            content_hash=_content_hash(draft_md),
            status="draft",
            audience="driver",
            change_note="种子草稿：新增冷静期试验条款，尚未提交审核",
            created_by="seed",
        )

        # coupon-purchase v2：灰度新版，挂 rel-0002 candidate（AC-4 锚点）。
        purchase_md = (settings.KNOWLEDGE_DIR / "coupon-purchase.md").read_text(encoding="utf-8")
        gray_md = _inject_before_section_end(purchase_md, "卡券购买资格与价格", GRAY_CLAUSE)
        repo.insert_version(
            connection,
            version_id="doc-coupon-purchase-v2",
            doc_id="doc-coupon-purchase",
            version_no=2,
            content_md=gray_md,
            content_hash=_content_hash(gray_md),
            status="published",
            audience="driver",
            change_note="种子灰度版：周末日卡买一赠一试运行",
            created_by="seed",
        )
        repo.insert_release(
            connection,
            release_id="rel-0002",
            release_no=2,
            label="周末日卡灰度版",
            status="candidate",
            gray_whitelist=GRAY_WHITELIST,
            gray_percent=GRAY_PERCENT,
            created_by="seed",
        )
        repo.attach_release_item(
            connection,
            release_id="rel-0002",
            version_id="doc-coupon-purchase-v2",
            content_hash=_content_hash(gray_md),
        )

        repo.write_audit(
            connection,
            actor="seed",
            action="seed.completed",
            target_type="database",
            target_id=str(path),
            after={
                "stable_release": "rel-0001",
                "candidate_release": "rel-0002",
                "gray_whitelist": GRAY_WHITELIST,
                "gray_percent": GRAY_PERCENT,
            },
            comment="演示数据播种完成",
        )

    return _collect_counts(path)


def _collect_counts(path: Path) -> dict[str, int]:
    with transaction(path) as connection:
        return {
            table: repo.count_rows(connection, table)
            for table in (
                "knowledge_bases",
                "documents",
                "doc_versions",
                "review_events",
                "releases",
                "release_items",
                "audit_log",
            )
        }


def reset(db_path: Path | str | None = None) -> dict[str, int]:
    """删库重建并播种。"""

    path = Path(db_path) if db_path is not None else settings.DB_PATH
    for suffix in ("", "-wal", "-shm"):
        db_file = Path(str(path) + suffix)
        if db_file.exists():
            db_file.unlink()
    return seed(path)


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:])[0:1] or ["seed"]
    if command[0] == "reset":
        counts = reset()
    elif command[0] == "seed":
        counts = seed()
    else:
        print(f"unknown command: {command[0]} (use seed|reset)")
        return 2
    print(json.dumps({"ok": True, "command": command[0], "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
