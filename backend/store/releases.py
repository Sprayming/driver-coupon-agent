"""发布 / 灰度 / 回滚服务（Task 6 起）。

Task 6：publish_release 不可变快照 + 门禁 hook；
Task 7：resolve_gray_route 灰度路由 + 灰度配置；
Task 8：rollback_release 回滚与缓存联动。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from api.schemas import (
    GateResult,
    GrayConfigRequest,
    GrayRoute,
    KnowledgeChunk,
    ReleaseCreateRequest,
    ReleasePublishResult,
    ReleaseView,
    RollbackRequest,
)
from config import settings
from rag.knowledge_base import build_chunks_from_content, parse_front_matter
from store import repository as repo
from store.db import transaction, utc_now_iso
from store.gate import run_release_gate


class ReleaseStateError(ValueError):
    """发布/灰度/回滚状态不满足要求（API 层映射 409）。"""


class GateBlockedError(ValueError):
    """质量门禁未通过（API 层映射 422，携带 GateResult 失败明细）。"""

    def __init__(self, gate: GateResult) -> None:
        super().__init__("release quality gate blocked")
        self.gate = gate


class SnapshotTamperedError(ValueError):
    """回滚目标快照的内容哈希与当前版本行不一致（API 层映射 409）。"""


# release 变更监听点：Task 9 在此注册「按 release 隔离的索引/检索缓存」失效回调。
# 监听器异常不影响发布/回滚主流程（只做缓存清理，失败最坏代价是缓存自然过期）。
_release_change_listeners: list[Callable[[dict], None]] = []


def add_release_change_listener(listener: Callable[[dict], None]) -> None:
    _release_change_listeners.append(listener)


def _emit_release_change(event: dict) -> None:
    for listener in list(_release_change_listeners):
        try:
            listener(event)
        except Exception:  # noqa: BLE001 — 缓存失效失败不得阻断发布/回滚
            continue


def _release_view(row, version_ids: list[str]) -> ReleaseView:
    return ReleaseView(
        release_id=row["release_id"],
        release_no=row["release_no"],
        label=row["label"],
        status=row["status"],
        gray_whitelist=json.loads(row["gray_whitelist"] or "[]"),
        gray_percent=int(row["gray_percent"]),
        created_by=row["created_by"],
        created_at=row["created_at"],
        activated_at=row["activated_at"],
        rolled_back_at=row["rolled_back_at"],
        rollback_reason=row["rollback_reason"],
        version_ids=version_ids,
    )


def list_releases(db_path: Path | str | None = None) -> list[ReleaseView]:
    with transaction(db_path or settings.DB_PATH) as connection:
        rows = repo.list_releases(connection)
        return [
            _release_view(row, repo.list_release_version_ids(connection, row["release_id"]))
            for row in rows
        ]


def publish_release(
    request: ReleaseCreateRequest,
    *,
    actor: str = "admin",
    db_path: Path | str | None = None,
) -> ReleasePublishResult:
    """把一组 approved 版本组成不可变发布快照（先过门，后置 stable/candidate）。"""

    if not request.version_ids:
        raise ReleaseStateError("version_ids 不能为空")

    with transaction(db_path or settings.DB_PATH) as connection:
        # —— 校验阶段 ——
        seen_docs: set[str] = set()
        version_rows = []
        for version_id in request.version_ids:
            row = repo.get_version(connection, version_id)
            if row is None:
                raise ReleaseStateError(f"版本不存在：{version_id}")
            if row["status"] != "approved":
                raise ReleaseStateError(
                    f"版本 {version_id} 当前状态 {row['status']}，只有 approved 才能发布"
                )
            # 防止同一文档两个版本进同一快照造成内容歧义。
            if row["doc_id"] in seen_docs:
                raise ReleaseStateError(f"同一文档不能在一次发布中出现两个版本：{row['doc_id']}")
            seen_docs.add(row["doc_id"])
            # 只允许发布文档最新版本；旧版"复活"应走回滚而不是新发布。
            latest = repo.get_latest_version(connection, row["doc_id"])
            if latest is None or latest["version_id"] != version_id:
                raise ReleaseStateError(
                    f"版本 {version_id} 不是文档 {row['doc_id']} 的最新版本，不能发布"
                )
            version_rows.append(row)

        target_status = "candidate" if request.as_candidate else "stable"
        if target_status == "candidate":
            existing_candidate = next(
                (r for r in repo.list_active_releases(connection) if r["status"] == "candidate"),
                None,
            )
            if existing_candidate is not None:
                raise ReleaseStateError(
                    f"已存在灰度发布 {existing_candidate['release_id']}，"
                    "请先通过回滚/转正处理后再发起新的灰度"
                )

        # —— 门禁阶段（Task 6 placeholder 恒过；Task 16 在此阻断坏版本）——
        gate = run_release_gate(request.version_ids, db_path=db_path)
        if not gate.passed:
            repo.write_audit(
                connection,
                actor=actor,
                action="gate.blocked",
                target_type="release",
                target_id="",
                after={"failures": [f.model_dump() for f in gate.failures]},
                comment="发布质量门禁未通过",
            )
            # 在任何 release 数据落库前抛出，事务回滚（仅审计事件保留）。
            raise GateBlockedError(gate)

        # —— 落库阶段 ——
        release_no = repo.max_release_no(connection) + 1
        release_id = f"rel-{release_no:04d}"
        repo.insert_release(
            connection,
            release_id=release_id,
            release_no=release_no,
            label=request.label,
            status=target_status,
            gray_whitelist=request.gray_whitelist,
            gray_percent=request.gray_percent,
            created_by=actor,
        )
        for row in version_rows:
            repo.attach_release_item(
                connection,
                release_id=release_id,
                version_id=row["version_id"],
                content_hash=row["content_hash"],  # 冗余快照哈希，回滚时校验
            )
            repo.update_version_status(
                connection, version_id=row["version_id"], status="published"
            )

        # 新 stable 全量切换：旧 stable/candidate 一律作废旧（灰度被全量切版自然终止）。
        superseded_ids: list[str] = []
        if target_status == "stable":
            for active in repo.list_active_releases(connection):
                if active["release_id"] == release_id:
                    continue
                repo.mark_release_rolled_back(
                    connection,
                    release_id=active["release_id"],
                    reason=f"superseded by {release_id}",
                )
                superseded_ids.append(active["release_id"])

        repo.write_audit(
            connection,
            actor=actor,
            action="release.published",
            target_type="release",
            target_id=release_id,
            after={
                "release_no": release_no,
                "status": target_status,
                "version_ids": request.version_ids,
                "gray_percent": request.gray_percent,
                "superseded": superseded_ids,
            },
            comment=request.label,
        )

        release_row = repo.get_release(connection, release_id)
        view = _release_view(release_row, list(request.version_ids))

    # 通知索引/检索缓存层（Task 9 接线）：stable 切换或新 candidate 都会改变路由结果。
    _emit_release_change({"type": "published", "release_id": release_id, "status": target_status})
    return ReleasePublishResult(release=view, gate=gate)


# —— Task 8：一键回滚 ——

def _verify_snapshot_integrity(connection, release_id: str) -> None:
    """回滚前校验目标快照明细哈希与当前版本行一致；不一致说明快照基线被破坏。"""

    rows = connection.execute(
        "SELECT ri.version_id AS version_id, ri.content_hash AS item_hash, "
        "dv.content_hash AS version_hash FROM release_items ri "
        "JOIN doc_versions dv ON dv.version_id = ri.version_id WHERE ri.release_id = ?",
        (release_id,),
    ).fetchall()
    mismatched = [
        row["version_id"] for row in rows if row["item_hash"] != row["version_hash"]
    ]
    if mismatched:
        raise SnapshotTamperedError(
            f"快照 {release_id} 内容完整性校验失败：{mismatched}"
        )


def _restore_release_as_stable(connection, release_id: str) -> None:
    connection.execute(
        "UPDATE releases SET status='stable', activated_at=?, rolled_back_at=NULL, "
        "rollback_reason='', gray_whitelist='[]', gray_percent=0 WHERE release_id=?",
        (utc_now_iso(), release_id),
    )


def rollback_release(
    release_id: str,
    request: RollbackRequest,
    *,
    actor: str = "admin",
    db_path: Path | str | None = None,
) -> ReleaseView:
    """回滚到指定历史快照（FR-4/AC-5）。

    - target=candidate：放弃灰度，candidate 作废，stable 不变；
    - target=rolled_back 历史快照：完整性校验通过后恢复为 stable，
      当前 stable 与 candidate 全部作废（灰度随回滚立即失效）；
    - target=当前 stable：拒绝（请明确选择要回到的历史快照）。
    """

    with transaction(db_path or settings.DB_PATH) as connection:
        target = repo.get_release(connection, release_id)
        if target is None:
            raise LookupError(release_id)

        if target["status"] == "candidate":
            repo.mark_release_rolled_back(
                connection,
                release_id=release_id,
                reason=f"gray discarded: {request.reason}",
            )
            repo.write_audit(
                connection,
                actor=actor,
                action="release.rolled_back",
                target_type="release",
                target_id=release_id,
                before={"status": "candidate"},
                after={"status": "rolled_back", "kind": "candidate_discarded"},
                comment=request.reason,
            )
            result_id = release_id

        elif target["status"] == "rolled_back":
            _verify_snapshot_integrity(connection, release_id)

            discarded: list[str] = []
            for active in repo.list_active_releases(connection):
                if active["status"] == "candidate":
                    kind = "candidate_discarded_on_rollback"
                    detail = f"discarded on rollback to {release_id}: {request.reason}"
                else:
                    kind = "stable_replaced_by_rollback"
                    detail = f"rolled back to {release_id}: {request.reason}"
                repo.mark_release_rolled_back(
                    connection, release_id=active["release_id"], reason=detail
                )
                repo.write_audit(
                    connection,
                    actor=actor,
                    action="release.rolled_back",
                    target_type="release",
                    target_id=active["release_id"],
                    before={"status": active["status"]},
                    after={"status": "rolled_back", "kind": kind, "restored": release_id},
                    comment=request.reason,
                )
                discarded.append(active["release_id"])

            _restore_release_as_stable(connection, release_id)
            repo.write_audit(
                connection,
                actor=actor,
                action="release.restored",
                target_type="release",
                target_id=release_id,
                before={"status": "rolled_back"},
                after={"status": "stable", "discarded": discarded},
                comment=request.reason,
            )
            result_id = release_id

        else:  # 当前 stable
            raise ReleaseStateError(
                f"{release_id} 是当前 stable 快照；回滚需要指定一个历史 rolled_back 快照作为目标"
            )

        result_row = repo.get_release(connection, result_id)
        view = _release_view(result_row, repo.list_release_version_ids(connection, result_id))

    # 回滚后所有按 release 隔离的检索索引/缓存必须立即失效（Task 9 接线）。
    _emit_release_change({"type": "rollback", "release_id": result_id})
    return view


# —— Task 7：灰度路由 ——

def gray_bucket(driver_id: str) -> int:
    """driver_id 的稳定哈希桶（0-99）：同一 ID 永远落同一桶，与会话无关。"""

    digest = hashlib.sha256(driver_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _latest_stable(connection) -> object | None:
    rows = connection.execute(
        "SELECT * FROM releases WHERE status='stable' ORDER BY release_no DESC LIMIT 1"
    ).fetchall()
    return rows[0] if rows else None


def _active_candidate(connection) -> object | None:
    rows = connection.execute(
        "SELECT * FROM releases WHERE status='candidate' ORDER BY release_no DESC LIMIT 1"
    ).fetchall()
    return rows[0] if rows else None


def resolve_gray_route(driver_id: str, db_path: Path | str | None = None) -> GrayRoute:
    """按司机身份解析本轮请求读哪个 release（FR-5）。

    优先级：白名单命中 → 哈希落桶 → stable；无 candidate 时 reason=none。
    """

    with transaction(db_path or settings.DB_PATH) as connection:
        stable = _latest_stable(connection)
        candidate = _active_candidate(connection)
        if stable is None:
            raise ReleaseStateError("系统中没有 stable 发布，无法路由")

        if candidate is None:
            return GrayRoute(
                release_id=stable["release_id"],
                release_no=stable["release_no"],
                channel="stable",
                reason="none",
                gray_percent=0,
            )

        whitelist = json.loads(candidate["gray_whitelist"] or "[]")
        percent = int(candidate["gray_percent"])
        if driver_id in whitelist:
            return GrayRoute(
                release_id=candidate["release_id"],
                release_no=candidate["release_no"],
                channel="candidate",
                reason="whitelist",
                gray_percent=percent,
            )
        if percent > 0 and gray_bucket(driver_id) < percent:
            return GrayRoute(
                release_id=candidate["release_id"],
                release_no=candidate["release_no"],
                channel="candidate",
                reason="percent",
                gray_percent=percent,
            )
        return GrayRoute(
            release_id=stable["release_id"],
            release_no=stable["release_no"],
            channel="stable",
            reason="stable",
            gray_percent=percent,
        )


def update_gray_config(
    release_id: str,
    config: GrayConfigRequest,
    *,
    actor: str = "admin",
    db_path: Path | str | None = None,
) -> ReleaseView:
    """调整 candidate 的灰度白名单/百分比（None 字段表示该项不改）。"""

    with transaction(db_path or settings.DB_PATH) as connection:
        row = repo.get_release(connection, release_id)
        if row is None:
            raise LookupError(release_id)
        if row["status"] != "candidate":
            raise ReleaseStateError(
                f"发布 {release_id} 状态为 {row['status']}，只有 candidate 可以配置灰度"
            )

        before = {
            "gray_whitelist": json.loads(row["gray_whitelist"] or "[]"),
            "gray_percent": int(row["gray_percent"]),
        }
        new_whitelist = config.gray_whitelist if config.gray_whitelist is not None else before["gray_whitelist"]
        new_percent = config.gray_percent if config.gray_percent is not None else before["gray_percent"]

        connection.execute(
            "UPDATE releases SET gray_whitelist = ?, gray_percent = ? WHERE release_id = ?",
            (json.dumps(new_whitelist, ensure_ascii=False), new_percent, release_id),
        )
        repo.write_audit(
            connection,
            actor=actor,
            action="release.gray_configured",
            target_type="release",
            target_id=release_id,
            before=before,
            after={"gray_whitelist": new_whitelist, "gray_percent": new_percent},
            comment="调整灰度配置",
        )
        updated = repo.get_release(connection, release_id)
        return _release_view(updated, repo.list_release_version_ids(connection, release_id))


# —— Task 11：角色 → 可检索知识库集合 ——

def get_allowed_kb_ids(role: str, db_path: Path | str | None = None) -> set[str]:
    """返回指定角色可检索的知识库 kb_id 集合（FR-8）。

    driver 不含 internal_playbook；agent/admin 可检索全部库。
    """

    allowed: set[str] = set()
    with transaction(db_path or settings.DB_PATH) as connection:
        rows = connection.execute(
            "SELECT kb_id, allowed_roles FROM knowledge_bases"
        ).fetchall()
        for row in rows:
            roles = json.loads(row["allowed_roles"] or "[]")
            if role in roles:
                allowed.add(row["kb_id"])
    return allowed


def get_allowed_audiences(role: str) -> set[str]:
    """返回指定角色可见的 audience 集合（FR-9）。

    driver 只能看 driver audience；agent/admin 可看 driver + internal。
    """

    if role == "driver":
        return {"driver"}
    return {"driver", "internal"}


# —— Task 9：按 release 快照构建知识 chunk ——

def load_release_chunks(
    release_id: str, db_path: Path | str | None = None
) -> list[KnowledgeChunk]:
    """从数据库读取指定 release 快照的全部版本，解析为知识 chunk。

    用于按 release 构建生产索引（Task 9）：每个 release 有独立的 chunk 集合，
    灰度司机看到 candidate 快照，其余司机看到 stable 快照。
    """

    chunks: list[KnowledgeChunk] = []
    with transaction(db_path or settings.DB_PATH) as connection:
        rows = connection.execute(
            "SELECT v.version_id, v.content_md, v.audience, v.effective_from, v.effective_to, "
            "d.source_path, d.title, d.kb_id "
            "FROM release_items ri "
            "JOIN doc_versions v ON v.version_id = ri.version_id "
            "JOIN documents d ON d.doc_id = v.doc_id "
            "WHERE ri.release_id = ? "
            "ORDER BY d.source_path, v.version_no",
            (release_id,),
        ).fetchall()
        for row in rows:
            metadata, _ = parse_front_matter(row["content_md"])
            # 把 db 层的 audience/kb_id/effective_* 覆盖到 metadata，保证 build_chunks_from_content
            # 透传到 chunk 的字段和发布快照的版本行一致（release 是内容真相源）。
            # 注意：md frontmatter 的 kb 字段是 domain（如 refund），而 knowledge_bases.kb_id
            # 是 kb-refund 格式；这里强制用数据库 kb_id 覆盖，让 chunk.kb_id 与 knowledge_bases
            # 表主键一致，RBAC 过滤时才能正确匹配。
            metadata["audience"] = row["audience"]
            if row["kb_id"]:
                metadata["kb"] = row["kb_id"]
            if row["effective_from"]:
                metadata["effective_from"] = row["effective_from"]
            if row["effective_to"]:
                metadata["effective_to"] = row["effective_to"]
            chunks.extend(
                build_chunks_from_content(
                    source_path=row["source_path"],
                    title=row["title"],
                    content_md=row["content_md"],
                    metadata=metadata,
                )
            )
    return chunks
