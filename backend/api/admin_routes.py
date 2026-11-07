"""RAG 平台管理 API（Task 4 起）：/admin/* 全部需要 X-Admin-Token。

Task 4：草稿同步、版本列表/详情、与线上差异、审计查询；
Task 5–8 继续在本 router 追加审核、发布、灰度、回滚端点。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from api.schemas import (
    AuditEventView,
    DocVersionView,
    GrayConfigRequest,
    GrayRoute,
    ReleaseCreateRequest,
    ReleasePublishResult,
    ReleaseView,
    ReviewActionRequest,
    RollbackRequest,
    SyncResult,
)
from config import settings
from store import lifecycle, releases


def require_admin_token(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> str:
    """管理接口鉴权：缺失或错误一律 401（FR-10；Task 12 补三类情形测试）。"""

    if not x_admin_token or x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-Admin-Token")
    return x_admin_token


def create_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.post("/kb/sync", response_model=SyncResult, dependencies=[Depends(require_admin_token)])
    def sync_knowledge() -> SyncResult:
        """扫描 knowledge/ 目录，把变更导入为 draft（线上版本不受影响）。"""

        return lifecycle.sync_knowledge_dir()

    @router.post(
        "/releases",
        response_model=ReleasePublishResult,
        dependencies=[Depends(require_admin_token)],
    )
    def create_release(request: ReleaseCreateRequest) -> ReleasePublishResult:
        """发起发布：approved 版本集合 → 质量门禁 → 不可变快照（candidate/stable）。"""

        try:
            return releases.publish_release(request)
        except releases.ReleaseStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except releases.GateBlockedError as exc:
            # 门禁拦截：422 + 失败用例明细（AC-12），无任何 release 落库。
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "release quality gate blocked",
                    "failures": [failure.model_dump() for failure in exc.gate.failures],
                },
            )

    @router.get(
        "/releases",
        response_model=list[ReleaseView],
        dependencies=[Depends(require_admin_token)],
    )
    def get_releases() -> list[ReleaseView]:
        return releases.list_releases()

    @router.post(
        "/releases/{release_id}/gray-config",
        response_model=ReleaseView,
        dependencies=[Depends(require_admin_token)],
    )
    def set_gray_config(release_id: str, config: GrayConfigRequest) -> ReleaseView:
        """调整 candidate 灰度白名单/百分比；非 candidate 返回 409。"""

        try:
            return releases.update_gray_config(release_id, config)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"release not found: {release_id}")
        except releases.ReleaseStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.post(
        "/releases/{release_id}/rollback",
        response_model=ReleaseView,
        dependencies=[Depends(require_admin_token)],
    )
    def rollback_release(release_id: str, request: RollbackRequest) -> ReleaseView:
        """一键回滚：candidate 作废，或历史 rolled_back 快照恢复为 stable（AC-5）。

        - reason 必填（RollbackRequest model_validator，空值 422）；
        - 回滚当前 stable → 409；快照哈希被篡改 → 409；不存在 → 404；
        - 回滚后触发索引/检索缓存失效（releases._emit_release_change）。
        """

        try:
            return releases.rollback_release(release_id, request)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"release not found: {release_id}")
        except releases.ReleaseStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except releases.SnapshotTamperedError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get(
        "/gray/route",
        response_model=GrayRoute,
        dependencies=[Depends(require_admin_token)],
    )
    def resolve_gray_route(driver_id: str = Query(..., description="司机 ID")) -> GrayRoute:
        """调试用：查看某司机当前灰度路由结果（candidate/stable + 理由）。"""

        try:
            return releases.resolve_gray_route(driver_id)
        except releases.ReleaseStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get(
        "/versions",
        response_model=list[DocVersionView],
        dependencies=[Depends(require_admin_token)],
    )
    def get_versions(
        status: str | None = Query(default=None, description="draft/submitted/approved/published/archived"),
    ) -> list[DocVersionView]:
        return lifecycle.list_versions(status=status)

    @router.post(
        "/versions/{version_id}/review",
        response_model=DocVersionView,
        dependencies=[Depends(require_admin_token)],
    )
    def review_version(version_id: str, request: ReviewActionRequest) -> DocVersionView:
        """审核动作：submit/approve/reject（reject 必须带 comment，否则 422）。"""

        try:
            return lifecycle.review_version(
                version_id,
                action=request.action,
                actor=request.actor,
                comment=request.comment,
            )
        except LookupError:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        except lifecycle.ReviewStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get(
        "/versions/{version_id}",
        response_model=DocVersionView,
        dependencies=[Depends(require_admin_token)],
    )
    def get_version(version_id: str) -> DocVersionView:
        view = lifecycle.get_version_detail(version_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        return view

    @router.get(
        "/versions/{version_id}/diff",
        dependencies=[Depends(require_admin_token)],
    )
    def get_version_diff(version_id: str) -> dict:
        result = lifecycle.diff_against_live(version_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        return result

    @router.get(
        "/audit",
        response_model=list[AuditEventView],
        dependencies=[Depends(require_admin_token)],
    )
    def get_audit(limit: int = Query(default=100, ge=1, le=500)) -> list[AuditEventView]:
        from store import repository as repo
        from store.db import transaction
        import json

        events: list[AuditEventView] = []
        with transaction(settings.DB_PATH) as connection:
            for row in repo.list_audit_events(connection, limit=limit):
                events.append(
                    AuditEventView(
                        id=row["id"],
                        actor=row["actor"],
                        action=row["action"],
                        target_type=row["target_type"],
                        target_id=row["target_id"],
                        before_state=json.loads(row["before_state"]) if row["before_state"] else None,
                        after_state=json.loads(row["after_state"]) if row["after_state"] else None,
                        comment=row["comment"],
                        created_at=row["created_at"],
                    )
                )
        return events

    return router
