"""RAG 平台管理 API（Task 4 起）：/admin/* 需要管理员身份。

鉴权（方案 A）：
- 人工操作：/auth/login 下发的 HttpOnly 会话 Cookie，按角色校验权限；
- 机器调用：保留 X-Admin-Token（CI/脚本），视作 publisher；
- 角色：viewer 只读 / editor 改草稿与提交审核 / reviewer 审核 / publisher 发布、灰度、回滚；
- 审计的操作人取自登录态，不再信任请求体里的 actor。
"""

from __future__ import annotations

import hmac
import os
import sqlite3
from dataclasses import dataclass

from fastapi import APIRouter, Cookie, Depends, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

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
from evals.report_reader import latest_intent_report
from rag.documents import parse_front_matter
from services.doc_import import docx_to_markdown
from store import lifecycle, releases, users
from store.db import transaction
from store.repository import write_audit

READ_ROLES: tuple[str, ...] = ("viewer", "editor", "reviewer", "publisher")


# —— 用户管理（仅 publisher） ——


class AdminAccountView(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: str
    status: str
    created_at: str


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: str
    display_name: str = Field(default="", max_length=64)


class UserUpdateRequest(BaseModel):
    role: str | None = None
    display_name: str | None = Field(default=None, max_length=64)
    status: str | None = None


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=8, max_length=128)


class AdminDocumentView(BaseModel):
    doc_id: str
    title: str
    source_path: str
    kb_id: str


class DraftCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    kb_code: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=20, max_length=200_000)
    audience: str = Field(default="driver", max_length=20)
    change_note: str = Field(default="", max_length=200)
    doc_id: str | None = None


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".md", ".docx"}

MAX_EVAL_FAILURES = 60


class IntentEvalProviderView(BaseModel):
    key: str
    status: str
    accuracy: float
    macro_f1: float
    avg_ms: float
    p95_ms: float
    failure_count: int


class IntentEvalFailureView(BaseModel):
    provider: str
    case_id: str
    message: str
    expected: str
    actual: str
    tags: str


class IntentEvalReportView(BaseModel):
    available: bool
    report_file: str = ""
    generated_at: str = ""
    dataset_version: str = ""
    dataset_sha256: str = ""
    cases_count: int = 0
    llm_model: str = ""
    providers: list[IntentEvalProviderView] = Field(default_factory=list)
    failures: list[IntentEvalFailureView] = Field(default_factory=list)


def build_intent_eval_view() -> IntentEvalReportView:
    """把最新报告压成前端需要的形状；报告缺失时返回 available=False。"""

    loaded = latest_intent_report()
    if loaded is None:
        return IntentEvalReportView(available=False)

    report_file, report = loaded
    providers: list[IntentEvalProviderView] = []
    failures: list[IntentEvalFailureView] = []
    for key, data in (report.get("providers") or {}).items():
        latency = data.get("latency") or {}
        provider_failures = data.get("failures") or []
        providers.append(
            IntentEvalProviderView(
                key=key,
                status=str(data.get("status") or "unknown"),
                accuracy=float(data.get("accuracy") or 0.0),
                macro_f1=float(data.get("macro_f1") or 0.0),
                avg_ms=float(latency.get("avg_ms") or 0.0),
                p95_ms=float(latency.get("p95_ms") or 0.0),
                failure_count=len(provider_failures),
            )
        )
        for failure in provider_failures:
            if len(failures) >= MAX_EVAL_FAILURES:
                continue
            failures.append(
                IntentEvalFailureView(
                    provider=key,
                    case_id=str(failure.get("case_id") or ""),
                    message=str(failure.get("message") or ""),
                    expected=str(failure.get("expected") or ""),
                    actual=str(failure.get("actual") or ""),
                    tags=str(failure.get("tags") or ""),
                )
            )

    return IntentEvalReportView(
        available=True,
        report_file=report_file,
        generated_at=str(report.get("generated_at") or ""),
        dataset_version=str(report.get("dataset_version") or ""),
        dataset_sha256=str(report.get("dataset_sha256") or ""),
        cases_count=int(report.get("cases_count") or 0),
        llm_model=str(report.get("llm_model") or ""),
        providers=providers,
        failures=failures,
    )


@dataclass(frozen=True)
class AdminIdentity:
    """当前后台操作人：会话用户，或持机器令牌的调用方。"""

    username: str
    role: str
    via: str


def require_admin(*allowed_roles: str):
    """鉴权依赖工厂：会话用户或 X-Admin-Token，且角色必须在白名单内。

    未登录/令牌错误一律 401；已登录但角色不足 403（前端隐藏按钮不算权限，服务端为准）。
    """

    def dependency(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        session_token: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    ) -> AdminIdentity:
        if x_admin_token:
            if not hmac.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
                raise HTTPException(status_code=401, detail="invalid or missing X-Admin-Token")
            identity = AdminIdentity(username="machine", role="publisher", via="token")
        else:
            user = users.resolve_session(settings.DB_PATH, session_token)
            if user is None:
                raise HTTPException(status_code=401, detail="未登录或会话已过期")
            identity = AdminIdentity(
                username=user["username"], role=user["role"], via="session"
            )

        if allowed_roles and identity.role not in allowed_roles:
            raise HTTPException(
                status_code=403, detail=f"角色 {identity.role} 无权执行该操作"
            )
        return identity

    return dependency


def create_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.post("/kb/sync", response_model=SyncResult, dependencies=[Depends(require_admin("editor", "publisher"))])
    def sync_knowledge() -> SyncResult:
        """扫描 knowledge/ 目录，把变更导入为 draft（线上版本不受影响）。"""

        return lifecycle.sync_knowledge_dir()

    @router.get(
        "/documents",
        response_model=list[AdminDocumentView],
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_documents() -> list[AdminDocumentView]:
        """已有文档列表，供「更新已有文档」下拉选择。"""

        return [AdminDocumentView(**row) for row in lifecycle.list_documents()]

    @router.get(
        "/evals/intent",
        response_model=IntentEvalReportView,
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_intent_eval() -> IntentEvalReportView:
        """最新一次意图识别对照评测（只读展示，不在后台触发跑分）。"""

        return build_intent_eval_view()

    @router.post("/kb/documents", response_model=DocVersionView)
    def create_document(
        request: DraftCreateRequest,
        identity: AdminIdentity = Depends(require_admin("editor", "publisher")),
    ) -> DocVersionView:
        """在线编辑生成草稿：只写 draft，不触碰线上版本。"""

        try:
            return lifecycle.create_draft_from_upload(
                title=request.title,
                kb_code=request.kb_code,
                content_md=request.content,
                audience=request.audience,
                change_note=request.change_note,
                doc_id=request.doc_id,
                actor=identity.username,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.post("/kb/upload", response_model=DocVersionView)
    async def upload_document(
        file: UploadFile = File(...),
        kb_code: str = Form(default=""),
        title: str = Form(default=""),
        audience: str = Form(default="driver"),
        change_note: str = Form(default=""),
        doc_id: str | None = Form(default=None),
        identity: AdminIdentity = Depends(require_admin("editor", "publisher")),
    ) -> DocVersionView:
        """上传 .md / .docx 生成草稿；解析结果不自动发布，仍需走审核流。"""

        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}；PDF 暂未开放，请先转成 Word/Markdown",
            )

        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 5 MB 上限")
        if not data:
            raise HTTPException(status_code=400, detail="文件为空")

        if suffix == ".md":
            body_md = data.decode("utf-8-sig", errors="replace")
            metadata, _ = parse_front_matter(body_md)
            title = title.strip() or str(metadata.get("title") or "")
            kb_code = kb_code.strip() or str(metadata.get("kb") or "")
            if audience == "driver" and metadata.get("audience"):
                audience = str(metadata["audience"])
        else:
            try:
                body_md = docx_to_markdown(data)
            except Exception as exc:  # noqa: BLE001 - 损坏的 docx 返回 400 而不是 500
                raise HTTPException(status_code=400, detail=f"Word 解析失败：{exc}") from exc

        if doc_id is None:
            if not title.strip():
                raise HTTPException(status_code=400, detail="请填写标题（或在上传的 md 中提供 title）")
            if not kb_code.strip():
                raise HTTPException(status_code=400, detail="请选择知识库（或在上传的 md 中提供 kb）")

        try:
            return lifecycle.create_draft_from_upload(
                title=title.strip(),
                kb_code=kb_code.strip(),
                content_md=body_md,
                audience=audience,
                change_note=change_note.strip(),
                doc_id=doc_id,
                actor=identity.username,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.post(
        "/releases",
        response_model=ReleasePublishResult,
        dependencies=[Depends(require_admin("publisher"))],
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
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_releases() -> list[ReleaseView]:
        return releases.list_releases()

    @router.post(
        "/releases/{release_id}/gray-config",
        response_model=ReleaseView,
        dependencies=[Depends(require_admin("publisher"))],
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
        dependencies=[Depends(require_admin("publisher"))],
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
        dependencies=[Depends(require_admin(*READ_ROLES))],
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
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_versions(
        status: str | None = Query(default=None, description="draft/submitted/approved/published/archived"),
    ) -> list[DocVersionView]:
        return lifecycle.list_versions(status=status)

    @router.post(
        "/versions/{version_id}/review",
        response_model=DocVersionView,
    )
    def review_version(
        version_id: str,
        request: ReviewActionRequest,
        identity: AdminIdentity = Depends(require_admin("editor", "reviewer", "publisher")),
    ) -> DocVersionView:
        """审核动作：submit（editor/publisher）、approve/reject（reviewer/publisher）。

        reject 必须带 comment，否则 422；审计的操作人取自登录态，不接受请求体自报。
        """

        required = ("editor", "publisher") if request.action == "submit" else ("reviewer", "publisher")
        if identity.role not in required:
            raise HTTPException(
                status_code=403, detail=f"角色 {identity.role} 无权执行 action={request.action}"
            )

        try:
            return lifecycle.review_version(
                version_id,
                action=request.action,
                actor=identity.username,
                comment=request.comment,
            )
        except LookupError:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        except lifecycle.ReviewStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get(
        "/versions/{version_id}",
        response_model=DocVersionView,
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_version(version_id: str) -> DocVersionView:
        view = lifecycle.get_version_detail(version_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        return view

    @router.get(
        "/versions/{version_id}/diff",
        dependencies=[Depends(require_admin(*READ_ROLES))],
    )
    def get_version_diff(version_id: str) -> dict:
        result = lifecycle.diff_against_live(version_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"version not found: {version_id}")
        return result

    @router.get(
        "/audit",
        response_model=list[AuditEventView],
        dependencies=[Depends(require_admin(*READ_ROLES))],
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

    # —— 用户管理（仅 publisher）：增删账号、改角色、重置密码、停用 ——

    def _audit_user_action(
        identity: AdminIdentity,
        action: str,
        target: dict[str, str],
        *,
        before: dict[str, str] | None = None,
        after: dict[str, str] | None = None,
    ) -> None:
        with transaction(settings.DB_PATH) as connection:
            write_audit(
                connection,
                actor=identity.username,
                action=action,
                target_type="user",
                target_id=target["username"],
                before=before,
                after=after,
            )

    def _require_account(user_id: str) -> dict[str, str]:
        account = users.get_user(settings.DB_PATH, user_id)
        if account is None:
            raise HTTPException(status_code=404, detail=f"user not found: {user_id}")
        return account

    @router.get(
        "/users",
        response_model=list[AdminAccountView],
        dependencies=[Depends(require_admin("publisher"))],
    )
    def list_accounts() -> list[AdminAccountView]:
        return [AdminAccountView(**row) for row in users.list_users(settings.DB_PATH)]

    @router.post("/users", response_model=AdminAccountView)
    def create_account(
        request: UserCreateRequest,
        identity: AdminIdentity = Depends(require_admin("publisher")),
    ) -> AdminAccountView:
        if request.role not in users.ROLES:
            raise HTTPException(status_code=422, detail=f"未知角色: {request.role}")
        try:
            created = users.create_user(
                settings.DB_PATH,
                username=request.username,
                role=request.role,
                password=request.password,
                display_name=request.display_name,
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="用户名已存在")

        account = _require_account(created["user_id"])
        _audit_user_action(
            identity, "rbac.user.create", account, after={"role": account["role"], "status": account["status"]}
        )
        return AdminAccountView(**account)

    @router.patch("/users/{user_id}", response_model=AdminAccountView)
    def update_account(
        user_id: str,
        request: UserUpdateRequest,
        identity: AdminIdentity = Depends(require_admin("publisher")),
    ) -> AdminAccountView:
        account = _require_account(user_id)

        # 不允许把自己停用或降级，避免管理员把自己锁在门外。
        self_target = account["username"] == identity.username
        role_changed = request.role is not None and request.role != account["role"]
        disabled = request.status is not None and request.status != "active"
        if self_target and (role_changed or disabled):
            raise HTTPException(status_code=400, detail="不能停用或修改自己的角色，请让另一位管理员操作")

        losing_publisher = (
            account["role"] == "publisher"
            and account["status"] == "active"
            and (request.role not in (None, "publisher") or request.status not in (None, "active"))
        )
        if losing_publisher and users.count_active_publishers(settings.DB_PATH, exclude_user_id=user_id) == 0:
            raise HTTPException(status_code=409, detail="至少保留一个在职 publisher，否则后台将无人可发布")

        try:
            updated = users.update_user(
                settings.DB_PATH,
                user_id,
                role=request.role,
                display_name=request.display_name,
                status=request.status,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        assert updated is not None

        _audit_user_action(
            identity,
            "rbac.user.update",
            updated,
            before={"role": account["role"], "status": account["status"]},
            after={"role": updated["role"], "status": updated["status"]},
        )
        return AdminAccountView(**updated)

    @router.post("/users/{user_id}/password", response_model=AdminAccountView)
    def reset_account_password(
        user_id: str,
        request: PasswordResetRequest,
        identity: AdminIdentity = Depends(require_admin("publisher")),
    ) -> AdminAccountView:
        account = _require_account(user_id)
        users.reset_password(settings.DB_PATH, user_id, request.password)
        # 审计只记录事实，绝不记录口令内容。
        _audit_user_action(identity, "rbac.user.password_reset", account)
        return AdminAccountView(**_require_account(user_id))

    @router.delete("/users/{user_id}")
    def delete_account(
        user_id: str,
        identity: AdminIdentity = Depends(require_admin("publisher")),
    ) -> dict[str, bool]:
        account = _require_account(user_id)
        if account["username"] == identity.username:
            raise HTTPException(status_code=400, detail="不能删除自己的账号")
        if (
            account["role"] == "publisher"
            and account["status"] == "active"
            and users.count_active_publishers(settings.DB_PATH, exclude_user_id=user_id) == 0
        ):
            raise HTTPException(status_code=409, detail="至少保留一个在职 publisher")

        users.delete_user(settings.DB_PATH, user_id)
        _audit_user_action(identity, "rbac.user.delete", account, before={"role": account["role"]})
        return {"ok": True}

    return router
