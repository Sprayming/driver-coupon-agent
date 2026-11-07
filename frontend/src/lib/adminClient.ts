// RAG 平台管理 API client（Task 17）。
// 所有 /admin/* 请求统一携带 X-Admin-Token（后端 Task 12 鉴权依赖校验，
// 缺失/错误一律 401）。Token 存 localStorage，默认 dev-admin-token（课程环境）。
import type {
  AuditEventView,
  DocVersionView,
  GateResult,
  GrayConfigRequest,
  GrayRoute,
  ReleaseCreateRequest,
  ReleasePublishResult,
  ReleaseView,
  ReviewActionRequest,
  RollbackRequest,
  SyncResult,
  VersionDiff,
} from "../types/api";

const TOKEN_STORAGE_KEY = "admin-token";
// 与后端 settings.ADMIN_TOKEN 缺省值一致，可在管理台右上角修改。
export const DEFAULT_ADMIN_TOKEN = "dev-admin-token";

export function getAdminToken(): string {
  return localStorage.getItem(TOKEN_STORAGE_KEY) || DEFAULT_ADMIN_TOKEN;
}

export function setAdminToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(`HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function adminFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/admin${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Token": getAdminToken(),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    // FastAPI 错误体形如 {detail: string | {...}}，原样抛给调用方展示。
    let detail: unknown = null;
    try {
      detail = (await response.json()).detail ?? null;
    } catch {
      detail = null;
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

// —— 知识同步与版本（Task 4） ——

export function syncKnowledge(): Promise<SyncResult> {
  return adminFetch<SyncResult>("/kb/sync", { method: "POST" });
}

export function listVersions(status?: string): Promise<DocVersionView[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return adminFetch<DocVersionView[]>(`/versions${query}`);
}

export function getVersion(versionId: string): Promise<DocVersionView> {
  return adminFetch<DocVersionView>(`/versions/${versionId}`);
}

export function getVersionDiff(versionId: string): Promise<VersionDiff> {
  return adminFetch<VersionDiff>(`/versions/${versionId}/diff`);
}

// —— 审核流（Task 5） ——

export function reviewVersion(
  versionId: string,
  request: ReviewActionRequest,
): Promise<DocVersionView> {
  return adminFetch<DocVersionView>(`/versions/${versionId}/review`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// —— 发布与门禁（Task 6/16） ——

export function createRelease(request: ReleaseCreateRequest): Promise<ReleasePublishResult> {
  return adminFetch<ReleasePublishResult>("/releases", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function listReleases(): Promise<ReleaseView[]> {
  return adminFetch<ReleaseView[]>("/releases");
}

// —— 回滚（Task 8） ——

export function rollbackRelease(releaseId: string, request: RollbackRequest): Promise<ReleaseView> {
  return adminFetch<ReleaseView>(`/releases/${releaseId}/rollback`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// —— 灰度（Task 7） ——

export function setGrayConfig(
  releaseId: string,
  request: GrayConfigRequest,
): Promise<ReleaseView> {
  return adminFetch<ReleaseView>(`/releases/${releaseId}/gray-config`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getGrayRoute(driverId: string): Promise<GrayRoute> {
  return adminFetch<GrayRoute>(`/gray/route?driver_id=${encodeURIComponent(driverId)}`);
}

// —— 审计（Task 4 起） ——

export function getAudit(limit = 100): Promise<AuditEventView[]> {
  return adminFetch<AuditEventView[]>(`/audit?limit=${limit}`);
}

/** 把 ApiError 的 detail 转成可展示文案（门禁拦截返回 {message, failures}）。 */
export function formatApiError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "鉴权失败（401）：请检查管理 Token";
    if (error.detail && typeof error.detail === "object" && "message" in (error.detail as Record<string, unknown>)) {
      const detail = error.detail as { message: string; failures?: Array<{ case_id: string; reason: string }> };
      const failures = (detail.failures ?? []).map((f) => `${f.case_id}: ${f.reason}`).join("；");
      return failures ? `${detail.message} —— ${failures}` : detail.message;
    }
    if (typeof error.detail === "string") return `${error.status}: ${error.detail}`;
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : "未知错误";
}

export type { GateResult };
