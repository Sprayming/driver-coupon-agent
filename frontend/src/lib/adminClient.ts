// RAG 平台管理 API client（Task 17 / 方案 A 账号体系）。
// 鉴权改为 HttpOnly 会话 Cookie（credentials: include），前端不再接触任何令牌；
// 机器调用仍可用 X-Admin-Token，但人工操作一律走 /auth/login。
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

export interface AdminUser {
  username: string;
  display_name: string;
  role: "viewer" | "editor" | "reviewer" | "publisher";
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // 上传走 FormData：必须让浏览器自己带 multipart boundary，不能手写 Content-Type。
  const isFormData = init?.body instanceof FormData;
  const response = await fetch(path, {
    ...init,
    credentials: "include",
    headers: isFormData
      ? (init?.headers ?? {})
      : {
          "Content-Type": "application/json",
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

function adminFetch<T>(path: string, init?: RequestInit): Promise<T> {
  return request<T>(`/api/admin${path}`, init);
}

// —— 账号（方案 A） ——

export function login(username: string, password: string): Promise<AdminUser> {
  return request<AdminUser>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function logout(): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/auth/logout", { method: "POST" });
}

/** 未登录时返回 null（401 不算异常），供管理台判断是否显示登录表单。 */
export async function fetchCurrentUser(): Promise<AdminUser | null> {
  try {
    return await request<AdminUser>("/api/auth/me");
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
}

/** 自助注册：后端固定创建 viewer（只读），提权必须由 publisher 在用户管理里做。 */
export function register(
  username: string,
  password: string,
  displayName: string,
): Promise<AdminUser> {
  return request<AdminUser>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password, display_name: displayName }),
  });
}

// —— 用户管理（仅 publisher） ——

export interface AdminAccount {
  user_id: string;
  username: string;
  display_name: string;
  role: AdminUser["role"];
  status: "active" | "disabled";
  created_at: string;
}

export function listAccounts(): Promise<AdminAccount[]> {
  return adminFetch<AdminAccount[]>("/users");
}

export function createAccount(input: {
  username: string;
  password: string;
  role: AdminUser["role"];
  display_name: string;
}): Promise<AdminAccount> {
  return adminFetch<AdminAccount>("/users", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updateAccount(
  userId: string,
  patch: Partial<Pick<AdminAccount, "role" | "display_name" | "status">>,
): Promise<AdminAccount> {
  return adminFetch<AdminAccount>(`/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function resetAccountPassword(userId: string, password: string): Promise<AdminAccount> {
  return adminFetch<AdminAccount>(`/users/${userId}/password`, {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

export function deleteAccount(userId: string): Promise<{ ok: boolean }> {
  return adminFetch<{ ok: boolean }>(`/users/${userId}`, { method: "DELETE" });
}

// —— 知识同步与版本（Task 4） ——

export function syncKnowledge(): Promise<SyncResult> {
  return adminFetch<SyncResult>("/kb/sync", { method: "POST" });
}

export function listVersions(status?: string): Promise<DocVersionView[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return adminFetch<DocVersionView[]>(`/versions${query}`);
}

// —— 文档创建/上传（最小版：只生成 draft） ——

export interface AdminDocument {
  doc_id: string;
  title: string;
  source_path: string;
  kb_id: string;
}

export function listDocuments(): Promise<AdminDocument[]> {
  return adminFetch<AdminDocument[]>("/documents");
}

// —— 评测报告（只读）：最新一次意图识别对照 ——

export interface IntentEvalProvider {
  key: string;
  status: string;
  accuracy: number;
  macro_f1: number;
  avg_ms: number;
  p95_ms: number;
  failure_count: number;
}

export interface IntentEvalFailure {
  provider: string;
  case_id: string;
  message: string;
  expected: string;
  actual: string;
  tags: string;
}

export interface IntentEvalReport {
  available: boolean;
  report_file: string;
  generated_at: string;
  dataset_version: string;
  dataset_sha256: string;
  cases_count: number;
  llm_model: string;
  providers: IntentEvalProvider[];
  failures: IntentEvalFailure[];
}

export function getIntentEvalReport(): Promise<IntentEvalReport> {
  return adminFetch<IntentEvalReport>("/evals/intent");
}

export interface DraftInput {
  title: string;
  kb_code: string;
  content: string;
  audience: string;
  change_note: string;
  doc_id?: string | null;
}

export function createDraft(input: DraftInput): Promise<DocVersionView> {
  return adminFetch<DocVersionView>("/kb/documents", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function uploadDraft(
  file: File,
  fields: Omit<DraftInput, "content">,
): Promise<DocVersionView> {
  const form = new FormData();
  form.append("file", file);
  for (const [key, value] of Object.entries(fields)) {
    if (value !== null && value !== undefined && value !== "") form.append(key, value);
  }
  return adminFetch<DocVersionView>("/kb/upload", { method: "POST", body: form });
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
    // 401 有两种含义：登录失败（用户名/密码错误）与会话过期。
    // 后端 detail 已经区分了，优先展示 detail；没有 detail 才用通用文案。
    if (typeof error.detail === "string" && error.detail.trim()) {
      return error.status === 401 ? error.detail : `${error.status}: ${error.detail}`;
    }
    if (error.status === 401) return "未登录或会话已过期（401），请重新登录";
    if (error.detail && typeof error.detail === "object" && "message" in (error.detail as Record<string, unknown>)) {
      const detail = error.detail as { message: string; failures?: Array<{ case_id: string; reason: string }> };
      const failures = (detail.failures ?? []).map((f) => `${f.case_id}: ${f.reason}`).join("；");
      return failures ? `${detail.message} —— ${failures}` : detail.message;
    }
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : "未知错误";
}

export type { GateResult };
