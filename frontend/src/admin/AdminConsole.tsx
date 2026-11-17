// 知识管理台（Task 18 / 方案 A）：先登录再操作，标签页按角色收敛。
// 角色：viewer 只读审计 / editor 草稿与提交 / reviewer 审核 / publisher 发布、灰度、回滚、用户管理。
// 自助注册只创建 viewer（只读），提权必须由 publisher 在「用户管理」里做。
// 注意：前端隐藏标签只是体验优化，权限由后端按会话角色强制校验。
import { useEffect, useState } from "react";
import {
  type AdminUser,
  fetchCurrentUser,
  formatApiError,
  login,
  logout,
  register,
} from "../lib/adminClient";
import { AuditTab } from "./AuditTab";
import { DraftsTab } from "./DraftsTab";
import { EvalTab } from "./EvalTab";
import { GrayTab } from "./GrayTab";
import { PublishTab } from "./PublishTab";
import { ReviewTab } from "./ReviewTab";
import { UsersTab } from "./UsersTab";

const TABS = [
  { key: "drafts", label: "草稿与同步", roles: ["editor", "publisher"] },
  { key: "review", label: "审核流", roles: ["reviewer", "publisher"] },
  { key: "publish", label: "发布与回滚", roles: ["publisher"] },
  { key: "gray", label: "灰度", roles: ["publisher"] },
  { key: "users", label: "用户管理", roles: ["publisher"] },
  { key: "audit", label: "审计日志", roles: ["viewer", "editor", "reviewer", "publisher"] },
  { key: "eval", label: "评测", roles: ["viewer", "editor", "reviewer", "publisher"] },
] as const;

type TabKey = (typeof TABS)[number]["key"];
type AuthMode = "login" | "register";

function visibleTabs(role: AdminUser["role"]) {
  return TABS.filter((tab) => (tab.roles as readonly string[]).includes(role));
}

export function AdminConsole() {
  const [user, setUser] = useState<AdminUser | null>(null);
  const [loadingUser, setLoadingUser] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>("audit");
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [loginDraft, setLoginDraft] = useState({ username: "admin", password: "" });
  const [registerDraft, setRegisterDraft] = useState({
    username: "",
    display_name: "",
    password: "",
  });
  const [loginError, setLoginError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchCurrentUser()
      .then((current) => {
        if (!alive) return;
        setUser(current);
        if (current) {
          const tabs = visibleTabs(current.role);
          if (tabs.length) setActiveTab(tabs[0].key);
        }
      })
      .catch(() => undefined)
      .finally(() => alive && setLoadingUser(false));
    return () => {
      alive = false;
    };
  }, []);

  function applyLogin(current: AdminUser) {
    setUser(current);
    const tabs = visibleTabs(current.role);
    if (tabs.length) setActiveTab(tabs[0].key);
  }

  async function handleLogin(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setLoginError("");
    try {
      applyLogin(await login(loginDraft.username.trim(), loginDraft.password));
    } catch (error) {
      setLoginError(formatApiError(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRegister(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setLoginError("");
    try {
      // 后端固定创建 viewer 并直接下发会话 Cookie，注册后即已登录。
      const created = await register(
        registerDraft.username.trim(),
        registerDraft.password,
        registerDraft.display_name.trim(),
      );
      applyLogin(created);
      setRegisterDraft({ username: "", display_name: "", password: "" });
    } catch (error) {
      setLoginError(formatApiError(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleLogout() {
    await logout().catch(() => undefined);
    setUser(null);
    setLoginDraft((draft) => ({ ...draft, password: "" }));
    setAuthMode("login");
  }

  if (loadingUser) {
    return <p className="admin-hint">正在校验登录状态…</p>;
  }

  if (!user) {
    const isRegister = authMode === "register";
    return (
      <div className="admin-auth-wrap">
        <form
          className="admin-login"
          onSubmit={isRegister ? handleRegister : handleLogin}
        >
          <h2>{isRegister ? "注册账号" : "管理台登录"}</h2>
          <p>
            {isRegister
              ? "自助注册仅创建只读账号；需要编辑、审核或发布权限，请联系管理员提权。"
              : "内部员工账号，按角色开放草稿、审核、发布、用户管理与审计权限。"}
          </p>
          <label>
            用户名
            <input
              value={isRegister ? registerDraft.username : loginDraft.username}
              onChange={(event) =>
                isRegister
                  ? setRegisterDraft((draft) => ({ ...draft, username: event.target.value }))
                  : setLoginDraft((draft) => ({ ...draft, username: event.target.value }))
              }
              autoComplete={isRegister ? "username" : "username"}
              minLength={isRegister ? 2 : undefined}
              maxLength={isRegister ? 32 : undefined}
              pattern={isRegister ? "[A-Za-z0-9_-]+" : undefined}
              title={isRegister ? "仅允许字母、数字、下划线和短横线" : undefined}
              required
            />
          </label>
          {isRegister && (
            <label>
              姓名（可空）
              <input
                value={registerDraft.display_name}
                onChange={(event) =>
                  setRegisterDraft((draft) => ({ ...draft, display_name: event.target.value }))
                }
                maxLength={64}
              />
            </label>
          )}
          <label>
            密码
            <input
              type="password"
              value={isRegister ? registerDraft.password : loginDraft.password}
              onChange={(event) =>
                isRegister
                  ? setRegisterDraft((draft) => ({ ...draft, password: event.target.value }))
                  : setLoginDraft((draft) => ({ ...draft, password: event.target.value }))
              }
              autoComplete={isRegister ? "new-password" : "current-password"}
              minLength={isRegister ? 8 : undefined}
              title={isRegister ? "至少 8 位" : undefined}
              required
            />
          </label>
          {loginError && <p className="admin-error">{loginError}</p>}
          <button className="submit-button" type="submit" disabled={submitting}>
            {submitting ? (isRegister ? "注册中..." : "登录中...") : isRegister ? "注册并登录" : "登录"}
          </button>
          <button
            type="button"
            className="link-button admin-auth-switch"
            onClick={() => {
              setAuthMode(isRegister ? "login" : "register");
              setLoginError("");
            }}
          >
            {isRegister ? "已有账号？返回登录" : "没有账号？注册一个"}
          </button>
        </form>
      </div>
    );
  }

  const tabs = visibleTabs(user.role);

  return (
    <div className="admin-console">
      <div className="admin-topbar">
        <div className="admin-tabs" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.key}
              className={activeTab === tab.key ? "active" : ""}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="account-box">
          <span>
            {user.display_name}（{user.role}）
          </span>
          <button type="button" className="mini-button" onClick={handleLogout}>
            退出
          </button>
        </div>
      </div>

      <div className="admin-tab-body" key={activeTab}>
        {activeTab === "drafts" && <DraftsTab />}
        {activeTab === "review" && <ReviewTab />}
        {activeTab === "publish" && <PublishTab />}
        {activeTab === "gray" && <GrayTab />}
        {activeTab === "users" && <UsersTab />}
        {activeTab === "audit" && <AuditTab />}
        {activeTab === "eval" && <EvalTab />}
      </div>
    </div>
  );
}
