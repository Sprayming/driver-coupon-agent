// 用户管理（方案 A）：publisher 才能进入。
// 支持新增账号、改角色、停用/启用、重置密码、删除；服务端二次校验并写审计。
import { useEffect, useState } from "react";
import {
  type AdminAccount,
  createAccount,
  deleteAccount,
  formatApiError,
  listAccounts,
  resetAccountPassword,
  updateAccount,
} from "../lib/adminClient";

const ROLES: AdminAccount["role"][] = ["viewer", "editor", "reviewer", "publisher"];

const ROLE_LABEL: Record<AdminAccount["role"], string> = {
  viewer: "只读",
  editor: "内容编辑",
  reviewer: "审核",
  publisher: "发布管理员",
};

export function UsersTab() {
  const [accounts, setAccounts] = useState<AdminAccount[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState({
    username: "",
    display_name: "",
    password: "",
    role: "viewer" as AdminAccount["role"],
  });

  async function refresh() {
    setAccounts(await listAccounts());
  }

  useEffect(() => {
    refresh().catch((err) => setError(formatApiError(err)));
  }, []);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await action();
      await refresh();
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    await run(async () => {
      await createAccount({
        username: draft.username.trim(),
        password: draft.password,
        role: draft.role,
        display_name: draft.display_name.trim(),
      });
      setDraft({ username: "", display_name: "", password: "", role: "viewer" });
      setCreating(false);
    });
  }

  return (
    <div className="users-tab">
      <div className="users-toolbar">
        <span className="users-hint">
          账号变更会写入审计；停用后该账号立即失效，重置密码会踢掉其全部会话。
        </span>
        <button type="button" className="mini-button" onClick={() => setCreating((v) => !v)}>
          {creating ? "取消" : "新增账号"}
        </button>
      </div>

      {creating && (
        <form className="users-create" onSubmit={handleCreate}>
          <input
            placeholder="用户名（字母/数字/_/-）"
            value={draft.username}
            onChange={(e) => setDraft({ ...draft, username: e.target.value })}
            minLength={2}
            maxLength={32}
            required
          />
          <input
            placeholder="姓名（可空）"
            value={draft.display_name}
            onChange={(e) => setDraft({ ...draft, display_name: e.target.value })}
          />
          <select
            value={draft.role}
            onChange={(e) => setDraft({ ...draft, role: e.target.value as AdminAccount["role"] })}
          >
            {ROLES.map((role) => (
              <option key={role} value={role}>
                {ROLE_LABEL[role]}
              </option>
            ))}
          </select>
          <input
            type="password"
            placeholder="初始密码（至少 8 位）"
            value={draft.password}
            onChange={(e) => setDraft({ ...draft, password: e.target.value })}
            minLength={8}
            required
          />
          <button className="submit-button" type="submit" disabled={busy}>
            创建
          </button>
        </form>
      )}

      {error && <p className="admin-error">{error}</p>}

      <table className="users-table">
        <thead>
          <tr>
            <th>账号</th>
            <th>姓名</th>
            <th>角色</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {accounts.map((account) => (
            <tr key={account.user_id}>
              <td className="users-name">{account.username}</td>
              <td>{account.display_name}</td>
              <td>
                <select
                  value={account.role}
                  disabled={busy}
                  onChange={(e) =>
                    run(() =>
                      updateAccount(account.user_id, {
                        role: e.target.value as AdminAccount["role"],
                      }),
                    )
                  }
                >
                  {ROLES.map((role) => (
                    <option key={role} value={role}>
                      {ROLE_LABEL[role]}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <span className={`status-pill ${account.status}`}>
                  {account.status === "active" ? "在职" : "已停用"}
                </span>
              </td>
              <td className="users-actions">
                <button
                  type="button"
                  className="mini-button"
                  disabled={busy}
                  onClick={() => {
                    const next = window.prompt(`为 ${account.username} 设置新密码（至少 8 位）`);
                    if (!next) return;
                    if (next.length < 8) {
                      setError("密码至少 8 位");
                      return;
                    }
                    void run(() => resetAccountPassword(account.user_id, next));
                  }}
                >
                  重置密码
                </button>
                <button
                  type="button"
                  className="mini-button"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      updateAccount(account.user_id, {
                        status: account.status === "active" ? "disabled" : "active",
                      }),
                    )
                  }
                >
                  {account.status === "active" ? "停用" : "启用"}
                </button>
                <button
                  type="button"
                  className="mini-button danger"
                  disabled={busy}
                  onClick={() => {
                    if (!window.confirm(`确认删除账号 ${account.username}？此操作不可恢复`)) return;
                    void run(() => deleteAccount(account.user_id));
                  }}
                >
                  删除
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
