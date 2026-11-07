import type { DemoUser, RuntimeRole } from "../types/api";

interface UserSwitcherProps {
  users: DemoUser[];
  selectedUserId: string;
  onSelectUser: (userId: string) => void;
  role: RuntimeRole;
  onRoleChange: (role: RuntimeRole) => void;
}

const ROLES: Array<{ key: RuntimeRole; label: string; hint: string }> = [
  { key: "driver", label: "司机", hint: "仅业务库，internal 物理隔离" },
  { key: "agent", label: "客服", hint: "可查 internal_playbook" },
  { key: "admin", label: "管理员", hint: "全部知识库" },
];

export function UserSwitcher({ users, selectedUserId, onSelectUser, role, onRoleChange }: UserSwitcherProps) {
  return (
    <div className="panel user-panel">
      <div className="panel-header">
        <span>调试用户与身份</span>
        <small>切换用户/角色会重置会话；runtime_role 只由这里注入，消息文本自称无效</small>
      </div>
      <div className="user-tabs">
        {users.map((user) => (
          <button
            key={user.profile.userId}
            type="button"
            className={user.profile.userId === selectedUserId ? "active" : ""}
            onClick={() => onSelectUser(user.profile.userId)}
          >
            <strong>{user.profile.nickname}</strong>
            <small>
              {user.profile.userId} · {user.profile.driverLevel} · risk {user.profile.riskLevel}
            </small>
            {user.profile.userId === "driver-gray-whitelist" && (
              <em className="gray-tag">灰度白名单</em>
            )}
          </button>
        ))}
      </div>
      <div className="role-tabs" role="radiogroup" aria-label="运行时角色">
        <span className="role-label">运行时角色（runtime_role）：</span>
        {ROLES.map((item) => (
          <button
            key={item.key}
            type="button"
            role="radio"
            aria-checked={role === item.key}
            className={role === item.key ? "active" : ""}
            title={item.hint}
            onClick={() => onRoleChange(item.key)}
          >
            {item.label}
          </button>
        ))}
        <small className="role-hint">{ROLES.find((r) => r.key === role)?.hint}</small>
      </div>
    </div>
  );
}
