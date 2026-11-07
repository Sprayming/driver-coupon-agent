import { FormEvent, useState } from "react";

import { AdminConsole } from "./admin/AdminConsole";
import { MessageList } from "./components/MessageList";
import { SampleQuestions } from "./components/SampleQuestions";
import { TracePanel } from "./components/TracePanel";
import { UserSwitcher } from "./components/UserSwitcher";
import { useAgentChat } from "./hooks/useAgentChat";

// Task 18：顶层视图切换——对话调试台 / 知识管理台（无路由依赖，状态切换即可）。
type AppView = "chat" | "admin";

export default function App() {
  const {
    users,
    selectedUser,
    selectedUserId,
    role,
    messages,
    isLoading,
    activeResponse,
    selectUser,
    selectRole,
    sendMessage,
  } = useAgentChat();

  const [view, setView] = useState<AppView>("chat");
  const [draft, setDraft] = useState("我的退款到账了吗");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (isLoading || !draft.trim()) return;
    await sendMessage(draft.trim());
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Driver Coupon Refund Agent Debug Console</p>
          <h1>卡券退款 Agent 调试后台</h1>
        </div>
        <p>观察向量检索、Citations 引用和成本信号如何支撑 Agent 回答。</p>
        <nav className="view-switch" aria-label="视图切换">
          <button
            type="button"
            className={view === "chat" ? "active" : ""}
            onClick={() => setView("chat")}
          >
            对话调试
          </button>
          <button
            type="button"
            className={view === "admin" ? "active" : ""}
            onClick={() => setView("admin")}
          >
            知识管理台
          </button>
        </nav>
      </header>

      {view === "admin" ? (
        <AdminConsole />
      ) : (
        <>
          <UserSwitcher
            users={users}
            selectedUserId={selectedUserId}
            onSelectUser={selectUser}
            role={role}
            onRoleChange={selectRole}
          />

          <section className="top-grid" aria-label="对话与调试">
            <MessageList messages={messages} isLoading={isLoading} />
            <TracePanel response={activeResponse} />
          </section>

          <section className="bottom-grid" aria-label="输入与示例">
            <form className="panel composer-panel" onSubmit={handleSubmit}>
              <div className="panel-header">
                <span>输入用户问题</span>
                <small>当前用户: {selectedUser.profile.nickname}</small>
              </div>

              <textarea
                id="question"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                disabled={isLoading}
                rows={4}
                placeholder="例如：这卡买完还没使，能退不"
              />

              <div className="composer-actions">
                <button className="submit-button" disabled={isLoading} type="submit">
                  {isLoading ? "处理中..." : "发送给 Agent"}
                </button>
              </div>
            </form>

            <SampleQuestions disabled={isLoading} onSelect={setDraft} />
          </section>
        </>
      )}
    </main>
  );
}
