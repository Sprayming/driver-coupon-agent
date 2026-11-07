// 知识管理台（Task 18）：五个标签页覆盖 6 类操作闭环——
// 草稿同步+差异 / 审核（驳回必填理由）/ 发布（门禁结果展示）/ 灰度（白名单+百分比）/ 回滚（必填原因）/ 审计历史。
// 切换标签页即重新挂载拉最新数据；各页操作成功后自行刷新本页列表。
// token 可在右上角修改（存 localStorage，保存后所有标签页重新拉数据）。
import { useState } from "react";
import { DEFAULT_ADMIN_TOKEN, getAdminToken, setAdminToken } from "../lib/adminClient";
import { AuditTab } from "./AuditTab";
import { DraftsTab } from "./DraftsTab";
import { GrayTab } from "./GrayTab";
import { PublishTab } from "./PublishTab";
import { ReviewTab } from "./ReviewTab";

const TABS = [
  { key: "drafts", label: "草稿与同步" },
  { key: "review", label: "审核流" },
  { key: "publish", label: "发布与回滚" },
  { key: "gray", label: "灰度" },
  { key: "audit", label: "审计日志" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export function AdminConsole() {
  const [activeTab, setActiveTab] = useState<TabKey>("drafts");
  const [tokenVersion, setTokenVersion] = useState(0);
  const [tokenDraft, setTokenDraft] = useState(getAdminToken());
  const [tokenSaved, setTokenSaved] = useState(false);

  function handleSaveToken() {
    setAdminToken(tokenDraft.trim() || DEFAULT_ADMIN_TOKEN);
    setTokenSaved(true);
    // token 变了，所有标签页的数据授权都变了：bump key 强制重挂载拉新数据。
    setTokenVersion((v) => v + 1);
    window.setTimeout(() => setTokenSaved(false), 2000);
  }

  return (
    <div className="admin-console">
      <div className="admin-topbar">
        <div className="admin-tabs" role="tablist">
          {TABS.map((tab) => (
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
        <div className="token-box">
          <input
            type="password"
            value={tokenDraft}
            onChange={(e) => setTokenDraft(e.target.value)}
            placeholder="X-Admin-Token"
            aria-label="管理 Token"
          />
          <button type="button" className="mini-button" onClick={handleSaveToken}>
            {tokenSaved ? "已保存" : "保存 Token"}
          </button>
        </div>
      </div>

      <div className="admin-tab-body" key={`${activeTab}-${tokenVersion}`}>
        {activeTab === "drafts" && <DraftsTab />}
        {activeTab === "review" && <ReviewTab />}
        {activeTab === "publish" && <PublishTab />}
        {activeTab === "gray" && <GrayTab />}
        {activeTab === "audit" && <AuditTab />}
      </div>
    </div>
  );
}
