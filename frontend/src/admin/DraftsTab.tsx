// 草稿与同步（Task 18）：md 目录同步入库为草稿、草稿列表、与线上差异、提交审核。
import { useCallback, useEffect, useState } from "react";
import type { DocVersionView, SyncResult, VersionDiff } from "../types/api";
import {
  formatApiError,
  getVersionDiff,
  listVersions,
  reviewVersion,
  syncKnowledge,
} from "../lib/adminClient";
import { ErrorBanner, NoticeBanner, StatusBadge } from "./ui";

export function DraftsTab() {
  const [drafts, setDrafts] = useState<DocVersionView[]>([]);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);
  const [diff, setDiff] = useState<{ versionId: string; data: VersionDiff } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      setDrafts(await listVersions("draft"));
    } catch (e) {
      setError(formatApiError(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function handleSync() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await syncKnowledge();
      setSyncResult(result);
      setNotice(
        result.changed === 0
          ? `扫描 ${result.scanned} 篇 md，无变更（unchanged=${result.unchanged}），未产生新草稿`
          : `扫描 ${result.scanned} 篇 md，新增 ${result.changed} 个草稿：${result.created_version_ids.join(", ")}`,
      );
      await reload();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleDiff(versionId: string) {
    setError(null);
    try {
      const data = await getVersionDiff(versionId);
      setDiff((prev) => (prev?.versionId === versionId ? null : { versionId, data }));
    } catch (e) {
      setError(formatApiError(e));
    }
  }

  async function handleSubmit(versionId: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await reviewVersion(versionId, { action: "submit", actor: "admin" });
      setNotice(`${versionId} 已提交审核（draft → submitted）`);
      setDiff(null);
      await reload();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="admin-tab">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      <NoticeBanner message={notice} onClose={() => setNotice(null)} />

      <div className="admin-toolbar">
        <button type="button" className="submit-button" disabled={busy} onClick={handleSync}>
          同步 knowledge/ 目录 → 草稿
        </button>
        <small>改动 md 后点同步：无变更不产生新版本；变更以 draft 状态入库，线上不受影响</small>
      </div>

      {syncResult && (
        <div className="sync-summary">
          扫描 {syncResult.scanned} · 未变更 {syncResult.unchanged} · 新草稿 {syncResult.changed}
        </div>
      )}

      <table className="admin-table">
        <thead>
          <tr>
            <th>版本</th>
            <th>标题</th>
            <th>知识库</th>
            <th>可见域</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {drafts.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">
                当前没有草稿
              </td>
            </tr>
          )}
          {drafts.map((v) => (
            <tr key={v.version_id}>
              <td>
                {v.version_id}
                <small className="cell-sub">
                  v{v.version_no} · {v.doc_id}
                </small>
              </td>
              <td>{v.title}</td>
              <td>{v.kb_id}</td>
              <td>
                <StatusBadge status={v.audience} />
              </td>
              <td>{v.created_at}</td>
              <td className="row-actions">
                <button type="button" className="mini-button" onClick={() => handleDiff(v.version_id)}>
                  {diff?.versionId === v.version_id ? "收起差异" : "查看差异"}
                </button>
                <button
                  type="button"
                  className="mini-button primary"
                  disabled={busy}
                  onClick={() => handleSubmit(v.version_id)}
                >
                  提交审核
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {diff && (
        <div className="diff-panel">
          <div className="diff-header">
            <span>
              差异对比：{diff.data.version_id} vs 线上{" "}
              {diff.data.baseline_version_id ?? "（无线上版本）"}
            </span>
          </div>
          {diff.data.added_lines.length === 0 && diff.data.removed_lines.length === 0 && (
            <p className="diff-empty">与线上版本内容一致</p>
          )}
          {diff.data.added_lines.map((line, i) => (
            <div key={`a${i}`} className="diff-line added">
              + {line}
            </div>
          ))}
          {diff.data.removed_lines.map((line, i) => (
            <div key={`r${i}`} className="diff-line removed">
              - {line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
