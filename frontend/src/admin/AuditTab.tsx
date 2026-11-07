// 审计日志（Task 18）：全部管理动作的审计流水，可展开 before/after 状态。
import { useCallback, useEffect, useState } from "react";
import type { AuditEventView } from "../types/api";
import { formatApiError, getAudit } from "../lib/adminClient";
import { ErrorBanner, JsonDetails } from "./ui";

export function AuditTab() {
  const [events, setEvents] = useState<AuditEventView[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    setBusy(true);
    try {
      setEvents(await getAudit(200));
      setError(null);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return (
    <div className="admin-tab">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      <div className="admin-toolbar">
        <button type="button" className="mini-button" disabled={busy} onClick={reload}>
          刷新
        </button>
        <small>所有写操作（同步/审核/发布/灰度/回滚/权限拦截）都会落审计表</small>
      </div>

      <table className="admin-table">
        <thead>
          <tr>
            <th>时间</th>
            <th>动作</th>
            <th>操作者</th>
            <th>对象</th>
            <th>备注</th>
            <th>状态变化</th>
          </tr>
        </thead>
        <tbody>
          {events.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">
                暂无审计事件
              </td>
            </tr>
          )}
          {events.map((e) => (
            <tr key={e.id}>
              <td>
                {e.created_at}
                <small className="cell-sub">#{e.id}</small>
              </td>
              <td>
                <code className="action-code">{e.action}</code>
              </td>
              <td>{e.actor}</td>
              <td>
                {e.target_type ? `${e.target_type}:` : ""}
                {e.target_id || "—"}
              </td>
              <td className="comment-cell">{e.comment || "—"}</td>
              <td>
                <JsonDetails label="before/after" data={{ before: e.before_state, after: e.after_state }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
