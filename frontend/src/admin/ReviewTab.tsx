// 审核流（Task 18）：submitted 待审核列表（通过/驳回弹窗必填理由）+ approved 待发布提示。
import { useCallback, useEffect, useState } from "react";
import type { DocVersionView } from "../types/api";
import { formatApiError, listVersions, reviewVersion } from "../lib/adminClient";
import { ErrorBanner, Modal, NoticeBanner, StatusBadge } from "./ui";

export function ReviewTab({ onChanged }: { onChanged?: () => void }) {
  const [submitted, setSubmitted] = useState<DocVersionView[]>([]);
  const [approved, setApproved] = useState<DocVersionView[]>([]);
  const [rejectTarget, setRejectTarget] = useState<DocVersionView | null>(null);
  const [rejectComment, setRejectComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [sub, appr] = await Promise.all([listVersions("submitted"), listVersions("approved")]);
      setSubmitted(sub);
      setApproved(appr);
    } catch (e) {
      setError(formatApiError(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function handleApprove(version: DocVersionView) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await reviewVersion(version.version_id, { action: "approve", actor: "admin" });
      setNotice(`${version.version_id} 审核通过（submitted → approved），可在「发布与回滚」页发布`);
      await reload();
      onChanged?.();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleRejectConfirm() {
    if (!rejectTarget) return;
    // 驳回理由必填（后端 422 兜底，这里前端先拦一层）。
    if (!rejectComment.trim()) {
      setError("驳回必须填写理由");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await reviewVersion(rejectTarget.version_id, {
        action: "reject",
        comment: rejectComment.trim(),
        actor: "admin",
      });
      setNotice(`${rejectTarget.version_id} 已驳回退回草稿（submitted → draft）`);
      setRejectTarget(null);
      setRejectComment("");
      await reload();
      onChanged?.();
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

      <h3 className="admin-section-title">待审核（submitted）</h3>
      <table className="admin-table">
        <thead>
          <tr>
            <th>版本</th>
            <th>标题</th>
            <th>知识库</th>
            <th>可见域</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {submitted.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">
                没有待审核版本
              </td>
            </tr>
          )}
          {submitted.map((v) => (
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
              <td>
                <StatusBadge status={v.status} />
              </td>
              <td className="row-actions">
                <button
                  type="button"
                  className="mini-button primary"
                  disabled={busy}
                  onClick={() => handleApprove(v)}
                >
                  通过
                </button>
                <button
                  type="button"
                  className="mini-button danger"
                  disabled={busy}
                  onClick={() => {
                    setRejectTarget(v);
                    setRejectComment("");
                  }}
                >
                  驳回
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="admin-section-title">审核通过待发布（approved）</h3>
      <table className="admin-table">
        <thead>
          <tr>
            <th>版本</th>
            <th>标题</th>
            <th>知识库</th>
            <th>可见域</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {approved.length === 0 && (
            <tr>
              <td colSpan={5} className="empty-cell">
                没有已通过待发布的版本
              </td>
            </tr>
          )}
          {approved.map((v) => (
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
              <td>
                <StatusBadge status={v.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {rejectTarget && (
        <Modal title={`驳回 ${rejectTarget.version_id}`} onClose={() => setRejectTarget(null)}>
          <p className="modal-hint">驳回理由必填，将写入 review_events 与审计（submitted → draft）。</p>
          <textarea
            rows={4}
            value={rejectComment}
            onChange={(e) => setRejectComment(e.target.value)}
            placeholder="例如：退款比例与现行政策冲突，请核对 P-REF-001"
          />
          <div className="modal-actions">
            <button type="button" className="mini-button" onClick={() => setRejectTarget(null)}>
              取消
            </button>
            <button
              type="button"
              className="mini-button danger"
              disabled={busy}
              onClick={handleRejectConfirm}
            >
              确认驳回
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
