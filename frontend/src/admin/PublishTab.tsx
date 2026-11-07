// 发布与回滚（Task 18）：勾选 approved 版本发起发布（展示门禁结果）+ release 列表 + 回滚弹窗必填原因。
import { useCallback, useEffect, useState } from "react";
import type { DocVersionView, GateResult, ReleaseView } from "../types/api";
import { createRelease, formatApiError, listReleases, listVersions, rollbackRelease } from "../lib/adminClient";
import { ErrorBanner, Modal, NoticeBanner, StatusBadge } from "./ui";

export function PublishTab({ onChanged }: { onChanged?: () => void }) {
  const [approved, setApproved] = useState<DocVersionView[]>([]);
  const [releases, setReleases] = useState<ReleaseView[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [label, setLabel] = useState("");
  const [asCandidate, setAsCandidate] = useState(true);
  const [whitelist, setWhitelist] = useState("");
  const [grayPercent, setGrayPercent] = useState(0);
  const [gateResult, setGateResult] = useState<GateResult | null>(null);
  const [rollbackTarget, setRollbackTarget] = useState<ReleaseView | null>(null);
  const [rollbackReason, setRollbackReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [appr, rel] = await Promise.all([listVersions("approved"), listReleases()]);
      setApproved(appr);
      setReleases(rel);
      // 已发布的版本从勾选里剔除，避免发起时被后端 409 拒绝。
      setSelected((prev) => prev.filter((id) => appr.some((v) => v.version_id === id)));
    } catch (e) {
      setError(formatApiError(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  function toggleVersion(versionId: string) {
    setSelected((prev) =>
      prev.includes(versionId) ? prev.filter((id) => id !== versionId) : [...prev, versionId],
    );
  }

  async function handlePublish() {
    if (selected.length === 0) {
      setError("请先勾选至少一个 approved 版本");
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    setGateResult(null);
    try {
      const result = await createRelease({
        version_ids: selected,
        label: label.trim(),
        as_candidate: asCandidate,
        gray_whitelist: asCandidate
          ? whitelist.split(/[\s,，;；]+/).filter(Boolean)
          : [],
        gray_percent: asCandidate ? grayPercent : 0,
      });
      setGateResult(result.gate);
      setNotice(
        `发布成功：${result.release.release_id}（${result.release.status}，门禁 ${result.gate.mode} 通过）`,
      );
      setSelected([]);
      setLabel("");
      await reload();
      onChanged?.();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleRollbackConfirm() {
    if (!rollbackTarget) return;
    if (!rollbackReason.trim()) {
      setError("回滚原因必填");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const release = await rollbackRelease(rollbackTarget.release_id, {
        reason: rollbackReason.trim(),
      });
      setNotice(`已回滚：${release.release_id} 现为 ${release.status}`);
      setRollbackTarget(null);
      setRollbackReason("");
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

      <h3 className="admin-section-title">发起发布</h3>
      <div className="publish-form">
        <table className="admin-table">
          <thead>
            <tr>
              <th>勾选</th>
              <th>版本</th>
              <th>标题</th>
              <th>知识库</th>
              <th>可见域</th>
            </tr>
          </thead>
          <tbody>
            {approved.length === 0 && (
              <tr>
                <td colSpan={5} className="empty-cell">
                  没有 approved 版本，先去「审核流」页通过审核
                </td>
              </tr>
            )}
            {approved.map((v) => (
              <tr key={v.version_id}>
                <td>
                  <input
                    type="checkbox"
                    checked={selected.includes(v.version_id)}
                    onChange={() => toggleVersion(v.version_id)}
                  />
                </td>
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
              </tr>
            ))}
          </tbody>
        </table>

        <div className="publish-options">
          <label>
            发布标签：
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="例如：周末日卡灰度版"
            />
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={asCandidate}
              onChange={(e) => setAsCandidate(e.target.checked)}
            />
            作为 candidate 灰度发布（不勾选则直接 stable，均需过质量门禁）
          </label>
          {asCandidate && (
            <>
              <label>
                灰度白名单（司机 ID，逗号/换行分隔）：
                <textarea
                  rows={2}
                  value={whitelist}
                  onChange={(e) => setWhitelist(e.target.value)}
                  placeholder="driver-gray-whitelist"
                />
              </label>
              <label>
                灰度百分比（0-100）：
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={grayPercent}
                  onChange={(e) => setGrayPercent(Number(e.target.value))}
                />
              </label>
            </>
          )}
          <button type="button" className="submit-button" disabled={busy} onClick={handlePublish}>
            {busy ? "发布中..." : `发布所选（${selected.length} 个版本）`}
          </button>
        </div>
      </div>

      {gateResult && (
        <div className={`gate-panel ${gateResult.passed ? "passed" : "blocked"}`}>
          <div className="gate-header">
            质量门禁：{gateResult.passed ? "通过" : "拦截"}（mode={gateResult.mode}）
          </div>
          <div className="gate-meta">检查版本：{gateResult.checked_versions.join(", ") || "无"}</div>
          {gateResult.failures.length > 0 && (
            <ul className="gate-failures">
              {gateResult.failures.map((f, i) => (
                <li key={i}>
                  {f.case_id}: {f.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <h3 className="admin-section-title">发布历史</h3>
      <table className="admin-table">
        <thead>
          <tr>
            <th>Release</th>
            <th>标签</th>
            <th>状态</th>
            <th>灰度配置</th>
            <th>包含版本</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {releases.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">
                还没有任何发布
              </td>
            </tr>
          )}
          {releases.map((r) => (
            <tr key={r.release_id}>
              <td>
                {r.release_id}
                <small className="cell-sub">{r.created_at}</small>
              </td>
              <td>{r.label || "—"}</td>
              <td>
                <StatusBadge status={r.status} />
              </td>
              <td>
                {r.status === "candidate"
                  ? `白名单 ${r.gray_whitelist.length} 人 · ${r.gray_percent}%`
                  : r.rollback_reason
                    ? `回滚原因：${r.rollback_reason}`
                    : "—"}
              </td>
              <td>{r.version_ids.join(", ") || "—"}</td>
              <td className="row-actions">
                <button
                  type="button"
                  className="mini-button danger"
                  disabled={busy}
                  onClick={() => {
                    setRollbackTarget(r);
                    setRollbackReason("");
                  }}
                >
                  回滚到此
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {rollbackTarget && (
        <Modal title={`回滚 ${rollbackTarget.release_id}`} onClose={() => setRollbackTarget(null)}>
          <p className="modal-hint">
            回滚原因必填（写入审计）。candidate 回滚后作废；历史快照回滚会恢复为 stable（完整性校验通过后）。
          </p>
          <textarea
            rows={3}
            value={rollbackReason}
            onChange={(e) => setRollbackReason(e.target.value)}
            placeholder="例如：灰度内容口径有误，先回稳定版"
          />
          <div className="modal-actions">
            <button type="button" className="mini-button" onClick={() => setRollbackTarget(null)}>
              取消
            </button>
            <button
              type="button"
              className="mini-button danger"
              disabled={busy}
              onClick={handleRollbackConfirm}
            >
              确认回滚
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
