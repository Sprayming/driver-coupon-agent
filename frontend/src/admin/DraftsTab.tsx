// 草稿与同步（Task 18）：在线编辑 / 上传 .md/.docx 生成草稿、md 目录同步、草稿列表、差异、提交审核。
import { useCallback, useEffect, useState } from "react";
import type { DocVersionView, SyncResult, VersionDiff } from "../types/api";
import type { AdminDocument } from "../lib/adminClient";
import {
  createDraft,
  formatApiError,
  getVersionDiff,
  listDocuments,
  listVersions,
  reviewVersion,
  syncKnowledge,
  uploadDraft,
} from "../lib/adminClient";
import { ErrorBanner, NoticeBanner, StatusBadge } from "./ui";

const KB_OPTIONS = [
  { value: "refund", label: "退款规则" },
  { value: "usage", label: "卡券使用" },
  { value: "complaint", label: "投诉处理" },
  { value: "purchase", label: "购买与商品" },
  { value: "internal_playbook", label: "内部作业手册" },
];

export function DraftsTab() {
  const [drafts, setDrafts] = useState<DocVersionView[]>([]);
  const [documents, setDocuments] = useState<AdminDocument[]>([]);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);
  const [diff, setDiff] = useState<{ versionId: string; data: VersionDiff } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [mode, setMode] = useState<"editor" | "upload">("editor");
  const [docId, setDocId] = useState("");
  const [title, setTitle] = useState("");
  const [kbCode, setKbCode] = useState("refund");
  const [audience, setAudience] = useState("driver");
  const [changeNote, setChangeNote] = useState("");
  const [content, setContent] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const reload = useCallback(async () => {
    try {
      const [versions, docs] = await Promise.all([listVersions("draft"), listDocuments()]);
      setDrafts(versions);
      setDocuments(docs);
    } catch (e) {
      setError(formatApiError(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function handleCreate() {
    if (mode === "upload" && !file) {
      setError("请先选择 .md 或 .docx 文件");
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const fields = {
        title: title.trim(),
        kb_code: kbCode,
        audience,
        change_note: changeNote.trim(),
        doc_id: docId || null,
      };
      const version =
        mode === "upload" && file
          ? await uploadDraft(file, fields)
          : await createDraft({ ...fields, content });

      setNotice(
        `已生成草稿 ${version.version_id}（v${version.version_no}，状态 draft）。` +
          "请到「审核」标签提交审批；发布前线上版本不受影响。",
      );
      setContent("");
      setFile(null);
      setChangeNote("");
      await reload();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

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

      <div className="doc-create">
        <div className="doc-create-head">
          <strong>更新文档 → 生成草稿</strong>
          <div className="doc-mode">
            <button
              type="button"
              className={mode === "editor" ? "active" : ""}
              onClick={() => setMode("editor")}
            >
              在线编辑
            </button>
            <button
              type="button"
              className={mode === "upload" ? "active" : ""}
              onClick={() => setMode("upload")}
            >
              上传文件
            </button>
          </div>
        </div>

        <div className="doc-form-grid">
          <label>
            更新范围
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              <option value="">新建文档</option>
              {documents.map((doc) => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.title}（{doc.source_path}）
                </option>
              ))}
            </select>
          </label>
          {!docId && (
            <>
              <label>
                标题
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="如：退款时效说明"
                />
              </label>
              <label>
                知识库
                <select value={kbCode} onChange={(e) => setKbCode(e.target.value)}>
                  {KB_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          <label>
            可见域
            <select value={audience} onChange={(e) => setAudience(e.target.value)}>
              <option value="driver">司机端</option>
              <option value="internal">内部</option>
            </select>
          </label>
        </div>

        {mode === "editor" ? (
          <label className="doc-full">
            正文（Markdown）
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={"# 退款时效\n\n- 未核销卡券可在购买后 7 天内申请退款\n- 已核销卡券需人工审核"}
            rows={8}
            />
          </label>
        ) : (
          <label className="doc-full">
            上传文件（.md / .docx，≤ 5 MB）
            <input
              key={file?.name ?? "empty"}
              type="file"
              accept=".md,.docx"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file && <small>已选择：{file.name}</small>}
          </label>
        )}

        <div className="doc-form-grid">
          <label>
            变更说明
            <input
              type="text"
              value={changeNote}
              onChange={(e) => setChangeNote(e.target.value)}
              placeholder="如：补充 48 小时退款口径"
            />
          </label>
        </div>

        <div className="doc-actions">
          <button type="button" className="submit-button" disabled={busy} onClick={handleCreate}>
            生成草稿
          </button>
          <small>只写 draft；之后必须经提交 → 审核 → 发布，线上版本不受影响</small>
        </div>
      </div>

      <h3 className="admin-section-title">目录同步（工程通道 / 批量导入）</h3>
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
