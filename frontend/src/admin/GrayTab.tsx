// 灰度配置（Task 18）：candidate 白名单/百分比调整 + 司机路由探测。
import { useCallback, useEffect, useState } from "react";
import type { GrayRoute, ReleaseView } from "../types/api";
import { formatApiError, getGrayRoute, listReleases, setGrayConfig } from "../lib/adminClient";
import { ErrorBanner, NoticeBanner, StatusBadge } from "./ui";

export function GrayTab({ onChanged }: { onChanged?: () => void }) {
  const [candidates, setCandidates] = useState<ReleaseView[]>([]);
  const [whitelistText, setWhitelistText] = useState<Record<string, string>>({});
  const [percent, setPercent] = useState<Record<string, number>>({});
  const [probeId, setProbeId] = useState("driver-gray-whitelist");
  const [probeResult, setProbeResult] = useState<GrayRoute | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const releases = await listReleases();
      const cands = releases.filter((r) => r.status === "candidate");
      setCandidates(cands);
      setWhitelistText((prev) => {
        const next = { ...prev };
        for (const r of cands) {
          if (!(r.release_id in next)) next[r.release_id] = r.gray_whitelist.join("\n");
        }
        return next;
      });
      setPercent((prev) => {
        const next = { ...prev };
        for (const r of cands) {
          if (!(r.release_id in next)) next[r.release_id] = r.gray_percent;
        }
        return next;
      });
    } catch (e) {
      setError(formatApiError(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function handleSave(releaseId: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const release = await setGrayConfig(releaseId, {
        gray_whitelist: (whitelistText[releaseId] ?? "")
          .split(/[\s,，;；]+/)
          .filter(Boolean),
        gray_percent: percent[releaseId],
      });
      setNotice(
        `${release.release_id} 灰度配置已生效：白名单 ${release.gray_whitelist.length} 人 · ${release.gray_percent}%（变更已写审计）`,
      );
      await reload();
      onChanged?.();
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleProbe() {
    setError(null);
    setProbeResult(null);
    try {
      setProbeResult(await getGrayRoute(probeId.trim()));
    } catch (e) {
      setError(formatApiError(e));
    }
  }

  const REASON_TEXT: Record<GrayRoute["reason"], string> = {
    whitelist: "白名单命中",
    percent: "哈希落桶命中",
    stable: "未命中灰度，走稳定版",
    none: "当前无 candidate，全走 stable",
  };

  return (
    <div className="admin-tab">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      <NoticeBanner message={notice} onClose={() => setNotice(null)} />

      <h3 className="admin-section-title">灰度中的 Release（candidate）</h3>
      {candidates.length === 0 && <p className="empty-cell">当前没有 candidate 发布</p>}
      {candidates.map((r) => (
        <div key={r.release_id} className="gray-card">
          <div className="gray-card-header">
            <span>
              {r.release_id} · {r.label || "未命名"}
            </span>
            <StatusBadge status={r.status} />
          </div>
          <label>
            白名单（司机 ID，换行/逗号分隔）：
            <textarea
              rows={3}
              value={whitelistText[r.release_id] ?? ""}
              onChange={(e) => setWhitelistText((prev) => ({ ...prev, [r.release_id]: e.target.value }))}
            />
          </label>
          <label>
            灰度百分比（0-100，sha256 哈希分桶）：
            <input
              type="number"
              min={0}
              max={100}
              value={percent[r.release_id] ?? 0}
              onChange={(e) => setPercent((prev) => ({ ...prev, [r.release_id]: Number(e.target.value) }))}
            />
          </label>
          <button type="button" className="mini-button primary" disabled={busy} onClick={() => handleSave(r.release_id)}>
            保存配置（即时生效）
          </button>
        </div>
      ))}

      <h3 className="admin-section-title">司机路由探测</h3>
      <div className="admin-toolbar">
        <input
          type="text"
          value={probeId}
          onChange={(e) => setProbeId(e.target.value)}
          placeholder="输入司机 ID"
        />
        <button type="button" className="mini-button" disabled={busy || !probeId.trim()} onClick={handleProbe}>
          查询路由
        </button>
      </div>
      {probeResult && (
        <div className="probe-result">
          <div>
            <span className="label">路由结果：</span>
            <StatusBadge status={probeResult.channel} />
          </div>
          <div>
            <span className="label">理由：</span>
            {REASON_TEXT[probeResult.reason]}（{probeResult.reason}）
          </div>
          <div>
            <span className="label">Release：</span>
            {probeResult.release_id}（#{probeResult.release_no}）
          </div>
          <div>
            <span className="label">灰度百分比：</span>
            {probeResult.gray_percent}%
          </div>
        </div>
      )}
    </div>
  );
}
