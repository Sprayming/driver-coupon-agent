// 评测（只读）：展示最新一次意图识别对照结果，不在后台触发跑分。
import { useEffect, useState } from "react";
import {
  type IntentEvalReport,
  formatApiError,
  getIntentEvalReport,
} from "../lib/adminClient";
import { ErrorBanner } from "./ui";

const PROVIDER_LABELS: Record<string, string> = {
  rules: "规则",
  hybrid: "规则 + 语义兜底",
  llm: "全量 LLM",
  jev: "Jev",
};

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function latency(value: number): string {
  return value > 0 ? `${Math.round(value)} ms` : "—";
}

export function EvalTab() {
  const [report, setReport] = useState<IntentEvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getIntentEvalReport()
      .then(setReport)
      .catch((e) => setError(formatApiError(e)));
  }, []);

  if (error) {
    return (
      <div className="admin-tab">
        <ErrorBanner message={error} onClose={() => setError(null)} />
      </div>
    );
  }

  if (!report) {
    return <p className="admin-hint">正在读取评测报告…</p>;
  }

  if (!report.available) {
    return (
      <div className="admin-tab">
        <h3 className="admin-section-title">意图识别对照评测</h3>
        <p className="admin-hint">
          暂无评测报告。在 backend 目录运行
          <code> python -m evals.intent_bakeoff --providers rules,hybrid,llm </code>
          生成后，这里会自动展示最新一次结果。
        </p>
      </div>
    );
  }

  return (
    <div className="admin-tab">
      <h3 className="admin-section-title">意图识别对照评测（最新一次）</h3>
      <p className="admin-hint">
        数据集 {report.dataset_version || "版本未记录（旧报告）"}
        {report.dataset_sha256 ? `（${report.dataset_sha256}）` : ""} · 样本{" "}
        {report.cases_count} 条 · 生成于 {report.generated_at} · 分类模型{" "}
        {report.llm_model} · 报告 {report.report_file}
      </p>

      <table className="admin-table">
        <thead>
          <tr>
            <th>方案</th>
            <th>准确率</th>
            <th>macro-F1</th>
            <th>平均延迟</th>
            <th>P95 延迟</th>
            <th>失败样本</th>
          </tr>
        </thead>
        <tbody>
          {report.providers.map((provider) => (
            <tr key={provider.key}>
              <td>
                <strong>{PROVIDER_LABELS[provider.key] ?? provider.key}</strong>
              </td>
              <td>{provider.status === "ok" ? percent(provider.accuracy) : "未执行"}</td>
              <td>{provider.status === "ok" ? provider.macro_f1.toFixed(4) : "—"}</td>
              <td>{latency(provider.avg_ms)}</td>
              <td>{latency(provider.p95_ms)}</td>
              <td>{provider.failure_count}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="admin-section-title">失败样本（{report.failures.length}）</h3>
      {report.failures.length === 0 ? (
        <p className="admin-hint">本次对照没有失败样本。</p>
      ) : (
        <table className="admin-table">
          <thead>
            <tr>
              <th>方案</th>
              <th>样本</th>
              <th>司机原话</th>
              <th>期望意图</th>
              <th>实际判定</th>
              <th>标签</th>
            </tr>
          </thead>
          <tbody>
            {report.failures.map((failure) => (
              <tr key={`${failure.provider}-${failure.case_id}`}>
                <td>{PROVIDER_LABELS[failure.provider] ?? failure.provider}</td>
                <td className="cell-sub">{failure.case_id}</td>
                <td>{failure.message}</td>
                <td>{failure.expected}</td>
                <td>{failure.actual}</td>
                <td className="cell-sub">{failure.tags}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p className="admin-hint">
        数据集为自建中文口语样本（含否定、同义改写、关键词冲突与业务外边界），非盲测；
        指标用于版本间横向对比，不代表线上真实分布。报告只读，跑分请在 backend 执行命令行。
      </p>
    </div>
  );
}
