interface SampleQuestionsProps {
  disabled: boolean;
  onSelect: (question: string) => void;
}

interface SampleRow {
  q: string;
  tag: string;      // 链路标签，如 "tool" / "hybrid" / "rag" / "edge"
  kind?: string;    // 命中的 BusinessFactNeed.kind（tool-calling 用）
}

// 扁平列表：编号 + 链路标签 + kind 徽章（方便在调试台一眼看出这条预期走哪条分支）。
// 所有带参数的问题对齐 driver_service_mock（8082）真实数据，确保点了就能拿到 found=True。
const SAMPLE_ROWS: SampleRow[] = [
  // ── ① tool-calling 纯实时业务 ──
  { q: "我今天跑了多少单",                         tag: "tool",   kind: "driver_order" },
  { q: "我在 ACT-12345 达标了吗",                  tag: "tool",   kind: "driver_activity" },
  { q: "我能不能参加 ACT-12345",                   tag: "tool",   kind: "driver_qualification" },
  { q: "我的退款到账了吗",                          tag: "tool",   kind: "driver_refund" },
  { q: "帮我查一下订单 88321 的退款进度",             tag: "tool",   kind: "driver_refund" },
  { q: "我最近三笔退款的状态",                       tag: "tool",   kind: "driver_refund" },
  { q: "我的账户里还有几张未使用的卡券",                tag: "tool",   kind: "driver_coupon" },
  { q: "卡券 CP-20260920-007 的核销记录",           tag: "tool",   kind: "driver_verification" },
  { q: "本月我的收入里卡券占了多少",                   tag: "tool",   kind: "driver_income" },

  // ── ② 工具 + RAG 混合 ──
  { q: "我这张卡昨天买的现在想退，还剩多少时间能退",    tag: "hybrid" },
  { q: "退款被拒了，帮我看看什么原因，该怎么申诉",       tag: "hybrid" },
  { q: "我有张卡快过期了，能续期吗",                   tag: "hybrid" },
  { q: "这张卡已经核销了，还能退吗，帮我查一下",         tag: "hybrid" },

  // ── ③ 纯 RAG 对照组 ──
  { q: "这卡买完还没使，能退不",                       tag: "rag" },
  { q: "月卡买了一个月没用，会过期吗",                 tag: "rag" },
  { q: "退款一般多长时间到账",                         tag: "rag" },
  { q: "卡券能转让给其他司机吗",                       tag: "rag" },
  { q: "平台抽佣比例是多少",                           tag: "rag" },

  // ── ④ 边界 / 寒暄 / 低置信 ──
  { q: "你好",                                       tag: "edge" },
  { q: "谢谢",                                       tag: "edge" },
  { q: "你们平台创始人是谁",                           tag: "edge" },
  { q: "你能帮我写首诗吗",                             tag: "edge" },
];

const TAG_LABEL: Record<string, string> = {
  tool:   "Tool",
  hybrid: "混合",
  rag:    "RAG",
  edge:   "边界",
};

export function SampleQuestions({ disabled, onSelect }: SampleQuestionsProps) {
  return (
    <div className="panel sample-panel">
      <div className="panel-header">
        <span>示例问题</span>
        <small>点击填入输入框（共 {SAMPLE_ROWS.length} 条）</small>
      </div>
      <ul className="sample-list">
        {SAMPLE_ROWS.map((row, i) => (
          <li key={`${row.tag}-${row.q}`} className="sample-row">
            <span className="sample-idx">{i + 1}.</span>
            <span className={`sample-tag sample-tag-${row.tag}`}>{TAG_LABEL[row.tag]}</span>
            <button
              type="button"
              className="sample-item"
              disabled={disabled}
              onClick={() => onSelect(row.q)}
            >
              {row.q}
            </button>
            {row.kind && <span className="sample-kind">{row.kind}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
