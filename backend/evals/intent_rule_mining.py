"""从评测集 tags 统计规则误判特征，输出候选升级条件。

用途：把「靠人总结失败类型」变成「靠数据筛失败类型」。
- 对每道题跑纯规则分类；
- 按 tags 分组统计命中率、准确率、平均升级分；
- 打印低准确率 tag 和高频错判对，作为新增升级条件 / 补词表的候选。

用法（在 backend 目录执行）：
    python -m evals.intent_rule_mining
"""

from __future__ import annotations

from collections import Counter, defaultdict

from evals.intent_bakeoff import load_dataset
from rag.intent import classify_intent
from rag.intent_semantic import ESCALATE_THRESHOLD, escalation_score


def main() -> int:
    version, digest, cases = load_dataset()
    print(f"数据集 {version}  hash={digest}  题数={len(cases)}\n")

    per_tag: dict[str, dict[str, float]] = defaultdict(
        lambda: {"total": 0, "hit": 0, "correct": 0, "score": 0.0}
    )
    misses: Counter[tuple[str, str]] = Counter()
    correct_total = 0
    hit_total = 0

    for case in cases:
        result = classify_intent(case.user_message)
        score, _ = escalation_score(case.user_message, result)
        correct = result.intent == case.expected_intent
        correct_total += int(correct)
        hit_total += int(result.intent != "unknown")
        for tag in case.tags or ["（无标签）"]:
            bucket = per_tag[tag]
            bucket["total"] += 1
            bucket["hit"] += int(result.intent != "unknown")
            bucket["correct"] += int(correct)
            bucket["score"] += score
        if not correct:
            misses[(case.expected_intent, result.intent)] += 1

    header = f"{'tag':<16}{'样本':>6}{'命中':>6}{'正确':>6}{'准确率':>9}{'平均升级分':>11}  建议"
    print(header)
    print("-" * len(header))
    for tag, bucket in sorted(
        per_tag.items(), key=lambda kv: kv[1]["correct"] / kv[1]["total"]
    ):
        total = bucket["total"]
        accuracy = bucket["correct"] / total
        avg_score = bucket["score"] / total
        if accuracy >= 0.9:
            advice = "规则已足够"
        elif avg_score >= ESCALATE_THRESHOLD:
            advice = "已有升级条件覆盖"
        else:
            advice = "候选：补词表或新增升级条件"
        print(
            f"{tag:<16}{int(total):>6}{int(bucket['hit']):>6}{int(bucket['correct']):>6}"
            f"{accuracy:>8.1%}{avg_score:>11.2f}  {advice}"
        )

    print("-" * len(header))
    total_all = len(cases)
    print(
        f"{'总计（去重）':<16}{total_all:>6}{hit_total:>6}{correct_total:>6}"
        f"{correct_total / total_all:>8.1%}{'':>11}  以此为准"
    )
    print("注：一题可带多个 tag，上方各 tag 行不可直接相加，准确率口径以总计行为准。")

    print("\n高频错判对（期望 → 实际）：")
    if not misses:
        print("  无")
    for (expected, actual), count in misses.most_common(10):
        print(f"  {expected:<20} → {actual:<20} {count} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
