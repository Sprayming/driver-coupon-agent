"""L38：评测回归框架——从 cases.yml 跑端到端评测。

和小哲电商 evals/runner.py 对齐：
- 加载评测用例（cases.yml）
- 调用 agent.chat() 跑每个用例
- 对比预期结果（intent / citations / next_action）
- 输出通过率 + 失败分类

本课次演化：
- L38 eval-runner：cases.yml 回归评测
- L41 final-rehearsal：终态评测框架

被谁调用：scripts/run_evals.py 或 pytest 测试
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class EvalCase:
    """单个评测用例。"""
    case_id: str
    user_message: str
    expected_intent: str
    expected_next_action: str | None = None
    expected_citations: list[str] = field(default_factory=list)
    runtime_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """单个用例的评测结果。"""
    case_id: str
    passed: bool
    actual_intent: str
    actual_next_action: str | None
    failures: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    """评测汇总。"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def load_eval_cases(cases_path: str | None = None) -> list[EvalCase]:
    """从 cases.yml 加载评测用例。

    被谁调用：run_evals()
    """
    if cases_path is None:
        cases_path = os.path.join(os.path.dirname(__file__), "cases.yml")
    if not os.path.exists(cases_path):
        return []
    with open(cases_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cases = []
    for item in raw.get("cases", []):
        cases.append(EvalCase(
            case_id=item["case_id"],
            user_message=item["user_message"],
            expected_intent=item["expected_intent"],
            expected_next_action=item.get("expected_next_action"),
            expected_citations=item.get("expected_citations", []),
            runtime_context=item.get("runtime_context", {}),
        ))
    return cases


def run_evals(agent, cases: list[EvalCase] | None = None) -> EvalSummary:
    """跑评测回归。

    Args:
        agent: CustomerServiceAgent 实例
        cases: 评测用例列表（None 时从 cases.yml 加载）

    Returns:
        EvalSummary（通过率 + 每个用例的结果）

    被谁调用：scripts/run_evals.py 或测试
    """
    from api.schemas import ChatRequest

    if cases is None:
        cases = load_eval_cases()

    summary = EvalSummary()
    for case in cases:
        request = ChatRequest(
            session_id=f"eval-{case.case_id}",
            runtime_user_id="eval-driver",
            user_message=case.user_message,
            runtime_context=case.runtime_context,
        )
        try:
            response = agent.chat(request)
            actual_intent = response.intent
            actual_next_action = response.next_action

            failures = []
            if actual_intent != case.expected_intent:
                failures.append(f"intent: expected={case.expected_intent}, actual={actual_intent}")
            if case.expected_next_action and actual_next_action != case.expected_next_action:
                failures.append(f"next_action: expected={case.expected_next_action}, actual={actual_next_action}")

            passed = len(failures) == 0
            result = EvalResult(
                case_id=case.case_id,
                passed=passed,
                actual_intent=actual_intent,
                actual_next_action=actual_next_action,
                failures=failures,
            )
        except Exception as e:
            result = EvalResult(
                case_id=case.case_id,
                passed=False,
                actual_intent="error",
                actual_next_action=None,
                failures=[f"exception: {type(e).__name__}: {str(e)[:100]}"],
            )

        summary.results.append(result)
        summary.total += 1
        if result.passed:
            summary.passed += 1
        else:
            summary.failed += 1

    return summary
