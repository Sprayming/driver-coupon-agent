"""意图分类对照评测：规则 / LLM(Qwen) / Jev 三组同题对比。

只评「一句话 → 主意图」，不跑完整 chat 链路，目的是回答：
「换 Jev 到底比现在的规则好多少？」

用法（在 backend 目录执行）：
    python -m evals.intent_bakeoff                       # 规则 + LLM（Jev 无 Key 自动跳过）
    python -m evals.intent_bakeoff --providers rules      # 只跑规则
    python -m evals.intent_bakeoff --model Qwen/Qwen2.5-7B-Instruct
    TYPESAFE_API_KEY=... python -m evals.intent_bakeoff --providers rules,llm,jev

Jev 接入说明：
- 需要 TYPESAFE_API_KEY；端点默认 https://api.typesafe.ai，可用 TYPESAFE_BASE_URL 覆盖；
- 走 POST {base}/v1/systemone，state 为待判文本，questions 里一个 choice 题；
- 返回的 choice / confidence 只用于路由，不直接执行业务动作。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

import httpx
import yaml

from api.schemas import Intent
from config import settings
from config.settings import api_key_is_missing, load_course_env
from rag.intent import classify_intent
from rag.intent_semantic import classify_intent_with_provider

LABELS: list[str] = list(get_args(Intent))

# Jev / LLM 两组的类别边界说明。规则组不读这段，它只认 intent.py 里的关键词表。
CRITERIA: dict[str, str] = {
    "refund_request": "申请退款、退卡、取消购买，或询问退款进度与退款规则（不包含单纯查询订单/核销状态）",
    "order_query": "查询卡券或订单的状态、核销记录、到账情况（不包含要求发起退款）",
    "promotion_consult": "咨询优惠券、活动、打折、奖励、福利等营销信息",
    "product_consult": "咨询卡种、价格、时长、推荐、不同卡的区别",
    "complaint": "表达投诉、举报、索赔、强烈不满，或威胁曝光",
    "general_chat": "寒暄、问候、感谢、询问助手身份等无业务诉求的闲聊",
    "unknown": "与卡券退款业务无关，或信息不足无法判断",
}


@dataclass
class Case:
    case_id: str
    user_message: str
    expected_intent: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    intent: str
    latency_ms: float = 0.0
    error: str = ""
    escalated: bool = False


@dataclass
class ProviderReport:
    name: str
    predictions: dict[str, Prediction]
    status: str = "ok"


def load_dataset(path: str | None = None) -> tuple[str, str, list[Case]]:
    """读取评测集，返回 (版本, 内容哈希, 用例)。

    版本来自 yml 顶层 version 字段；哈希用于确认「两次跑分用的是同一份考卷」。
    """

    resolved = Path(path or os.path.join(os.path.dirname(__file__), "intent_cases.yml"))
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    version = str(raw.get("version") or "unversioned")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:12]
    cases = [
        Case(
            case_id=item["case_id"],
            user_message=item["user_message"],
            expected_intent=item["expected_intent"],
            tags=item.get("tags", []),
        )
        for item in raw.get("cases", [])
    ]
    unknown_labels = {case.expected_intent for case in cases} - set(LABELS)
    if unknown_labels:
        raise ValueError(f"评测集存在非法意图标签: {sorted(unknown_labels)}")
    return version, digest, cases


def load_cases(path: str | None = None) -> list[Case]:
    return load_dataset(path)[2]


# —— 第一组：现有规则 ——


def run_rules(cases: list[Case]) -> ProviderReport:
    predictions: dict[str, Prediction] = {}
    for case in cases:
        started = time.perf_counter()
        result = classify_intent(case.user_message)
        predictions[case.case_id] = Prediction(
            intent=result.intent,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return ProviderReport(name="rules", predictions=predictions)


# —— 第二组半：hybrid（规则优先 + 低置信语义兜底） ——


def run_hybrid(cases: list[Case], *, model: str | None = None) -> ProviderReport:
    load_course_env()
    original_provider = settings.INTENT_PROVIDER
    original_model = os.environ.get("AGENT_CLASSIFIER_MODEL")
    settings.INTENT_PROVIDER = "hybrid"
    if model:
        os.environ["AGENT_CLASSIFIER_MODEL"] = model

    predictions: dict[str, Prediction] = {}
    try:
        for case in cases:
            started = time.perf_counter()
            result, escalated = classify_intent_with_provider(case.user_message)
            predictions[case.case_id] = Prediction(
                intent=result.intent,
                latency_ms=(time.perf_counter() - started) * 1000,
                escalated=escalated,
            )
    finally:
        settings.INTENT_PROVIDER = original_provider
        if original_model is None:
            os.environ.pop("AGENT_CLASSIFIER_MODEL", None)
        else:
            os.environ["AGENT_CLASSIFIER_MODEL"] = original_model

    return ProviderReport(name="hybrid", predictions=predictions)


# —— 第二组：LLM 单次分类（模拟「让 Qwen 输出 JSON」） ——


LLM_SYSTEM_PROMPT = (
    "你是卡券退款业务的意图分类器。只输出一个 JSON 对象，形如 {\"intent\": \"类别\"}，"
    "不要解释、不要多余文字。类别必须从下面选一个：\n"
    + "\n".join(f"- {label}: {desc}" for label, desc in CRITERIA.items())
    + '\n补充规则：只有明确的问候、感谢、询问助手身份才算 general_chat；'
    "天气、写诗、股票、美食等与卡券业务无关的话题一律归 unknown。"
)


def _call_llm(
    message: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
        "max_tokens": 32,
        "enable_thinking": False,
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_intent_json(text: str) -> str:
    """先按 JSON 解析；失败时退化到「文本里出现哪个类别名」。"""

    match = text.find("{")
    if match != -1:
        end = text.rfind("}")
        if end > match:
            try:
                data = json.loads(text[match : end + 1])
                intent = str(data.get("intent", "")).strip()
                if intent in LABELS:
                    return intent
            except json.JSONDecodeError:
                pass
    for label in LABELS:
        if label in text:
            return label
    raise ValueError(f"无法解析意图: {text[:80]}")


def run_llm(
    cases: list[Case],
    *,
    model: str | None = None,
    delay: float = 0.0,
    max_retries: int = 3,
    timeout: float = 30.0,
) -> ProviderReport:
    load_course_env()
    api_key = os.getenv("AGENT_OPENAI_API_KEY", "")
    if api_key_is_missing(api_key):
        return ProviderReport(name="llm", predictions={}, status="skipped: AGENT_OPENAI_API_KEY 未配置")

    base_url = os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    resolved_model = model or os.getenv("AGENT_CLASSIFIER_MODEL") or os.getenv(
        "AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B"
    )

    predictions: dict[str, Prediction] = {}
    for index, case in enumerate(cases):
        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                text = _call_llm(
                    case.user_message,
                    base_url=base_url,
                    api_key=api_key,
                    model=resolved_model,
                    timeout=timeout,
                )
                predictions[case.case_id] = Prediction(
                    intent=parse_intent_json(text),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - 评测要记录单条失败而不是中断整轮
                last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 8))
        else:
            predictions[case.case_id] = Prediction(
                intent="error", latency_ms=(time.perf_counter() - started) * 1000, error=last_error
            )
        if delay > 0 and index < len(cases) - 1:
            time.sleep(delay)

    status = "ok" if all(p.intent != "error" for p in predictions.values()) else "partial: 存在失败用例"
    return ProviderReport(name="llm", predictions=predictions, status=status)


# —— 第三组：Jev（托管 System One 决策模型） ——


JEV_INSTRUCTIONS = (
    "判断司机当前这句话最主要的业务意图；只选一个类别，"
    "如果与卡券退款业务无关或信息不足，选 unknown。"
)


def run_jev(cases: list[Case], *, timeout: float = 20.0) -> ProviderReport:
    api_key = os.getenv("TYPESAFE_API_KEY", "")
    if api_key_is_missing(api_key):
        return ProviderReport(
            name="jev",
            predictions={},
            status="skipped: TYPESAFE_API_KEY 未配置（可用 TYPESAFE_API_KEY=... 重跑）",
        )

    base_url = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    model = os.getenv("TYPESAFE_MODEL", "jev-latest")

    predictions: dict[str, Prediction] = {}
    for case in cases:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{base_url}/v1/systemone",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "state": case.user_message,
                    "questions": {
                        "intent": {
                            "type": "choice",
                            "instructions": JEV_INSTRUCTIONS,
                            "criteria": CRITERIA,
                        }
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            answer = response.json()["answers"]["intent"]
            choice = answer["choice"] if isinstance(answer, dict) else str(answer)
            if choice not in LABELS:
                raise ValueError(f"Jev 返回未知类别: {choice}")
            predictions[case.case_id] = Prediction(
                intent=choice, latency_ms=(time.perf_counter() - started) * 1000
            )
        except Exception as exc:  # noqa: BLE001
            predictions[case.case_id] = Prediction(
                intent="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {str(exc)[:120]}",
            )
    return ProviderReport(name="jev", predictions=predictions, status="ok")


# —— 指标与报告 ——


def compute_metrics(cases: list[Case], report: ProviderReport) -> dict[str, Any]:
    """计算指标。

    关键口径（L38 踩坑修正）：
    - error（没拿到模型结果，如 429/超时）与 wrong（拿到结果但判错）必须分开统计；
    - accuracy 的分母只算有效样本，避免限流把准确率假性拉低；
    - 同时额外给出 accuracy_all（分母含 error），方便对照历史报告。
    """

    gold = {case.case_id: case.expected_intent for case in cases}
    total = len(cases)

    error_cases = [case for case in cases if report.predictions[case.case_id].error]
    valid_cases = [case for case in cases if not report.predictions[case.case_id].error]
    error_count = len(error_cases)
    valid_total = len(valid_cases)

    correct = sum(
        1 for case in valid_cases
        if report.predictions[case.case_id].intent == gold[case.case_id]
    )
    per_class: dict[str, dict[str, float]] = {}
    failures: list[dict[str, str]] = []

    labels_in_gold = [label for label in LABELS if label in set(gold.values())]
    for label in labels_in_gold:
        tp = sum(
            1
            for case in valid_cases
            if gold[case.case_id] == label
            and report.predictions[case.case_id].intent == label
        )
        fp = sum(
            1
            for case in valid_cases
            if gold[case.case_id] != label
            and report.predictions[case.case_id].intent == label
        )
        fn = sum(
            1
            for case in valid_cases
            if gold[case.case_id] == label
            and report.predictions[case.case_id].intent != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    for case in valid_cases:
        predicted = report.predictions[case.case_id]
        if predicted.intent != gold[case.case_id]:
            failures.append(
                {
                    "case_id": case.case_id,
                    "message": case.user_message,
                    "expected": gold[case.case_id],
                    "actual": predicted.intent,
                    "tags": ",".join(case.tags),
                    "kind": "wrong",
                    "error": "",
                }
            )
    for case in error_cases:
        predicted = report.predictions[case.case_id]
        failures.append(
            {
                "case_id": case.case_id,
                "message": case.user_message,
                "expected": gold[case.case_id],
                "actual": predicted.intent,
                "tags": ",".join(case.tags),
                "kind": "error",
                "error": predicted.error,
            }
        )

    latencies = sorted(
        report.predictions[case.case_id].latency_ms for case in valid_cases
    )
    if latencies:
        p95_index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
        latency = {
            "avg_ms": round(statistics.mean(latencies), 1),
            "p95_ms": round(latencies[p95_index], 1),
        }
    else:
        latency = {"avg_ms": None, "p95_ms": None}

    return {
        "total": total,
        "valid_total": valid_total,
        "error_count": error_count,
        "error_rate": round(error_count / total, 4) if total else 0.0,
        "correct": correct,
        # 主口径：只算有效样本（error 不计入分母）
        "accuracy": round(correct / valid_total, 4) if valid_total else 0.0,
        # 对照口径：分母含 error，用于和历史报告对齐
        "accuracy_all": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(
            statistics.mean(entry["f1"] for entry in per_class.values()), 4
        ) if per_class else 0.0,
        "latency": latency,
        "per_class": per_class,
        "failures": failures,
    }


def print_report(report: ProviderReport, metrics: dict[str, Any] | None) -> None:
    print(f"\n=== {report.name} === status: {report.status}")
    if metrics is None:
        print("（本组未执行，不计入指标）")
        return
    print(
        f"accuracy={metrics['accuracy']:.2%}（有效 {metrics['valid_total']}/{metrics['total']}）"
        f"  macro_f1={metrics['macro_f1']:.4f}  "
        f"avg={metrics['latency']['avg_ms']}ms  p95={metrics['latency']['p95_ms']}ms"
    )
    if metrics["error_count"]:
        print(
            f"⚠️ 执行失败 {metrics['error_count']} 条（{metrics['error_rate']:.1%}），"
            f"已从准确率分母剔除；含失败样本口径 accuracy_all={metrics['accuracy_all']:.2%}"
        )
    escalated = sum(1 for prediction in report.predictions.values() if prediction.escalated)
    if escalated:
        print(f"语义兜底触发 {escalated}/{len(report.predictions)} 条")
    print("按类别：")
    for label, entry in metrics["per_class"].items():
        print(
            f"  {label:<18} P={entry['precision']:.2f} R={entry['recall']:.2f} "
            f"F1={entry['f1']:.2f} n={entry['support']}"
        )
    if metrics["failures"]:
        wrong = [f for f in metrics["failures"] if f.get("kind") != "error"]
        errors = [f for f in metrics["failures"] if f.get("kind") == "error"]
        if wrong:
            print(f"判错 {len(wrong)} 条：")
            for failure in wrong:
                print(
                    f"  [{failure['case_id']}] {failure['message']} | "
                    f"expected={failure['expected']} actual={failure['actual']}"
                )
        if errors:
            print(f"执行失败 {len(errors)} 条：")
            for failure in errors[:5]:
                print(f"  [{failure['case_id']}] {failure['error'][:80]}")
            if len(errors) > 5:
                print(f"  …… 其余 {len(errors) - 5} 条见报告 JSON")


def check_gate(
    results: dict[str, dict[str, Any]],
    *,
    min_accuracy: float,
    min_macro_f1: float,
    max_p95_ms: float,
    max_error_rate: float = 0.05,
) -> list[str]:
    """对照 CI 门禁：只判已完成（有样本结果）的组，返回不达标原因列表。

    除准确率/F1/延迟外，还检查 error_rate：
    限流导致大量失败时，即使有效样本准确率很高，也不能算通过。
    """

    executed = {name: entry for name, entry in results.items() if entry.get("total")}
    if not executed:
        return ["没有任何一组完成评测（检查 API Key 与网络）"]

    problems: list[str] = []
    for name, entry in executed.items():
        status = str(entry.get("status") or "")
        if status != "ok":
            problems.append(f"{name}: 状态 {status}")

        error_rate = float(entry.get("error_rate") or 0.0)
        if error_rate > max_error_rate:
            problems.append(
                f"{name}: 执行失败率 {error_rate:.1%} > 门禁 {max_error_rate:.1%}"
                f"（{entry.get('error_count', 0)} 条）"
            )

        accuracy = float(entry.get("accuracy") or 0.0)
        if accuracy < min_accuracy:
            problems.append(
                f"{name}: accuracy {accuracy:.2%} < 门禁 {min_accuracy:.2%}"
            )

        macro_f1 = float(entry.get("macro_f1") or 0.0)
        if macro_f1 < min_macro_f1:
            problems.append(
                f"{name}: macro_f1 {macro_f1:.4f} < 门禁 {min_macro_f1:.4f}"
            )

        p95 = (entry.get("latency") or {}).get("p95_ms")
        if p95 is not None and float(p95) > max_p95_ms:
            problems.append(f"{name}: p95 {float(p95):.0f}ms > 门禁 {max_p95_ms:.0f}ms")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="意图分类三组对照评测")
    parser.add_argument("--providers", default="rules,llm,jev", help="rules,hybrid,llm,jev 逗号分隔")
    parser.add_argument("--cases", default=None, help="评测集路径（默认 intent_cases.yml）")
    parser.add_argument("--model", default=None, help="LLM 组使用的模型名")
    parser.add_argument("--delay", type=float, default=0.0, help="LLM 组每条之间的间隔秒数")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    parser.add_argument("--gate", action="store_true", help="指标低于阈值时返回非 0 退出码")
    parser.add_argument("--min-accuracy", type=float, default=0.90, help="门禁：最低准确率")
    parser.add_argument("--min-macro-f1", type=float, default=0.85, help="门禁：最低 macro-F1")
    parser.add_argument("--max-p95-ms", type=float, default=3000.0, help="门禁：最大 P95 延迟")
    parser.add_argument(
        "--max-error-rate", type=float, default=0.05,
        help="门禁：允许的执行失败率（429/超时等），默认 5%%",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="严格模式：只要出现执行失败（error）就以非 0 退出，避免带病出结论",
    )
    args = parser.parse_args()

    load_course_env()
    dataset_version, dataset_sha256, cases = load_dataset(args.cases)
    providers = {item.strip() for item in args.providers.split(",") if item.strip()}

    results: dict[str, dict[str, Any]] = {}
    if "rules" in providers:
        report = run_rules(cases)
        metrics = compute_metrics(cases, report) if report.predictions else None
        results["rules"] = {"status": report.status, **(metrics or {})}
        print_report(report, metrics)
    if "llm" in providers:
        report = run_llm(cases, model=args.model, delay=args.delay)
        metrics = compute_metrics(cases, report) if report.predictions else None
        results["llm"] = {"status": report.status, **(metrics or {})}
        print_report(report, metrics)
    if "hybrid" in providers:
        report = run_hybrid(cases, model=args.model)
        metrics = compute_metrics(cases, report) if report.predictions else None
        results["hybrid"] = {"status": report.status, **(metrics or {})}
        print_report(report, metrics)
    if "jev" in providers:
        report = run_jev(cases)
        metrics = compute_metrics(cases, report) if report.predictions else None
        results["jev"] = {"status": report.status, **(metrics or {})}
        print_report(report, metrics)

    output = args.output or os.path.join(
        os.path.dirname(__file__), "reports",
        f"intent_bakeoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": {
                    "version": dataset_version,
                    "sha256": dataset_sha256,
                    "cases_count": len(cases),
                },
                "dataset_version": dataset_version,
                "dataset_sha256": dataset_sha256,
                "cases_count": len(cases),
                "llm_model": args.model or os.getenv("AGENT_CLASSIFIER_MODEL"),
                "providers": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n报告已写入: {output}")

    if args.strict:
        total_errors = sum(int(entry.get("error_count") or 0) for entry in results.values())
        if total_errors:
            print(f"\n严格模式：本轮共有 {total_errors} 条执行失败，结论不可信，退出。")
            return 1

    if args.gate:
        problems = check_gate(
            results,
            min_accuracy=args.min_accuracy,
            min_macro_f1=args.min_macro_f1,
            max_p95_ms=args.max_p95_ms,
            max_error_rate=args.max_error_rate,
        )
        if problems:
            print("\n门禁未通过：")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\n门禁通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
