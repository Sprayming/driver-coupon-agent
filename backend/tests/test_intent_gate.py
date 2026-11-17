"""评测集版本与 CI 门禁：同一份考卷可追溯，指标不达标时退出码非 0。"""

from __future__ import annotations

import json
import sys

from evals.intent_bakeoff import check_gate, load_dataset, main


def test_dataset_has_version_and_hash():
    version, digest, cases = load_dataset()

    assert version == "v1"
    assert len(digest) == 12
    assert len(cases) == 56
    assert len({case.case_id for case in cases}) == 56


def test_gate_passes_when_thresholds_met():
    results = {
        "rules": {
            "status": "ok",
            "total": 56,
            "accuracy": 0.9821,
            "macro_f1": 0.9813,
            "latency": {"avg_ms": 250.0, "p95_ms": 560.0},
        }
    }

    problems = check_gate(results, min_accuracy=0.95, min_macro_f1=0.95, max_p95_ms=3000)

    assert problems == []


def test_gate_reports_multiple_failures():
    results = {
        "rules": {
            "status": "ok",
            "total": 56,
            "accuracy": 0.5893,
            "macro_f1": 0.6225,
            "latency": {"avg_ms": 0.0, "p95_ms": 0.0},
        },
        "llm": {
            "status": "partial: 存在失败用例",
            "total": 56,
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "latency": {"avg_ms": 900.0, "p95_ms": 5000.0},
        },
    }

    problems = check_gate(results, min_accuracy=0.95, min_macro_f1=0.9, max_p95_ms=3000)

    text = "\n".join(problems)
    assert "rules: accuracy" in text
    assert "rules: macro_f1" in text
    assert "llm: 状态 partial" in text
    assert "llm: p95" in text


def test_gate_fails_when_nothing_ran():
    problems = check_gate({}, min_accuracy=0.9, min_macro_f1=0.9, max_p95_ms=3000)

    assert problems and "没有任何一组" in problems[0]


def test_cli_writes_dataset_metadata(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["intent_bakeoff", "--providers", "rules", "--output", str(output)],
    )

    assert main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["version"] == "v1"
    assert payload["dataset"]["sha256"]
    assert payload["dataset"]["cases_count"] == 56
    assert payload["dataset_version"] == "v1"


def test_cli_gate_returns_nonzero_below_threshold(tmp_path, monkeypatch):
    output = tmp_path / "report_gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intent_bakeoff",
            "--providers",
            "rules",
            "--output",
            str(output),
            "--gate",
            "--min-accuracy",
            "0.95",
        ],
    )

    assert main() == 1
    assert output.exists()
