"""读取评测报告（只读）：为管理后台「评测」页提供最新一次对照结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def latest_intent_report(
    reports_dir: Path | str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """取最新的 intent_bakeoff*.json；没有报告时返回 None。"""

    directory = Path(reports_dir or DEFAULT_REPORTS_DIR)
    files = sorted(directory.glob("intent_bakeoff*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    latest = files[-1]
    return latest.name, json.loads(latest.read_text(encoding="utf-8"))
