"""Prompt Registry：按执行阶段和业务信号选择、排序并渲染稳定规则片段。

对标小哲 L41 `prompts/loader.py`：

- registry 只登记「规则片段」（security / role / contract / policy / capability）
- 片段正文放在同名 `.md`，加载时做目录穿越校验
- 调用方用 signals 选择片段，模型只拿到渲染后的 system prompt，不接触注册表

动态业务事实（工具 Observation、RAG 引用、Runtime Context）仍由各自模块提供，
不允许写进 prompt md，避免把易变事实固化成规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROMPTS_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = PROMPTS_DIR / "prompt_registry.yml"


@dataclass(frozen=True)
class PromptFragment:
    """一个可按执行阶段和业务信号选择的稳定 Prompt 片段。"""

    name: str
    level: str
    version: str
    load: str
    priority: int
    status: str = "active"
    description: str = ""


class PromptManager:
    """管理稳定规则片段；不承载业务事实，也不替调用方做路由决策。"""

    def __init__(self, prompts_dir: Path = PROMPTS_DIR) -> None:
        self.prompts_dir = Path(prompts_dir).resolve()
        self._registry = self._load_registry()

    @lru_cache(maxsize=None)
    def load(self, name: str) -> str:
        """只允许读取注册目录中的 Markdown，避免自由文件名跳出 Prompt 边界。"""

        path = (self.prompts_dir / f"{name}.md").resolve()
        if path.parent != self.prompts_dir:
            raise ValueError(f"Invalid prompt name: {name}")
        return path.read_text(encoding="utf-8").strip()

    def select_fragments(self, signals: dict[str, Any]) -> list[PromptFragment]:
        """按受控信号选择片段；priority 只控制拼接顺序，不代表模型置信度。"""

        selected = [
            fragment
            for fragment in self._registry
            if fragment.status == "active" and self._matches(fragment.load, signals)
        ]
        return sorted(selected, key=lambda fragment: (-fragment.priority, fragment.name))

    def render_system_prompt(self, fragments: list[PromptFragment]) -> str:
        """把已选择片段渲染成一条 system message，不混入动态业务事实。"""

        return "\n\n".join(
            f"[{fragment.name} | {fragment.level} | {fragment.version} | priority={fragment.priority}]\n"
            f"{self.load(fragment.name)}"
            for fragment in fragments
        )

    def selection_summary(
        self, fragments: list[PromptFragment], *, phase: str
    ) -> list[dict[str, Any]]:
        """返回可公开的片段元数据，不暴露完整 system prompt。"""

        return [
            {
                "name": fragment.name,
                "level": fragment.level,
                "version": fragment.version,
                "load": fragment.load,
                "priority": fragment.priority,
                "phase": phase,
                "estimated_tokens": max(1, len(self.load(fragment.name)) // 4),
            }
            for fragment in fragments
        ]

    def _load_registry(self) -> list[PromptFragment]:
        payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
        return [
            PromptFragment(
                name=str(item["name"]),
                level=str(item.get("level", "capability")),
                version=str(item.get("version", "v1")),
                load=str(item.get("load", "always")),
                priority=int(item.get("priority", 50)),
                status=str(item.get("status", "active")),
                description=str(item.get("description", "")),
            )
            for item in payload.get("prompts", [])
        ]

    @staticmethod
    def _matches(load: str, signals: dict[str, Any]) -> bool:
        """把注册表里的 load 条件翻译成对 signals 的判断。"""

        if load == "always":
            return True
        if load == "when_route":
            return signals.get("phase") == "route"
        if load == "when_clarification":
            return signals.get("phase") == "clarification"
        if load == "when_final_answer":
            return signals.get("phase") == "final_answer"
        if load == "when_high_risk_after_sale":
            return bool(signals.get("high_risk_after_sale"))
        if load == "when_business_tool":
            return bool(signals.get("needs_business_tools"))
        if load == "when_rag":
            return bool(signals.get("needs_rag"))
        return False


# 模块级单例：与 L41 同名，供 answer_client / router_client / langchain_runtime 共用。
prompt_manager = PromptManager()


def load_prompt(name: str) -> str:
    """兼容课程里按文件名直接读取的场景；新链路统一使用 PromptManager。"""

    return prompt_manager.load(name.removesuffix(".md"))


def get_prompt_manager() -> PromptManager:
    """获取 PromptManager 单例（保留既有调用方式）。"""

    return prompt_manager
