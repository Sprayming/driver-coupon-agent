"""L06：Prompt 加载器 + 注册表。

和小哲电商 prompts/loader.py 对齐：
- PromptManager：从 prompt_registry.yml 加载 prompt 模板
- 支持按场景路由（faq / refund / general_chat 等）
- 支持版本号（方便灰度和回滚）

本课次演化：
- L05 prompt-loader：PromptManager 基础加载
- L06 prompt-versioning：版本号 + 场景路由
- L41 final-rehearsal：终态 Prompt 管理

被谁调用：agents/customer_service_agent.py chat() 阶段 4/5（生成回答时选 prompt）
"""

from __future__ import annotations

import os
from typing import Any

import yaml


# 默认 prompt 注册表路径
_DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "prompt_registry.yml")


class PromptManager:
    """Prompt 模板管理器。

    从 prompt_registry.yml 加载所有 prompt 模板，按场景名 + 版本号查找。
    """

    def __init__(self, registry_path: str | None = None) -> None:
        self.registry_path = registry_path or _DEFAULT_REGISTRY_PATH
        self._registry: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """加载 prompt_registry.yml。"""
        if not os.path.exists(self.registry_path):
            # 文件不存在时用内置默认值（不阻断主流程）
            self._registry = self._default_registry()
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._registry = yaml.safe_load(f) or {}
        except Exception:
            self._registry = self._default_registry()

    def _default_registry(self) -> dict[str, Any]:
        """内置默认 prompt（注册表文件不存在时的兜底）。"""
        return {
            "scenarios": {
                "general_chat": {
                    "system": "你是网约车司机端卡券客服 Agent，请用友好、准确的语气回答司机的问题。",
                    "version": "1.0",
                },
                "faq": {
                    "system": "你是网约车司机端卡券客服 Agent。请基于提供的政策知识回答，不要编造信息。",
                    "version": "1.0",
                },
                "refund": {
                    "system": "你是网约车司机端卡券客服 Agent。退款是高风险操作，你只能解释政策和判断资格，不能直接执行退款。",
                    "version": "1.0",
                },
            }
        }

    def get_prompt(self, scenario: str, version: str | None = None) -> str | None:
        """按场景名获取 system prompt。

        Args:
            scenario: 场景名（general_chat / faq / refund 等）
            version: 版本号（None 表示用最新版）

        Returns:
            system prompt 字符串，找不到返回 None
        """
        scenarios = self._registry.get("scenarios", {})
        scenario_cfg = scenarios.get(scenario)
        if not scenario_cfg:
            return None
        if version and scenario_cfg.get("version") != version:
            # 版本不匹配时返回 None（调用方可以降级）
            return None
        return scenario_cfg.get("system")

    def list_scenarios(self) -> list[str]:
        """列出所有可用场景。"""
        return list(self._registry.get("scenarios", {}).keys())


# 模块级单例
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """获取 PromptManager 单例。"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
