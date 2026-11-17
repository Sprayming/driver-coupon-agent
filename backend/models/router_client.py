"""路由候选模型客户端：让模型提议，代码裁决（对标小哲 L41 的 route_model_client）。

调用时机：只在规则置信不足时调用，规则高置信直接短路，省一次模型调用。
失败策略：没配 Key、超时、JSON 解析失败、跨字段校验失败 → 一律返回 None，
调用方退回规则基线。模型永远不会直接决定执行哪条路径。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from api.schemas import RoutePlan, RoutePlanCandidate
from config.settings import api_key_is_missing, load_course_env
from prompts.loader import prompt_manager

logger = logging.getLogger(__name__)

# 知识域白名单：模型只能在这些域里补，防止凭空发明域把检索路由带偏。
ALLOWED_DOMAINS = {"refund", "usage", "promotion", "complaint"}

# 熔断：模型不可用（限流/超时）时不反复打，退避期间直接走规则基线。
# 这不是优化，是保护——上游 429 时如果每条低置信请求都重试，会把整条链路拖慢。
FAILURE_COOLDOWN_SECONDS = 60.0
# 路由是前置判断，不能等满 90 秒：模型慢就尽快退回规则。
ROUTER_TIMEOUT_SECONDS = 6.0
_last_failure_at: float = 0.0


def _in_cooldown() -> bool:
    return _last_failure_at > 0 and (time.monotonic() - _last_failure_at) < FAILURE_COOLDOWN_SECONDS


def _note_failure() -> None:
    global _last_failure_at
    _last_failure_at = time.monotonic()


def _note_success() -> None:
    global _last_failure_at
    _last_failure_at = 0.0


def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔包裹的 ```json ... ``` 代码块。"""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped[3:]
    if body.rstrip().endswith("```"):
        body = body.rsplit("```", 1)[0]
    return body.strip()


def plan_route_candidate(
    user_message: str,
    rule_plan: RoutePlan,
    *,
    llm_caller: Callable[..., Any] | None = None,
) -> RoutePlanCandidate | None:
    """让模型给出路由候选；任何异常都返回 None，由调用方退回规则基线。"""

    load_course_env()
    if api_key_is_missing(os.getenv("AGENT_OPENAI_API_KEY")):
        # 没配 Key 就直接走规则：不产生网络调用，测试与离线环境保持确定性。
        return None
    if _in_cooldown():
        return None

    if llm_caller is None:
        from models.llm_client import call_chat_model

        llm_caller = call_chat_model

    # L41 对齐：路由契约不再硬编码在客户端，由 Prompt Registry 按 phase=route 选择。
    fragments = prompt_manager.select_fragments({"phase": "route"})
    messages = [
        {"role": "system", "content": prompt_manager.render_system_prompt(fragments)},
        {
            "role": "user",
            "content": f"用户问题：{user_message}\n规则基线给出的 intent：{rule_plan.intent}",
        },
    ]

    try:
        result = llm_caller(messages, timeout=ROUTER_TIMEOUT_SECONDS)
        payload = json.loads(_strip_code_fence(result.answer or ""))
        if not isinstance(payload, dict):
            logger.warning("路由候选不是 JSON 对象，退回规则基线: %r", str(payload)[:120])
            return None
        candidate = RoutePlanCandidate.model_validate(payload)
        _note_success()
    except Exception as exc:  # noqa: BLE001 - 候选只是增强，失败一律退回规则
        _note_failure()
        logger.warning("路由候选生成失败，退回规则基线: %s", exc)
        return None

    # 知识域白名单在模型层就收紧；工具白名单在 planner 层做，因为要用工具目录判断。
    unknown_domains = [d for d in candidate.knowledge_domains if d not in ALLOWED_DOMAINS]
    if unknown_domains:
        logger.info("剔除白名单外的知识域: %s", unknown_domains)
        candidate = candidate.model_copy(
            update={
                "knowledge_domains": [
                    d for d in candidate.knowledge_domains if d in ALLOWED_DOMAINS
                ]
            }
        )
    return candidate
