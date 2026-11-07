"""L19 司机域：LLM 澄清规划客户端。

模型只产出 ClarificationPlan 草案，不直接调工具、不读数据库、
不能替用户从候选活动里选一个；后端仍按 ToolSpec 和当前司机身份硬校验。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from api.schemas import ChatRequest, ClarificationPlan, ToolSpec
from config.settings import api_key_is_missing, load_course_env
from tools.clarification_planning import build_validated_clarification_plan
from tools.contracts import TOOL_SPECS


def strip_json_fence(content: str) -> str:
    """剥掉模型可能包上的 ```json 围栏。"""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def tool_specs_for_prompt() -> list[dict[str, Any]]:
    """把工具说明书转成 Prompt 里的可读清单。

    L24：把工具自带的能力边界（read_only / boundary）一起推给模型，
    让模型知道哪些动作这个工具做不了，而不是只看到参数 schema。
    """
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "required": spec.required,
            "parameters_schema": spec.parameters_schema,
            "read_only": spec.read_only,
            "boundary": spec.boundary,
        }
        for spec in TOOL_SPECS.values()
    ]


def build_clarification_planner_messages(user_message: str) -> list[dict[str, str]]:
    """构造澄清规划 Prompt。

    模型负责理解司机话术、选候选工具、抽已知参数、指出缺失必填参数；
    真正的参数校验和工具执行全在后端。
    """
    schema_hint = {
        "intent": "general_chat | promotion_consult | order_query | refund_request | complaint | unknown",
        "tool_name": "get_activity_progress | get_activity_qualification | get_verification_record | ... | null",
        "known_arguments": {"activity_id": "ACT-12345", "coupon_id": "CP-001", "time_range": "本月"},
        "missing_required": ["activity_id"],
        "clarification_question": "需要向司机追问的问题，没有则为 null",
        "confidence": 0.0,
        "reason": "一句话说明为什么这样规划",
    }
    system_prompt = (
        "你是网约车司机端卡券客服 Agent 的澄清规划器，只能输出 JSON，不要输出解释。\n"
        "你的任务是识别司机意图、选择候选工具、抽取已知参数、指出缺失必填参数，并给出澄清问题草案。\n"
        "重要边界：不要输出 user_id / driver_id；不要替司机从多个活动或卡券里做选择；"
        "活动进度和活动资格类工具必须有 activity_id 才能查；核销记录工具必须有 coupon_id。\n"
        "如果司机说的是“我这个活动达标了吗”但没有活动编号，选择目标工具并把 activity_id 放进 missing_required。\n"
        "如果司机问单量、退款、卡券、收入这类不依赖编号的实时数据，选择对应工具并把缺失字段留空。\n"
        "如果司机想查看或操作自己的卡券但没给卡券号，选择 get_coupon_status 列出名下卡券，后续由用户确认目标是哪一张。\n"
        "如果司机问的是活动规则、卡券使用规则、退款规则这类稳定知识，tool_name 填 null。\n"
        "输出 JSON 必须符合这个字段形状："
        f"{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    user_prompt = (
        "可用工具：\n"
        f"{json.dumps(tool_specs_for_prompt(), ensure_ascii=False)}\n\n"
        f"司机消息：{user_message}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


class ClarificationPlannerModelClient:
    """真实调用 LLM，生成司机域的 ClarificationPlan。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        load_course_env()
        self.api_key = api_key if api_key is not None else os.getenv("AGENT_OPENAI_API_KEY")
        self.base_url = (
            base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("AGENT_CLASSIFIER_MODEL")
            or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
        )
        self.http_client = http_client

    def plan_clarification(
        self,
        request: ChatRequest,
        tool_specs: dict[str, ToolSpec] | None = None,
    ) -> ClarificationPlan:
        """产出经过后端校验的 ClarificationPlan；模型不可用时抛异常交给调用方降级。"""
        if api_key_is_missing(self.api_key):
            raise RuntimeError("AGENT_OPENAI_API_KEY 未配置，无法调用 LLM 生成澄清规划。")

        request_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": self.model,
                "messages": build_clarification_planner_messages(request.user_message),
                # temperature=0：规划要输出稳定 JSON，不要发挥。
                "temperature": 0,
                "max_tokens": 300,
            },
        }
        if "siliconflow.cn" in self.base_url:
            request_kwargs["json"]["enable_thinking"] = False
        if self.http_client is not None:
            response = self.http_client.post(f"{self.base_url}/chat/completions", **request_kwargs)
        else:
            response = httpx.post(
                f"{self.base_url}/chat/completions", **request_kwargs, timeout=60
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(strip_json_fence(content))
        if not isinstance(payload, dict):
            raise RuntimeError("LLM 澄清规划不是 JSON 对象。")
        return build_validated_clarification_plan(payload, tool_specs or TOOL_SPECS, self.model)
