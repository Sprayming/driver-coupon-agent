"""模型客户端层，封装 OpenAI-compatible 聊天模型调用和错误边界。"""

from __future__ import annotations

import os

import httpx
from langchain_openai import ChatOpenAI

from api.schemas import ChatModelResult
from config.settings import api_key_is_missing, load_course_env
from cost.observer import parse_model_usage


def call_chat_model(
    messages: list[dict[str, str]],
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> ChatModelResult:
    """调用聊天模型，并保留平台 usage 供成本观察模块使用。"""

    load_course_env()
    resolved_api_key = api_key or os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_api_key):
        # 没有模型 usage 时也要继续返回 ChatModelResult，让成本模块能走本地估算路径。
        return ChatModelResult(
            answer="统一运行配置未设置有效的 AGENT_OPENAI_API_KEY。当前版本会用本地估算观察 Prompt token 趋势。"
        )

    resolved_base_url = (base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    resolved_model = model or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    # 第 07 课开始保留 usage；但 usage 解析仍交给 cost/observer.py，不塞进 HTTP 调用层。
    # enable_thinking=False：客服问答不需要思考链，省几百 reasoning token，平台高峰期也更快返回。
    # 注意：该参数是硅基流动对 Qwen3 系列的扩展参数，换其他模型平台时可能需要去掉。
    request_kwargs = {
        "headers": {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"},
        "json": {"model": resolved_model, "messages": messages, "enable_thinking": False},
    }
    if http_client is not None:
        response = http_client.post(f"{resolved_base_url}/chat/completions", **request_kwargs)
    else:
        response = httpx.post(f"{resolved_base_url}/chat/completions", **request_kwargs, timeout=90)
    response.raise_for_status()
    payload = response.json()
    return ChatModelResult(answer=payload["choices"][0]["message"]["content"], usage=parse_model_usage(payload))


def create_tool_calling_model() -> ChatOpenAI:
    """构造支持工具调用的 LangChain ChatOpenAI 模型（供 create_agent 使用）。

    和 call_chat_model 共用同一套 API Key/Base URL/Model 配置，
    但走 LangChain 的消息格式（AIMessage/ToolMessage）而非裸 HTTP。
    模型不可用时抛异常，由调用方决定降级策略。
    """
    load_course_env()
    api_key = os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        raise RuntimeError("AGENT_OPENAI_API_KEY 未配置，无法启动 Tool Calling 模型。")
    base_url = os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    model = os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
    )
