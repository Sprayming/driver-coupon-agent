"""模型客户端层，封装 OpenAI-compatible 聊天模型调用和错误边界。"""

from __future__ import annotations

import os

import httpx
from langchain_openai import ChatOpenAI

from api.schemas import ChatModelResult
from config.settings import api_key_is_missing, load_course_env
from cost.governance import parse_model_usage


def candidate_models(primary: str) -> list[str]:
    """返回候选模型列表：显式指定的主模型 + 环境变量里的候补模型。

    环境变量：AGENT_OPENAI_MODELS=Qwen/Qwen3-8B,Qwen/Qwen2.5-7B-Instruct
    主模型排第一，其余去重后有顺序地作为候补。
    """

    raw = os.getenv("AGENT_OPENAI_MODELS", "")
    fallbacks = [item.strip() for item in raw.split(",") if item.strip()]
    ordered = [primary] + [item for item in fallbacks if item != primary]
    # 去重但保持顺序
    seen: set[str] = set()
    result: list[str] = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _is_retryable(exc: Exception) -> bool:
    """判断错误是否值得换下一个候选模型重试。

    只对 429（限流）和 5xx（服务端不稳定）以及网络/超时类错误切换；
    400/401/403 这类参数或鉴权错误换模型也没用，直接抛出。
    """

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def call_chat_model(
    messages: list[dict[str, str]],
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> ChatModelResult:
    """调用聊天模型，并保留平台 usage 供成本观察模块使用。

    timeout：调用方可传更短的超时。路由这类前置判断不该等满 90 秒，
    模型慢就该尽快降级走规则。

    候补模型：主模型遇到 429/5xx/超时，会按 AGENT_OPENAI_MODELS 顺序自动切换，
    全部失败才抛最后一个异常，由调用方决定降级。
    """

    load_course_env()
    resolved_api_key = api_key or os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_api_key):
        # 没有模型 usage 时也要继续返回 ChatModelResult，让成本模块能走本地估算路径。
        return ChatModelResult(
            answer="统一运行配置未设置有效的 AGENT_OPENAI_API_KEY。当前版本会用本地估算观察 Prompt token 趋势。"
        )

    resolved_base_url = (base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    primary_model = model or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")

    last_error: Exception | None = None
    for candidate in candidate_models(primary_model):
        # 第 07 课开始保留 usage；但 usage 解析仍交给 cost/observer.py，不塞进 HTTP 调用层。
        # enable_thinking=False：客服问答不需要思考链，省几百 reasoning token，平台高峰期也更快返回。
        # 只对 Qwen 系列带这个扩展参数，换其他平台/模型时避免被拒。
        payload: dict[str, object] = {"model": candidate, "messages": messages}
        if candidate.lower().startswith("qwen"):
            payload["enable_thinking"] = False
        if temperature is not None:
            payload["temperature"] = temperature
        request_kwargs = {
            "headers": {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"},
            "json": payload,
        }
        try:
            if http_client is not None:
                response = http_client.post(f"{resolved_base_url}/chat/completions", **request_kwargs)
            else:
                response = httpx.post(
                    f"{resolved_base_url}/chat/completions",
                    **request_kwargs,
                    timeout=timeout if timeout is not None else 90,
                )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - 需要分类判断是否换模型
            last_error = exc
            if _is_retryable(exc):
                continue
            raise
        payload_json = response.json()
        return ChatModelResult(
            answer=payload_json["choices"][0]["message"]["content"],
            usage=parse_model_usage(payload_json),
        )

    # 所有候选都失败：抛出最后一个错误，交给上层降级
    assert last_error is not None
    raise last_error


def create_tool_calling_model():
    """构造支持工具调用的 LangChain 模型（供 create_agent 使用）。

    和 call_chat_model 共用同一套 API Key/Base URL/Model 配置，
    但走 LangChain 的消息格式（AIMessage/ToolMessage）而非裸 HTTP。

    候补模型：配置了 AGENT_OPENAI_MODELS 时，用 with_fallbacks 串成链，
    主模型 429/5xx/超时后自动切下一个，agent 的工具调用循环不需要改。
    模型不可用时抛异常，由调用方决定降级策略。
    """

    load_course_env()
    api_key = os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        raise RuntimeError("AGENT_OPENAI_API_KEY 未配置，无法启动 Tool Calling 模型。")
    base_url = os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    primary = os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    candidates = candidate_models(primary)

    def _build(model_name: str) -> ChatOpenAI:
        # max_retries=1：先用一次重试兜住偶发抖动，再交给候补模型，避免长时间卡住
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_retries=1,
        )

    primary_model = _build(candidates[0])
    if len(candidates) == 1:
        return primary_model
    return primary_model.with_fallbacks([_build(name) for name in candidates[1:]])
