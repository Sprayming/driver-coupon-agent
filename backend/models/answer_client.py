"""回答组装客户端（对标小哲 L41 `models/answer_client.py`）。

职责边界：
- `models/llm_client.py`：最底层 HTTP 调用 + usage 解析
- `prompts/loader.py`：按阶段和业务信号选择稳定 system prompt 片段
- 本模块：回答阶段的门面，统一拼装 prompt、注入客户端，让 `agent.chat()` 不直接碰底层

后续如果要加"回答前统一裁剪上下文""回答后统一清洗"，都收在这一层。
"""

from __future__ import annotations

from typing import Any

import httpx

from models.llm_client import ChatModelResult, call_chat_model
from prompts.loader import prompt_manager


def build_answer_messages(
    user_payload: str,
    *,
    phase: str = "final_answer",
    needs_rag: bool = False,
    needs_business_tools: bool = False,
    high_risk_after_sale: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """按业务信号渲染 system prompt，并返回可公开的片段元数据。

    system prompt 只来自注册表；`user_payload` 才是本轮动态事实
    （工具 Observation / RAG 引用 / Runtime Context），两者不能混在一起。
    """

    signals = {
        "phase": phase,
        "needs_rag": needs_rag,
        "needs_business_tools": needs_business_tools,
        "high_risk_after_sale": high_risk_after_sale,
    }
    fragments = prompt_manager.select_fragments(signals)
    messages = [
        {"role": "system", "content": prompt_manager.render_system_prompt(fragments)},
        {"role": "user", "content": user_payload},
    ]
    return messages, prompt_manager.selection_summary(fragments, phase=phase)


def compose_answer(
    messages: list[dict[str, str]],
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> ChatModelResult:
    """生成最终回答。"""

    return call_chat_model(
        messages,
        http_client=http_client,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
