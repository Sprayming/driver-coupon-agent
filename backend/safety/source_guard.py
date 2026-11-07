"""L36：Prompt Injection 防御 + 工具结果安全检查。

和小哲电商 safety/source_guard.py 对齐：
- inspect_source：检查文本里有没有 Prompt Injection 特征（"忽略上文"/"system"/"ignore"）
- 用于工具结果 + RAG 引用的安全检查

本课次演化：
- L36 source-guard：正则检测 Prompt Injection
- L41 final-rehearsal：终态安全检查 + 所有工具结果过 inspect_source

被谁调用：agents/customer_service_agent.py chat() 阶段 5（工具结果 + RAG 引用检查）
"""

from __future__ import annotations

import re

# Prompt Injection 特征正则（中英文混合）
PROMPT_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"忽略(以上|上文|之前|前面|上述)(所有|全部|的)?(指令|提示|规则|系统)", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior|system)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"你现在(是|扮演|变成)", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"```(system|developer)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>\s*system", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above)\s+", re.IGNORECASE),
]

# 高风险关键词（用于工具结果里检测异常指令）
HIGH_RISK_KEYWORDS: list[str] = [
    "rm -rf",
    "drop table",
    "delete from",
    "exec(",
    "eval(",
    "__import__",
    "os.system",
    "subprocess",
]


def inspect_source(text: str | None) -> tuple[bool, str]:
    """检查文本是否包含 Prompt Injection 或高风险指令。

    Args:
        text: 待检查的文本（工具结果 / RAG 引用 / 用户消息）

    Returns:
        (is_safe, reason)
        - is_safe=True：文本安全
        - is_safe=False：检测到风险，reason 说明命中了哪个规则

    被谁调用：agents/customer_service_agent.py chat() 阶段 5
    """
    if not text:
        return True, "empty_text"

    # 检查 Prompt Injection 正则
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return False, f"prompt_injection_pattern:{pattern.pattern[:40]}"

    # 检查高风险关键词
    text_lower = text.lower()
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_lower:
            return False, f"high_risk_keyword:{keyword}"

    return True, "safe"


def inspect_tool_result(result: dict | str | None) -> tuple[bool, str]:
    """检查工具返回结果是否安全。

    工具结果可能是 dict 或 str，统一转成字符串后检查。

    被谁调用：agents/customer_service_agent.py chat() 阶段 5
    """
    if result is None:
        return True, "none"
    if isinstance(result, dict):
        text = str(result)
    else:
        text = str(result)
    return inspect_source(text)
