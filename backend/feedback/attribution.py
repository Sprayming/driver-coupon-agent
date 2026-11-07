"""L39：失败归因——分析用户负反馈，定位是哪个环节出了问题。

和小哲电商 feedback/attribution.py 对齐：
- FailureAttributor：根据用户反馈 + 本轮 Trace 推断失败原因
- 6 个归因规则：检索失败 / 工具失败 / 模型幻觉 / 安全拦截 / 澄清超时 / 其他

本课次演化：
- L39 feedback-attribution：失败归因规则
- L41 final-rehearsal：终态归因 + case 回填

被谁调用：api/routes.py /feedback/submit 端点（或后续反馈处理流程）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureAttribution:
    """失败归因结果。"""
    category: str  # retrieval_failure / tool_failure / model_hallucination / safety_block / clarification_timeout / unknown
    reason: str
    confidence: float  # 0.0 ~ 1.0
    suggested_fix: str = ""


class FailureAttributor:
    """失败归因器。

    根据用户负反馈的关键词 + 本轮 Trace 事件，推断失败原因。
    """

    # 归因规则：(关键词列表, 归因类别, 置信度, 建议修复)
    RULES: list[tuple[list[str], str, float, str]] = [
        (["没回答", "答非所问", "不知道", "不相关"], "retrieval_failure", 0.7, "检查 RAG 召回是否覆盖该问题"),
        (["查不到", "没有数据", "工具报错", "超时"], "tool_failure", 0.8, "检查工具调用是否成功，后端服务是否正常"),
        (["胡说", "编造", "不对", "错误信息"], "model_hallucination", 0.75, "加强 RAG 引用约束，降低模型温度"),
        (["拒绝", "不能", "不让", "拦截"], "safety_block", 0.6, "检查安全规则是否过于严格"),
        (["一直问", "反复问", "不解决"], "clarification_timeout", 0.65, "检查澄清逻辑是否正确恢复原意图"),
    ]

    def attribute(self, feedback_text: str, trace_events: list[dict] | None = None) -> FailureAttribution:
        """归因。

        Args:
            feedback_text: 用户负反馈文本
            trace_events: 本轮 Trace 事件列表（可选）

        Returns:
            FailureAttribution（归因类别 + 原因 + 置信度 + 建议）

        被谁调用：api/routes.py /feedback/submit 端点
        """
        feedback_lower = feedback_text.lower()

        # 1. 关键词匹配
        for keywords, category, confidence, fix in self.RULES:
            if any(kw in feedback_lower for kw in keywords):
                return FailureAttribution(
                    category=category,
                    reason=f"反馈中包含关键词：{[kw for kw in keywords if kw in feedback_lower]}",
                    confidence=confidence,
                    suggested_fix=fix,
                )

        # 2. Trace 事件分析（如果有）
        if trace_events:
            error_events = [e for e in trace_events if e.get("status") == "error"]
            if error_events:
                return FailureAttribution(
                    category="tool_failure",
                    reason=f"Trace 中有 {len(error_events)} 个错误事件",
                    confidence=0.7,
                    suggested_fix="检查工具调用和后端服务",
                )

        # 3. 兜底
        return FailureAttribution(
            category="unknown",
            reason="无法从反馈关键词和 Trace 事件中推断失败原因",
            confidence=0.3,
            suggested_fix="人工查看完整对话记录",
        )
