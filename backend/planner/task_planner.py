"""L25 司机域 TaskPlanner：把 chat() 里硬编码的分支判断抽象成可校验的 RoutePlan。

设计原则（对照小哲电商 L25）：
1. 规则优先：plan_by_rules() 用确定性词表生成高置信 RoutePlan，confidence >= 0.85 直接返回，省模型调用。
2. 低置信时才调分类模型补充（diver 当前环境模型调用成本高，默认走规则，模型接口预留）。
3. _apply_safety_constraints() 统一高风险覆盖：退款/补偿类诉求必须走受控路径，模型不能把写动作降成轻路径。
4. _constrain_required_tools() 用 TOOL_SPECS 白名单收窄 required_tools，避免开放未知工具。

司机域 intent 取值（和 tools/planning.py 的 detect_business_fact_need 对齐）：
- driver_refund: 退款（发起退款、退款进度、到账）
- driver_coupon: 卡券状态（未使用张数、列表）
- driver_verification: 核销记录
- driver_activity: 活动完成度（达标/差几单）
- driver_qualification: 活动资格
- driver_income: 收入统计
- driver_order: 单量统计
- promotion_consult: 活动规则咨询（纯 RAG）
- complaint: 投诉
- general_chat: 寒暄
- unknown: 兜底
"""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    RoutePlan,
    RouteSource,
    ToolCandidate,
    PlannerTrace,
    RiskLevel,
)
from tools.catalog import DRIVER_TOOL_CATALOG


class ToolCatalog:
    """从 DriverToolCatalog 生成工具候选列表，供 TaskPlanner 收窄 required_tools。

    L24：工具候选完全从 MCP-style 目录派生（domain / read_only / risk_level / boundary / resource_uris），
    不再在 Planner 里硬编码"哪个工具属于哪个域、哪个工具风险高"。
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolCandidate] = {}
        for spec in DRIVER_TOOL_CATALOG.list_tools():
            self._tools[spec.name] = ToolCandidate(
                name=spec.name,
                domain=spec.domain,
                # 只读工具才允许在轻路径调用；写工具必须走受控工作流。
                allowed_in_light_path=spec.read_only,
                risk_level=spec.risk_level,
                reason=spec.description[:60],
            )

    def candidate_summaries(self, route_plan: RoutePlan) -> list[ToolCandidate]:
        """根据 route_plan 的 intent 收窄候选工具范围，不全量暴露。"""
        intent_tools = self._intent_tools(route_plan.intent)
        if intent_tools:
            return [self._tools[name] for name in intent_tools if name in self._tools]
        # 兜底：如果 intent 没匹配到特定工具集，返回全部只读工具候选。
        return list(self._tools.values())

    def allowed_required_tools(
        self, proposed: list[str], candidates: list[ToolCandidate]
    ) -> list[str]:
        """白名单收窄：只保留候选列表里存在的工具名，避免模型开放未知工具。"""
        allowed_names = {c.name for c in candidates}
        return [name for name in proposed if name in allowed_names]

    @staticmethod
    def _intent_tools(intent: str) -> list[str]:
        """司机域 intent → 候选工具映射。"""
        mapping: dict[str, list[str]] = {
            "driver_refund": ["get_refund_status", "get_coupon_status"],
            "driver_coupon": ["get_coupon_status", "get_verification_record"],
            "driver_verification": ["get_verification_record", "get_coupon_status"],
            "driver_activity": ["get_activity_progress"],
            "driver_qualification": ["get_activity_qualification", "get_activity_progress"],
            "driver_income": ["get_income_summary"],
            "driver_order": ["get_driver_order_count"],
            "promotion_consult": [],
            "complaint": [],
            "general_chat": [],
            "unknown": [],
        }
        return mapping.get(intent, [])


class TaskPlanner:
    """司机域路由规划器。"""

    # 卡券号/订单号识别（diver 司机域：CP-001、订单88321、8位数字）
    COUPON_ID_PATTERN = re.compile(r"\bCP[-_]?\d{2,}\b", re.IGNORECASE)
    ORDER_ID_PATTERN = re.compile(r"\b(?:订单|ORD)?[-_]?\d{6,}\b")

    # 否定模式：用户说"不是退款""不需要补偿"时，不应该命中高风险售后。
    NEGATED_REFUND_PATTERN = re.compile(
        r"(?:不是|并非|不想|不打算|不需要|无需|不用|不要)"
        r"(?:想要|想|要|申请|办理)?(?:退款|补偿|退钱)"
    )

    MIN_MODEL_CONFIDENCE = 0.65

    def __init__(self, tool_catalog: ToolCatalog | None = None) -> None:
        self.tool_catalog = tool_catalog or ToolCatalog()

    async def plan(
        self,
        user_message: str,
        runtime_context: dict[str, Any] | None = None,
    ) -> tuple[RoutePlan, PlannerTrace]:
        """生成本轮 RoutePlan，并把规则和工具候选约束收口到统一结果。"""
        rule_plan = self._apply_safety_constraints(
            user_message,
            self.plan_by_rules(user_message, runtime_context or {}),
        )
        candidates = self.tool_catalog.candidate_summaries(rule_plan)

        # 高置信直接返回规则结果，省模型调用（diver 当前模型成本高）。
        if rule_plan.confidence >= 0.85:
            constrained = self._constrain_required_tools(rule_plan, candidates)
            return constrained, self._build_trace(rule_plan, constrained, candidates)

        # 低置信时预留分类模型接口；当前 diver 环境默认走规则兜底。
        fallback_plan = rule_plan.model_copy(update={"source": "rules_fallback"})
        constrained = self._constrain_required_tools(fallback_plan, candidates)
        return constrained, self._build_trace(rule_plan, constrained, candidates)

    def _build_trace(
        self,
        rule_plan: RoutePlan,
        constrained: RoutePlan,
        candidates: list[ToolCandidate],
    ) -> PlannerTrace:
        return PlannerTrace(
            source=constrained.source,
            rule_confidence=rule_plan.confidence,
            candidate_tools=candidates,
            constrained_required_tools=constrained.required_tools,
            public_reason=self._public_reason(constrained),
        )

    def plan_by_rules(
        self, user_message: str, runtime_context: dict[str, Any]
    ) -> RoutePlan:
        """用确定性规则生成高置信 RoutePlan。"""
        text = user_message.strip()
        has_coupon_id = bool(self.COUPON_ID_PATTERN.search(text))
        has_order_id = bool(self.ORDER_ID_PATTERN.search(text))
        signals = self._route_signals(text)

        # 优先级 1：高风险退款/补偿诉求（带卡券号/订单号 → 走受控工作流）
        if signals["refund_action"] and (has_coupon_id or has_order_id):
            return self._decision(
                intent="driver_refund",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=self._rewrite_rag_query(text, "driver_refund"),
                confidence=0.95,
                source="rules",
                intents=["driver_refund"],
                required_tools=["get_refund_status", "get_coupon_status"],
                knowledge_domains=["refund"],
                has_realtime_fact=True,
                risk_level="high",
                requires_workflow=True,
                entity_refs=["coupon_id"] if has_coupon_id else ["order_id"],
                fallback_policy="workflow_first",
            )

        # 优先级 2：退款进度/到账查询（只读，不触发工作流）
        if signals["refund_query"]:
            return self._decision(
                intent="driver_refund",
                needs_rag=False,
                needs_business_tools=True,
                rag_query=text,
                confidence=0.9,
                source="rules",
                intents=["driver_refund"],
                required_tools=["get_refund_status"],
                has_realtime_fact=True,
                risk_level="low",
                fallback_policy="tool_first",
            )

        # 优先级 3：核销记录查询
        if signals["verification"]:
            tools = ["get_verification_record"]
            if not has_coupon_id:
                tools = ["get_coupon_status", "get_verification_record"]
            return self._decision(
                intent="driver_verification",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=self._rewrite_rag_query(text, "driver_verification"),
                confidence=0.9 if has_coupon_id else 0.8,
                source="rules",
                intents=["driver_verification"],
                required_tools=tools,
                knowledge_domains=["usage"],
                has_realtime_fact=True,
                entity_refs=["coupon_id"] if has_coupon_id else [],
                fallback_policy="tool_first_then_policy_caveat",
            )

        # 优先级 4：卡券状态查询
        if signals["coupon"]:
            return self._decision(
                intent="driver_coupon",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=self._rewrite_rag_query(text, "driver_coupon"),
                confidence=0.9,
                source="rules",
                intents=["driver_coupon"],
                required_tools=["get_coupon_status"],
                knowledge_domains=["usage"],
                has_realtime_fact=True,
                fallback_policy="tool_first_then_policy_caveat",
            )

        # 优先级 5：活动完成度（"活动资格"这类问题要落到优先级 6，不能被"活动"抢先命中）
        if signals["activity"] and not signals["qualification"]:
            tools = ["get_activity_progress"]
            return self._decision(
                intent="driver_activity",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=self._rewrite_rag_query(text, "driver_activity"),
                confidence=0.85 if signals["activity_id"] else 0.75,
                source="rules",
                intents=["driver_activity"],
                required_tools=tools,
                knowledge_domains=["promotion"],
                has_realtime_fact=True,
                entity_refs=["activity_id"] if signals["activity_id"] else [],
                fallback_policy="tool_first_then_policy_caveat",
            )

        # 优先级 6：活动资格
        if signals["qualification"]:
            return self._decision(
                intent="driver_qualification",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=self._rewrite_rag_query(text, "driver_qualification"),
                confidence=0.85 if signals["activity_id"] else 0.75,
                source="rules",
                intents=["driver_qualification"],
                required_tools=["get_activity_qualification"],
                knowledge_domains=["promotion"],
                has_realtime_fact=True,
                entity_refs=["activity_id"] if signals["activity_id"] else [],
                fallback_policy="tool_first_then_policy_caveat",
            )

        # 优先级 7：收入统计
        if signals["income"]:
            return self._decision(
                intent="driver_income",
                needs_rag=False,
                needs_business_tools=True,
                rag_query=text,
                confidence=0.9,
                source="rules",
                intents=["driver_income"],
                required_tools=["get_income_summary"],
                has_realtime_fact=True,
                fallback_policy="tool_first",
            )

        # 优先级 8：单量统计
        if signals["order_count"]:
            return self._decision(
                intent="driver_order",
                needs_rag=False,
                needs_business_tools=True,
                rag_query=text,
                confidence=0.9,
                source="rules",
                intents=["driver_order"],
                required_tools=["get_driver_order_count"],
                has_realtime_fact=True,
                fallback_policy="tool_first",
            )

        # 优先级 9：活动规则咨询（纯 RAG，不调工具）
        if signals["promotion_rule"]:
            return self._decision(
                intent="promotion_consult",
                needs_rag=True,
                needs_business_tools=False,
                rag_query=self._rewrite_rag_query(text, "promotion_consult"),
                confidence=0.85,
                source="rules",
                intents=["promotion_consult"],
                knowledge_domains=["promotion"],
                fallback_policy="knowledge_only",
            )

        # 优先级 10：投诉
        if signals["complaint"]:
            return self._decision(
                intent="complaint",
                needs_rag=True,
                needs_business_tools=False,
                rag_query=self._rewrite_rag_query(text, "complaint"),
                confidence=0.8,
                source="rules",
                intents=["complaint"],
                knowledge_domains=["complaint"],
                risk_level="medium",
                fallback_policy="knowledge_only",
            )

        # 兜底：低置信 general_chat
        return self._decision(
            intent="general_chat",
            needs_rag=False,
            needs_business_tools=False,
            rag_query=text,
            confidence=0.4,
            source="rules_fallback",
            intents=["general_chat"],
        )

    @classmethod
    def _route_signals(cls, text: str) -> dict[str, bool]:
        """收集司机域路由信号。注意：退款要区分"发起退款动作"和"查退款进度"。"""
        # 先剥离否定模式，避免"不是退款"被误判。
        active_text = cls.NEGATED_REFUND_PATTERN.sub("", text)

        # 发起退款动作（写操作意图，高风险）
        refund_action = any(
            kw in active_text
            for kw in ("发起退款", "申请退款", "我要退款", "给我退", "帮我退", "退款申请", "退钱", "补偿", "赔偿")
        )
        # 查退款进度（只读，低风险）
        refund_query = any(
            kw in active_text
            for kw in ("退款进度", "退款到账", "到账了吗", "退款状态", "退款记录", "我的退款")
        ) and not refund_action

        verification = any(
            kw in active_text
            for kw in ("核销", "核销记录", "核销了吗", "什么时候核销", "核销时间")
        )
        coupon = any(
            kw in active_text
            for kw in ("卡券", "优惠券", "券", "还有几张", "未使用", "已过期")
        ) and not verification
        activity = any(
            kw in active_text
            for kw in ("活动", "达标", "完成度", "差几单", "还差", "进度")
        )
        qualification = any(
            kw in active_text
            for kw in ("资格", "能参加", "报名条件", "符合条件", "门槛", "可以参加")
        )
        income = any(
            kw in active_text
            for kw in ("收入", "卡券收入", "本月收入", "收入统计", "赚了多少")
        )
        order_count = any(
            kw in active_text
            for kw in ("单量", "跑了多少单", "今日单", "本周单", "接了多少")
        )
        promotion_rule = any(
            kw in active_text
            for kw in ("活动规则", "活动怎么算", "满减", "优惠规则", "会员价")
        )
        complaint = any(
            kw in active_text
            for kw in ("投诉", "举报", "曝光", "315", "差评")
        )
        activity_id = bool(re.search(r"ACT[-_]?\d{2,}", text, re.IGNORECASE)) or "活动" in text and any(
            c.isdigit() for c in text
        )

        return {
            "refund_action": refund_action,
            "refund_query": refund_query,
            "verification": verification,
            "coupon": coupon,
            "activity": activity,
            "qualification": qualification,
            "income": income,
            "order_count": order_count,
            "promotion_rule": promotion_rule,
            "complaint": complaint,
            "activity_id": activity_id,
        }

    def _apply_safety_constraints(self, user_message: str, route_plan: RoutePlan) -> RoutePlan:
        """对规则结果统一执行高风险最终覆盖。模型不能把写动作降成轻路径。"""
        signals = self._route_signals(user_message)
        high_risk = (
            signals["refund_action"]
            or route_plan.risk_level == "high"
            or route_plan.requires_workflow
        )
        if not high_risk:
            return route_plan

        intents = list(dict.fromkeys(["driver_refund", *route_plan.intents]))
        domains = list(dict.fromkeys(["refund", *route_plan.knowledge_domains]))
        return route_plan.model_copy(
            update={
                "intent": "driver_refund",
                "needs_rag": True,
                "needs_business_tools": True,
                "rag_query": self._rewrite_rag_query(user_message, "driver_refund"),
                "confidence": max(route_plan.confidence, 0.95),
                "intents": intents,
                "required_tools": ["get_refund_status", "get_coupon_status"],
                "knowledge_domains": domains,
                # 高风险路径必然需要实时事实支撑资格判断。
                "has_realtime_fact": True,
                "risk_level": "high",
                "requires_workflow": True,
                "fallback_policy": "workflow_first",
            }
        )

    def _constrain_required_tools(
        self, route_plan: RoutePlan, candidates: list[ToolCandidate]
    ) -> RoutePlan:
        """用工具目录白名单收窄 required_tools。"""
        allowed = self.tool_catalog.allowed_required_tools(route_plan.required_tools, candidates)
        needs_business_tools = bool(allowed) or route_plan.has_realtime_fact or route_plan.requires_workflow
        if allowed == route_plan.required_tools and needs_business_tools == route_plan.needs_business_tools:
            return route_plan
        return route_plan.model_copy(
            update={"required_tools": allowed, "needs_business_tools": needs_business_tools}
        )

    @staticmethod
    def _decision(
        *,
        intent: str,
        needs_rag: bool,
        needs_business_tools: bool,
        rag_query: str,
        confidence: float,
        source: RouteSource,
        intents: list[str],
        entity_refs: list[str] | None = None,
        required_context: list[str] | None = None,
        required_tools: list[str] | None = None,
        knowledge_domains: list[str] | None = None,
        has_realtime_fact: bool = False,
        risk_level: RiskLevel = "low",
        requires_workflow: bool = False,
        fallback_policy: str = "default",
    ) -> RoutePlan:
        return RoutePlan(
            intent=intent,
            needs_rag=needs_rag,
            needs_business_tools=needs_business_tools or bool(required_tools) or has_realtime_fact,
            rag_query=rag_query,
            confidence=confidence,
            source=source,
            intents=intents,
            entity_refs=entity_refs or [],
            required_context=required_context or [],
            required_tools=required_tools or [],
            knowledge_domains=knowledge_domains or [],
            has_realtime_fact=has_realtime_fact,
            risk_level=risk_level,
            requires_workflow=requires_workflow,
            fallback_policy=fallback_policy,
        )

    @staticmethod
    def _rewrite_rag_query(text: str, intent: str) -> str:
        """为 RAG 检索生成更稳定的查询词。"""
        parts = [text]
        if intent == "driver_refund":
            parts.append("司机卡券退款政策 退款条件 退款期限")
        elif intent == "driver_verification":
            parts.append("卡券核销规则 核销记录查询")
        elif intent == "driver_coupon":
            parts.append("卡券使用规则 有效期 卡券状态")
        elif intent in ("driver_activity", "driver_qualification"):
            parts.append("司机活动规则 参与资格 达标条件")
        elif intent == "promotion_consult":
            parts.append("活动规则 满减 会员优惠")
        elif intent == "complaint":
            parts.append("投诉处理流程 客服响应")
        return " ".join(dict.fromkeys(parts))

    @staticmethod
    def _public_reason(route_plan: RoutePlan) -> str:
        """把路由原因改写成可展示摘要，不暴露内部策略细节。"""
        if route_plan.requires_workflow:
            return "本轮命中高风险售后路径，只返回资格判断和路由信号，不在轻路径执行写动作。"
        if route_plan.needs_rag and route_plan.needs_business_tools:
            return "本轮同时需要知识依据和实时业务事实，走 Tool + RAG 路由。"
        if route_plan.needs_business_tools:
            return "本轮需要实时业务事实，先收窄到对应只读工具候选。"
        if route_plan.needs_rag:
            return "本轮属于稳定规则咨询，走 RAG 知识路径。"
        return "本轮没有命中业务知识或工具路径，按普通客服对话收口。"
