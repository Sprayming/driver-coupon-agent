"""Agent 编排层，承接请求并决定本课要调用哪些模型、检索或观察能力。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from api.schemas import ChatRequest, ChatResponse, Intent, IntentResult, KnowledgeHit, Citation, RetrievalCacheEntry, RerankOutcome, BusinessFactNeed, BusinessFactResult, ToolAction, ToolCallRecord, ToolObservation, ClarificationPlan, ClarificationRequest, CostSummary

from models.llm_client import call_chat_model, create_tool_calling_model
# L15：向量召回之外新增关键词通道和 hybrid 融合，都住在 vector_store。
from rag.vector_store import retrieve_by_vector, get_vector_store, merge_candidates, keyword_retrieve, merge_hybrid_hits
# Task 13：BM25 硬关键词专路。
from rag.search_engine import BM25Local, SearchEngine
# Task 14：RAG-Fusion 多查询 + RRF 融合。
from rag.fusion import generate_sub_queries, rrf_fuse
from rag.prompting import render_vector_rag_messages
from embeddings.client import EmbeddingClient
from cost.observer import build_cost_summary
from config import settings
from config.settings import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
    TOP_K,
    # L14：宽召回候选池大小 + 重排后真正喂给回答的条数。
    CANDIDATE_K,
    FINAL_TOP_K,
)
from rag.quality import is_low_confidence, get_quality_summary, peek_quality_summary
# L13：意图分类抽到 rag/intent（质量评测也要用，避免循环导入）；查询改写是本课核心。
from rag.intent import classify_intent
from rag.query_rewrite import normalize_query, rewrite_retrieval_query
# L14：reranker 精排独立成模块，只吃候选列表、只吐排序结果，不侵入召回层。
from rag.reranker import rerank_candidates
# L15：检索前计划，复用意图和改写结果，产出场景路由 + 主题范围 + 关键词通道用词。
# L16：planning 再加实时业务问题识别（缓存边界 + 保守回答分支的判断依据）。
from rag.planning import pre_retrieval_plan
# L16：索引确认 + 检索缓存（版本指纹 / 缓存策略 / LRU 读写）。
# Task 9：get_release_index 按 release 构建并缓存索引。
from rag.index_cache import (
    get_knowledge_index,
    get_release_index,
    cache_key_for,
    cache_policy_for,
    cache_get,
    cache_put,
)
# Task 9：按 driver_id 解析灰度路由，得到本轮请求的 release_id。
# Task 11：RBAC 角色过滤 + audience 硬隔离。
from store.releases import (
    resolve_gray_route,
    get_allowed_kb_ids,
    get_allowed_audiences,
)
# L17：司机域实时业务事实接入——detect_business_fact_need 产出 BusinessFactNeed，
# BusinessFactService.lookup 执行三道安全闸 + 调 driver_service_mock（8082端口）。
from tools.planning import detect_business_fact_need
from services.business_facts import BusinessFactService
# L18：LangChain Tool Calling——工具说明书 + StructuredTool 包装层。
from tools.contracts import TOOL_SPECS
from tools.catalog import DRIVER_TOOL_CATALOG
# L24：MCP 客户端——按配置发现 Server、握手、拉工具池、按命名空间路由调用。
from mcp_client.manager import MCPClientManager
from tools.langchain_tools import build_langchain_tools, tool_action_reason
# L19：工具澄清——LLM 产出 ClarificationPlan 草案，后端校验后决定「先反问」还是「调工具」。
from models.clarification_planner import ClarificationPlannerModelClient
from tools.clarification_planning import (
    enrich_plan_with_runtime_context,
    plan_tool_action,
    post_tool_clarification,
    pre_tool_clarification,
)
from tools.tool_runtime import execute_tool_action
# L20：Observation 压缩层——工具原始结果不能直接进模型，先压缩成安全 Observation。
from observability.observation import build_observation
# L21：错误降级——高风险写操作拦截 + 工具超时/模型不可用兜底。
from degradation.fallbacks import fallback_answer, classify_risk, ModelServiceError
# L23：Hooks 治理层——pre/post/error/completion 统一卡口。
from hooks.manager import HookManager
from api.schemas import ErrorCategory, DegradationState, Observation as _Observation
# L25：TaskPlanner 路由规划——把硬编码分支判断抽象成可校验的 RoutePlan。
from planner.task_planner import TaskPlanner
# L26：高风险动作边界——退款/补偿只做资格判断，不执行写动作。
from policies.refund_policy import assess_refund_boundary, BLOCKED_WRITE_ACTIONS
# L32-L34：Session Memory + 上下文构建——多轮对话记忆 + 澄清闭环
from context.builder import (
    consume_pending_clarification,
    find_context_coupon,
    remember_pending_clarification,
    update_memory,
)
# L27-L29：LangGraph 退款审批 Workflow + L30 HITL resume
from workflows.refund_graph import run_refund_approval
# L36：Prompt Injection 防御 + 工具结果安全检查
from safety.source_guard import inspect_tool_result

import httpx


def build_citations(hits:list[KnowledgeHit])-> list[Citation]:
    """将检索到的知识片段转换为引用格式"""
    return [
        Citation(
            citation_id=f"C{index}",
            source_path=hit.chunk.source_path,
            source_title=hit.chunk.title,
            chunk_id=hit.chunk.chunk_id,
            score=hit.score,
            snippet=hit.chunk.text,
        )
        for index, hit in enumerate(hits, start=1)
    ]

# def build_fallback_answer () -> str :
#     """没有可靠命中时生成固定兜底回答。""" 
#     return ( "我暂时没有在当前知识库里找到可靠依据，不能直接给出规则结论。" "如果你有其他问题，请告诉我，我会尽力帮你。")
def build_fallback_answer() -> str:
    """低置信或无可靠依据时的兜底回答。"""
    return (
        "我现在没有找到足够可靠的卡券规则依据，不能直接给出结论。"
        "请你补充卡券类型、购买时间或订单状态；如果涉及退款争议，我可以帮你转人工继续核验。"
    )

def build_general_chat_answer () -> str : 
    """普通客服寒暄不需要 RAG 依据。""" 
    return ( "你好，我是网约车司机端卡券客服 Agent。你可以直接描述卡券购买、使用、退款或投诉问题；" "涉及具体规则时，我会优先依据当前知识库回答，找不到可靠依据时不会编造。" )

def build_clarification_answer(clarification: ClarificationRequest) -> str:
    """把结构化澄清请求渲染成确定性问句。

    L19：需要澄清时不调模型，避免模型改写追问或替司机挑候选。
    """
    if clarification.candidates:
        choices = "；".join(
            f"{candidate.value}（{candidate.label}）" for candidate in clarification.candidates
        )
        return f"{clarification.message} 我查到你名下这些：{choices}。"
    return clarification.message


class CustomerServiceAgent:
    """客户服务中心 Agent：向量检索 + token 成本观察"""

    def __init__(
        self,
        *,
        chat_http_client: httpx.Client | None = None,
        chat_api_key: str | None = None,
        chat_base_url: str | None = None,
        chat_model_name: str | None = None,
        clarification_planner: ClarificationPlannerModelClient | None = None,
    ) -> None:
        """初始化向量检索 Agent，并准备会话级 token 观察日志。"""
        self._message_count_by_session: dict[str, int] = {}
        self._cost_events_by_session: dict[str, list[dict]] = {}
        self._chat_http_client = chat_http_client
        self._chat_api_key = chat_api_key
        self._chat_base_url = chat_base_url
        self._chat_model_name = chat_model_name
        # Embedding 客户端，用于向量检索
        self._embedding_client = EmbeddingClient(
            base_url=DEFAULT_EMBEDDING_BASE_URL,
            model=DEFAULT_EMBEDDING_MODEL,
            api_key=chat_api_key,
        )
        # L17：司机域实时业务事实服务（三道安全闸 + driver_service_mock）。
        self._business_facts = BusinessFactService()
        # L19：澄清规划器（模型不可用时由 chat() 降级到 L17 直查）。
        self._clarification_planner = clarification_planner or ClarificationPlannerModelClient()
        # L24：MCP 客户端。读配置 → 连 Server → 握手 → 拉工具池，
        # 成功则用工具池重建本地目录；失败则保持静态 TOOL_SPECS，不影响主链路。
        self._mcp_client = MCPClientManager()
        try:
            self._mcp_ready = self._mcp_client.ensure_ready_sync()
        except Exception:
            self._mcp_ready = False
        if self._mcp_ready:
            DRIVER_TOOL_CATALOG.load_tool_specs(self._mcp_client.local_tool_specs())

    def warm_up_offline_caches(self) -> None:
        """预热向量库和质量摘要，让首批 /chat 不必支付冷启动 embedding。

        必须传入本实例的 embedding client：向量库缓存按客户端实例做 key，
        换成别的实例预热会和请求路径的缓存对不上，等于白预热。
        """

        get_vector_store(embedding_client=self._embedding_client)
        get_quality_summary(embedding_client=self._embedding_client)

    def _run_langchain_tool_calling(
        self,
        request: ChatRequest,
    ) -> tuple[str, list[ToolCallRecord], dict[str, Any]]:
        """用 LangChain create_agent 执行司机域工具调用闭环。

        L18 核心：LangChain 负责生成 AIMessage.tool_calls、执行 StructuredTool、
        产生 ToolMessage。工具执行收口到 execute_tool_action（三道安全闸）。
        返回的 tool_calls 是调试台观察格式，从 LangChain 消息整理而来。
        """
        from tools.langchain_tools import build_langchain_tools as _build

        tools = _build(request)
        try:
            model = create_tool_calling_model()
        except RuntimeError:
            # 模型不可用 → 降级：不调工具，直接返回提示
            return "当前工具调用模型未配置，暂时无法查询实时业务数据。", [], {"create_agent": False, "skip_reason": "model_not_configured"}

        langchain_agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "你是网约车司机端卡券客服 Agent。需要查询司机活动完成度、单量、活动资格、"
                "退款进度、卡券状态、核销记录或收入统计等实时业务事实时，必须优先调用给定工具确认事实。"
                "不要编造单量、达标状态、退款金额或卡券数量。"
                "工具执行前后都由后端校验参数和当前司机身份，你只能调当前司机自己的工具。"
            ),
        )
        result = langchain_agent.invoke(
            {"messages": [{"role": "user", "content": request.user_message}]},
            config={"recursion_limit": 4},
        )
        messages: list[BaseMessage] = result["messages"]
        final_answer = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            "工具调用没有返回可用结果。",
        )
        tool_calls = self._tool_calls_from_langchain_messages(messages)
        langchain_state = {
            "model": self._model_label(model),
            "create_agent": True,
            "available_tools": [t.name for t in tools],
            "message_types": [m.__class__.__name__ for m in messages],
        }
        return final_answer, tool_calls, langchain_state

    @staticmethod
    def _model_label(model: BaseChatModel) -> str:
        """返回公开可展示的模型名称，不含 Key 或私有配置。"""
        for attr in ("model_name", "model"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return model.__class__.__name__

    def _tool_calls_from_langchain_messages(self, messages: list[BaseMessage]) -> list[ToolCallRecord]:
        """把 LangChain AIMessage.tool_calls / ToolMessage 配对成调试台 ToolCallRecord。"""
        records: list[ToolCallRecord] = []
        observations_by_id: dict[str, ToolObservation] = {}
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                observations_by_id[str(message.tool_call_id)] = ToolObservation.model_validate_json(str(message.content))
            except Exception:
                continue

        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                tool_name = str(tool_call["name"])
                observation = observations_by_id.get(str(tool_call["id"]))
                if observation is None:
                    continue
                records.append(
                    ToolCallRecord(
                        action=ToolAction(
                            tool_name=tool_name,
                            arguments=dict(tool_call.get("args") or {}),
                            reason=tool_action_reason(tool_name),
                        ),
                        observation=observation,
                    )
                )
        return records

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        处理客户请求：向量检索 → 调用模型 → 记录成本。
        """
        # L37: 全链路 Trace——每个关键阶段都写事件
        from observability.trace import get_trace_store
        trace_store = get_trace_store()
        # L23: 初始化 Hooks 治理管理器
        hooks = HookManager()
        # L25: 初始化 TaskPlanner（路由规划）
        task_planner = TaskPlanner()
        # L23: 主链路 return 前统一赋值；高风险提前 return 分支会自己覆盖这个 None
        hooks_on_completion = None
        # L25/L26: 路由规划 + 高风险评估，在分支判断前统一计算
        route_plan = None
        planner_trace = None
        risk_assessment = None

        # 1. 会话计数
        self._message_count_by_session[request.session_id] = (
            self._message_count_by_session.get(request.session_id, 0) + 1
        )
        message_count = self._message_count_by_session[request.session_id]

        
        # # 2. 向量检索（L10 核心变化：按语义召回，不再按意图全选）
        # hits = retrieve_by_vector(
        #     request.user_message,
        #     embedding_client=self._embedding_client,
        # )
        # citations = build_citations(hits)
        # # 3. 意图分类（先检索在分类）
        # intent_result = classify_intent(request.user_message,hits)

        # 2. L13：先用归一化后的文本做纯规则意图分类（改写依赖意图，必须在检索之前）。
        normalized_message = normalize_query(request.user_message)
        intent_result = classify_intent(normalized_message)
        # L37: 记录意图识别 Trace
        trace_store.add(request.session_id, "intent", {
            "user_message": request.user_message,
            "classified_intent": intent_result.intent,
        })

        # 2.1 L32-L34：Session Memory 澄清消费——上一轮让用户补参数，这一轮用户补了？
        #   consume_pending_clarification 会检查 pending_clarification 是否过期/被取代/补完。
        #   如果补完了，返回上一轮的真实意图（覆盖这一轮可能不准的 classify_intent 结果）。
        #   场景：上一轮用户说"我要退款"→ 系统反问"请提供卡券号"→ 这一轮用户只补了 CP...
        #        classify_intent("CP...") 可能识别成 general_chat，但真实意图是 refund_request
        import re as _re
        _coupon_match = _re.search(r'CP\d{6,}', normalized_message)
        explicit_coupon_id = _coupon_match.group() if _coupon_match else None
        resolved_clarification, _clarification_transition = consume_pending_clarification(
            session_id=request.session_id,
            runtime_user_id=request.runtime_user_id,
            current_intent=intent_result.intent,
            supplied_fields={"coupon_id": explicit_coupon_id},
            current_message_count=message_count,
        )
        if resolved_clarification is not None:
            # 上一轮澄清已解决 → 用恢复的真实意图覆盖 classify_intent 的结果
            intent_result = IntentResult(intent=resolved_clarification["intent"])

        # 3. L13 核心：用户原话不动，只为检索生成 rewritten_query。
        rewrite = rewrite_retrieval_query(request.user_message, intent_result.intent)

        # 4. L15 核心：检索前计划。复用意图和改写结果，产出场景路由、主题范围和关键词通道用词。
        #    注意执行顺序较 L14 有调整：plan 必须在两路向量召回之前生成（召回要带主题过滤），
        #    所以 raw_hits 的计算从「改写之前」挪到「plan 之后」，语义不变（原话基线，独立检索）。
        plan = pre_retrieval_plan(request.user_message, intent_result, rewrite)

        # 4.1 L25: TaskPlanner 路由规划——把硬编码分支判断抽象成可校验的 RoutePlan。
        #    plan_by_rules 用司机域词表生成高置信决策，confidence >= 0.85 直接返回。
        #    route_plan.needs_business_tools 替代原来的 driver_domain_realtime。
        import asyncio
        try:
            route_plan, planner_trace = asyncio.run(
                task_planner.plan(request.user_message, request.runtime_context or {})
            )
        except Exception:
            # TaskPlanner 失败时降级为通用兜底，不让规划异常阻断聊天。
            route_plan = None
            planner_trace = None

        # 4.5 L16：索引确认。知识库内容的 sha256 指纹是本轮所有缓存的「身份证」。
        # Task 9：生命周期开关开启时，按 runtime_user_id 解析灰度路由，
        #   取对应 release 快照的索引（灰度司机看 candidate，其余看 stable）；
        #   开关关闭时回退 L16 全局单索引（与课程行为一致）。
        release_route = None
        if settings.KNOWLEDGE_LIFECYCLE_ENABLED:
            release_route = resolve_gray_route(request.runtime_user_id)
            index = get_release_index(release_route.release_id)
        else:
            index = get_knowledge_index()

        # Task 11：RBAC 过滤。只信请求顶层 runtime_role（消息体提权无效）。
        # 开关关闭时 allowed_kb_ids/allowed_audiences 为 None，检索层不过滤（L16 旧行为）。
        allowed_kb_ids: set[str] | None = None
        allowed_audiences: set[str] | None = None
        if settings.RBAC_ENABLED:
            allowed_kb_ids = get_allowed_kb_ids(request.runtime_role)
            allowed_audiences = get_allowed_audiences(request.runtime_role)

        # 4.6 L17 → L25：实时业务问题识别。
        # L25 用 route_plan.needs_business_tools 替代原来的 driver_domain_realtime。
        # route_plan 不可用时回退到 L17 的 detect_business_fact_need（兼容降级）。
        business_fact_need: BusinessFactNeed = detect_business_fact_need(request)
        driver_domain_realtime = (
            route_plan.needs_business_tools
            if route_plan is not None
            else business_fact_need.requires_realtime
        )

        # L21 → L25：风险等级优先取 route_plan.risk_level（TaskPlanner 统一口径），
        # route_plan 不可用时回退到 L21 classify_risk。
        risk_level = (
            route_plan.risk_level
            if route_plan is not None
            else classify_risk(intent_result.intent, request.user_message)
        )
        degradation = DegradationState(degraded=False, error_category="none")

        # L26：高风险退款/补偿诉求——先做资格判断，不直接转人工，也不执行写动作。
        # 用 route_plan.requires_workflow 判断是否进入受控售后路径。
        requires_workflow = (
            route_plan.requires_workflow if route_plan is not None else risk_level == "high"
        )
        if requires_workflow:
            # 提取卡券号 / 订单号（从用户消息）
            import re
            coupon_id_match = re.search(r"\bCP[-_]?\d{2,}\b", request.user_message, re.IGNORECASE)
            order_id_match = re.search(r"\b(?:订单|ORD)?[-_]?\d{6,}\b", request.user_message)
            coupon_id = coupon_id_match.group(0) if coupon_id_match else None
            order_id = order_id_match.group(0) if order_id_match else None

            # 有卡券号时尝试查卡券状态（只读工具，不执行写动作）
            coupon_status = None
            verified_at = None
            tool_calls_for_risk: list[ToolCallRecord] = []
            if coupon_id:
                try:
                    # L24：从 DriverToolCatalog 读工具，而不是直接读 TOOL_SPECS
                    spec = DRIVER_TOOL_CATALOG.get_tool("get_coupon_status")
                    action = ToolAction(
                        tool_name="get_coupon_status",
                        arguments={"status": "unused"},
                        reason="L26 高风险资格判断：查询卡券状态以判定退款资格。",
                    )
                    if spec:
                        hooks.pre_tool_call(action, request, spec)
                    tool_obs = execute_tool_action(action, request)
                    # L36: 工具结果安全检查——防止 Prompt Injection 通过工具结果注入
                    if tool_obs and tool_obs.status == "success":
                        is_safe, _reason = inspect_tool_result(tool_obs.data)
                        if not is_safe:
                            tool_obs = None  # 不安全的结果直接丢弃，不进后续逻辑
                    if tool_obs and tool_obs.status == "success":
                        # 从工具返回里找该卡券的状态
                        coupons = tool_obs.data.get("coupons", [])
                        for c in coupons:
                            if str(c.get("coupon_id", "")).upper() == coupon_id.upper():
                                coupon_status = c.get("status")
                                verified_at = c.get("verified_at")
                                break
                        if coupon_status is None and coupons:
                            coupon_status = coupons[0].get("status")
                            verified_at = coupons[0].get("verified_at")
                    raw_obs = build_observation(tool_obs) if tool_obs else None
                    if raw_obs:
                        safe_obs = hooks.post_tool_call(raw_obs)
                        tool_calls_for_risk.append(ToolCallRecord(action=action, observation=tool_obs))
                except Exception:
                    pass  # 工具失败不阻断资格判断，coupon_status 保持 None → blocked

            # L27-L29: 评估退款资格边界——跑 LangGraph 退款审批图
            #   run_refund_approval 内部调 assess_refund_boundary（逻辑一致），
            #   如果 LangGraph 可用且需要人工审批 → 图会在 pause_for_human_approval 处 interrupt
            risk_assessment = run_refund_approval(
                coupon_id=coupon_id or "",
                coupon_status=coupon_status,
                thread_id=request.session_id,
            )
            # L37: 记录退款 Workflow Trace
            trace_store.add(request.session_id, "refund_workflow", {
                "coupon_id": coupon_id,
                "coupon_status": coupon_status,
                "eligibility_status": risk_assessment.eligibility_status,
                "needs_human_approval": risk_assessment.needs_human_approval,
            })

            # 根据资格状态生成回答
            if risk_assessment.eligibility_status == "eligible_for_application":
                target = coupon_id or order_id or "该卡券"
                reason_text = risk_assessment.reasons[0]
                answer = (
                    f"关于 {target} 的退款申请：{reason_text} "
                    "我不会因为你一句『帮我退』就直接执行退款；后续必须进入受控审批流程。"
                )
                next_action = "route_to_controlled_workflow"
            elif risk_assessment.eligibility_status == "needs_clarification":
                # L32：写入 Session Memory，下一轮用户补卡券号后 consume_pending_clarification 会恢复原意图
                remember_pending_clarification(
                    session_id=request.session_id,
                    runtime_user_id=request.runtime_user_id,
                    intent=intent_result.intent,
                    required_field="coupon_id",
                    current_message_count=message_count,
                )
                answer = (
                    "退款属于高风险操作，必须先提供卡券号或订单号。"
                    "我可以帮你判断退款资格，但不会直接退款、取消或补偿。"
                )
                next_action = "ask_clarification"
            else:
                target = coupon_id or order_id or "该卡券"
                answer = (
                    f"关于 {target} 的退款：{risk_assessment.reasons[0]}"
                    "我只能解释依据并建议转人工复核，不执行任何写动作。"
                )
                next_action = "transfer_to_human"

            degradation = DegradationState(
                degraded=True,
                error_category="high_risk_write_blocked",
                fallback_message=answer,
            )
            # L23: Hooks completion
            hooks_on_completion = hooks.on_completion(
                next_action=next_action, risk_level="high", degradation=degradation
            )
            return ChatResponse(
                session_id=request.session_id,
                answer=answer,
                intent=intent_result.intent,
                intent_result=intent_result,
                citations=risk_assessment.policy_basis,
                cost_summary=CostSummary(
                    prompt_tokens=0, answer_tokens=0, total_tokens=0,
                    token_source="local_estimate", context_chars=0,
                    estimated_input_cost_cny=0.0, estimated_output_cost_cny=0.0,
                    estimated_total_cost_cny=0.0,
                    pricing_note="high_risk_assessment_no_model_call",
                ),
                reasoning_summary=[
                    "[L25 RoutePlan] 命中高风险售后路径，requires_workflow=True。",
                    "[L26 高风险边界] 只做退款资格判断，不执行写动作。",
                    f"[L26] blocked_write_actions={BLOCKED_WRITE_ACTIONS}",
                    "[L23 Hooks] pre/post/on_completion 卡口记录治理摘要。",
                ],
                session_state={
                    "agent_version": "lesson-26-high-risk-action-boundary",
                    "message_count": message_count,
                    "degradation": degradation.model_dump(),
                    "route_plan": route_plan.model_dump() if route_plan else None,
                    "planner_trace": planner_trace.model_dump() if planner_trace else None,
                    "risk_assessment": risk_assessment.model_dump(),
                    "hooks": {
                        "events": [e.model_dump() for e in hooks.events],
                        "completion": hooks_on_completion.model_dump() if hooks_on_completion else None,
                    },
                    # L24：高风险分支也输出 MCP 客户端视图，保持调试台口径一致。
                    "mcp_client": {
                        "ready": self._mcp_ready,
                        **self._mcp_client.summary(),
                    },
                    "next_gap": "高风险资格判断已实现；受控审批工作流（LangGraph/HITL）在 L27-L31。",
                },
                tool_calls=tool_calls_for_risk,
                clarification=None,
                risk_level="high",
                needs_human_approval=True,
                degraded=True,
                hook_events=hooks.events,
                hook_completion=hooks_on_completion,
                route_plan=route_plan,
                planner_trace=planner_trace,
                risk_assessment=risk_assessment,
            )

        # 4.7 L17：缓存策略扩展——司机域实时问题也不可缓存。
        # cache_policy_for 内部只认 RAG 检索口径，这里手动把 driver_domain_realtime 纳入不可缓存红线。
        cache_policy = cache_policy_for(plan)
        if driver_domain_realtime and cache_policy.get("cacheable"):
            cache_policy = {
                "cacheable": False,
                "scope": "retrieval_hits_only",
                "reason": "司机域实时业务事实（活动/单量/资格）不能缓存为知识库答案。",
            }
        cache_key = cache_key_for(plan, index)
        cached_entry = cache_get(cache_key)
        cache_hit = cache_policy["cacheable"] and cached_entry is not None

        # 5-8. 检索 + 精排。L14/L15 链路只在缓存未命中时执行，命中时直接复用上次的精排结果。
        # （L13 旧链路注释见下方 else 分支内的对应位置，注释保留，不删。）
        if cache_hit:
            reranked_hits = list(cached_entry.hits)
            rerank_outcome = RerankOutcome(
                hits=reranked_hits,
                mode=cached_entry.rerank_mode,
                model=cached_entry.rerank_model,
                error=cached_entry.rerank_error,
            )
            # 命中时三路召回明细留空：本次请求没真的跑召回，调试台看 cache.cache_hit 区分。
            raw_hits, rewritten_hits, keyword_hits = [], [], []
            vector_candidates, candidates = [], []
            # Task 21 回归修复：BM25/Fusion 明细同属召回产物，命中时一并留空——
            # 此前漏初始化导致缓存命中直接 UnboundLocalError（500），
            # 旧测试从未重复问同一问题所以没暴露（TR-21.3 缓存命中计时首次踩中）。
            bm25_hits = []
            fusion_sub_queries = []
        else:
            # 5. 第一次检索：用户原话，只作改写前的对比基线，raw_hits 不参与回答。
            # L14：从「只取 top 2」放宽成「按 CANDIDATE_K 宽召回」，让更多候选先进池子再交给 reranker 精排。
            # L15：再叠加 plan 的主题路由过滤，其他场景的知识不进候选池。
            # raw_hits = retrieve_by_vector(request.user_message, embedding_client=self._embedding_client)
            raw_hits = retrieve_by_vector(
                request.user_message,
                top_k=CANDIDATE_K,
                embedding_client=self._embedding_client,
                allowed_topics=plan.allowed_topics,
                index=index,
                allowed_kb_ids=allowed_kb_ids,
                allowed_audiences=allowed_audiences,
            )

            # 6. 第二次检索：改写后问题，真正用于回答。L14/L15 改动同上。
            # rewritten_hits = retrieve_by_vector(
            #     rewrite.rewritten_query, embedding_client=self._embedding_client
            # )
            rewritten_hits = retrieve_by_vector(
                rewrite.rewritten_query,
                top_k=CANDIDATE_K,
                embedding_client=self._embedding_client,
                allowed_topics=plan.allowed_topics,
                index=index,
                allowed_kb_ids=allowed_kb_ids,
                allowed_audiences=allowed_audiences,
            )

            # 7. L15 新增：关键词通道。用 plan.keyword_terms 撞 chunk 的 keywords/text，
            #    专补向量召回够不着的长尾精确词（规则编号、稀有业务词）。
            keyword_hits = keyword_retrieve(
                plan,
                embedding_client=self._embedding_client,
                index=index,
                allowed_kb_ids=allowed_kb_ids,
                allowed_audiences=allowed_audiences,
            )

            # 8. L12 第二道线作用在改写后的分数上。
            # low_confidence = is_low_confidence(rewritten_hits)
            # reliable_hits = [] if low_confidence else rewritten_hits
            # citations = build_citations(reliable_hits)
            # ↑ 以上是 L13 旧链路（注释保留，不删）：直接拿改写后的检索结果当依据，
            #   既没有候选池合并，也没有精排，多个相似片段同时召回时 top-k 顺序不精。
            # ↓ L14/L15 新链路：向量双路合并 → 与关键词通道 hybrid 融合 → reranker 精排
            #   → 用「重排后的分数」判低置信 → 只取 FINAL_TOP_K 条。
            #   分工：向量+关键词负责「宽」（互补召回，该进池的别漏），重排负责「精」（进池的排对顺序）。
            vector_candidates = merge_candidates(raw_hits, rewritten_hits)

            # Task 13：BM25 硬关键词专路（政策编号/商品名精确命中）。
            bm25_engine: SearchEngine | None = None
            if settings.BM25_ENABLED:
                bm25_engine = BM25Local(list(index.chunks_by_id.values()))

            def _bm25_for(query: str) -> list[KnowledgeHit]:
                if bm25_engine is None:
                    return []
                hits = bm25_engine.search(query, top_k=CANDIDATE_K)
                if allowed_kb_ids is not None or allowed_audiences is not None:
                    hits = [
                        h for h in hits
                        if (allowed_kb_ids is None or not h.chunk.kb_id or h.chunk.kb_id in allowed_kb_ids)
                        and (allowed_audiences is None or h.chunk.audience in allowed_audiences)
                    ]
                return hits

            bm25_hits = _bm25_for(rewrite.rewritten_query)

            # Task 14：RAG-Fusion——多子查询各跑 hybrid 后 RRF 融合。
            fusion_sub_queries: list[str] = []
            if settings.RAG_FUSION_ENABLED:
                fusion_sub_queries = generate_sub_queries(
                    rewrite.rewritten_query, count=settings.FUSION_QUERY_COUNT
                )
                sub_ranked: list[list[KnowledgeHit]] = []
                for sq in fusion_sub_queries:
                    sq_vec = retrieve_by_vector(
                        sq,
                        top_k=CANDIDATE_K,
                        embedding_client=self._embedding_client,
                        allowed_topics=plan.allowed_topics,
                        index=index,
                        allowed_kb_ids=allowed_kb_ids,
                        allowed_audiences=allowed_audiences,
                    )
                    sq_kw = keyword_retrieve(
                        plan,
                        embedding_client=self._embedding_client,
                        index=index,
                        allowed_kb_ids=allowed_kb_ids,
                        allowed_audiences=allowed_audiences,
                    )
                    sq_bm25 = _bm25_for(sq)
                    sub_ranked.append(merge_hybrid_hits(plan, sq_vec, sq_kw, sq_bm25))
                candidates = rrf_fuse(sub_ranked)
            else:
                candidates = merge_hybrid_hits(plan, vector_candidates, keyword_hits, bm25_hits)
            rerank_outcome = rerank_candidates(rewrite.rewritten_query, candidates)
            reranked_hits = rerank_outcome.hits
            # L16：把「索引版本 + 检索计划 + rerank 后命中」整体存入缓存；
            # 只存检索结果，最终回答永不入缓存。
            cache_put(
                cache_key,
                RetrievalCacheEntry(
                    index_version=index.version,
                    plan=plan,
                    hits=reranked_hits,
                    rerank_mode=rerank_outcome.mode,
                    rerank_model=rerank_outcome.model,
                    rerank_error=rerank_outcome.error,
                ),
            )

        # L14：低置信判断——只有 is_low_confidence 打分口径。
        low_confidence = is_low_confidence(reranked_hits)
        reliable_hits = [] if low_confidence else reranked_hits[:FINAL_TOP_K]

        # Task 11：prompt 构建前最终断言——深度防御，确保喂给模型的依据不含越权内容。
        # 检索层已过滤，这里兜底：若仍有越权 chunk，写审计 rbac.denied 并清空引用。
        # Task 19：拦截明细同步进 session_state（rbac.denied_chunk_ids），TracePanel 可展示权限拦截事件。
        rbac_denied_chunk_ids: list[str] = []
        if settings.RBAC_ENABLED:
            leaked = [
                hit for hit in reliable_hits
                if (allowed_audiences is not None and hit.chunk.audience not in allowed_audiences)
                or (allowed_kb_ids is not None and hit.chunk.kb_id and hit.chunk.kb_id not in allowed_kb_ids)
            ]
            if leaked:
                from store.db import transaction
                from store import repository as repo

                with transaction(settings.DB_PATH) as conn:
                    repo.write_audit(
                        conn,
                        actor=request.runtime_user_id,
                        action="rbac.denied",
                        target_type="query",
                        target_id=request.session_id,
                        after={
                            "role": request.runtime_role,
                            "leaked_chunk_ids": [h.chunk.chunk_id for h in leaked],
                            "leaked_audiences": [h.chunk.audience for h in leaked],
                        },
                        comment="RBAC 最终断言拦截越权知识片段",
                    )
                rbac_denied_chunk_ids = [h.chunk.chunk_id for h in leaked]
                reliable_hits = []

        citations = build_citations(reliable_hits)
        

        # #4.三条answer_path 分支（L11 核心）
        # if hits:
        #     messages=render_vector_rag_messages(request, intent_result, hits)
        #     model_result = call_chat_model(
        #         messages,
        #         http_client=self._chat_http_client,
        #         api_key=self._chat_api_key,
        #         base_url=self._chat_base_url,
        #         model=self._chat_model_name,
        #     )
        #     answer = model_result.answer
        #     answer_path = "answer_with_citations"
        #     model_usage = model_result.usage
        # elif intent_result.intent == "general_chat":
        # # 普通寒暄：固定话术，不调模型 
        #     messages = [
        #             { "role" : "system" , "content" : "你是网约车司机端卡券客服 Agent。普通寒暄可以直接接住，业务规则结论必须有知识依据。" },
        #             { "role" : "user" , "content" : f"用户原话： {request.user_message} \n粗意图： {intent_result.intent} \n当前分支：普通客服寒暄，不需要 citations。" },
        #     ]
        #     answer = build_general_chat_answer()
        #     answer_path = "general_chat_without_citations"
        #     model_usage = None
        # else :
        #      # 无命中且非寒暄：固定兜底话术，不调模型 
        #     messages = render_vector_rag_messages(request, intent_result, hits)
        #     answer = build_fallback_answer()
        #     answer_path = "no_reliable_knowledge_fallback"
        #     model_usage = None

        # L17：business_fact_result 在 driver_domain_realtime 分支里赋值，
        # 分支外 session_state 需要读取，这里先初始化为 None 哨兵。
        business_fact_result: BusinessFactResult | None = None
        # L18：工具调用记录 + LangChain 状态（RAG 分支为空/未创建）。
        tool_calls: list[ToolCallRecord] = []
        langchain_state: dict[str, Any] = {
            "create_agent": False,
            "skip_reason": "not_realtime_routed",
            "available_tools": [spec.name for spec in DRIVER_TOOL_CATALOG.list_tools()],
            "message_types": [],
        }
        # L19：规划器产物（非司机域实时分支保持 None）。
        clarification_plan: ClarificationPlan | None = None
        clarification: ClarificationRequest | None = None
        planned_action: ToolAction | None = None
        observation: ToolObservation | None = None
        # L20：原始 ToolObservation（压缩前）+ 压缩后的安全 Observation。
        tool_observation: ToolObservation | None = None
        compressed_observation: "Observation | None" = None

        # L12/L13 旧判断（注释保留，不删）：只看有没有可靠依据，寒暄排在后面。
        # if reliable_hits:                       # ← L11 这里是 if hits:
        # L14 改动：寒暄要排除在外。原因：低置信线是按分数口径校出来的，
        # 重排（尤其商业 reranker）的分和向量分不是同一量纲——实测纯寒暄「你好」也会被重排抬到 0.49 以上，
        # 于是被错判成「有依据可引用」，既白调一次模型，又会给寒暄硬挂上不相干的 citation。
        # 寒暄本来就不该依赖检索依据，所以这里显式排掉 general_chat。
        # L16 回答分支注释：实时/寒暄/兜底三条岔路。
        # L17 新增司机域实时业务事实双轨——先看有没有工具可用，再看有没有依据。
        # 优先级：driver_domain_realtime（有工具链）> 正常 RAG > 寒暄 > 兜底。
        if driver_domain_realtime:
            # L19：先让 LLM 产出 ClarificationPlan 草案，后端校验后再决定
            # 「先反问补参」还是「生成 ToolAction 调工具」。
            # L18 的 LangChain 工具循环（_run_langchain_tool_calling）保留在本文件留档，
            # 不接入主链路——课程 L20/L21 会把澄清和工具循环重新合回去。
            planner_failed = False

            try:
                clarification_plan = enrich_plan_with_runtime_context(
                    request,
                    self._clarification_planner.plan_clarification(request, TOOL_SPECS),
                )
            except Exception as exc:
                # 规划器不可用（无 Key / 超时 / JSON 解析失败）→ 降级 L17 直查。
                planner_failed = True
                langchain_state = {
                    "create_agent": False,
                    "skip_reason": "planner_failed_fallback_to_l17_lookup",
                    "available_tools": [spec.name for spec in DRIVER_TOOL_CATALOG.list_tools()],
                    "message_types": [],
                    "planner_error": str(exc),
                }

            if planner_failed:
                business_fact_result = self._business_facts.lookup(business_fact_need, request)
            else:
                langchain_state = {
                    "create_agent": False,
                    "skip_reason": "lesson19_clarification_path",
                    "available_tools": [spec.name for spec in DRIVER_TOOL_CATALOG.list_tools()],
                    "message_types": [],
                }
                # 工具调用前澄清：后端按 ToolSpec 重算缺失字段，缺参就止步。
                clarification = pre_tool_clarification(request, clarification_plan)
                planned_action = None if clarification else plan_tool_action(clarification_plan)
                # L23: pre_tool_call 卡口——校验参数、脱敏摘要
                if planned_action:
                    # L24：从 DriverToolCatalog 读工具，而不是直接读 TOOL_SPECS
                    spec = DRIVER_TOOL_CATALOG.get_tool(planned_action.tool_name)
                    if spec:
                        hooks.pre_tool_call(planned_action, request, spec)
                
                tool_observation = execute_tool_action(planned_action, request) if planned_action else None
                # L37: 记录工具调用 Trace
                if planned_action:
                    trace_store.add(request.session_id, "tool_call", {
                        "tool_name": planned_action.tool_name,
                        "status": tool_observation.status if tool_observation else "none",
                    })
                # L36: 工具结果安全检查——防止 Prompt Injection 通过工具结果注入模型上下文
                if tool_observation and tool_observation.status == "success":
                    is_safe, _reason = inspect_tool_result(tool_observation.data)
                    if not is_safe:
                        tool_observation = None  # 不安全的结果直接丢弃
                # L20：把原始 ToolObservation（含完整业务数据）压缩成安全 Observation，
                # 只有压缩后的 summary/facts 才能进模型上下文。
                raw_observation = build_observation(tool_observation) if tool_observation else None
                # L23: post_tool_call 卡口——脱敏 + 提示注入检测
                observation = hooks.post_tool_call(raw_observation) if raw_observation else None
                # L21：给 Observation 标记错误分类，供降级层判断
                if observation and observation.status == "error":
                    observation.error_category = tool_observation.data.get("source", "unknown") if tool_observation else "unknown"
                    # L23: on_error 卡口——工具异常归一成降级事件
                    hooks.on_error(
                        tool_name=observation.tool_name,
                        error_category=observation.error_category,
                        message=observation.summary,
                        attempts=getattr(tool_observation, "attempts", 1) if tool_observation else 1,
                    )
                # 工具调用后澄清：列候选型工具（get_coupon_status）查回多张卡券，
                # 结果本身就是候选列表，让用户确认目标，模型不许代选。
                # 注意：post_tool_clarification 仍吃原始 tool_observation（需要看完整候选列表）。
                if not clarification:
                    clarification = post_tool_clarification(planned_action, tool_observation)

                if planned_action and observation:
                    # ToolCallRecord 保留原始 tool_observation 供调试台审计；
                    # 但 business_fact_result 只用压缩后的 observation.facts。
                    tool_calls = [ToolCallRecord(action=planned_action, observation=tool_observation)]
                    business_fact_result = BusinessFactResult(
                        source=tool_observation.data.get("source", "driver_service"),
                        found=observation.status == "success",
                        user_matched=tool_observation.data.get("user_matched", True),
                        summary=observation.summary,
                        facts=observation.facts,  # L20：压缩后的关键 facts，不含敏感字段
                    )
                elif business_fact_result is None:
                    # 模型没识别出可执行工具（tool_name=null）→ 与 L17 found=False 同口径处理。
                    business_fact_result = BusinessFactResult(
                        source="planner",
                        found=False,
                        summary="本轮没有匹配到可执行的司机域实时工具。",
                    )

            if clarification:
                # L32：写入 Session Memory，下一轮用户补参数后 consume_pending_clarification 会恢复原意图
                _required_field = clarification.clarification_field or "coupon_id"
                remember_pending_clarification(
                    session_id=request.session_id,
                    runtime_user_id=request.runtime_user_id,
                    intent=intent_result.intent,
                    required_field=_required_field,
                    current_message_count=message_count,
                )
                # 澄清态：保留确定性追问，本轮不调模型。
                # 工具前澄清时工具未调；工具后澄清时工具已调（tool_calls 有记录），
                # 但回答仍是让用户确认目标的问句，不用 Observation 组织答案。
                answer = build_clarification_answer(clarification)
                answer_path = "clarification_required"
                citations = []
                model_usage = None
                messages = [
                    {"role": "system", "content": "clarification_required"},
                    {"role": "user", "content": request.user_message},
                ]
            elif business_fact_result.found:
                # 事实拿到了：实时事实 + RAG 规则片段融合回答（L17 起保留的设计）。
                fact_block = f"【司机实时业务事实（来自运力后端，可信）】\n{business_fact_result.summary}"
                rag_context = "\n\n".join(
                    f"[{hit.chunk.chunk_id} | score={hit.score}]\n{hit.chunk.text}" for hit in reliable_hits
                ) if reliable_hits else "当前问题没有匹配到卡券规则片段。"
                system_msg = (
                    "你是网约车司机端卡券客服 Agent。本轮拿到了司机的实时业务事实（已由运力后端校验，可信），"
                    "同时也匹配到了一些卡券规则知识片段。回答时：\n"
                    "1. 实时业务事实必须直接引用，不能编造或改数字；\n"
                    "2. 卡券规则片段作为补充背景（如退款流程、资格要求等）；\n"
                    "3. 如果事实和规则有冲突，以实时业务事实为准。"
                )
                user_msg = f"用户原话：{request.user_message}\n\n{fact_block}\n\n【卡券规则知识片段（来自向量检索）】\n{rag_context}\n"
                messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
                model_result = call_chat_model(messages, http_client=self._chat_http_client,
                                               api_key=self._chat_api_key, base_url=self._chat_base_url,
                                               model=self._chat_model_name)
                answer = model_result.answer
                model_usage = model_result.usage
                answer_path = "answer_with_tool_fact"
                citations = []
            else:
                # 事实没拿到：a) RAG 有命中回退 RAG；b) 停 tool_fact_fallback
                if reliable_hits and intent_result.intent != "general_chat":
                    rag_context = "\n\n".join(
                        f"[{hit.chunk.chunk_id} | score={hit.score}]\n{hit.chunk.text}" for hit in reliable_hits
                    )
                    system_msg = (
                        "你是网约车司机端卡券客服 Agent。当前尝试查询司机实时业务事实，但工具未返回数据"
                        f"（原因：{business_fact_result.summary}）。\n"
                        "不过你仍然匹配到了一些卡券规则知识片段。请基于规则片段回答用户问题，不要编造实时数据。"
                    )
                    messages = [{"role": "system", "content": system_msg},
                                {"role": "user", "content": f"用户原话：{request.user_message}\n\n【卡券规则知识片段】\n{rag_context}"}]
                    model_result = call_chat_model(messages, http_client=self._chat_http_client,
                                                   api_key=self._chat_api_key, base_url=self._chat_base_url,
                                                   model=self._chat_model_name)
                    answer = model_result.answer
                    answer_path = "answer_with_citations"
                    citations = [Citation(citation_id=f"C{idx}", source_path=hit.chunk.source_path,
                                          source_title=hit.chunk.title, chunk_id=hit.chunk.chunk_id,
                                          score=hit.score, snippet=hit.chunk.text)
                                 for idx, hit in enumerate(reliable_hits, start=1)]
                    model_usage = model_result.usage
                else:
                    answer = business_fact_result.summary
                    answer_path = "tool_fact_fallback"
                    citations = []
                    model_usage = None
                    messages = [{"role": "system", "content": "fallback"}, {"role": "user", "content": request.user_message}]
        elif reliable_hits and intent_result.intent != "general_chat":
            messages = render_vector_rag_messages(request, intent_result, reliable_hits, rewrite=rewrite)
            model_result = call_chat_model(
                messages,
                http_client=self._chat_http_client,
                api_key=self._chat_api_key,
                base_url=self._chat_base_url,
                model=self._chat_model_name,
            )
            answer = model_result.answer
            answer_path = "answer_with_citations"
            model_usage = model_result.usage
        elif intent_result.intent == "general_chat":
            messages = [
                {"role": "system", "content": "你是网约车司机端卡券客服 Agent。普通寒暄可以直接接住，业务规则结论必须有知识依据。"},
                {"role": "user", "content": f"用户原话：{request.user_message}\n粗意图：{intent_result.intent}\n当前分支：普通客服寒暄，不需要 citations。"},
            ]
            answer = build_general_chat_answer()
            answer_path = "general_chat_without_citations"
            # L14：寒暄分支清空引用，和 answer_path 保持一致（否则会出现「无引用回答」却挂着 citations）。
            citations = []
            model_usage = None
        else:
            messages = render_vector_rag_messages(request, intent_result, reliable_hits, rewrite=rewrite)
            answer = build_fallback_answer()
            answer_path = "clarify_or_handoff"          # ← L11 叫 no_reliable_knowledge_fallback
            model_usage = None

        # # 4. 拼装 Prompt（检索结果注入）
        # messages = render_vector_rag_messages(request, intent_result, hits)

        # # 5. 调用模型，拿回 answer 和 token 成本
        # model_result = call_chat_model(
        #     messages,
        #     http_client=self._chat_http_client,
        #     api_key=self._chat_api_key,
        #     base_url=self._chat_base_url,
        #     model=self._chat_model_name,
        # )

        # L13：top_score 口径是改写后检索的最高分；raw_top_* 只留作改写前后对比。
        # L14：top_score 口径改成「重排后的最终分」——低置信判断和回答依据都看重排结果，
        #      改写后、重排前的分数单独留一个 rewritten_top_score 做链路对照。
        # top_score = rewritten_hits[0].score if rewritten_hits else 0.0
        rewritten_top_score = rewritten_hits[0].score if rewritten_hits else 0.0
        top_score = reranked_hits[0].score if reranked_hits else 0.0
        raw_top_score = raw_hits[0].score if raw_hits else 0.0
        # quality_summary = run_rag_quality_check(embedding_client=self._embedding_client) 
        # L12 修正：质量摘要在后台线程预热，这里只做非阻塞读取，绝不触发重算。
        quality_summary = peek_quality_summary()
        confidence_level = (
            "not_applicable"
            if intent_result.intent == "general_chat"
            else ("low" 
                  if low_confidence 
                  else "high")
        ) # ← L12 加这行

        # 6. 记录 token 成本（保留 L07 的 usage 传递）
        cost_summary = build_cost_summary(messages, answer, model_usage)
        event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
            "cost_summary": cost_summary.model_dump(),
        }
        self._cost_events_by_session.setdefault(request.session_id, []).append(event)

        # reasoning_summary = [
        #     f"识别为{intent_result.intent}意图，命中关键词：{intent_result.matched_keywords}",
        #     f"向量检索召回 {len(hits)} 条卡券规则，token来源为{cost_summary.token_source}",
        #     "向量检索已上线，下一步可优化切片大小或调整相似度阈值",
        # ]
        # —— L13 旧 reasoning_summary（注释保留，不删）——
        # reasoning_summary = [ 
        #     f"识别为 {intent_result.intent} 意图，命中关键词：{intent_result.matched_keywords}",
        #     (
        #         f"原始问题 top 命中 {raw_hits[0].chunk.chunk_id if raw_hits else 'none'}（{raw_top_score:.3f}）；"
        #         f"改写后 top 命中 {rewritten_hits[0].chunk.chunk_id if rewritten_hits else 'none'}（{top_score:.3f}），"
        #         f"低置信线 {LOW_CONFIDENCE_THRESHOLD}。"
        #     ),
        #     f"检索改写：{rewrite.rewritten_query}" if rewrite.applied else "本轮未触发查询改写，原问题直接检索。",
        #     f"判定 confidence_level={confidence_level}，走 {answer_path} 分支。",
        # ] # ← L13 更新：展示改写前后 top 命中对比
        # ↓ L14：在改写对比之外，补一条「候选池 → 重排后 top」的说明。
        #   原因：重排会把顺序换掉，只说改写前后 top 无法判断 reranker 到底起没起作用。
        # ↓ L15：再补一条检索前计划说明（场景路由 + 关键词通道），并展示三路召回各自命中数。
        reasoning_summary = [
            f"识别为 {intent_result.intent} 意图，命中关键词：{intent_result.matched_keywords}",
            f"检索前计划：{plan.reason}",
            (
                f"索引版本 {index.version}（{index.chunk_count} 篇知识），"
                f"检索缓存{'命中，直接复用上次精排结果' if cache_hit else '未命中，执行完整检索'}"
                + ("" if cache_policy["cacheable"] else f"，不可缓存原因：{cache_policy['reason']}")
                + "。"
            ),
            (
                f"原始问题 top 命中 {raw_hits[0].chunk.chunk_id if raw_hits else 'none'}（{raw_top_score:.3f}）；"
                f"改写后、重排前 top 命中 {rewritten_hits[0].chunk.chunk_id if rewritten_hits else 'none'}"
                f"（{rewritten_top_score:.3f}）。"
            ),
            (
                f"向量合并 {len(vector_candidates)} 条 + 关键词通道 {len(keyword_hits)} 条，"
                f"hybrid 融合后候选池 {len(candidates)} 条，经 {rerank_outcome.mode} reranker 精排后 top 命中 "
                f"{reranked_hits[0].chunk.chunk_id if reranked_hits else 'none'}（{top_score:.3f}），"
                f"低置信线 {LOW_CONFIDENCE_THRESHOLD}。"
            ),
            f"检索改写：{rewrite.rewritten_query}" if rewrite.applied else "本轮未触发查询改写，原问题直接检索。",
            f"判定 confidence_level={confidence_level}，走 {answer_path} 分支。",
        ] # ← L15 更新：展示「计划 + 三路召回 + 重排」全链路
        session_state = {
            "agent_version": "driver-coupon-L16-index-cache",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_driver_name,
                "driver_level": request.runtime_driver_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            # "rag": {
            #     "mode": "vector_search",
            #     "score_threshold": 0.28,
            #     "top_k": 2,
            #     "answer_path" : answer_path, # ← L11加这行 
            #     "citation_count" : len (citations), # ← L11加这行
            #     "matched_chunk_ids": [hit.chunk.chunk_id for hit in hits],
            #     "scores": [hit.score for hit in hits],
            # },
                "rag": {
                    # L16：链路名补上 index_cache，调试台一眼能看出本课加了索引版本和检索缓存。
                    # "mode": "query_rewrite_vector_retrieval",
                    # "mode": "query_rewrite_vector_retrieval_rerank",
                    # "mode": "hybrid_rag_rerank",
                    "mode": "hybrid_rag_with_index_cache",
                    "score_threshold": RETRIEVAL_SCORE_THRESHOLD,
                    "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                    # "top_k": TOP_K,
                    "top_k": FINAL_TOP_K,          # L14：回答只吃重排后的 top 2
                    "candidate_k": CANDIDATE_K,     # L14：候选池宽召回条数
                    "final_top_k": FINAL_TOP_K,     # L14：重排后真正喂给 Prompt 的条数
                    # L14：reranker 运行模式，前端和审计都要看得见（lightweight / commercial / commercial_fallback）
                    "rerank_mode": rerank_outcome.mode,
                    "rerank_model": rerank_outcome.model,
                    "rerank_error": rerank_outcome.error,
                    # L16：索引版本 + 检索缓存状态（前端/审计要能看见缓存命中与不可缓存原因）
                    "index": {
                        "version": index.version,
                        "fingerprint": index.fingerprint,
                        "chunk_count": index.chunk_count,
                        "inverted_term_count": len(index.inverted_index),
                        # Task 19：历史索引标记——索引内 historical chunk 数量（方案Y 宽进候选池）。
                        "historical_chunk_count": sum(
                            1
                            for c in index.chunks_by_id.values()
                            if c.effective_status == "historical"
                        ),
                        # Task 9：release 路由信息（灰度司机看 candidate，其余看 stable）。
                        # 开关关闭时为 None（走 L16 全局单索引）。
                        "release": (
                            release_route.model_dump() if release_route is not None else None
                        ),
                    },
                    "cache": {**cache_policy, "cache_hit": cache_hit, "cache_key": cache_key},
                    # L15：检索前计划（场景/主题范围/关键词通道用词/可读原因）
                    "plan": plan.model_dump(),
                    # L13：改写明细 + 改写前后基线对比
                    "rewrite": rewrite.model_dump(),
                    "raw_top_chunk_id": raw_hits[0].chunk.chunk_id if raw_hits else None,
                    "raw_top_score": raw_top_score,
                    "raw_retrieved_count": len(raw_hits),
                    "rewritten_retrieved_count": len(rewritten_hits),
                    "rewritten_top_score": rewritten_top_score,
                    # L15：三路召回各自的命中清单（向量原话+向量改写合并 / 关键词通道）
                    "keyword_chunk_ids": [hit.chunk.chunk_id for hit in keyword_hits],
                    "keyword_retrieved_count": len(keyword_hits),
                    # Task 13：BM25 专路命中清单。
                    "bm25_chunk_ids": [hit.chunk.chunk_id for hit in bm25_hits],
                    "bm25_retrieved_count": len(bm25_hits),
                    # Task 14：RAG-Fusion 子查询清单。
                    "fusion_sub_queries": fusion_sub_queries,
                    "fusion_enabled": settings.RAG_FUSION_ENABLED,
                    # Task 11/19：RBAC 角色与授权视图（TracePanel 展示角色、可检索库/audience、拦截事件）。
                    "rbac": {
                        "runtime_role": request.runtime_role,
                        "rbac_enabled": settings.RBAC_ENABLED,
                        # None 表示开关关闭未过滤（L16 旧行为），空集表示授权后无库可查。
                        "allowed_kb_ids": sorted(allowed_kb_ids) if allowed_kb_ids is not None else None,
                        "allowed_audiences": sorted(allowed_audiences) if allowed_audiences is not None else None,
                        # 最终断言拦截的越权片段（空=本轮无拦截）。
                        "denied_chunk_ids": rbac_denied_chunk_ids,
                    },
                    # L17：司机域实时业务事实（TracePanel 展示本轮工具调用链路）。
                    # 不在 driver_domain_realtime 分支下时 business_fact_result 为 None，整块省略。
                    "business_fact": {
                        "need": business_fact_need.model_dump(),
                        "driver_domain_realtime": driver_domain_realtime,
                        "source": business_fact_result.source if business_fact_result else None,
                        "found": business_fact_result.found if business_fact_result else None,
                        "user_matched": business_fact_result.user_matched if business_fact_result else None,
                        "summary": business_fact_result.summary if business_fact_result else None,
                    } if driver_domain_realtime else None,
                    # L18/L19：Tool Calling 调试区块（候选工具 / 澄清计划 / 实际动作 / Observation）。
                    "tool_calling": {
                        "available_tools": [spec.model_dump() for spec in DRIVER_TOOL_CATALOG.list_tools()],
                        "clarification_plan": clarification_plan.model_dump() if clarification_plan else None,
                        "clarification": clarification.model_dump() if clarification else None,
                        "planned_action": planned_action.model_dump() if planned_action else None,
                        "observation": observation.model_dump() if observation else None,
                        # L20：Observation 压缩治理可观测性——
                        # raw_payload_keys 是工具原始返回的所有字段名，
                        # omitted_fields 是被压缩层丢弃、不进模型的字段。
                        "raw_payload_keys": list(tool_observation.data.keys()) if tool_observation else [],
                        "omitted_fields": observation.omitted_fields if observation else [],
                        # L19 课程形态：工具后澄清单独标记（仅列候选型工具触发）。
                        "post_tool_clarification": (
                            clarification.model_dump()
                            if planned_action and planned_action.tool_name == "get_coupon_status" and clarification
                            else None
                        ),
                        "tool_call_count": len(tool_calls),
                        "langchain": langchain_state,
                    },
                    # L24：MCP-style 目录绑定关系——调试台看"这个工具挂了哪些边界文档"。
                    "mcp_binding": DRIVER_TOOL_CATALOG.binding_summary(
                        planned_action, risk_level
                    ).model_dump(),
                    # L24：MCP 客户端视图——连了哪些 Server、工具池路由表、连接错误。
                    "mcp_client": {
                        "ready": self._mcp_ready,
                        **self._mcp_client.summary(),
                    },
                    "vector_merged_chunk_ids": [hit.chunk.chunk_id for hit in vector_candidates],
                    # L14：候选池 → 重排后 → 最终采用 的三段链路（L15 起候选池是 hybrid 融合结果）
                    "candidate_chunk_ids": [hit.chunk.chunk_id for hit in candidates],
                    "reranked_chunk_ids": [hit.chunk.chunk_id for hit in reranked_hits],
                    "rerank_reasons": {hit.chunk.chunk_id: hit.rerank_reasons for hit in reranked_hits},
                    "score_breakdown": {
                        hit.chunk.chunk_id: {
                            "vector": hit.vector_score,
                            # L15：关键词通道分和召回来源也进 breakdown，双路贡献一眼可查
                            "keyword": hit.keyword_score,
                            # Task 13：BM25 专路分。
                            "bm25": hit.bm25_score,
                            "sources": hit.sources,
                            "matched_keywords": hit.matched_keywords,
                            "rerank": hit.rerank_score,
                            "final": hit.score,
                        }
                        for hit in reranked_hits
                    },
                    "retrieved_count": len(reliable_hits),
                    "citation_count": len(citations),
                    "confidence_level": confidence_level,
                    "top_score": top_score,
                    "answer_path": answer_path,
                    "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
                    "scores": [hit.score for hit in reliable_hits],
                },
            "cost_log": {
                "event_count": len(self._cost_events_by_session[request.session_id]),
                "latest": event,
            },
            # L16 后检索链路有了版本化缓存；L15 的 gap（词表匹配漏同义词/标注依赖）依然成立。
            # 旧 gap（备查）："关键词通道目前是词表匹配，同义词、错别字仍会漏召回；主题路由依赖 md 注释里的 topic 标注，新知识漏标会被场景过滤误伤（当前对空 topic 保守放行）。"
            # L20：Observation 已经能压缩工具结果，控制哪些字段进模型上下文；
            # L21：错误降级能兜住系统不稳，但商品咨询还需要把实时库存价格和稳定知识一起组织成答案。
            "next_gap": "L21 已实现高风险拦截 + 超时/模型不可用降级，但多工具串行、跨会话状态管理和审批流还需要更复杂的编排。",
            # L21 新增：降级状态（是否触发错误分类 + 兜底话术）
            "degradation": degradation.model_dump(),
            # L23 新增：Hooks 治理状态（completion 在 return 前统一调用后再塞回来）
            "hooks": {
                "events": [e.model_dump() for e in hooks.events],
            },
        }
        # 质量摘要只做非阻塞读取：预热线程还没跑完时整段省略，前端 TracePanel 会自动隐藏。
        if quality_summary is not None:
            session_state["rag_quality"] = {
                "case_count": quality_summary.total_cases,
                "passed_cases": quality_summary.passed_cases,
                "average_recall_at_k": quality_summary.average_recall_at_k,
                "average_precision_at_k": quality_summary.average_precision_at_k,
            }

        # L34：把验证过的业务实体写回 Session Memory（下一轮 find_context_coupon 会优先用它）
        update_memory(
            session_id=request.session_id,
            runtime_user_id=request.runtime_user_id,
            intent=intent_result.intent,
            verified_coupon_id=explicit_coupon_id,
            user_message=request.user_message,
        )

        # L23: hooks_on_completion 主链路统一赋值（高风险提前 return 分支已自己赋值）
        if hooks_on_completion is None:
            hooks_on_completion = hooks.on_completion(
                next_action=("ask_clarification" if clarification
                             else (observation.next_action if observation else "fallback_answer")),
                risk_level=risk_level,
                degradation=degradation,
            )

        # 7. 返回模型回复
        # L21：risk_level / degraded / needs_human_approval 是 ChatResponse 新增字段
        # L23：hook_events / hook_completion 是 Hooks 治理卡口记录
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            citations=citations, # ← L11加这行
            cost_summary=cost_summary,
            reasoning_summary=reasoning_summary,
            session_state=session_state,
            tool_calls=tool_calls,  # L18：工具调用记录（调试台观察 Action / Observation）
            clarification=clarification,  # L19：需要补参数时的结构化澄清请求
            risk_level=risk_level,
            needs_human_approval=risk_level == "high",
            degraded=degradation.degraded,
            # L23 新增：Hooks 治理事件和摘要
            hook_events=hooks.events,
            hook_completion=hooks_on_completion,
            # L25 新增：TaskPlanner 路由决策 + 公开轨迹
            route_plan=route_plan,
            planner_trace=planner_trace,
            # L26 新增：高风险资格评估（非高风险请求为 None）
            risk_assessment=risk_assessment,
        )
