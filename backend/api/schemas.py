"""Pydantic 请求响应模型。早期课程从这里建立稳定 API 契约。"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

ReasoningView = Literal["default", "off", "summary", "teaching"]

Intent = Literal[
    "general_chat",
    "promotion_consult",
    "product_consult",
    "order_query",
    "refund_request",
    "complaint",
    "unknown",
    ]

# L15：检索场景（scene）是「检索前计划」层的概念，和 Intent 不是一回事——
# intent 回答「用户想干什么」，scene 回答「该去哪些主题的知识里找」。
# 取值按卡券业务定（refund/usage/promotion/complaint），unknown 保留全主题兜底。
RetrievalScene = Literal["refund", "usage", "promotion", "complaint", "unknown"]

# —— RAG 平台化（Task 2）新增的领域字面量 ——
# Role：请求方角色。课程环境由前端切换器注入，生产由网关注入；
# 后端只信 ChatRequest 顶层的 runtime_role，用户消息文本里自称管理员无效（AC-8）。
Role = Literal["driver", "agent", "admin"]
# DocStatus：文档版本生命周期状态（与 store/schema.sql 的注释口径保持一致）。
# draft 草稿 → submitted 待审核 → approved 审核通过待发布 → published 已发布；archived 归档。
DocStatus = Literal["draft", "submitted", "approved", "published", "archived"]
# ReleaseStatus：candidate 灰度中 / stable 全量稳定 / rolled_back 已回滚（历史留存）。
ReleaseStatus = Literal["candidate", "stable", "rolled_back"]
# Audience：driver 司机可见；internal 仅内部角色，对司机全通道物理不可达（FR-9）。
Audience = Literal["driver", "internal"]

class ChatRequest(BaseModel):
    """调试后台发送给 `/chat` 的最小请求格式"""

    session_id: str = Field(..., description="会话 ID，用于关联用户请求和响应")
    # runtime_* 是系统确认过的事实，不能让用户在聊天里自称会员或冒充身份。
    runtime_user_id: str = Field(..., description="运行时司机ID，用于关联用户请求和响应")
    runtime_driver_name: str| None = Field(default=None, description="运行时司机姓名，用于关联用户请求和响应")
    runtime_driver_phone: str| None = Field(default=None, description="运行时司机手机号，用于关联用户请求和响应")
    runtime_driver_level: str| None = Field(default=None, description="运行时司机等级，用于关联用户请求和响应")
    runtime_risk_level:str|None = Field(default=None, description="调试后台当前选择的风险等级")
    # user_message 才是真正交给模型的用户输入，必须和 runtime_* 分开放。
    user_message: str = Field(..., description="用户输入的问题")
    runtime_context:dict[str,Any]|None =None
    reasoning_view: ReasoningView="default"# 推理视图，默认默认
    debug: bool = True # 是否开启调试模式
    # —— RAG 平台化（Task 2）——
    # 请求角色，默认 driver。旧前端不传该字段时按司机处理（TR-2.1，不产生 422）。
    # 安全口径：只信任这个由前端切换器/生产网关传入的顶层字段，
    # 禁止从 user_message 文本里识别身份（AC-8 伪造提权测试依据）。
    runtime_role: Role = "driver"

class IntentResult(BaseModel):
    """意图识别结果"""

    intent: Intent
    matched_keywords: list[str] = Field(default_factory=list)
    explanation: str

class SourceDocument(BaseModel):
    """一篇 Markdown 知识文档"""
    source_path: str
    title: str
    metadata: dict[str, Any]
    body: str

class KnowledgeSection(BaseModel):
    """Markdown 中的一个 ## 章节"""
    source_path: str
    document_title: str
    section_index: int
    section: str
    chunk_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    effective_status: str = "active"
    text: str
    metadata: dict[str, Any]

class KnowledgeChunk(BaseModel):
    """用于 embedding 和向量召回的知识片段"""
    chunk_id: str
    title: str
    source_path: str
    section: str
    keywords: list[str]
    effective_status: str = "active"
    # L14 新增：topic 来自 markdown 章节注释（如 topic: complaint）。
    # 之前它只活在 KnowledgeSection.metadata 里，没有落到 chunk 上，reranker 读不到这个业务信号。
    # 注意用 str 而不是 Literal：本项目 md 里的 topic 取值（complaint / refund_request / product_consult）
    # 和课程版电商取值不同，写死 Literal 会直接校验失败。
    topic: str = ""
    text: str
    # —— RAG 平台化（Task 2）新增字段 ——
    # 全部带默认值：L10–L16 旧链路构造 KnowledgeChunk 时不传这些字段也不破（向后兼容）。
    kb_id: str = ""  # 所属知识库 ID（多知识库隔离，FR-8）
    audience: str = "driver"  # driver / internal（internal 对司机物理不可达，FR-9）
    policy_codes: list[str] = Field(default_factory=list)  # 政策编号，如 P-REF-001（BM25 专路）
    product_names: list[str] = Field(default_factory=list)  # 商品名，如 周末日卡（BM25 专路）
    effective_from: str | None = None  # 生效起点 ISO（展示/预留，本期不按它硬过滤，见 FR-7）
    effective_to: str | None = None  # 生效终点 ISO（同上，historical 走方案Y 软降权）

class VectorRecord(BaseModel):
    """向量库中的一条记录：知识片段加对应 embedding"""
    chunk: KnowledgeChunk
    embedding: list[float]

class KnowledgeHit(BaseModel):
    """向量检索命中的知识片段和相似度分数"""
    chunk: KnowledgeChunk
    score: float = Field(ge=0, le=1)
    # L14：把「向量初始分」和「重排后的最终分」拆开看。
    # score 仍然是本片段最终用于排序和回答的分数；vector_score 保留初召回信号，
    # rerank_score / rerank_reasons 记录 reranker 做了什么，方便调试台和审计。
    vector_score: float = Field(default=0.0, ge=0, le=1)
    rerank_score: float | None = Field(default=None, ge=0, le=1)
    rerank_reasons: list[str] = Field(default_factory=list)
    # L15：hybrid 召回补三个字段——
    # keyword_score 记录关键词通道的独立打分（和 vector_score 并列，任一路强都能让片段进池）；
    # sources 记录本片段由哪几路召回（vector / keyword）；
    # matched_keywords 记录关键词通道命中的具体词，调试台可直接展示。
    keyword_score: float = Field(default=0.0, ge=0, le=1)
    # Task 13：BM25 硬关键词专路打分（政策编号/商品名精确命中通道）。
    bm25_score: float = Field(default=0.0, ge=0, le=1)
    sources: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)

class CostSummary(BaseModel):
    """成本总结"""

    prompt_tokens: int
    answer_tokens: int
    total_tokens: int
    # token_source 让你知道本轮成本来自平台 usage，还是本地估算兜底。
    token_source: Literal["model_usage", "local_estimate"]
    usage_details: dict[str, Any] = Field(default_factory=dict)
    estimated_input_cost_cny: float
    estimated_output_cost_cny: float
    estimated_total_cost_cny: float
    context_chars: int
    pricing_note: str

class TokenUsage(BaseModel):
    """token使用情况"""

    prompt_tokens: int
    answer_tokens: int
    total_tokens: int
    details: dict[str, Any] = Field(default_factory=dict)

class ChatModelResult(BaseModel):
    """模型模型结果"""

    answer: str
    usage: TokenUsage | None = None




class Citation(BaseModel):
    """引用知识片段"""
    citation_id: str
    source_path: str
    source_title: str
    chunk_id: str
    score: float
    snippet: str

class QueryRewrite(BaseModel):
    """L13 查询改写结果：原话只用于对话，rewritten_query 只用于检索。"""
    original_query: str
    rewritten_query: str
    applied: bool
    added_terms: list[str] = Field(default_factory=list)
    reason: str

class RetrievalPlan(BaseModel):
    """L15 检索前计划：场景路由、主题范围和关键词通道用词。

    复用 L13 的 QueryRewrite 结果（rewritten_query 直接来自改写层），
    本层只新增「场景 + 主题范围 + 关键词补词」三个检索信号，不改写用户原话。
    """
    original_query: str
    rewritten_query: str
    scene: RetrievalScene
    # 项目 chunk.topic 是自由 str 而不是 Literal（见 KnowledgeChunk 上的注释），
    # 所以 allowed_topics 用 list[str]，写死 Literal 会校验失败。
    allowed_topics: list[str]
    keyword_terms: list[str] = Field(default_factory=list)
    reason: str


class KnowledgeIndex(BaseModel):
    """L16 内存中的知识索引快照。

    version 是全部 chunk 序列化后的 sha256 指纹（截 12 位）——知识库改一个字就变，
    它是所有缓存条目的「身份证」：缓存 key 带上它，旧知识产生的缓存才不会冒充新知识。
    """
    version: str
    fingerprint: str
    chunk_count: int
    chunks_by_id: dict[str, KnowledgeChunk]
    # 倒排表 keyword -> chunk_ids。本课只构建并暴露给调试台（inverted_term_count），
    # 关键词通道仍走词表撞词，和课程版保持一致。
    inverted_index: dict[str, list[str]]


class RetrievalCacheEntry(BaseModel):
    """L16 检索缓存条目：只存检索+精排后的命中结果，永不存最终自然语言回答。

    用户拍板（L16）：缓存放 rerank 之后——商业 rerank 是确定性打分（无采样），
    对同一 query 是确定性变换，缓存连 rerank API 调用一起省。
    """
    index_version: str
    plan: RetrievalPlan
    hits: list[KnowledgeHit]
    # 缓存命中时用这些字段还原 rerank_outcome 的调试信息，session_state 口径不变
    rerank_mode: str
    rerank_model: str | None = None
    rerank_error: str | None = None

class RagQualityCase(BaseModel):
    """固定问题集里的一条用例。

    Task 15 扩展：
    - forbidden_chunk_ids：命中即算错（越权/草稿内容）；
    - runtime_role：以指定角色跑检索（多角色评测）；
    - require_historical_demoted：historical chunk 应被软降权（方案Y 回归）。
    """
    case_id: str
    question: str
    expected_chunk_ids: list[str]
    forbidden_chunk_ids: list[str] = Field(default_factory=list)
    must_fallback: bool = False
    runtime_role: str = "driver"
    require_historical_demoted: bool = False

class RagQualityCaseResult(BaseModel):
    """一条用例的评测结果。

    Task 15 扩展：
    - forbidden_hit：是否命中 forbidden_chunk_ids 中的越权片段；
    - runtime_role：评测时使用的角色。
    """
    case_id: str
    retrieved_chunk_ids: list[str]
    expected_chunk_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    fallback: bool
    forbidden_hit: bool = False
    runtime_role: str = "driver"
    passed: bool

class RagQualitySummary(BaseModel):
    """整批固定问题集的汇总。"""
    total_cases: int
    passed_cases: int
    average_recall_at_k: float
    average_precision_at_k: float
    results: list[RagQualityCaseResult]

class ChatResponse(BaseModel):
    """调试后台返回给 `/chat` 的最小响应格式"""
    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    cost_summary: CostSummary
    reasoning_summary: list[str]
    session_state: dict[str, Any]
    # L11 新增：引用证据列表，C1/C2 编号和正文 [C1] 一一对应
    citations: list[Citation] = Field(default_factory=list)
    # L18 新增：工具调用记录（调试台观察 Action / Observation，RAG 分支为空列表）
    tool_calls: list["ToolCallRecord"] = Field(default_factory=list)
    # L19 新增：需要用户补参数/确认候选时的结构化澄清请求，不需要澄清时为 None。
    clarification: "ClarificationRequest | None" = None
    # L21 新增：风险等级、降级标记、转人工审批
    risk_level: str = "low"
    needs_human_approval: bool = False
    degraded: bool = False
    # L23 新增：Hooks 治理事件（统一卡口记录）
    hook_events: list["HookEvent"] = Field(default_factory=list)
    hook_completion: "HookCompletion | None" = None
    # L25 新增：TaskPlanner 路由决策 + 公开轨迹
    route_plan: "RoutePlan | None" = None
    planner_trace: "PlannerTrace | None" = None
    # L26 新增：高风险动作资格评估（非高风险请求为 None）
    risk_assessment: "HighRiskAssessment | None" = None

# ===========================================================================
# L17 实时业务事实接入：BusinessFactNeed / BusinessFactResult
# 小哲 L17 的双链路分叉契约——先判断"需要实时事实吗"，再决定走实时链还是 RAG 链。
# ===========================================================================

class BusinessFactNeed(BaseModel):
    """工具规划层输出：描述本轮需要什么实时业务事实。

    司机域取值：
    - driver_activity: 活动完成度（我在XX活动达标了吗？还差多少？）
    - driver_order: 今日/本周单量
    - driver_qualification: 活动资格（我能不能参加XX活动？）
    - driver_refund: 退款进度/状态（我的退款到账了吗、最近退款状态）
    - driver_coupon: 卡券状态（还有几张未使用、已核销记录）
    - driver_verification: 核销记录（这张卡券的核销记录）
    - driver_income: 收入统计（本月收入、卡券占比）
    - unknown: 不需要实时事实，走 RAG 链
    """
    kind: Literal[
        "driver_activity", "driver_order", "driver_qualification",
        "driver_refund", "driver_coupon", "driver_verification", "driver_income",
        "unknown",
    ]
    requires_realtime: bool
    # 已抽取参数（planning.py 用正则/关键词识别后填入）
    activity_id: str | None = None
    time_range: str | None = None  # 今日/本周/本月 等
    # 退款/卡券类的额外参数
    order_id: str | None = None  # 订单号（退款进度查询用）
    coupon_id: str | None = None  # 卡券号（核销/状态查询用）
    reason: str = ""


class BusinessFactResult(BaseModel):
    """实时业务事实服务 lookup 的返回值。

    found=False 三态（小哲 L17 的三道安全闸直接映射）：
    - 缺参数 → summary 解释缺什么
    - user_matched=False → 越权，事实不属于当前司机
    - source=driver_service / source=self → driver_service 查不到

    found=True → summary 是可直接回答用户的摘要（三态：达标 / 未达标 / 数据缺失）
    """
    source: str
    found: bool
    user_matched: bool = True
    summary: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# L18 Tool Calling 契约：模型看到工具名/描述/参数 schema，真正执行仍由后端控制。
# ToolSpec 是「工具说明书」形状；TOOL_SPECS 注册表放在 tools/contracts.py。
# ===========================================================================

class ToolSpec(BaseModel):
    """LLM 可见的工具说明书：名字、描述、必填参数、参数示例。

    L24 增强：工具定义要自带能力边界，而不是让 Hooks / Planner 各自硬编码。
      - read_only: 是否只读（写工具必须经过受控工作流）
      - risk_level: 工具自身的风险等级，供 pre_tool_call 和 TaskPlanner 直接读取
      - boundary: 工具不能做什么（能力边界，随工具一起推给模型）
    """
    name: str
    description: str
    required: list[str] = Field(default_factory=list)
    parameters_schema: dict[str, str] = Field(default_factory=dict)
    domain: str = "unknown"
    read_only: bool = True
    risk_level: "RiskLevel" = "low"
    boundary: list[str] = Field(default_factory=list)
    # L24：MCP-style 绑定——每个工具挂 resource_uri 指向边界文档，prompt_id 指向口径模板。
    resource_uris: list[str] = Field(default_factory=list)
    prompt_ids: list[str] = Field(default_factory=list)


class ToolAction(BaseModel):
    """LLM 产出的工具调用动作：工具名 + 参数 + 调用原因。"""
    tool_name: str
    arguments: dict[str, Any]
    reason: str = ""


class ToolObservation(BaseModel):
    """后端受控执行层返回的工具观察结果（success / error + 摘要 + 原始数据）。"""
    tool_name: str
    status: Literal["success", "error"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# L20 ToolResult 与 Observation：工具返回的原始 ToolObservation（含完整业务数据）
# 不能直接进模型上下文，必须压缩成安全 Observation（只保留摘要/关键 facts/被丢弃字段/下一步动作）。
# ===========================================================================

# L26 新增：高风险资格判断后，把请求路由到受控工作流，而不是直接转人工。
NextAction = Literal["answer_user", "fallback_answer", "ask_clarification", "route_to_controlled_workflow", "transfer_to_human"]


ErrorCategory = Literal[
    "none",
    "timeout",
    "validation_error",
    "not_found",
    "forbidden",
    "business_error",
    "model_unavailable",
    "system_error",
    "high_risk_write_blocked",
]


class Observation(BaseModel):
    """L20：压缩后的安全 Observation，可进入模型回答上下文。

    - summary：一句话摘要，直接可进 prompt
    - facts：结构化关键事实键值对，模型可引用
    - omitted_fields：显式记录哪些原始字段被丢弃（便于调试和审计）
    - next_action：answer_user / fallback_answer / ask_clarification
    - error_category：L21 新增，错误分类供降级层判断
    """
    tool_name: str
    status: Literal["success", "error"]
    summary: str
    facts: dict[str, Any] = Field(default_factory=dict)
    omitted_fields: list[str] = Field(default_factory=list)
    next_action: NextAction = "answer_user"
    # L21 新增：错误分类，工具层 / Observation 压缩层共同标记
    error_category: ErrorCategory = "none"


class ToolCallRecord(BaseModel):
    """调试台观察格式：Action / Observation 配对，方便前端展示工具调用链路。"""
    action: ToolAction
    observation: ToolObservation


# ===========================================================================
# L19 工具澄清契约：LLM 只产出 ClarificationPlan 草案，
# 后端按 ToolSpec 重算必填字段，缺参数时先反问用户而不是硬调工具。
# ===========================================================================

class ClarificationCandidate(BaseModel):
    """候选值（如司机正在参加的活动），只能来自后端校验过的运行时上下文。"""
    value: str
    label: str
    hint: str = ""


class ClarificationRequest(BaseModel):
    """需要用户补充参数时的结构化澄清请求（可带候选让用户选，模型不许代选）。"""
    clarification_field: str
    message: str
    candidates: list[ClarificationCandidate] = Field(default_factory=list)


class ClarificationPlan(BaseModel):
    """LLM 输出的澄清规划草案。

    模型负责理解话术、抽取字段、提出澄清问题；后端仍用 ToolSpec 和
    当前司机身份做硬校验，missing_required 以后端重算结果为准。
    """
    intent: Intent
    tool_name: str | None = None
    known_arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    confidence: float = 0.7
    reason: str = ""
    source: Literal["llm"] = "llm"
    model_name: str | None = None


# —— L14 reranker 数据结构 ——
# 和 KnowledgeHit 不同，这两个不是「每次请求都要序列化给前端」的契约，
# 只在后端 reranker 内部传递，所以用 dataclass 而不是 pydantic BaseModel。
@dataclass(frozen=True)
class RerankConfig:
    """商业 reranker 调用配置。"""
    api_key: str
    base_url: str
    model: str
    instruction: str

@dataclass(frozen=True)
class RerankOutcome:
    """reranker 返回的排序结果和运行模式（lightweight / commercial / commercial_fallback）。"""
    hits: list[KnowledgeHit]
    mode: str
    model: str | None = None
    error: str | None = None


# ===========================================================================
# RAG 平台化（Task 2 起）：生命周期/发布/灰度/审计的 API 契约
# 这些模型服务于 /admin/* 管理接口与运行时 ReleaseView 展示，与聊天旧契约并存。
# ===========================================================================

class KnowledgeBaseView(BaseModel):
    """知识库视图。"""
    kb_id: str
    name: str
    domain: str
    allowed_roles: list[str] = Field(default_factory=list)


class DocVersionView(BaseModel):
    """文档版本视图（列表/详情用，content_md 仅详情接口按需返回）。"""
    version_id: str
    doc_id: str
    version_no: int
    title: str
    source_path: str
    kb_id: str
    status: str
    audience: str
    content_hash: str
    change_note: str = ""
    created_by: str = "system"
    created_at: str
    status_updated_at: str
    effective_from: str | None = None
    effective_to: str | None = None
    content_md: str | None = None


class ReviewActionRequest(BaseModel):
    """审核动作请求：submit/approve/reject；reject 时 comment 必填（API 层 422 校验）。"""
    action: Literal["submit", "approve", "reject"]
    comment: str = ""
    actor: str = "admin"

    @model_validator(mode="after")
    def _reject_requires_comment(self) -> "ReviewActionRequest":
        # AC-2：驳回必须填理由，pydantic 校验失败 FastAPI 直接返回 422。
        if self.action == "reject" and not self.comment.strip():
            raise ValueError("驳回（reject）必须填写 comment 理由")
        return self


class ReleaseCreateRequest(BaseModel):
    """发起发布：勾选一组已 approved 的 version_id 组成不可变快照。"""
    version_ids: list[str]
    label: str = ""
    # True=candidate 灰度版；False=直接 stable（仍必须通过质量门禁）。
    as_candidate: bool = True
    gray_whitelist: list[str] = Field(default_factory=list)
    gray_percent: int = 0


class ReleaseView(BaseModel):
    """发布版本视图。"""
    release_id: str
    release_no: int
    label: str
    status: str
    gray_whitelist: list[str] = Field(default_factory=list)
    gray_percent: int = 0
    created_by: str = "admin"
    created_at: str
    activated_at: str | None = None
    rolled_back_at: str | None = None
    rollback_reason: str = ""
    version_ids: list[str] = Field(default_factory=list)


class GrayConfigRequest(BaseModel):
    """灰度配置：白名单 + 百分比（0-100）。"""
    gray_whitelist: list[str] | None = None
    gray_percent: int | None = Field(default=None, ge=0, le=100)


class RollbackRequest(BaseModel):
    """一键回滚请求：reason 必填（API 层 422 校验，AC-5 留痕）。"""
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_non_blank_reason(self):
        if not self.reason.strip():
            raise ValueError("回滚理由不能为空白")
        return self


class GrayRoute(BaseModel):
    """单次请求的灰度路由解析结果（调试台展示 candidate/stable 身份）。"""
    release_id: str
    release_no: int
    channel: Literal["candidate", "stable"]
    # whitelist=白名单命中；percent=哈希落桶；stable=未命中灰度，走稳定版；none=该域无 candidate
    reason: Literal["whitelist", "percent", "stable", "none"]
    gray_percent: int = 0


class AuditEventView(BaseModel):
    """审计事件视图（7 类事件统一出口）。"""
    id: int
    actor: str
    action: str
    target_type: str = ""
    target_id: str = ""
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    comment: str = ""
    created_at: str


class SyncResult(BaseModel):
    """md → 草稿同步结果。"""
    scanned: int
    unchanged: int
    changed: int
    created_version_ids: list[str] = Field(default_factory=list)
    details: list[dict[str, Any]] = Field(default_factory=list)


class GateFailure(BaseModel):
    """质量门禁的单条失败明细（Task 16 充实）。"""
    case_id: str
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


class GateResult(BaseModel):
    """发布质量门禁结果。Task 6 阶段为 placeholder（恒过），Task 16 接真实评测。"""
    passed: bool
    mode: str  # placeholder / quality_eval
    checked_versions: list[str] = Field(default_factory=list)
    failures: list[GateFailure] = Field(default_factory=list)
    ran_at: str


class ReleasePublishResult(BaseModel):
    """发布动作结果：新版本视图 + 门禁结果。"""
    release: ReleaseView
    gate: GateResult



# ===========================================================================
# L21 错误降级契约：DegradationState / ToolExecutionError / ModelServiceError
# ===========================================================================

class DegradationState(BaseModel):
    """L21：降级状态，记录本轮是否触发错误分类 + 兜底话术。"""
    degraded: bool
    error_category: ErrorCategory = "none"
    retry_count: int = 0
    fallback_message: str | None = None


class ToolExecutionError(Exception):
    """工具执行错误，带 ErrorCategory 分类。"""
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category



# ===========================================================================
# L23 Hooks 治理契约
# ===========================================================================

HookType = Literal["pre_tool_call", "post_tool_call", "on_error", "on_completion"]
# L24：工具自身的风险等级也需要 medium（如投诉、涉及用户权益的只读查询）。
RiskLevel = Literal["low", "medium", "high"]


class HookEvent(BaseModel):
    """单次 Hook 卡口事件。"""
    hook_type: HookType
    target_name: str
    action: str
    result: str
    reason: str
    safe_summary: dict[str, Any]
    redacted: bool = False
    degraded: bool = False


class HookCompletion(BaseModel):
    """一轮请求结束时的 Hooks 治理摘要。"""
    hook_count: int
    tool_count: int
    touched_tools: list[str]
    redacted_count: int
    degraded_count: int
    risk_hit_count: int
    safe_summary: dict[str, Any]


# ===========================================================================
# L24 司机域 MCP 目录契约：多团队协作时，工具边界有标准位置。
# 每个工具挂 resource_uri 指向独立的边界文档，prompt_id 指向统一的口径模板。
# ===========================================================================

class MCPResource(BaseModel):
    """MCP-style 资源文档：工具能力边界的独立说明。"""
    uri: str
    title: str
    content: str


class MCPPrompt(BaseModel):
    """MCP-style Prompt 模板：统一工具返回的口径。"""
    prompt_id: str
    title: str
    content: str


class MCPBindingSummary(BaseModel):
    """工具 + 资源 + Prompt 的绑定关系，供调试台和审计观察。"""
    tool_source: str
    selected_tool: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    boundary: str = ""


# ===========================================================================
# L25 TaskPlanner 路由契约：RoutePlan 是可校验的入口决策，不是模型长计划。
# 它告诉后续链路要不要 RAG、要不要工具、允许哪些工具、风险有多高，
# 以及是否应该交给后续受控售后路径。
# ===========================================================================

RouteSource = Literal["rules", "classifier", "rules_fallback"]


class RoutePlan(BaseModel):
    """执行器前的结构化路由信号。"""
    intent: str
    needs_rag: bool
    needs_business_tools: bool
    rag_query: str
    confidence: float
    source: RouteSource
    intents: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    knowledge_domains: list[str] = Field(default_factory=list)
    has_realtime_fact: bool = False
    risk_level: RiskLevel = "low"
    requires_workflow: bool = False
    fallback_policy: str = "default"


class ToolCandidate(BaseModel):
    """工具候选摘要：只暴露工具名、域、是否允许在轻路径调用、风险等级。"""
    name: str
    domain: str
    allowed_in_light_path: bool
    risk_level: RiskLevel
    reason: str


class PlannerTrace(BaseModel):
    """公开 Planner 轨迹：让路由原因可观察，但不暴露隐藏推理。"""
    source: RouteSource
    rule_confidence: float
    candidate_tools: list[ToolCandidate]
    constrained_required_tools: list[str]
    public_reason: str


# ===========================================================================
# L26 高风险动作边界契约：HighRiskAssessment 只做资格判断，不执行写动作。
# 司机域：退款/补偿类诉求必须先判断资格，明确列出被禁止的写动作。
# ===========================================================================

class HighRiskAssessment(BaseModel):
    """高风险售后动作的资格评估结果。"""
    action_type: Literal["refund", "compensation", "unknown"]
    # 司机域用 coupon_id（卡券号）或 order_id（退款订单号）
    coupon_id: str | None = None
    order_id: str | None = None
    eligibility_status: Literal[
        "eligible_for_application", "not_eligible", "needs_clarification", "blocked"
    ]
    risk_level: RiskLevel
    needs_human_approval: bool
    evidence_checklist: list[str]
    policy_basis: list[Citation]
    reasons: list[str]
    # 明确列出本轮被禁止的写动作，调试台和审计直接可见。
    blocked_write_actions: list[str] = Field(default_factory=list)
