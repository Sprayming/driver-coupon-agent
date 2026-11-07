export interface ChatRequest {
  session_id: string;
  runtime_user_id: string;
  // Task 11：RBAC 角色只信顶层字段，消息体自称无效。
  runtime_role?: RuntimeRole;
  runtime_driver_name?: string;
  runtime_driver_phone?: string;
  runtime_driver_level?: string;
  runtime_risk_level?: string;
  user_message: string;
  reasoning_view?: string;
  debug?: boolean;
  runtime_context?: Record<string, unknown>;
}

export interface Citation {
  citation_id: string;
  source_title: string;
  source_path: string;
  chunk_id: string;
  score: number;
  snippet: string;
}

export interface CostSummary {
  prompt_tokens: number;
  answer_tokens: number;
  total_tokens: number;
  token_source: string;
  estimated_input_cost_cny: number;
  estimated_output_cost_cny: number;
  estimated_total_cost_cny: number;
  context_chars: number;
  pricing_note: string;
}

export interface IntentResult {
  intent: string;
  matched_keywords: string[];
  explanation: string;
}

export interface SessionState {
  agent_version: string;
  message_count: number;
  runtime_context: {
    user_id: string;
    nickname?: string;
    driver_level?: string;
    risk_level?: string;
    page_context?: Record<string, unknown>;
  };
  rag: {
    mode: string;
    score_threshold: number;
    low_confidence_threshold: number;
    top_k: number;
    raw_retrieved_count: number;
    rewritten_retrieved_count?: number;
    retrieved_count: number;
    citation_count: number;
    confidence_level: string;
    top_score: number;
    answer_path: string;
    matched_chunk_ids: string[];
    scores: Record<string, number> | number[];
    // L13 查询改写
    rewrite?: {
      original_query: string;
      rewritten_query: string;
      applied: boolean;
      added_terms: string[];
      reason: string;
    };
    raw_top_chunk_id?: string | null;
    raw_top_score?: number;
    rewritten_top_score?: number;
    // L14 reranker 精排
    candidate_k?: number;
    final_top_k?: number;
    rerank_mode?: string;
    rerank_model?: string | null;
    rerank_error?: string | null;
    candidate_chunk_ids?: string[];
    reranked_chunk_ids?: string[];
    rerank_reasons?: Record<string, string[]>;
    score_breakdown?: Record<
      string,
      {
        vector: number;
        rerank: number | null;
        final: number;
        // L15 hybrid 召回：关键词通道分、召回来源、命中词
        keyword?: number;
        // Task 13：BM25 硬关键词专路分
        bm25?: number;
        sources?: string[];
        matched_keywords?: string[];
      }
    >;
    // L15 检索前计划 + 关键词通道
    plan?: {
      original_query: string;
      rewritten_query: string;
      scene: string;
      allowed_topics: string[];
      keyword_terms: string[];
      reason: string;
    };
    keyword_chunk_ids?: string[];
    keyword_retrieved_count?: number;
    vector_merged_chunk_ids?: string[];
    // Task 13：BM25 硬关键词专路
    bm25_chunk_ids?: string[];
    bm25_retrieved_count?: number;
    // Task 14：RAG-Fusion 多查询 + RRF
    fusion_sub_queries?: string[];
    fusion_enabled?: boolean;
    // Task 9：release 索引版本 + Task 11 RBAC 过滤信息
    release_id?: string;
    // Task 19：角色与授权视图（后端开关关闭时 allowed_* 为 null，denied 为本轮拦截明细）
    rbac?: {
      runtime_role: RuntimeRole;
      rbac_enabled: boolean;
      allowed_kb_ids: string[] | null;
      allowed_audiences: string[] | null;
      denied_chunk_ids: string[];
    };
    // L16 索引版本 + 检索缓存状态 + 实时缺口
    index?: {
      version: string;
      fingerprint: string;
      chunk_count: number;
      inverted_term_count: number;
      // Task 19：历史索引标记（方案Y：historical chunk 宽进候选池）
      historical_chunk_count?: number;
      // Task 9：release 路由（开关关闭时为 null，走 L16 全局单索引）
      release?: GrayRoute | null;
    };
    cache?: {
      cacheable: boolean;
      scope: string;
      reason: string;
      cache_hit: boolean;
      cache_key: string;
    };
    realtime_gap?: boolean;
  };
  rag_quality?: {
    case_count: number;
    passed_cases: number;
    average_recall_at_k: number;
    average_precision_at_k: number;
  };
  cost_log: {
    event_count: number;
    latest: Record<string, unknown>;
  };
  next_gap: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  intent: string;
  intent_result: IntentResult;
  citations: Citation[];
  cost_summary: CostSummary;
  reasoning_summary: string[];
  session_state: SessionState;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  answer_path?: string;
  cost_summary?: CostSummary;
  session_state?: SessionState;
  latency_ms?: number;
}

export interface DemoUser {
  profile: {
    userId: string;
    nickname: string;
    driverLevel?: string;
    riskLevel?: string;
  };
}

// ===========================================================================
// RAG 平台管理契约（Task 17）：与后端 api/schemas.py 管理模型一一对应。
// 旧字段全部保留，仅追加，不做删除性修改（TR-17.1）。
// ===========================================================================

export type RuntimeRole = "driver" | "agent" | "admin";
export type DocStatus = "draft" | "submitted" | "approved" | "published" | "archived";
export type ReleaseStatus = "candidate" | "stable" | "rolled_back";

/** 灰度路由解析结果（后端 GrayRoute，Task 7）。 */
export interface GrayRoute {
  release_id: string;
  release_no: number;
  channel: "candidate" | "stable";
  /** whitelist=白名单命中；percent=哈希落桶；stable=未命中灰度；none=该域无 candidate */
  reason: "whitelist" | "percent" | "stable" | "none";
  gray_percent: number;
}

/** 文档版本视图（后端 DocVersionView，content_md 仅详情接口返回）。 */
export interface DocVersionView {
  version_id: string;
  doc_id: string;
  version_no: number;
  title: string;
  source_path: string;
  kb_id: string;
  status: DocStatus;
  audience: "driver" | "internal";
  content_hash: string;
  change_note: string;
  created_by: string;
  created_at: string;
  status_updated_at: string;
  effective_from: string | null;
  effective_to: string | null;
  content_md?: string | null;
}

/** 与线上版本的行级差异（GET /admin/versions/{id}/diff）。 */
export interface VersionDiff {
  version_id: string;
  baseline_version_id: string | null;
  added_lines: string[];
  removed_lines: string[];
}

/** 发布版本视图（后端 ReleaseView）。 */
export interface ReleaseView {
  release_id: string;
  release_no: number;
  label: string;
  status: ReleaseStatus;
  gray_whitelist: string[];
  gray_percent: number;
  created_by: string;
  created_at: string;
  activated_at: string | null;
  rolled_back_at: string | null;
  rollback_reason: string;
  version_ids: string[];
}

/** 审计事件（后端 AuditEventView，7 类事件统一出口）。 */
export interface AuditEventView {
  id: number;
  actor: string;
  action: string;
  target_type: string;
  target_id: string;
  before_state: Record<string, unknown> | null;
  after_state: Record<string, unknown> | null;
  comment: string;
  created_at: string;
}

/** md → 草稿同步结果（POST /admin/kb/sync）。 */
export interface SyncResult {
  scanned: number;
  unchanged: number;
  changed: number;
  created_version_ids: string[];
  details: Array<Record<string, unknown>>;
}

/** 质量门禁失败明细（Task 16）。 */
export interface GateFailure {
  case_id: string;
  reason: string;
  detail: Record<string, unknown>;
}

/** 发布质量门禁结果。 */
export interface GateResult {
  passed: boolean;
  mode: string;
  checked_versions: string[];
  failures: GateFailure[];
  ran_at: string;
}

/** 发布动作结果：新 release + 门禁结果。 */
export interface ReleasePublishResult {
  release: ReleaseView;
  gate: GateResult;
}

// —— 管理操作请求体 ——

export interface ReviewActionRequest {
  action: "submit" | "approve" | "reject";
  /** reject 时必填（后端 422 校验） */
  comment?: string;
  actor?: string;
}

export interface ReleaseCreateRequest {
  version_ids: string[];
  label?: string;
  /** true=candidate 灰度版；false=直接 stable */
  as_candidate?: boolean;
  gray_whitelist?: string[];
  gray_percent?: number;
}

export interface GrayConfigRequest {
  gray_whitelist?: string[] | null;
  gray_percent?: number | null;
}

export interface RollbackRequest {
  reason: string;
}
