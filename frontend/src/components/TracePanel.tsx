import type { ChatResponse } from "../types/api";

interface TracePanelProps {
  response: ChatResponse | null;
}

export function TracePanel({ response }: TracePanelProps) {
  if (!response) {
    return (
      <div className="panel trace-panel empty">
        <div className="empty-state">
          <div className="icon">🔍</div>
          <p>发送请求后，这里会显示检索和成本详情</p>
        </div>
      </div>
    );
  }

  const rag = response.session_state?.rag;
  const quality = response.session_state?.rag_quality;
  const cost = response.cost_summary;

  return (
    <div className="panel trace-panel">
      <div className="panel-header">
        <span>运行详情</span>
        <small>{response.session_state?.agent_version}</small>
      </div>

      <div className="trace-section">
        <div className="trace-title">意图识别</div>
        <div className="trace-content">
          <div className="trace-row">
            <span className="label">意图:</span>
            <span className="value">{response.intent}</span>
          </div>
          <div className="trace-row">
            <span className="label">命中关键词:</span>
            <span className="value">
              {response.intent_result.matched_keywords.length > 0
                ? response.intent_result.matched_keywords.join(", ")
                : "无"}
            </span>
          </div>
          <div className="trace-row">
            <span className="label">说明:</span>
            <span className="value">{response.intent_result.explanation}</span>
          </div>
        </div>
      </div>

      {rag && (
        <>
          {/* Task 19：发布版本 + 灰度身份 + RBAC 角色视图 + 权限拦截事件 */}
          {(rag.index?.release || rag.rbac) && (
            <div className="trace-section">
              <div className="trace-title">发布与权限</div>
              <div className="trace-content">
                {rag.index?.release ? (
                  <>
                    <div className="trace-row">
                      <span className="label">知识版本:</span>
                      <span className="value">
                        {rag.index.release.release_id}（#{rag.index.release.release_no}）
                      </span>
                    </div>
                    <div className="trace-row">
                      <span className="label">灰度身份:</span>
                      <span className={`value channel-badge ${rag.index.release.channel}`}>
                        {rag.index.release.channel} ·{" "}
                        {rag.index.release.reason === "whitelist"
                          ? "白名单命中"
                          : rag.index.release.reason === "percent"
                            ? `哈希落桶（${rag.index.release.gray_percent}%）`
                            : rag.index.release.reason === "none"
                              ? "无 candidate，走 stable"
                              : "未命中灰度，走 stable"}
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="trace-row">
                    <span className="label">知识版本:</span>
                    <span className="value">全局单索引（生命周期开关未开启）</span>
                  </div>
                )}
                {typeof rag.index?.historical_chunk_count === "number" && (
                  <div className="trace-row">
                    <span className="label">历史索引标记:</span>
                    <span className="value">
                      含 {rag.index.historical_chunk_count} 条 historical chunk（方案Y 软降权，不硬过滤）
                    </span>
                  </div>
                )}
                {rag.rbac && (
                  <>
                    <div className="trace-row">
                      <span className="label">运行时角色:</span>
                      <span className="value">
                        {rag.rbac.runtime_role}
                        {rag.rbac.rbac_enabled ? "" : "（RBAC 未开启，未过滤）"}
                      </span>
                    </div>
                    {rag.rbac.allowed_kb_ids !== null && (
                      <div className="trace-row">
                        <span className="label">可检索知识库:</span>
                        <span className="value">
                          {rag.rbac.allowed_kb_ids.length > 0
                            ? rag.rbac.allowed_kb_ids.join(", ")
                            : "（空）"}
                        </span>
                      </div>
                    )}
                    {rag.rbac.allowed_audiences !== null && (
                      <div className="trace-row">
                        <span className="label">可见 audience:</span>
                        <span className="value">{rag.rbac.allowed_audiences.join(", ")}</span>
                      </div>
                    )}
                    {rag.rbac.denied_chunk_ids.length > 0 && (
                      <div className="trace-row">
                        <span className="label permission-denied">权限拦截:</span>
                        <span className="value permission-denied">
                          最终断言拦截 {rag.rbac.denied_chunk_ids.length} 条越权片段（已写审计
                          rbac.denied）: {rag.rbac.denied_chunk_ids.join(", ")}
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          <div className="trace-section">
            <div className="trace-title">RAG 检索</div>
            <div className="trace-content">
              <div className="trace-row">
                <span className="label">模式:</span>
                <span className="value">{rag.mode}</span>
              </div>
              {/* Task 19：检索通道贡献——向量合并/关键词/BM25/fusion 子查询 */}
              {(rag.bm25_chunk_ids?.length || rag.fusion_sub_queries?.length || rag.vector_merged_chunk_ids?.length) && (
                <>
                  <div className="trace-row">
                    <span className="label">向量通道:</span>
                    <span className="value">
                      {rag.vector_merged_chunk_ids?.length ?? 0} 条：
                      {rag.vector_merged_chunk_ids?.join(", ") || "无"}
                    </span>
                  </div>
                  <div className="trace-row">
                    <span className="label">BM25 专路:</span>
                    <span className="value">
                      {rag.bm25_retrieved_count ?? 0} 条：
                      {rag.bm25_chunk_ids?.join(", ") || "无"}
                    </span>
                  </div>
                  {rag.fusion_sub_queries && rag.fusion_sub_queries.length > 0 && (
                    <div className="trace-row">
                      <span className="label">Fusion 子查询:</span>
                      <span className="value">
                        <ol className="fusion-list">
                          {rag.fusion_sub_queries.map((q, i) => (
                            <li key={i}>{q}</li>
                          ))}
                        </ol>
                      </span>
                    </div>
                  )}
                  {typeof rag.fusion_enabled === "boolean" && (
                    <div className="trace-row">
                      <span className="label">Fusion 开关:</span>
                      <span className="value">{rag.fusion_enabled ? "开启" : "关闭"}</span>
                    </div>
                  )}
                </>
              )}
              {rag.rewrite && (
              <>
                <div className="trace-row">
                  <span className="label">原始问题:</span>
                  <span className="value">{rag.rewrite.original_query}</span>
                </div>
                <div className="trace-row">
                  <span className="label">检索改写:</span>
                  <span className="value">
                    {rag.rewrite.applied
                      ? rag.rewrite.rewritten_query
                      : "未触发（原问题直接检索）"}
                  </span>
                </div>
                {rag.rewrite.applied && (
                  <>
                    <div className="trace-row">
                      <span className="label">补充词:</span>
                      <span className="value">
                        {rag.rewrite.added_terms.join("、") || "无（仅口语归一化）"}
                      </span>
                    </div>
                    <div className="trace-row">
                      <span className="label">初始 Top:</span>
                      <span className="value">
                        {rag.raw_top_chunk_id ?? "无"}（{rag.raw_top_score?.toFixed(3) ?? "—"}）
                      </span>
                    </div>
                    <div className="trace-row">
                      <span className="label">改写原因:</span>
                      <span className="value">{rag.rewrite.reason}</span>
                    </div>
                  </>
                )}
              </>
            )}
            {/* L15 新增：检索前计划（场景路由 + 主题范围 + 关键词通道） */}
            {rag.plan && (
              <>
                <div className="trace-row">
                  <span className="label">检索场景:</span>
                  <span className="value">
                    {rag.plan.scene}（主题：{rag.plan.allowed_topics.join(", ")}）
                  </span>
                </div>
                <div className="trace-row">
                  <span className="label">计划原因:</span>
                  <span className="value">{rag.plan.reason}</span>
                </div>
                {/* L16 新增：索引版本 + 检索缓存状态 */}
                {rag.index && (
                  <div className="trace-row">
                    <span className="label">索引版本:</span>
                    <span className="value">
                      {rag.index.version}（{rag.index.chunk_count} 篇 / 倒排 {rag.index.inverted_term_count} 词）
                    </span>
                  </div>
                )}
                {rag.cache && (
                  <div className="trace-row">
                    <span className="label">检索缓存:</span>
                    <span className="value">
                      {rag.cache.cacheable
                        ? rag.cache.cache_hit
                          ? "命中，复用上次精排结果"
                          : "未命中，本轮结果已写入缓存"
                        : `不可缓存（${rag.cache.reason}）`}
                    </span>
                  </div>
                )}
                {rag.realtime_gap && (
                  <div className="trace-row">
                    <span className="label">实时缺口:</span>
                    <span className="value">实时状态问题，知识库不作答，等待业务工具接入</span>
                  </div>
                )}
                <div className="trace-row">
                  <span className="label">关键词通道:</span>
                  <span className="value">
                    {rag.keyword_retrieved_count ?? 0} 条：
                    {rag.keyword_chunk_ids?.join(", ") || "无"}
                  </span>
                </div>
              </>
            )}
            <div className="trace-row">
              <span className="label">置信度:</span>
              <span className="value">{rag.confidence_level}</span>
            </div>
            <div className="trace-row">
              <span className="label">最高分:</span>
              <span className="value">
                {rag.top_score.toFixed(3)}（改写后，低置信线 {rag.low_confidence_threshold}）
              </span>
            </div>
            <div className="trace-row">
              <span className="label">回答路径:</span>
              <span className={`value path-badge ${rag.answer_path}`}>
                {rag.answer_path}
              </span>
            </div>
            <div className="trace-row">
              <span className="label">命中 chunk:</span>
              <span className="value">{rag.matched_chunk_ids.join(", ") || "无"}</span>
            </div>
            <div className="trace-row">
              <span className="label">候选(原→改/采用):</span>
              <span className="value">
                {rag.raw_retrieved_count} → {rag.rewritten_retrieved_count ?? rag.raw_retrieved_count} / {rag.retrieved_count}
              </span>
            </div>
            <div className="trace-row">
              <span className="label">相似度:</span>
              <span className="value">
                {Array.isArray(rag.scores)
                  ? rag.scores.map((s) => s.toFixed(3)).join(", ")
                  : Object.entries(rag.scores)
                      .map(([id, score]) => `${id}: ${score.toFixed(3)}`)
                      .join(", ")}
              </span>
            </div>
            {/* L14 新增：reranker 精排链路，方便判断重排到底改变了什么 */}
            {rag.rerank_mode && (
              <>
                <div className="trace-row">
                  <span className="label">重排模式:</span>
                  <span className="value">
                    {rag.rerank_mode}
                    {rag.rerank_model ? `（${rag.rerank_model}）` : ""}
                  </span>
                </div>
                {rag.rerank_error && (
                  <div className="trace-row">
                    <span className="label">重排回退原因:</span>
                    <span className="value">{rag.rerank_error}</span>
                  </div>
                )}
                <div className="trace-row">
                  <span className="label">候选片段:</span>
                  <span className="value">
                    {rag.candidate_chunk_ids?.length ?? 0} 条：
                    {rag.candidate_chunk_ids?.join(", ") || "无"}
                  </span>
                </div>
                <div className="trace-row">
                  <span className="label">重排后:</span>
                  <span className="value">{rag.reranked_chunk_ids?.join(", ") || "无"}</span>
                </div>
                <div className="trace-row">
                  <span className="label">分数(向量/词/BM25→重排):</span>
                  <span className="value">
                    {rag.score_breakdown
                      ? Object.entries(rag.score_breakdown)
                          .map(
                            ([id, item]) =>
                              `${id}: ${item.vector.toFixed(3)}/${(item.keyword ?? 0).toFixed(3)}/${(item.bm25 ?? 0).toFixed(3)} → ${item.final.toFixed(3)}${item.sources?.length ? `（${item.sources.join("+")}）` : ""}`,
                          )
                          .join("；")
                      : "无"}
                  </span>
                </div>
                <div className="trace-row">
                  <span className="label">重排原因:</span>
                  <span className="value">
                    {rag.rerank_reasons && Object.keys(rag.rerank_reasons).length > 0
                      ? Object.entries(rag.rerank_reasons)
                          .map(([id, reasons]) => `${id}: ${reasons.join("、")}`)
                          .join("；")
                      : "无"}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
        </>
      )}

      {quality && (
        <div className="trace-section">
          <div className="trace-title">RAG 质量评测</div>
          <div className="trace-content">
            <div className="trace-row">
              <span className="label">通过用例:</span>
              <span className="value">
                {quality.passed_cases} / {quality.case_count}
              </span>
            </div>
            <div className="trace-row">
              <span className="label">平均 recall@k:</span>
              <span className="value">{quality.average_recall_at_k.toFixed(3)}</span>
            </div>
            <div className="trace-row">
              <span className="label">平均 precision@k:</span>
              <span className="value">{quality.average_precision_at_k.toFixed(3)}</span>
            </div>
          </div>
        </div>
      )}

      {cost && (
        <div className="trace-section">
          <div className="trace-title">成本观察</div>
          <div className="trace-content">
            <div className="trace-row">
              <span className="label">Token 来源:</span>
              <span className="value">{cost.token_source}</span>
            </div>
            <div className="trace-row">
              <span className="label">Prompt tokens:</span>
              <span className="value">{cost.prompt_tokens}</span>
            </div>
            <div className="trace-row">
              <span className="label">Answer tokens:</span>
              <span className="value">{cost.answer_tokens}</span>
            </div>
            <div className="trace-row">
              <span className="label">总成本:</span>
              <span className="value">¥{cost.estimated_total_cost_cny.toFixed(4)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
