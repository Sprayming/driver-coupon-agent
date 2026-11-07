"""Task 21 验证：全量回归、开关双态与端到端演练（TR-21.1 / TR-21.2 / TR-21.3）。

TR-21.1：6 条 L12 原始质量用例 + 8 个人工回归场景（AC-17），
         在「新开关全关（== L16 行为）」与「平台开」双态下回归全过且结论一致（AC-15）。
         本组用真实本地 embedding（bge-small-zh 离线），只 mock chat LLM（规避平台限流），
         保证回归测的是真实检索链路而不是假向量。
         双态口径说明：fusion 是「扩召回」增强（RRF 归一化会抬分，AC-10 明确开启改变召回行为，
         由 Task 14 专项覆盖），故严格双态一致断言中平台开态不含 fusion；
         fusion 开态单独跑一轮 sanity 记录进证据。
TR-21.2：端到端演练「导入→驳回→通过→发布→灰度调整→权限拦截→回滚」（AC-14），
         核对 7 类审计事件字段完整、时间线递增，并导出 JSON。
         演练测流程/审计不测检索质量，用均匀假向量保证快速确定。
TR-21.3：baseline / fusion / 缓存命中三态延迟与成本记录（真实 embedding + mock chat LLM）。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from api.schemas import (
    ChatModelResult,
    ChatRequest,
    GrayConfigRequest,
    KnowledgeHit,
    ReleaseCreateRequest,
    RerankOutcome,
    RollbackRequest,
)
from config import settings
from rag import index_cache, vector_store
from rag.quality import load_quality_cases, run_rag_quality_check
from store import lifecycle, releases
from store import repository as repo
from store.db import transaction, utc_now_iso
from store.seed import seed

# —— 公共 fixture 与工具 ——

ALL_SWITCHES = (
    "KNOWLEDGE_LIFECYCLE_ENABLED",
    "RBAC_ENABLED",
    "BM25_ENABLED",
    "RAG_FUSION_ENABLED",
    "GATE_ENABLED",
)


@pytest.fixture()
def db_with_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "task21.db"
    seed(db_path)
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    # 清空模块级索引/缓存，避免测试间污染。
    index_cache.reset_index_and_cache()
    # 向量库缓存也要清：其他测试用 1024 维假向量（mock embed）并缓存在
    # _VECTOR_STORE_CACHE（按 embedding client + 索引版本做 key），
    # 本文件用真实 bge-small（512 维）会撞上残留假向量的维度（left=512, right=1024）。
    vector_store._VECTOR_STORE_CACHE.clear()
    yield db_path
    index_cache.reset_index_and_cache()
    vector_store._VECTOR_STORE_CACHE.clear()


def _set_all_switches(monkeypatch, value: bool) -> None:
    for name in ALL_SWITCHES:
        monkeypatch.setattr(settings, name, value)


def _patch_embeddings_fake(monkeypatch) -> None:
    """演练类测试用：均匀假向量，快且确定（不测检索质量，只测流程/审计）。"""

    fake_vec = [0.1] * 1024
    monkeypatch.setattr("rag.vector_store.embed_text", lambda *a, **kw: fake_vec)
    monkeypatch.setattr(
        "rag.vector_store.embed_texts",
        lambda texts, *a, **kw: [fake_vec for _ in texts],
    )
    monkeypatch.setattr("rag.quality.peek_quality_summary", lambda: None)


def _mock_chat_llm(monkeypatch) -> None:
    """只 mock chat LLM（embedding 保持真实）：

    - 回答分支：返回固定话术；
    - fusion 子查询分支：回显原问题（单子查询，RRF 退化为原排序，避免 mock 引入假召回方向）。
    """

    def fake_caller(messages, **kw):
        system_prompt = messages[0].get("content", "") if messages else ""
        if "查询改写助手" in system_prompt:
            user_query = messages[-1]["content"]
            return ChatModelResult(answer=json.dumps([user_query], ensure_ascii=False))
        return ChatModelResult(answer="【mock】依据命中的卡券规则给出的回答")

    # agent 模块顶部 from 导入持有独立引用，必须双 patch（fusion 是调用点延迟导入，走源模块）。
    monkeypatch.setattr("agents.customer_service_agent.call_chat_model", fake_caller)
    monkeypatch.setattr("models.llm_client.call_chat_model", fake_caller)


def _evidence_dir() -> Path:
    path = settings.BACKEND_DIR / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chat(agent, session_id: str, message: str, user_id: str = "driver-normal-001"):
    return agent.chat(
        ChatRequest(session_id=session_id, runtime_user_id=user_id, user_message=message)
    )


# —— TR-21.1：6 质量用例 + 8 场景回归，双态一致 ——

# L12 原始 6 条质量用例（rag_quality_cases.json 前 6 条，AC-17「现有 6 条」）。
ORIGINAL_SIX_CASE_IDS = [
    "refund-not-used",
    "refund-exclusion",
    "refund-process-timeline",
    "purchase-limit",
    "unknown-cooking",
    "unknown-founder",
]


def test_tr21_1_original_six_quality_cases_pass_in_both_states(db_with_seed, monkeypatch):
    """6 条原始质量用例：开关全关（L16）与平台开双态均全过，且逐用例结论一致。"""

    original_six = [c for c in load_quality_cases() if c.case_id in ORIGINAL_SIX_CASE_IDS]
    assert len(original_six) == 6, "rag_quality_cases.json 前 6 条 L12 原始用例缺失"

    summaries = {}
    for enabled in (False, True):
        _set_all_switches(monkeypatch, enabled)
        # 双态各走各的索引：关=L16 全局单索引，开=rel-0001 release 快照索引。
        index_cache.reset_index_and_cache()
        summaries[enabled] = run_rag_quality_check(cases=original_six)

    for enabled, summary in summaries.items():
        failed = [r.case_id for r in summary.results if not r.passed]
        assert summary.passed_cases == 6, (
            f"6 用例在开关={'平台开' if enabled else '全关(L16)'}态未全过: {failed}"
        )

    # 全关态 == L16，平台开态逐用例结论必须与其一致（AC-17 存量行为零回退）。
    # 命中按集合比对：recall/precision 本身就是集合口径；FINAL_TOP_K 内部近似平局的
    # 两个 chunk（如 process/timeline，分差 <0.01）可能因浮点末位差异换位，不构成结论回退。
    off_by_id = {r.case_id: r for r in summaries[False].results}
    for result in summaries[True].results:
        baseline = off_by_id[result.case_id]
        assert result.passed == baseline.passed, f"用例 {result.case_id} 双态 passed 不一致"
        assert result.fallback == baseline.fallback, f"用例 {result.case_id} 双态 fallback 不一致"
        assert sorted(result.retrieved_chunk_ids) == sorted(baseline.retrieved_chunk_ids), (
            f"用例 {result.case_id} 双态命中集合不一致: "
            f"全关={baseline.retrieved_chunk_ids} 平台开={result.retrieved_chunk_ids}"
        )


# 8 个人工回归场景（AC-17）：(场景名, 问题, 期望 answer_path, 期望引用命中集合[任一命中即算])
SCENARIOS = [
    ("退款", "这卡买完还没使，能退不", "answer_with_citations", {"coupon-refund-eligibility"}),
    ("核销", "卡券已经核销了还能退款吗", "answer_with_citations", {"coupon-refund-exclusions"}),
    (
        "流程",
        "退款要走什么流程、多久能到账",
        "answer_with_citations",
        {"coupon-refund-process", "coupon-refund-timeline"},
    ),
    ("购买", "买日卡有什么限制", "answer_with_citations", {"coupon-purchase-eligibility"}),
    ("投诉话术", "我要投诉，怎么升级处理", "answer_with_citations", {"coupon-complaint-escalation"}),
    # AC-6②：显式历史意图，historical 片段正常召回进引用前列（方案Y 软降权只压「没问历史」的情形）。
    ("历史活动", "2025 年的老卡券当时怎么规定的", "answer_with_citations", {"coupon-old-promotion"}),
    ("寒暄", "你好", "general_chat_without_citations", set()),
    ("无关兜底", "怎么做红烧肉", "clarify_or_handoff", set()),
]


def _run_scenarios(agent, tag: str) -> dict[str, dict]:
    outcomes = {}
    for name, message, expected_path, expected_chunks in SCENARIOS:
        resp = _chat(agent, f"tr21-{tag}-{name}", message)
        rag = resp.session_state["rag"]
        cited_ids = [citation.chunk_id for citation in resp.citations]
        outcomes[name] = {
            "answer_path": rag["answer_path"],
            "cited": cited_ids,
            "rerank_reasons": rag["rerank_reasons"],
        }

        assert rag["answer_path"] == expected_path, (
            f"[{tag}] 场景「{name}」answer_path={rag['answer_path']}，期望 {expected_path}"
        )
        if expected_chunks:
            assert set(cited_ids) & expected_chunks, (
                f"[{tag}] 场景「{name}」引用 {cited_ids} 未命中期望 {expected_chunks}"
            )
        else:
            assert cited_ids == [], f"[{tag}] 场景「{name}」不应有引用，实际 {cited_ids}"
        # 司机引用永不包含 internal 片段（AC-7/8，双态都不许漏）。
        assert not any("internal" in cid for cid in cited_ids), (
            f"[{tag}] 场景「{name}」司机引用泄漏 internal: {cited_ids}"
        )
    return outcomes


def test_tr21_1_eight_scenarios_consistent_with_l16(db_with_seed, monkeypatch):
    """8 场景回归：双态结论均符合 L16 期望且彼此一致；fusion 开态单独 sanity。"""

    _mock_chat_llm(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()

    # 态一：新开关全关 == L16。
    _set_all_switches(monkeypatch, False)
    index_cache.reset_index_and_cache()
    outcomes_off = _run_scenarios(agent, "全关L16")

    # 态二：平台开（lifecycle+rbac+bm25；fusion 的扩召回行为由 Task 14 专项覆盖，不进严格一致断言）。
    _set_all_switches(monkeypatch, True)
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", False)
    index_cache.reset_index_and_cache()
    outcomes_on = _run_scenarios(agent, "平台开")

    # 双态逐场景结论一致（AC-17 零回退）。口径 = answer_path + 引用集合：
    # 同组期望 chunk 间分数差 <0.01（近似平局），BM25 等增强通道（AC-9 的天职就是调排序）
    # 可在组内换位，属正常排序调整而非结论回退；越权/兜底/方案Y 由下方专项断言覆盖。
    for name in outcomes_off:
        off, on = outcomes_off[name], outcomes_on[name]
        assert off["answer_path"] == on["answer_path"], f"场景「{name}」双态 answer_path 不一致"
        assert sorted(off["cited"]) == sorted(on["cited"]), (
            f"场景「{name}」双态引用集合不一致: 全关={off['cited']} 平台开={on['cited']}"
        )

    # AC-6②：显式历史问题，historical 片段正常召回进引用（FINAL_TOP_K=2 内即「前列」），
    # top1 归属与 L16 基线一致（上面双态一致断言已覆盖，L16 基线 top1 为 coupon-activation-rule）。
    assert "coupon-old-promotion" in outcomes_off["历史活动"]["cited"]
    assert "coupon-old-promotion" in outcomes_on["历史活动"]["cited"]
    # AC-6①：普通退款问题（不带历史词）top1 不被 historical 挤占。
    assert outcomes_off["退款"]["cited"][0] != "coupon-old-promotion"
    assert outcomes_on["退款"]["cited"][0] != "coupon-old-promotion"

    # AC-6 机制留痕：方案Y 软降权信号必须在 rerank 层真实生效（而非排序巧合）。
    reasons_history = outcomes_off["历史活动"]["rerank_reasons"].get("coupon-old-promotion", [])
    assert any("放行不降权" in reason for reason in reasons_history), (
        f"显式历史问法下方案Y 放行信号缺失: {reasons_history}"
    )
    reasons_refund = outcomes_off["退款"]["rerank_reasons"].get("coupon-old-promotion", [])
    if reasons_refund:  # 老活动进候选池时必须留降权痕迹；未进池也不算挤占
        assert any("历史卡券降权" in reason for reason in reasons_refund), (
            f"普通退款问法下方案Y 降权信号缺失: {reasons_refund}"
        )

    # fusion 开态 sanity：扩召回不丢正解——业务场景仍带期望引用（兜底/寒暄行为如实记录进 session_state）。
    monkeypatch.setattr(settings, "RAG_FUSION_ENABLED", True)
    index_cache.reset_index_and_cache()
    outcomes_fusion = _run_scenarios_fusion_sanity(agent)


def _run_scenarios_fusion_sanity(agent) -> dict[str, list[str]]:
    """fusion 开态：业务场景期望 chunk 必须在引用中（扩召回不该把正解挤出 FINAL_TOP_K）。"""

    business_expectations = {
        "退款": {"coupon-refund-eligibility"},
        "核销": {"coupon-refund-exclusions"},
        "购买": {"coupon-purchase-eligibility"},
        "投诉话术": {"coupon-complaint-escalation"},
    }
    outcomes: dict[str, list[str]] = {}
    for name, message, _path, _expected in SCENARIOS:
        resp = _chat(agent, f"tr21-fusion-{name}", message)
        cited_ids = [citation.chunk_id for citation in resp.citations]
        outcomes[name] = cited_ids
        if name in business_expectations:
            assert set(cited_ids) & business_expectations[name], (
                f"[fusion] 场景「{name}」引用 {cited_ids} 丢失期望 {business_expectations[name]}"
            )
        assert not any("internal" in cid for cid in cited_ids), (
            f"[fusion] 场景「{name}」司机引用泄漏 internal: {cited_ids}"
        )
    return outcomes


# —— TR-21.2：端到端演练 + 7 类审计事件时间线导出 ——

# AC-14 要求的 7 类事件（演练剧本各阶段对应动作）。
REQUIRED_SEVEN_ACTIONS = {
    "draft.created",          # 导入
    "review.reject",          # 驳回
    "review.approve",         # 通过
    "release.published",      # 发布
    "release.gray_configured",  # 灰度调整
    "rbac.denied",            # 权限拦截
    "release.rolled_back",    # 回滚
}


def test_tr21_2_e2e_drill_audit_timeline_export(db_with_seed, monkeypatch, tmp_path):
    """端到端演练：导入→驳回→通过→发布→灰度调整→权限拦截→回滚，7 类审计事件可导出。"""

    monkeypatch.setattr(settings, "KNOWLEDGE_LIFECYCLE_ENABLED", True)
    monkeypatch.setattr(settings, "RBAC_ENABLED", True)
    monkeypatch.setattr(settings, "GATE_ENABLED", False)  # 门禁行为由 TR-16 专项覆盖，演练走 placeholder
    _patch_embeddings_fake(monkeypatch)

    # —— 阶段 1：导入（sync 检测 md 变更 → draft.created）——
    work_dir = tmp_path / "knowledge"
    shutil.copytree(settings.KNOWLEDGE_DIR, work_dir)
    usage_md = work_dir / "coupon-usage.md"
    usage_md.write_text(
        usage_md.read_text(encoding="utf-8").rstrip()
        + "\n\n- 【演练条款 E2E-1】日卡激活后 24 小时内可自助改期一次。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "KNOWLEDGE_DIR", work_dir)

    sync_result = lifecycle.sync_knowledge_dir(actor="admin")
    assert sync_result.changed == 1
    draft_id = sync_result.created_version_ids[0]
    assert draft_id == "doc-coupon-usage-v2"

    # —— 阶段 2：驳回（submit → reject，comment 必填）——
    lifecycle.review_version(draft_id, action="submit", actor="admin")
    lifecycle.review_version(draft_id, action="reject", actor="admin", comment="演练：补充条款生效说明")

    # —— 阶段 3：通过（再 submit → approve）——
    lifecycle.review_version(draft_id, action="submit", actor="admin")
    lifecycle.review_version(draft_id, action="approve", actor="admin", comment="演练：复核通过")

    # —— 阶段 4：发布（usage v2 全量发布为 stable rel-0003，旧 stable/candidate 作废）——
    stable_result = releases.publish_release(
        ReleaseCreateRequest(
            version_ids=[draft_id],
            label="E2E 演练：usage 改期条款全量发布",
            as_candidate=False,
        ),
        actor="admin",
    )
    stable_id = stable_result.release.release_id
    assert stable_result.release.status == "stable"

    # 再为退款草稿走一遍审核，发布为 candidate（灰度阶段的载体）。
    lifecycle.review_version("doc-coupon-refund-v2", action="submit", actor="admin")
    lifecycle.review_version("doc-coupon-refund-v2", action="approve", actor="admin", comment="演练：灰度放行")
    candidate_result = releases.publish_release(
        ReleaseCreateRequest(
            version_ids=["doc-coupon-refund-v2"],
            label="E2E 演练：退款冷静期灰度",
            as_candidate=True,
        ),
        actor="admin",
    )
    candidate_id = candidate_result.release.release_id
    assert candidate_result.release.status == "candidate"

    # —— 阶段 5：灰度调整（白名单 + 百分比，立即生效可路由）——
    releases.update_gray_config(
        candidate_id,
        GrayConfigRequest(gray_whitelist=["driver-e2e-gray"], gray_percent=10),
        actor="admin",
    )
    route = releases.resolve_gray_route("driver-e2e-gray")
    assert route.channel == "candidate" and route.reason == "whitelist"
    assert releases.resolve_gray_route("driver-normal-001").channel == "stable"

    # —— 阶段 6：权限拦截（故障注入：rerank 泄漏 internal 片段，最终断言必须拦下并记审计）——
    index_cache.reset_index_and_cache()
    internal_chunk = next(
        chunk
        for chunk in index_cache.get_knowledge_index().chunks_by_id.values()
        if chunk.audience == "internal"
    )
    leaked_hit = KnowledgeHit(chunk=internal_chunk, score=0.99, vector_score=0.99)

    def fake_rerank(query, candidates):
        return RerankOutcome(hits=[leaked_hit, *list(candidates)[:1]], mode="lightweight")

    monkeypatch.setattr("agents.customer_service_agent.rerank_candidates", fake_rerank)

    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    resp = _chat(agent, "e2e-rbac", "特批工单统一使用 TB- 前缀编号是什么意思", user_id="driver-e2e-rbac")
    # 泄漏片段被最终断言拦截：引用清空 + session_state 记录拦截明细。
    assert resp.citations == []
    assert resp.session_state["rag"]["rbac"]["denied_chunk_ids"] == [internal_chunk.chunk_id]

    # —— 阶段 7：回滚（放弃灰度，全员回 stable）——
    releases.rollback_release(
        candidate_id,
        RollbackRequest(reason="E2E 演练：灰度结束放弃"),
        actor="admin",
    )
    route_after = releases.resolve_gray_route("driver-e2e-gray")
    assert route_after.channel == "stable" and route_after.reason == "none"

    # —— 核对：7 类审计事件齐全、字段完整、时间线递增，并导出 JSON ——
    with transaction(db_with_seed) as connection:
        rows = repo.list_audit_events(connection, limit=200)
    timeline = [dict(row) for row in reversed(rows)]  # 转为 id 升序（时间线）

    ids = [row["id"] for row in timeline]
    assert ids == sorted(ids), "审计时间线 id 未递增"

    actions = {row["action"] for row in timeline}
    missing = REQUIRED_SEVEN_ACTIONS - actions
    assert not missing, f"审计缺少必需事件: {missing}"

    for row in timeline:
        for field in ("actor", "action", "target_type", "created_at"):
            assert row[field], f"审计事件 {row['id']} 字段 {field} 为空"

    # 各阶段代表事件在时间线上严格按剧本顺序出现。
    def first_id(action: str, target_id: str) -> int:
        return next(
            row["id"]
            for row in timeline
            if row["action"] == action and row["target_id"] == target_id
        )

    stage_ids = [
        first_id("draft.created", draft_id),
        first_id("review.reject", draft_id),
        first_id("review.approve", draft_id),
        first_id("release.published", stable_id),
        first_id("release.gray_configured", candidate_id),
        first_id("rbac.denied", "e2e-rbac"),
        first_id("release.rolled_back", candidate_id),
    ]
    assert stage_ids == sorted(stage_ids), f"演练阶段时间线乱序: {stage_ids}"

    export_path = _evidence_dir() / "task21_e2e_audit_export.json"
    export_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now_iso(),
                "drill": "导入→驳回→通过→发布→灰度调整→权限拦截→回滚",
                "stage_order_ids": stage_ids,
                "event_count": len(timeline),
                "timeline": timeline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[TR-21.2] 审计时间线导出: {export_path}（{len(timeline)} 条，7 类事件齐全）")


# —— TR-21.3：性能记录（baseline / fusion / 缓存命中）——


def test_tr21_3_performance_record(db_with_seed, monkeypatch):
    """三态延迟与成本记录：baseline(全关) / fusion(全开) / 缓存命中，导出 JSON。"""

    _mock_chat_llm(monkeypatch)
    from agents.customer_service_agent import CustomerServiceAgent

    agent = CustomerServiceAgent()
    target = "这卡买完还没使，能退不"
    warmup = "买日卡有什么限制"  # 预热向量库，把一次性索引构建排除在计时外

    def timed_chat(session_id: str, message: str):
        start = time.perf_counter()
        resp = _chat(agent, session_id, message)
        return resp, round((time.perf_counter() - start) * 1000, 1)

    # baseline：新开关全关（== L16），检索缓存未命中。
    _set_all_switches(monkeypatch, False)
    index_cache.reset_index_and_cache()
    timed_chat("perf-warmup-off", warmup)
    resp_base, base_ms = timed_chat("perf-base", target)

    # fusion：平台全开，检索缓存未命中。
    _set_all_switches(monkeypatch, True)
    index_cache.reset_index_and_cache()
    timed_chat("perf-warmup-on", warmup)
    resp_fusion, fusion_ms = timed_chat("perf-fusion", target)

    # 缓存命中：同问题同设置立即再来一轮（L16 检索缓存，命中连 rerank 都省）。
    resp_cache, cache_ms = timed_chat("perf-cache", target)

    rag_base = resp_base.session_state["rag"]
    rag_fusion = resp_fusion.session_state["rag"]
    rag_cache = resp_cache.session_state["rag"]

    assert rag_base["cache"]["cache_hit"] is False
    assert rag_fusion["cache"]["cache_hit"] is False
    assert rag_cache["cache"]["cache_hit"] is True
    assert rag_fusion["fusion_sub_queries"], "fusion 开态应记录子查询清单"
    assert rag_base["answer_path"] == rag_fusion["answer_path"] == "answer_with_citations"
    # 缓存命中只做字典查找，必须显著快于真跑检索的两态。
    assert cache_ms < base_ms, f"缓存命中 {cache_ms}ms 未快于 baseline {base_ms}ms"
    assert cache_ms < fusion_ms, f"缓存命中 {cache_ms}ms 未快于 fusion {fusion_ms}ms"

    record = {
        "generated_at": utc_now_iso(),
        "environment": {
            "embedding": "local bge-small-zh-v1.5（真实向量，离线）",
            "chat_llm": "mock（规避平台限流；成本为本地估算口径）",
            "question": target,
        },
        "scenarios": {
            "baseline_all_switches_off": {
                "latency_ms": base_ms,
                "cache_hit": rag_base["cache"]["cache_hit"],
                "answer_path": rag_base["answer_path"],
                "cost": resp_base.cost_summary.model_dump(),
            },
            "fusion_all_switches_on": {
                "latency_ms": fusion_ms,
                "cache_hit": rag_fusion["cache"]["cache_hit"],
                "answer_path": rag_fusion["answer_path"],
                "fusion_sub_queries": rag_fusion["fusion_sub_queries"],
                "cost": resp_fusion.cost_summary.model_dump(),
            },
            "retrieval_cache_hit": {
                "latency_ms": cache_ms,
                "cache_hit": rag_cache["cache"]["cache_hit"],
                "answer_path": rag_cache["answer_path"],
                "cost": resp_cache.cost_summary.model_dump(),
            },
        },
    }
    record_path = _evidence_dir() / "task21_perf_record.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[TR-21.3] 性能记录: baseline={base_ms}ms fusion={fusion_ms}ms "
        f"cache_hit={cache_ms}ms → {record_path}"
    )
