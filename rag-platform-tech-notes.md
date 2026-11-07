# RAG 平台技术要点与代码路径速查

> 用途：课前口头复盘 / 面试讲解的速查笔记。每条技术点标注代码实现路径（文件 → 函数）。
> 最后更新：2026-10-18（发布预热方案 A+B 落地后）

---

## 1. 总览：两侧一交点

平台 = **供给侧（新增）+ 消费侧（L16 原样）+ 唯一交点：SQLite 已发布快照**。

```
编辑源头          供给侧（SQLite 生命周期）                消费侧（问答）
md 文件 ──sync──▶ doc_versions(draft→submitted→approved)
                  └─publish─▶ release + release_items（不可变快照）
                                                     │
                              唯一交点：快照内容      ▼
                              resolve_gray_route(司机) → 选 release
                              → 按快照建索引/向量 → L16 检索全链路 → 回答
```

- 供给侧代码：`backend/store/`（db / repository / lifecycle / releases / gate / seed）+ `backend/api/admin_routes.py`
- 消费侧代码：`backend/agents/customer_service_agent.py` + `backend/rag/`
- 关键认知：管理台一切操作只写 SQLite，**从不碰向量索引**；索引由消费侧按快照懒构建/预热构建

## 2. 索引来源之变（一句话核心）

**L16：md → 全局单索引。平台化后：md →（sync）→ SQLite →（发布）→ 快照 → 按 release 建索引。**

- 快照内容读取：`backend/store/releases.py` → `load_release_chunks(release_id)`
- 按 release 的索引缓存：`backend/rag/index_cache.py` → `get_release_index`（key=release_id）
- 向量库缓存：`backend/rag/vector_store.py` → `get_vector_store`（key=(client, 索引版本)，全量 chunk embedding 在这里发生，是冷启动大头）
- 切片/embedding 代码两侧复用同一套：`backend/rag/knowledge_base.py`

## 3. 灰度路由（不是 HTTP 路由，是 /chat 内部一次函数调用）

实现：`backend/store/releases.py` → `resolve_gray_route(driver_id)`，按序判断：

1. driver_id ∈ candidate.gray_whitelist → candidate（channel=whitelist）
2. sha256(driver_id) 前 8 位 % 100 < gray_percent → candidate（channel=percent，同司机恒定同桶）
3. 否则 → stable

- 灰度配置即时生效：`update_gray_config`（白名单/百分比，审计 release.gray_configured）
- 全量发布新 stable 会同时作废旧 stable 与 candidate（rollback_reason=superseded）
- 回滚：`rollback_release` 恢复历史快照为 stable，校验 release_items.content_hash 防篡改
- 缓存联动：发布/回滚触发 `_emit_release_change` → 监听器失效缓存 + 预热（见 §6）

## 4. 权限域分离（两把互不相干的钥匙）

| 域 | 凭证 | 管什么 | 代码路径 |
|---|---|---|---|
| 消费侧 RBAC | `runtime_role`（driver/agent/admin） | 检索可见哪些知识库/audience | `customer_service_agent.py` 顶层参数（不信任消息体提权）；`store/releases.py` → `get_allowed_kb_ids / get_allowed_audiences` |
| 供给侧管理 | `X-Admin-Token` 头 | 能否调 /admin/* 写接口 | `backend/api/admin_routes.py` → `require_admin_token`（FastAPI Depends 前置卡口，401） |

- 前端角色选择器只影响 /chat 的 runtime_role；管理台可见性不受角色限制，没 token 全部 401
- 审计 actor 是自报字段（共享 token 无法区分人）；生产应接 IAM 会话——平台预留了 runtime_role 与 audit actor 两个身份挂点
- 每写必审计：`store/repository.py` → `write_audit`，7 类 action：seed.completed / review.* / release.published / release.gray_configured / release.rolled_back / gate.blocked / rbac.denied

## 5. 全局开关与加载顺序（踩过的坑）

开关常量定义：`backend/config/settings.py`（KNOWLEDGE_LIFECYCLE_ENABLED / RBAC_ENABLED / BM25_ENABLED / RAG_FUSION_ENABLED / GATE_ENABLED，缺省全 False = L16 行为）。

**坑**：`load_course_env()` 只在 LLM/embedding 客户端发起调用时懒执行，而开关常量在 settings 模块 import 时就定值——course.env 里写开关**不生效**，平台开关必须用真实环境变量启动：

```powershell
$env:AGENT_KNOWLEDGE_LIFECYCLE_ENABLED='true'; $env:AGENT_RBAC_ENABLED='true'; python main.py
```

- 测试不受影响的原因：测试 monkeypatch `settings.X`（动态读处生效），模块级常量保持 False
- 开关没生效时的现象：管理台发布照常成功（供给侧不吃开关），但 /chat 走 L16 全局索引，发布"无感"

## 6. 发布后预热（方案 A+B，2026-10-18 落地）

问题：首问触发懒构建（索引 + 全量 chunk embedding），司机白等 ~30s。

实现：`backend/rag/index_cache.py` → `prewarm_active_release_indexes()`
- 遍历当前 stable/candidate → 调 `get_release_index` + `get_vector_store`（复用懒函数，不另写构建逻辑）
- 挂载点 A（发布/回滚后）：`on_release_change` 失效旧缓存后同步调用——管理员多等几秒，司机首问毫秒级
- 挂载点 B（启动时）：`backend/main.py` → `warm_up_offline_caches` 后台线程追加调用
- 容错：监听器吞异常，预热失败最坏退回老行为（首问慢一次），不阻断发布
- 测试免疫：读模块级开关常量，pytest monkeypatch 时自动跳过，不打真实 embedding API
- 实测：启动预热 7.7s（rel-0001,19 chunks）；发布后司机首问 2751ms（原 ~30s）

## 7. 已知设计取舍 / 坑位清单

1. **单文档发布会缩小知识面**：快照=勾选版本集合，不是增量。只发 refund-v2 → usage/purchase 等退出线上 → 相关问题兜底（平台如实承认缺口，是正确行为）。防线=质量门禁（GATE_ENABLED=true 时 expected_not_recalled 拦截）；门禁 placeholder 时靠人
2. **检索缓存放大瞬时失败**：`customer_service_agent.py` cache_put 无条件执行——限流拿到的空结果/兜底会被缓存反复复放。改进方向：兜底结果不入缓存
3. **cache_hit 分支留痕缺失**：bm25_hits / fusion_sub_queries 在缓存命中时为空（Task 21 修复了未初始化 500，留空是近似处理）
4. **管理员无账号体系**：单一共享 token，无分级；审计 actor 自报。生产接上游 IAM
5. **索引/向量全进程内**：重启即冷（预热缓解）；彻底方案=embedding 按 content_hash 持久化（方案 C，未做）

## 8. 演示环境速查

- 后端：`cd backend && python main.py`（:8000）；平台开关见 §5
- 前端：`cd frontend && npm run dev`（:5173，**必须走 vite 代理**，直接开 8000 或 html 文件会 Failed to fetch）
- 管理台 token：`dev-admin-token`（AGENT_ADMIN_TOKEN 覆盖）
- 演示库重建：`cd backend && python -m store.seed reset`（幂等：5库/7版本/2 releases）
- 全量回归：`cd backend && python -m pytest tests -q`（92 passed，2026-10-18）
