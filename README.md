# 司机卡券 AI 审批助手（driver-coupon-refund-agent）

面向网约车司机卡券核销 / 退款 / 核销记录 / 收入查询等高频审批场景的 AI 审批助手。**个人独立项目**：以真实业务痛点为原型，通过 Mock 司机域微服务复刻场景做技术验证。

## 核心设计

- **业务契约驱动**：将"这券能不能核销/退款"抽象为 `BusinessFactNeed` 契约（7 类事实查询：订单 / 卡券 / 核销 / 退款 / 计价 / 收入 / 身份），LLM 仅负责生成草案，后端硬校验锁定审批口径
- **6 层决策链**：意图识别 → 业务事实契约 → 证据校验 → 风险与权限门禁 → 草案生成 → 终审校验，每层输入输出结构化，单层异常不拖垮全链路
- **双轨 RAG**：稳定知识（退款规则 / 话术）走向量检索，实时业务事实走 Tool-Calling 查询 Mock 微服务，杜绝模型编造
- **全链路 Trace 调试台**：记录每轮对话的决策轨迹与工具调用，线上问题可逐层定位
- **知识生命周期**：草稿 → 审核 → 发布 → 灰度 → 回滚，配套 RBAC 角色可见性与管理后台

## 技术栈

- 后端：Python / FastAPI / SQLite（知识发布与审计）/ ChromaDB（向量检索）/ BM25 + RRF 融合
- 前端：React 19 / TypeScript / Vite（审批对话台 + 管理后台 + Trace 面板）
- 质量：pytest 98 项自动化测试 / 22 条评估样本（复杂查询 / 投诉 / 边界案例）

## 快速开始

```bash
# 后端
cd backend
pip install -r requirements.txt
cp ../course.env.example ../course.env  # 填入你的 OpenAI 兼容接口 Key
uvicorn api.main:app --reload --port 8000

# 前端
cd frontend
npm install
npm run dev

# 测试
cd backend && pytest
```

> 配置说明：模型走 OpenAI 兼容接口（硅基流动 / DeepSeek 等均可），将 `course.env.example` 复制为 `course.env` 并填入自己的 Key。

## 目录结构

```
backend/
  agents/        # Agent 编排
  api/           # FastAPI 路由（含管理端）
  planner/       # 意图识别与任务规划
  rag/           # 双轨检索（向量 + BM25 + RRF 融合 + Rerank）
  tools/         # Tool-Calling 契约与运行时
  store/         # 知识生命周期（发布/灰度/回滚/审计）
  policies/      # 审批策略硬校验
  evals/         # 评估样本与回归
  tests/         # 98 项自动化测试
frontend/
  src/           # 对话台 / 管理后台 / Trace 调试台
```
