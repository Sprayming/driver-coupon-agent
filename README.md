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
uvicorn main:app --reload --port 8000

# Mock 司机域微服务(另开终端,端口 8082)
cd backend
uvicorn driver_service_mock:app --port 8082

# 前端
cd frontend
npm install
npm run dev

# 测试
cd backend && pytest
```

> 配置说明：模型走 OpenAI 兼容接口（硅基流动 / DeepSeek 等均可），将 `course.env.example` 复制为 `course.env` 并填入自己的 Key。

### 管理后台登录（内部员工）

管理台按角色开放权限，服务端强制校验：`viewer` 只读审计 / `editor` 改草稿并提交 / `reviewer` 审核 / `publisher` 发布、灰度、回滚。

首次启动会自动播种四个演示账号（密码为 `AGENT_AUTH_SEED_PASSWORD`，默认 `demo123456`）：

| 账号 | 角色 | 能做什么 |
|---|---|---|
| `admin` | publisher | 全部操作（含发布与回滚） |
| `reviewer` | reviewer | 审核通过/驳回 |
| `editor` | editor | 同步草稿、提交审核 |
| `viewer` | viewer | 只读审计日志 |

会话使用 HttpOnly Cookie（8 小时），前端不接触任何令牌；机器调用仍可用 `X-Admin-Token`。

登录页支持自助注册：注册账号固定为 `viewer`（只读），注册后自动登录；需要编辑/审核/发布权限，由 `publisher` 在「用户管理」里改角色。用户管理（增删账号、改角色、重置密码、停用）仅 `publisher` 可见，所有变更写入审计。生产环境可设 `AGENT_ALLOW_SELF_REGISTER=false` 关闭注册入口。


## Docker 一键部署

```bash
cp course.env.example course.env   # 先填入自己的 Key(只读挂载进容器,不打包进镜像)
docker compose up -d --build
```

- 前端对话台：http://localhost （默认 80 端口，`FRONTEND_PORT=8080 docker compose up -d` 可改）
- 后端 API + Trace 调试台：http://localhost:8000
- 三个容器：`frontend`（nginx 托管 React 产物，`/api` 反代后端）→ `backend`（FastAPI，首启自动播种 SQLite 平台库）→ `driver-mock`（Mock 司机域微服务，仅容器内网可见）
- 接真实司机域后端时，只需把 `docker-compose.yml` 里 `DRIVER_SERVICE_BASE_URL` 改为真实地址

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
