# 本项目是AI驱动的司机卡券退款助手，
# 旨在帮助司机在卡券退款、核销查询等场景获得自动化处理。

"""薄入口：只负责组装 FastAPI app 并启动司机卡券退款 Agent 后端。"""

from __future__ import annotations

import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.customer_service_agent import CustomerServiceAgent
from api.admin_routes import create_admin_router
from api.routes import create_router
from rag import index_cache
from store import releases


agent = CustomerServiceAgent()


def warm_up_offline_caches() -> None:
    """后台预热向量库和质量摘要，避免首批 /chat 支付冷启动 embedding。"""

    started_at = time.perf_counter()
    try:
        agent.warm_up_offline_caches()
        # 方案 B（2026-10-18 用户拍板）：启动预热同步覆盖当前活跃 stable/candidate 的
        # release 索引与向量库；发版/回滚后由 on_release_change 接力预热（方案 A）。
        index_cache.prewarm_active_release_indexes()
        print(f"[warmup] 离线缓存预热完成，用时 {time.perf_counter() - started_at:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        # 预热失败不能让进程挂掉：缓存留空，rag_quality 字段自动隐藏即可。
        print(f"[warmup] 预热失败，缓存留空：{type(exc).__name__}: {exc}", flush=True)


# daemon 线程预热：服务器立即可服务，启动阶段不被 embedding 阻塞。
threading.Thread(target=warm_up_offline_caches, name="offline-cache-warmup", daemon=True).start()


def create_app() -> FastAPI:
    """组装司机卡券退款 Agent 应用。"""

    app = FastAPI(title="Driver Coupon Refund Agent v0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_router(lambda: agent))
    # RAG 平台化（Task 4）：管理接口 /admin/*，X-Admin-Token 鉴权。
    app.include_router(create_admin_router())
    # Task 8：发布/回滚后清空全局索引与检索缓存（Task 9 升级为按 release 隔离失效）。
    releases.add_release_change_listener(index_cache.on_release_change)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
