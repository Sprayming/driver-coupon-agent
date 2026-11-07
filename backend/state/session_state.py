"""L32：会话状态存储——模块级全局 dict，按 session_id + runtime_user_id 两级隔离。

和小哲电商 L32 state/session_state.py 对齐，但司机域用 coupon_id / activity_id / order_id。

存储内容：
- pending_clarification：上一轮让用户补的参数（卡券号/活动ID/订单号）
- verified_coupon_id / verified_activity_id / verified_order_id：验证过的业务实体
- last_intent / last_user_message：上一轮的意图和消息
- history_messages：最近几轮对话（压缩后）

本课次演化：
- L32 session-memory：模块级全局 dict，支持 pending_clarification 读写
- L33 runtime-context：从 Runtime Context 拿当前司机的卡券/活动列表做澄清候选
- L34 context-builder：上下文组装 + 历史压缩
- L35 context-compression：历史消息压缩

被谁调用：context/builder.py（current_memory / get_memory_value / set_memory_value）
"""

from __future__ import annotations

from typing import Any

# 会话记忆：{session_id: {runtime_user_id: memory_dict}}
SESSION_MEMORIES: dict[str, dict[str, dict[str, Any]]] = {}

# 消息计数：{session_id: count}
MESSAGE_COUNT_BY_SESSION: dict[str, int] = {}

# 成本事件：{session_id: [event]}
COST_EVENTS_BY_SESSION: dict[str, list[dict]] = {}


def current_memory(session_id: str, runtime_user_id: str) -> dict[str, Any]:
    """获取（或创建）当前会话的 memory dict。

    两级隔离：同一个 session_id 下不同 runtime_user_id 的 memory 互不干扰。
    被谁调用：context/builder.py 的所有函数
    """
    session = SESSION_MEMORIES.setdefault(session_id, {})
    return session.setdefault(runtime_user_id, {})


def get_memory_value(session_id: str, runtime_user_id: str, key: str, default: Any = None) -> Any:
    """安全读取 memory 中的某个字段。

    被谁调用：agents/customer_service_agent.py chat()（读取 verified_coupon_id 等）
    """
    return current_memory(session_id, runtime_user_id).get(key, default)


def set_memory_value(session_id: str, runtime_user_id: str, key: str, value: Any) -> None:
    """写入 memory 中的某个字段。

    被谁调用：context/builder.py update_memory（写 verified_coupon_id 等）
    """
    current_memory(session_id, runtime_user_id)[key] = value
