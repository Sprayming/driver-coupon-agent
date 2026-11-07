import { useCallback, useState } from "react";
import type { ChatRequest, ChatResponse, DemoUser, Message, RuntimeRole } from "../types/api";

const DEFAULT_USERS: DemoUser[] = [
  {
    profile: {
      userId: "driver_001",
      nickname: "张师傅",
      driverLevel: "gold",
      riskLevel: "low",
    },
  },
  {
    profile: {
      userId: "driver_002",
      nickname: "李师傅",
      driverLevel: "silver",
      riskLevel: "medium",
    },
  },
  // Task 19：灰度样本司机——seed 数据里 rel-0002 candidate 的白名单 ID，
  // 选中后请求走 candidate 索引，TracePanel 可看到灰度身份（channel=whitelist 命中）。
  {
    profile: {
      userId: "driver-gray-whitelist",
      nickname: "王灰度",
      driverLevel: "gold",
      riskLevel: "low",
    },
  },
];

export function useAgentChat() {
  const [users] = useState<DemoUser[]>(DEFAULT_USERS);
  const [selectedUserId, setSelectedUserId] = useState("driver-001");
  // Task 19：调试台角色切换（driver/agent/admin）。后端只信请求顶层 runtime_role，
  // agent/admin 可见 internal_playbook 库，driver 被硬隔离（TR-19.2 验证点）。
  const [role, setRole] = useState<RuntimeRole>("driver");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeResponse, setActiveResponse] = useState<ChatResponse | null>(null);
  const [sessionId, setSessionId] = useState("session-001");

  const selectedUser = users.find((u) => u.profile.userId === selectedUserId) || users[0];

  const selectUser = useCallback((userId: string) => {
    setSelectedUserId(userId);
    setMessages([]);
    setActiveResponse(null);
    setSessionId(`session-${Date.now()}`);
  }, []);

  const selectRole = useCallback((nextRole: RuntimeRole) => {
    // 切角色即换身份：清空会话避免上一角色的上下文串台。
    setRole(nextRole);
    setMessages([]);
    setActiveResponse(null);
    setSessionId(`session-${Date.now()}`);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim()) return;

      const userMessage: Message = { role: "user", content };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      const requestBody: ChatRequest = {
        session_id: sessionId,
        runtime_user_id: selectedUser.profile.userId,
        runtime_role: role,
        runtime_driver_name: selectedUser.profile.nickname,
        runtime_driver_level: selectedUser.profile.driverLevel,
        runtime_risk_level: selectedUser.profile.riskLevel,
        user_message: content,
        debug: true,
        reasoning_view: "default",
      };

      // 调试台观察：端到端总耗时（毫秒）。performance.now() 是单调时钟，亚毫秒精度，
      // 覆盖 前端 → Vite 代理 → 后端整条检索/精排/模型 链路。
      const startedAt = performance.now();

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data: ChatResponse = await response.json();
        const latency_ms = performance.now() - startedAt;
        setActiveResponse(data);

        const assistantMessage: Message = {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          answer_path: data.session_state?.rag?.answer_path,
          cost_summary: data.cost_summary,
          session_state: data.session_state,
          latency_ms,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error) {
        const errorMessage: Message = {
          role: "assistant",
          content: `请求失败: ${error instanceof Error ? error.message : "未知错误"}`,
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [selectedUser, sessionId, role],
  );

  return {
    users,
    selectedUser,
    selectedUserId,
    role,
    messages,
    isLoading,
    activeResponse,
    selectUser,
    selectRole,
    sendMessage,
  };
}
