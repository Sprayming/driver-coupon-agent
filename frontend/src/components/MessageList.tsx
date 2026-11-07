import type { Message } from "../types/api";
import { CitationCard } from "./CitationCard";

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="panel message-panel empty">
        <div className="empty-state">
          <div className="icon">💬</div>
          <p>发送问题后，这里会显示对话记录</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel message-panel">
      <div className="panel-header">
        <span>对话记录</span>
        <small>{messages.length} 条消息</small>
      </div>
      <div className="message-list">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-header">
              <span className="role">{msg.role === "user" ? "司机" : "Agent"}</span>
              {msg.answer_path && (
                <span className={`path-badge ${msg.answer_path}`}>
                  {msg.answer_path}
                </span>
              )}
            </div>
            <div className="message-content">{msg.content}</div>
            {msg.citations && msg.citations.length > 0 && (
              <div className="citations-section">
                <div className="citations-title">引用证据 ({msg.citations.length})</div>
                {msg.citations.map((c) => (
                  <CitationCard key={c.citation_id} citation={c} />
                ))}
              </div>
            )}
            {msg.cost_summary && (
              <div className="cost-info">
                <small>
                  耗时: {msg.latency_ms != null ? `${(msg.latency_ms / 1000).toFixed(2)} s` : "-"} |
                  tokens: {msg.cost_summary.total_tokens} |
                  cost: ¥{msg.cost_summary.estimated_total_cost_cny.toFixed(4)} |
                  source: {msg.cost_summary.token_source}
                </small>
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-content">正在思考...</div>
          </div>
        )}
      </div>
    </div>
  );
}
