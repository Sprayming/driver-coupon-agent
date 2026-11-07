// 管理台共享小组件（Task 18）：错误横幅、状态徽标、确认弹窗、JSON 折叠块。
import { ReactNode, useState } from "react";

export function ErrorBanner({ message, onClose }: { message: string | null; onClose?: () => void }) {
  if (!message) return null;
  return (
    <div className="admin-error" role="alert">
      <span>{message}</span>
      {onClose && (
        <button type="button" className="link-button" onClick={onClose}>
          关闭
        </button>
      )}
    </div>
  );
}

export function NoticeBanner({ message, onClose }: { message: string | null; onClose?: () => void }) {
  if (!message) return null;
  return (
    <div className="admin-notice" role="status">
      <span>{message}</span>
      {onClose && (
        <button type="button" className="link-button" onClick={onClose}>
          关闭
        </button>
      )}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge status-${status}`}>{status}</span>;
}

/** 操作确认弹窗：children 为表单体，title 为标题。 */
export function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>{title}</span>
          <button type="button" className="link-button" onClick={onClose}>
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

/** before/after 状态 JSON 折叠展示（审计/发布明细用）。 */
export function JsonDetails({ label, data }: { label: string; data: Record<string, unknown> | null }) {
  const [open, setOpen] = useState(false);
  if (!data || Object.keys(data).length === 0) return null;
  return (
    <div className="json-details">
      <button type="button" className="link-button" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} {label}
      </button>
      {open && <pre className="json-pre">{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
}
