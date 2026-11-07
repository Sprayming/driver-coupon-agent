-- RAG 知识平台 SQLite 表结构（Task 1）。
-- 设计约定：
-- 1) 全部主键用业务字符串 ID（应用层生成），仅审计/事件表用自增 rowid；
-- 2) 时间戳一律由应用层传 UTC ISO-8601 字符串，列只声明 TEXT NOT NULL；
-- 3) 发布快照不可变性由应用层保证（published/archived 版本行禁止 UPDATE/DELETE），
--    release_items 额外冗余 content_hash，回滚时校验快照未被篡改；
-- 4) JSON 结构（角色集合、灰度白名单、状态前后像）以 TEXT 存 JSON。

-- 知识库：按业务域分库，allowed_roles 声明哪些角色可检索（FR-8）。
CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id         TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    domain        TEXT NOT NULL,
    allowed_roles TEXT NOT NULL DEFAULT '["driver","agent","admin"]',  -- JSON 数组
    created_at    TEXT NOT NULL
);

-- 文档：一个 md 源文件对应一个文档，归属于一个知识库。
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    kb_id       TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    source_path TEXT NOT NULL UNIQUE,   -- 如 coupon-refund.md，sync 时按它定位
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 文档版本：md 全文快照。每次编辑产生新版本行，旧版本永不覆盖（审核流 + 不可变发布的基础）。
-- status: draft（草稿）/ submitted（待审核）/ approved（审核通过待发布）/ published（已发布）/ archived（归档过期）
-- audience: driver（司机可见）/ internal（仅内部角色，FR-9）
CREATE TABLE IF NOT EXISTS doc_versions (
    version_id      TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
    version_no      INTEGER NOT NULL,           -- 同一文档内递增
    content_md      TEXT NOT NULL,               -- md 全文快照
    content_hash    TEXT NOT NULL,               -- sha256(content_md)，sync diff 与快照校验用
    status          TEXT NOT NULL DEFAULT 'draft',
    audience        TEXT NOT NULL DEFAULT 'driver',
    effective_from  TEXT,                        -- 生效起点（ISO），NULL 表示发布即生效
    effective_to    TEXT,                        -- 生效终点（ISO），NULL 表示长期有效；注意：本期不按它硬过滤——
                                                 -- historical/过期内容沿用 L14 方案Y（宽进+reranker 软降权，见 spec FR-7），
                                                 -- 该字段仅用于管理台展示与未来定时切换预留
    change_note     TEXT NOT NULL DEFAULT '',
    created_by      TEXT NOT NULL DEFAULT 'system',
    created_at      TEXT NOT NULL,
    status_updated_at TEXT NOT NULL,
    UNIQUE(doc_id, version_no)
);

-- 审核事件：submit/approve/reject 全留痕（FR-3 / AC-2）。
CREATE TABLE IF NOT EXISTS review_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL REFERENCES doc_versions(version_id),
    action     TEXT NOT NULL CHECK (action IN ('submit','approve','reject')),
    actor      TEXT NOT NULL,
    comment    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

-- 发布：一次发布是一组不可变版本快照的集合（FR-4/5/6）。
-- status: candidate（灰度中）/ stable（全量稳定版）/ rolled_back（已回滚，历史留存）
CREATE TABLE IF NOT EXISTS releases (
    release_id     TEXT PRIMARY KEY,
    release_no     INTEGER NOT NULL UNIQUE,      -- 递增发布号
    label          TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'candidate'
                       CHECK (status IN ('candidate','stable','rolled_back')),
    gray_whitelist TEXT NOT NULL DEFAULT '[]',   -- JSON 数组：命中即走 candidate
    gray_percent   INTEGER NOT NULL DEFAULT 0   -- 0-100，按 driver_id 哈希截桶
                       CHECK (gray_percent BETWEEN 0 AND 100),
    created_by     TEXT NOT NULL DEFAULT 'admin',
    created_at     TEXT NOT NULL,
    activated_at   TEXT,
    rolled_back_at TEXT,
    rollback_reason TEXT NOT NULL DEFAULT ''
);

-- 发布明细：release 与版本快照的多对多；content_hash 冗余存一份做回滚校验。
CREATE TABLE IF NOT EXISTS release_items (
    release_id   TEXT NOT NULL REFERENCES releases(release_id),
    version_id   TEXT NOT NULL REFERENCES doc_versions(version_id),
    content_hash TEXT NOT NULL,
    PRIMARY KEY (release_id, version_id)
);

-- 统一审计日志：审核/驳回/发布/灰度/回滚/权限拒绝/门禁拦截 7 类事件（NFR-2）。
-- action 取值不在 DB 层强约束（平台演进会新增事件类型），由应用层常量保证口径。
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    actor       TEXT NOT NULL,                   -- 操作人/触发身份（driver_id 或 admin 账号）
    action      TEXT NOT NULL,                   -- 如 review.reject / release.publish / rbac.denied / gate.blocked
    target_type TEXT NOT NULL DEFAULT '',        -- document / version / release / query
    target_id   TEXT NOT NULL DEFAULT '',
    before_state TEXT,                           -- JSON：变更前状态（可空）
    after_state  TEXT,                           -- JSON：变更后状态（可空）
    comment     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_versions_doc_status ON doc_versions(doc_id, status);
CREATE INDEX IF NOT EXISTS idx_review_events_version ON review_events(version_id);
CREATE INDEX IF NOT EXISTS idx_release_items_version ON release_items(version_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_target ON audit_log(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);
