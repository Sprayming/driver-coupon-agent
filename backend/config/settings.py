"""运行配置层，集中读取课程环境变量、能力声明和模型连接参数。"""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
# Prompt Registry 作为业务规则文件存放在 backend 根目录。
PROMPT_REGISTRY_PATH = BACKEND_DIR / "prompt_registry.json"
# 本项目 backend 只有一层嵌套（项目根/backend），parents[0] 即项目根。
# L07 是 课程根/lesson目录/backend 两层嵌套才用 parents[1]，结构不同不能照抄。
DEFAULT_COURSE_ENV_PATH = BACKEND_DIR.parents[0] / "course.env"
PLACEHOLDER_API_KEYS = {"", "你的模型平台 Key", "your-api-key", "YOUR_API_KEY"}
# 默认价格只用于本地趋势观察，不替代模型平台真实账单。
DEFAULT_INPUT_CNY_PER_1K = 0.001
DEFAULT_OUTPUT_CNY_PER_1K = 0.002

# RAG 配置
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
TOP_K = 2
# L14：把 TOP_K 拆成两个角色。
# 上面的 TOP_K 保留给 L10~L13 旧链路回退用（不删，便于对比实验），新链路不再直接引用它。
# CANDIDATE_K：宽召回时每个查询各取几条进候选池（配合 merge 去重）；
# FINAL_TOP_K：reranker 精排之后，真正喂给回答 Prompt 的片段条数。
CANDIDATE_K = 4
FINAL_TOP_K = 2
# —— L15 hybrid 召回 ——
# 向量宽召回条数沿用上面的 CANDIDATE_K（即课程版的 VECTOR_TOP_K，L14 已就位，不重复定义）。
# KEYWORD_TOP_K：关键词通道每轮召回条数，和向量通道等宽，两路是互补关系而非主从。
KEYWORD_TOP_K = 4
# SCORE_THRESHOLD = 0.28
# —— RAG 质量与低置信（L12）——
QUALITY_CASES_PATH = BACKEND_DIR / "rag_quality_cases.json"

# 第一道线：候选入场线。只决定「片段要不要进候选池」，不放行 = 当作没检索到。
RETRIEVAL_SCORE_THRESHOLD = 0.20

# 第二道线：低置信回答线。看的是最高分，决定「敢不敢把这些片段当依据回答」。
# bge-small-zh-v1.5 校准结果（固定问题集 6 条）：反例最高 0.338 < 阈值 < 正例最低 0.646，取 0.49。
# L13 上查询改写后分数分布会变，需用固定问题集重新校准。
LOW_CONFIDENCE_THRESHOLD = 0.49

# —— L14 reranker 精排 ——
# 默认走「轻量规则重排」：纯本地规则加减分，零网络延迟，方便每轮请求直接跑。
# 商业 reranker 要显式打开开关 + 配好 Key 才会启用，失败自动回退到轻量规则。
RERANK_ENABLED_ENV = "AGENT_RAG_RERANK_ENABLED"
RERANK_MODEL_DEFAULT = "Qwen/Qwen3-Reranker-8B"
RERANK_INSTRUCTION_DEFAULT = (
    "Given a customer-service question of the XiaoZhe driver coupon desk, rank passages that directly "
    "answer it higher. Prefer currently effective coupon rules over retired 2025 campaign notes, and "
    "prefer passages that contain the exact refund, usage, activation or complaint constraints needed "
    "to answer the user."
)
RERANK_TIMEOUT_SECONDS = 10.0

# —— L13 查询改写：口语归一化规则（源词, 目标词, 原因）——
# 只改检索问题，不改用户原话；长词规则放前面，避免被短词规则提前截断。
NORMALIZATION_RULES: list[tuple[str, str, str]] = [
    ("还没使", "未使用", "把口语“还没使”对齐到退款条件里的“未使用”"),
    ("还没用", "未使用", "把口语“还没用”对齐到退款条件里的“未使用”"),
    ("能退不", "可以退款吗", "把省略问法“能退不”展开为退款条件问法"),
    ("能退吗", "可以退款吗", "把口语“能退吗”归一到退款条件问法"),
    ("如何退款", "如何申请退款", "对齐退款申请流程的知识库用词"),
    ("咋退", "如何申请退款", "把口语“咋退”对齐到退款申请流程用词"),
    ("怎么退", "如何申请退款", "对齐退款流程的知识库用词"),
    ("几天到账", "退款到账时效", "把口语到账问法对齐到时效规则"),
    ("钱什么时候回来", "退款到账时效", "对齐退款到账时效规则"),
    ("用过了", "已经核销使用", "对齐“不支持退款情形”里的已使用措辞"),
    ("这卡", "卡券", "把指代词“这卡”对齐到业务对象“卡券”"),
    ("那卡", "卡券", "把指代词“那卡”对齐到业务对象“卡券”"),
]

# #query改写
# NORMALIZATION_RULE = (
#     ("那个", "", "去掉缺少上下文的指代词“那个”"),
#     ("这个", "", "去掉缺少上下文的指代词“这个”"),
#     ("那卡", "", "去掉缺少上下文的指代词“那卡”"),
#     ("这卡", "", "去掉缺少上下文的指代词“这卡”"),
#     ("还没使", "未使用", "口语对齐"),
#     ("能退不", "能否退款", "省略问法展开"),
#     ("咋退", "如何退款", "口语对齐"),
# )


# Embedding 配置
DEFAULT_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
# L12 性能优化：从 Qwen3-Embedding-4B（84s/次）换到 0.6B（暖调用 ~42s/次）。
# 注意：换模型后向量维度从 2560 变为 1024，分数分布也会变，LOW_CONFIDENCE_THRESHOLD
# 和 RETRIEVAL_SCORE_THRESHOLD 必须用固定问题集重新校准后才能定稿。
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# 本地 embedding（L12 性能优化第二步）：用 sentence-transformers 跑小模型，单次 <500ms，
# 彻底消除远程 API 的 42s 延迟。设为 False 可回退到远程 SiliconFlow embedding。
USE_LOCAL_EMBEDDING = True
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def load_course_env() -> Path | None:
    """读取全课共享配置，让每节课不用重复声明模型 Key。"""

    env_path = Path(os.getenv("AGENT_COURSE_ENV", str(DEFAULT_COURSE_ENV_PATH))).expanduser()
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)
    return env_path


def api_key_is_missing(api_key: str | None) -> bool:
    """判断模型 Key 是否缺失或仍是课程占位值。"""

    # 占位 Key 要当成未配置处理，不能让它一路打到模型平台才失败。
    return api_key is None or api_key.strip() in PLACEHOLDER_API_KEYS


def env_flag_enabled(name: str, default: bool = False) -> bool:
    """读取布尔环境变量开关（L14 用来控制商业 reranker 是否启用）。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# —— L16 索引与检索缓存 ——
# 必须放在 env_flag_enabled 定义之后（模块自上而下求值）。
# RAG_CACHE_ENABLED：检索缓存总开关，course.env 里 AGENT_RAG_CACHE_ENABLED 控制，方便回退对比。
RAG_CACHE_ENABLED = env_flag_enabled("AGENT_RAG_CACHE_ENABLED", default=True)
# 检索缓存条数上限（课程版是无限 dict，内存随问题数无界增长；用户拍板加简单上限 128 条）。
RAG_CACHE_MAX_ENTRIES = 128

# —— RAG 平台化（生命周期/多知识库/BM25/Fusion）开关与配置 ——
# 安全默认策略：四个平台开关缺省全部 False，系统行为与 L16 完全一致（NFR-1 回退承诺，TR-2.2）；
# 需要平台能力时在 course.env 显式置 true。
# KNOWLEDGE_LIFECYCLE_ENABLED：草稿/审核/发布/灰度/回滚生命周期；关闭时聊天链路走 L16 全局单索引。
KNOWLEDGE_LIFECYCLE_ENABLED = env_flag_enabled("AGENT_KNOWLEDGE_LIFECYCLE_ENABLED", default=False)
# RBAC_ENABLED：多知识库角色过滤 + audience 硬隔离（FR-8/9）。
RBAC_ENABLED = env_flag_enabled("AGENT_RBAC_ENABLED", default=False)
# BM25_ENABLED：政策编号/商品名硬关键词 BM25 专路（FR-11）。
BM25_ENABLED = env_flag_enabled("AGENT_BM25_ENABLED", default=False)
# RAG_FUSION_ENABLED：多子查询 + RRF 融合扩召回（FR-12）。
RAG_FUSION_ENABLED = env_flag_enabled("AGENT_RAG_FUSION_ENABLED", default=False)
# Task 16 GATE_ENABLED：发布质量门禁开关。默认 False（保持 Task 6 placeholder 行为）。
GATE_ENABLED = env_flag_enabled("AGENT_GATE_ENABLED", default=False)
# 管理 API 共享密钥（FR-10）。课程本地默认值仅用于调试，生产由环境变量注入。
ADMIN_TOKEN = os.getenv("AGENT_ADMIN_TOKEN", "dev-admin-token")
# SQLite 数据库文件路径（Task 1 store 层默认值与之对齐）。
DB_PATH = Path(os.getenv("AGENT_DB_PATH", str(BACKEND_DIR / "data" / "rag_platform.db")))
# RAG-Fusion 参数：子查询条数（默认 3，spec 假设项，评测后再调）与 RRF 平滑常数 k=60。
FUSION_QUERY_COUNT = int(os.getenv("AGENT_RAG_FUSION_QUERY_COUNT", "3"))
RRF_K = 60

# ── L17 实时业务事实接入：司机运力业务后端 ──
# 默认指向本地 mock 服务（driver_service_mock.py，端口 8082）。
# 替换为真实后端地址时只需改环境变量 DRIVER_SERVICE_BASE_URL。
DRIVER_SERVICE_BASE_URL = os.getenv(
    "DRIVER_SERVICE_BASE_URL",
    os.getenv("AGENT_DRIVER_SERVICE_BASE_URL", "http://localhost:8082"),
)
