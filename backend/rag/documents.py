"""知识文档层：读取 knowledge/ 下的 Markdown、解析 frontmatter、转成公开 Citation。

对标小哲 L41 `rag/documents.py`。职责边界：

- 本模块只做「文件 → 元数据 + 正文 / Citation」
- 切块、建索引、检索分别在 `knowledge_base` / `index_cache` / `vector_store`

文件读取带 lru_cache：同名文件只解析一次，避免每次检索都碰磁盘。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from api.schemas import Citation, SourceDocument
from config.settings import KNOWLEDGE_DIR


def parse_metadata_value(value: str) -> Any:
    """解析 Markdown 元数据里的简单值。"""

    parsed_value = value.strip()
    if parsed_value.startswith("[") and parsed_value.endswith("]"):
        return [
            item.strip().strip("\"'")
            for item in parsed_value[1:-1].split(",")
            if item.strip()
        ]
    if "," in parsed_value:
        return [
            item.strip().strip("\"'")
            for item in parsed_value.split(",")
            if item.strip()
        ]
    return parsed_value


def parse_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """拆出 Markdown 顶部 front matter 和正文。"""

    if not markdown.startswith("---"):
        return {}, markdown

    end_marker = markdown.find("\n---", 3)
    if end_marker == -1:
        return {}, markdown

    raw_metadata = markdown[3:end_marker].strip()
    body = markdown[end_marker + 4 :].lstrip()
    metadata: dict[str, Any] = {}
    for raw_line in raw_metadata.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_metadata_value(value)
    return metadata, body


def iter_knowledge_paths() -> list[Path]:
    """列出 knowledge 目录下的全部 Markdown（排序保证结果稳定）。"""

    return sorted(KNOWLEDGE_DIR.glob("*.md"))


@lru_cache(maxsize=None)
def read_document_text(path_str: str) -> str:
    """读取单个知识文件；同名文件只读一次。"""

    return Path(path_str).read_text(encoding="utf-8")


def load_source_documents() -> list[SourceDocument]:
    """读取 knowledge 目录里的所有 Markdown 原文。"""

    documents: list[SourceDocument] = []
    for path in iter_knowledge_paths():
        metadata, body = parse_front_matter(read_document_text(str(path)))
        title = str(metadata.get("title") or path.stem)
        documents.append(
            SourceDocument(source_path=path.name, title=title, metadata=metadata, body=body)
        )
    return documents


@lru_cache(maxsize=None)
def load_knowledge_citation(name: str) -> Citation:
    """按文件名加载知识文件并转成 Citation（L41 同款 API）。

    `name` 只允许是 knowledge/ 下的文件名，不接受 `../` 或绝对路径。
    """

    base = Path(KNOWLEDGE_DIR).resolve()
    path = (base / name).resolve()
    # resolve() 已经消掉 ../：只要父目录不是 knowledge 目录就判为路径穿越。
    if path.parent != base:
        raise ValueError(f"Invalid knowledge name: {name}")

    metadata, body = parse_front_matter(read_document_text(str(path)))
    return Citation(
        citation_id=f"knowledge/{name}",
        source_path=f"knowledge/{name}",
        source_title=str(metadata.get("title") or path.stem),
        chunk_id=path.stem,
        score=float(metadata.get("score") or 0.0),
        snippet=body.strip(),
    )
