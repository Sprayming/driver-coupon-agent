"""Word 文档转 Markdown（后台上传通道的最小实现）。

只做保结构不保样式的转换：标题层级、段落、表格行。
样式（字体/颜色/编号）不进入检索，避免把版式噪声喂给 RAG。
"""

from __future__ import annotations

import io
import re

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

_HEADING_RE = re.compile(r"(heading|标题)\s*([1-6])", re.IGNORECASE)


def _heading_level(style_name: str) -> int:
    match = _HEADING_RE.search(style_name or "")
    return int(match.group(2)) if match else 0


def _cell_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def docx_to_markdown(data: bytes) -> str:
    """把 .docx 字节流转成 Markdown 文本。"""

    document = Document(io.BytesIO(data))
    blocks: list[str] = []

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _heading_level(paragraph.style.name if paragraph.style else "")
            blocks.append(f"{'#' * level} {text}" if level else text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows: list[str] = []
            for row in table.rows:
                cells = [_cell_text(cell.text) for cell in row.cells]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header = rows[0]
                separator = "| " + " | ".join(["---"] * len(rows[0].split("|")[1:-1])) + " |"
                blocks.append("\n".join([header, separator] + rows[1:]))

    return "\n\n".join(blocks)
