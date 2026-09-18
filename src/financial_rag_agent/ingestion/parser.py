import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADING_RE = re.compile(r"^(PART\s+[IVX]+\b|Item\s+\d+[A-Za-z]?\.?\s)", re.IGNORECASE)
HEADING_MAX_LEN = 150

BlockType = Literal["heading", "table", "paragraph"]


@dataclass
class Block:
    type: BlockType
    text: str
    table_rows: list[list[str]] | None = None


def _is_leaf_text_block(tag) -> bool:
    return tag.find(["div", "p", "table"]) is None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def dedupe_adjacent(cells: list[str]) -> list[str]:
    """Collapses consecutive duplicate cells, which colspan expansion
    creates on purpose (the same text repeated across every column it
    spans) — useful when flattening a row to compact text for embedding
    or display, without losing the column-aligned structure in table_rows
    itself."""
    result: list[str] = []
    for cell in cells:
        if not result or result[-1] != cell:
            result.append(cell)
    return result


def _int_attr(tag, name: str) -> int:
    try:
        return max(1, int(tag.get(name, 1)))
    except (TypeError, ValueError):
        return 1


def _extract_table_rows(table_tag) -> list[list[str]]:
    """Expands colspan/rowspan into a proper rectangular grid, repeating a
    spanned cell's text into every column/row it visually covers. SEC
    filings render financial tables with colspan extremely heavily (e.g. a
    header cell spanning three data sub-columns for "$", the digits, and a
    trailing separator) — without expansion, rows end up with wildly
    inconsistent lengths and no reliable column correspondence."""
    rows: list[list[str]] = []
    pending: dict[int, list] = {}  # col_index -> [text, rows_remaining]

    for tr in table_tag.find_all("tr"):
        row: list[str] = []
        col = 0
        for cell in tr.find_all(["td", "th"]):
            while col in pending:
                text, _ = pending[col]
                row.append(text)
                pending[col][1] -= 1
                if pending[col][1] <= 0:
                    del pending[col]
                col += 1

            text = _clean_text(cell.get_text(" ", strip=True))
            colspan = _int_attr(cell, "colspan")
            rowspan = _int_attr(cell, "rowspan")
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col] = [text, rowspan - 1]
                col += 1

        while col in pending:
            text, _ = pending[col]
            row.append(text)
            pending[col][1] -= 1
            if pending[col][1] <= 0:
                del pending[col]
            col += 1

        if any(cell for cell in row):
            rows.append(row)

    return rows


def parse_filing_html(path: Path) -> list[Block]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup

    blocks: list[Block] = []
    for tag in body.find_all(["div", "p", "table"]):
        if tag.find_parent("table") is not None:
            continue

        if tag.name == "table":
            rows = _extract_table_rows(tag)
            if rows:
                text = "\n".join(" | ".join(dedupe_adjacent(row)) for row in rows)
                blocks.append(Block(type="table", text=text, table_rows=rows))
            continue

        if not _is_leaf_text_block(tag):
            continue

        text = _clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        if len(text) <= HEADING_MAX_LEN and HEADING_RE.match(text):
            blocks.append(Block(type="heading", text=text))
        else:
            blocks.append(Block(type="paragraph", text=text))

    return blocks
