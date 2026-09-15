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


def _is_leaf_text_block(tag) -> bool:
    return tag.find(["div", "p", "table"]) is None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _flatten_table(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [_clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_filing_html(path: Path) -> list[Block]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup

    blocks: list[Block] = []
    for tag in body.find_all(["div", "p", "table"]):
        if tag.find_parent("table") is not None:
            continue

        if tag.name == "table":
            text = _flatten_table(tag)
            if text:
                blocks.append(Block(type="table", text=text))
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
