import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_rag_agent.core.config import get_settings
from financial_rag_agent.ingestion.parser import Block, dedupe_adjacent

PART_RE = re.compile(r"^PART\s+[IVX]+\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^(Item\s+\d+[A-Za-z]?)\.?\s*(.*)$", re.IGNORECASE)
_NOISE_RE = re.compile(r"^\d{1,4}$")  # bare page numbers
_TOC_RE = re.compile(r"^Table of Contents$", re.IGNORECASE)


@dataclass
class ChunkDraft:
    chunk_index: int
    part_label: str | None
    item_label: str | None
    item_heading: str | None
    section_path: str | None
    text: str
    modality: str = "text"
    table_data: list[list[str]] | None = None


def _section_path(part_label: str | None, item_heading: str | None) -> str | None:
    parts = [p for p in (part_label, item_heading) if p]
    return " > ".join(parts) if parts else None


def _flatten_rows(rows: list[list[str]], title: str | None = None) -> str:
    body = "\n".join(" | ".join(dedupe_adjacent(row)) for row in rows)
    return f"{title}\n\n{body}" if title else body


def _nearest_title(buffer: list[str]) -> str | None:
    """Walks backward through the section's accumulated prose to find the
    nearest real lead-in text for a table (e.g. "Revenue by Reportable
    Segments"), skipping running-header noise like bare page numbers and
    "Table of Contents" — SEC filings have no semantic heading tags, so a
    table row embedded alone has zero context without this."""
    for text in reversed(buffer):
        stripped = text.strip()
        if _NOISE_RE.match(stripped) or _TOC_RE.match(stripped):
            continue
        return text
    return None


def _group_table_rows(rows: list[list[str]], target_chars: int) -> list[list[list[str]]]:
    """Splits a table's rows into groups that each stay near target_chars,
    without ever breaking a row in half. A single oversized row still forms
    its own group rather than being truncated."""
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_len = 0

    for row in rows:
        row_text = " | ".join(row)
        row_len = len(row_text) + 1  # + separator newline
        if current and current_len + row_len > target_chars:
            groups.append(current)
            current = []
            current_len = 0
        current.append(row)
        current_len += row_len

    if current:
        groups.append(current)

    return groups


class SECFilingChunker:
    """Splits a filing's parsed Blocks into Chunks, propagating the last-seen
    Part/Item heading onto every chunk of that section (not just the first),
    since SEC filings only render the heading once in the source HTML."""

    def __init__(self, target_tokens: int | None = None, overlap_tokens: int | None = None):
        settings = get_settings()
        resolved_target = target_tokens if target_tokens is not None else settings.chunk_target_tokens
        resolved_overlap = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens
        self._target_chars = resolved_target * 4
        overlap_chars = resolved_overlap * 4
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, blocks: list[Block]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        chunk_index = 0

        current_part: str | None = None
        current_item_label: str | None = None
        current_item_heading: str | None = None
        last_table_title: str | None = None
        buffer: list[str] = []

        def flush_buffer() -> None:
            nonlocal chunk_index
            if not buffer:
                return
            section_text = "\n\n".join(buffer)
            for piece in self._splitter.split_text(section_text):
                piece = piece.strip()
                if not piece:
                    continue
                drafts.append(
                    ChunkDraft(
                        chunk_index=chunk_index,
                        part_label=current_part,
                        item_label=current_item_label,
                        item_heading=current_item_heading,
                        section_path=_section_path(current_part, current_item_heading),
                        text=piece,
                    )
                )
                chunk_index += 1
            buffer.clear()

        def emit_table(block: Block) -> None:
            nonlocal chunk_index, last_table_title
            # A table with no prose before it (back-to-back with another
            # table, e.g. Item 15's exhibit sub-tables) has nothing in
            # buffer to draw a title from — fall back to the previous
            # table's title rather than leaving this one with no context.
            title = _nearest_title(buffer) or last_table_title
            last_table_title = title
            flush_buffer()
            for group in _group_table_rows(block.table_rows or [], self._target_chars):
                drafts.append(
                    ChunkDraft(
                        chunk_index=chunk_index,
                        part_label=current_part,
                        item_label=current_item_label,
                        item_heading=current_item_heading,
                        section_path=_section_path(current_part, current_item_heading),
                        text=_flatten_rows(group, title=title),
                        modality="table",
                        table_data=group,
                    )
                )
                chunk_index += 1

        for block in blocks:
            if block.type == "heading":
                if PART_RE.match(block.text):
                    flush_buffer()
                    current_part = block.text
                    current_item_label = None
                    current_item_heading = None
                    last_table_title = None
                    continue

                item_match = ITEM_RE.match(block.text)
                if item_match:
                    flush_buffer()
                    current_item_label = item_match.group(1).strip()
                    current_item_heading = block.text
                    last_table_title = None
                    continue

            if block.type == "table":
                emit_table(block)
                continue

            buffer.append(block.text)

        flush_buffer()
        return drafts
