import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_rag_agent.config import get_settings
from financial_rag_agent.ingestion.parser import Block

PART_RE = re.compile(r"^PART\s+[IVX]+\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^(Item\s+\d+[A-Za-z]?)\.?\s*(.*)$", re.IGNORECASE)


@dataclass
class ChunkDraft:
    chunk_index: int
    part_label: str | None
    item_label: str | None
    item_heading: str | None
    section_path: str | None
    text: str


def _section_path(part_label: str | None, item_heading: str | None) -> str | None:
    parts = [p for p in (part_label, item_heading) if p]
    return " > ".join(parts) if parts else None


class SECFilingChunker:
    """Splits a filing's parsed Blocks into Chunks, propagating the last-seen
    Part/Item heading onto every chunk of that section (not just the first),
    since SEC filings only render the heading once in the source HTML."""

    def __init__(self, target_tokens: int | None = None, overlap_tokens: int | None = None):
        settings = get_settings()
        target_chars = (target_tokens or settings.chunk_target_tokens) * 4
        overlap_chars = (overlap_tokens or settings.chunk_overlap_tokens) * 4
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, blocks: list[Block]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        chunk_index = 0

        current_part: str | None = None
        current_item_label: str | None = None
        current_item_heading: str | None = None
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

        for block in blocks:
            if block.type == "heading":
                if PART_RE.match(block.text):
                    flush_buffer()
                    current_part = block.text
                    current_item_label = None
                    current_item_heading = None
                    continue

                item_match = ITEM_RE.match(block.text)
                if item_match:
                    flush_buffer()
                    current_item_label = item_match.group(1).strip()
                    current_item_heading = block.text
                    continue

            buffer.append(block.text)

        flush_buffer()
        return drafts
