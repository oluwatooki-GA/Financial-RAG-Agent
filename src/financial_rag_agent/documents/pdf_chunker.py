import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_rag_agent.core.config import get_settings
from financial_rag_agent.ingestion.chunker import ChunkDraft, _flatten_rows, _group_table_rows
from financial_rag_agent.ingestion.parser import Block

PAGE_RE = re.compile(r"^Page (\d+)$")


class PDFChunker:
    """Splits a discovered PDF's parsed Blocks into Chunks. A generic PDF
    has no Part/Item structure to key off (unlike SEC's "ITEM 7" pattern),
    so the page number is the only real positional metadata available —
    reused via the existing item_label/item_heading columns (rather than
    adding a new column for a single extra concept) so citations still
    get a human-readable locator ("Page 126") regardless of which
    chunker produced them.

    Shares its table-grouping/flattening helpers with SECFilingChunker
    (ingestion/chunker.py) instead of duplicating them. If a third,
    meaningfully different document structure shows up later, that's the
    right moment to promote these into a shared module behind a formal
    interface — not before, per the same reasoning as discovery/base.py's
    Strategy pattern (built once there were real sources to compare, not
    speculatively for one).
    """

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
        current_page_label: str | None = None
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
                        part_label=None,
                        item_label=current_page_label,
                        item_heading=current_page_label,
                        section_path=current_page_label,
                        text=piece,
                    )
                )
                chunk_index += 1
            buffer.clear()

        for block in blocks:
            if block.type == "heading" and PAGE_RE.match(block.text):
                flush_buffer()
                current_page_label = block.text
                continue

            if block.type == "table":
                flush_buffer()
                for group in _group_table_rows(block.table_rows or [], self._target_chars):
                    drafts.append(
                        ChunkDraft(
                            chunk_index=chunk_index,
                            part_label=None,
                            item_label=current_page_label,
                            item_heading=current_page_label,
                            section_path=current_page_label,
                            text=_flatten_rows(group, title=current_page_label),
                            modality="table",
                            table_data=group,
                        )
                    )
                    chunk_index += 1
                continue

            buffer.append(block.text)

        flush_buffer()
        return drafts
