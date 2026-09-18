from pathlib import Path

import pdfplumber

from financial_rag_agent.ingestion.parser import Block


def _is_real_table(rows: list[list[str]]) -> bool:
    """pdfplumber's table detection false-positives on visually grid-like
    non-tabular content — verified live on a real annual report, where a
    table-of-contents page was detected as a "table" whose rows crammed
    everything into one cell. A genuine multi-column financial table has
    most rows with at least two real, non-empty cells; a false positive
    like the TOC mostly doesn't."""
    if not rows:
        return False
    multi_cell_rows = sum(1 for row in rows if sum(1 for cell in row if cell.strip()) >= 2)
    return multi_cell_rows >= len(rows) / 2


def parse_pdf(path: Path) -> list[Block]:
    """Extracts text and tables per page from a downloaded PDF.

    Unlike SEC HTML filings, a generic PDF has no reliable heading markup
    to key section-tracking off (no "ITEM 7" pattern) — structure here is
    page-based: each page gets a "Page N" heading block, its detected
    real tables (filtered by _is_real_table) become table blocks with
    genuine extracted rows, and the rest of that page's text — with the
    table regions excluded first, so a table's numbers aren't duplicated
    into a paragraph block too — becomes a paragraph block.

    Known real limitation, verified live on an actual annual report: real
    thousands-separated figures come through uncorrupted, but pdfplumber's
    cell-boundary detection is less reliable than SEC HTML's explicit
    <td> tags — some rows split cleanly across columns, others merge
    several values into one cell. The numbers themselves are never wrong;
    column alignment just isn't guaranteed the way it is for SEC tables.
    """
    blocks: list[Block] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            blocks.append(Block(type="heading", text=f"Page {page_number}"))

            page_without_tables = page
            for table in page.find_tables():
                rows = [[cell or "" for cell in row] for row in table.extract()]
                if _is_real_table(rows):
                    text = "\n".join(" | ".join(row) for row in rows)
                    blocks.append(Block(type="table", text=text, table_rows=rows))
                    page_without_tables = page_without_tables.outside_bbox(table.bbox)

            text = (page_without_tables.extract_text() or "").strip()
            if text:
                blocks.append(Block(type="paragraph", text=text))

    return blocks
