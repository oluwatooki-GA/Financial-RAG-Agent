from financial_rag_agent.documents.pdf_chunker import PDFChunker
from financial_rag_agent.ingestion.parser import Block


def _paragraph(n: int) -> Block:
    sentence = f"This is filler sentence number {n} describing the annual report in detail. "
    return Block(type="paragraph", text=sentence * 20)


def _table(rows: list[list[str]]) -> Block:
    text = "\n".join(" | ".join(row) for row in rows)
    return Block(type="table", text=text, table_rows=rows)


def test_page_label_propagates_to_every_chunk_on_that_page():
    blocks = [
        Block(type="heading", text="Page 126"),
        _paragraph(1),
        _paragraph(2),
        _paragraph(3),
    ]

    chunker = PDFChunker(target_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(blocks)

    assert len(chunks) >= 2, "fixture should produce multiple chunks to prove propagation"
    for c in chunks:
        assert c.item_label == "Page 126"
        assert c.item_heading == "Page 126"
        assert c.section_path == "Page 126"
        assert c.part_label is None


def test_new_page_heading_resets_the_label():
    blocks = [
        Block(type="heading", text="Page 1"),
        _paragraph(1),
        Block(type="heading", text="Page 2"),
        _paragraph(2),
    ]

    chunker = PDFChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    labels = {c.item_label for c in chunks}
    assert labels == {"Page 1", "Page 2"}


def test_chunk_index_is_sequential_across_the_document():
    blocks = [
        Block(type="heading", text="Page 1"),
        _paragraph(1),
        Block(type="heading", text="Page 2"),
        _paragraph(2),
    ]

    chunker = PDFChunker(target_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(blocks)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_table_never_merges_with_surrounding_prose():
    table_rows = [["Share capital", "18,275,115", "17,069,475"], ["Share premium", "500,604,865", "329,229,161"]]
    blocks = [
        Block(type="heading", text="Page 126"),
        _paragraph(1),
        _table(table_rows),
        _paragraph(2),
    ]

    chunker = PDFChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_data == table_rows
    assert "500,604,865" in table_chunks[0].text

    text_chunks = [c for c in chunks if c.modality == "text"]
    assert all("500,604,865" not in c.text for c in text_chunks)


def test_table_chunk_still_carries_the_page_label():
    blocks = [
        Block(type="heading", text="Page 126"),
        _table([["Revenue", "100,000"], ["Profit", "50,000"]]),
    ]

    chunker = PDFChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].item_label == "Page 126"


def test_large_table_splits_by_row_group_not_mid_row():
    rows = [[f"Line item {i}", "A" * 40, "B" * 40] for i in range(10)]
    blocks = [Block(type="heading", text="Page 130"), _table(rows)]

    chunker = PDFChunker(target_tokens=20, overlap_tokens=0)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) > 1, "fixture should force a multi-group split"

    all_rows_in_chunks = [row for c in table_chunks for row in c.table_data]
    assert all_rows_in_chunks == rows
