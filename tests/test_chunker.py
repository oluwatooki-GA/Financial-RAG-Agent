from financial_rag_agent.ingestion.chunker import SECFilingChunker
from financial_rag_agent.ingestion.parser import Block


def _paragraph(n: int) -> Block:
    sentence = f"This is filler sentence number {n} describing risk factors in detail. "
    return Block(type="paragraph", text=sentence * 20)


def test_heading_propagates_to_every_chunk_in_section_not_just_first():
    blocks = [
        Block(type="heading", text="Item 1A. Risk Factors"),
        _paragraph(1),
        _paragraph(2),
        _paragraph(3),
        _paragraph(4),
    ]

    chunker = SECFilingChunker(target_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(blocks)

    assert len(chunks) >= 3, "fixture should produce multiple chunks to prove propagation"
    for c in chunks:
        assert c.item_label == "Item 1A"
        assert c.item_heading == "Item 1A. Risk Factors"
        assert c.section_path == "Item 1A. Risk Factors"


def test_new_heading_resets_section_state():
    blocks = [
        Block(type="heading", text="Part I"),
        Block(type="heading", text="Item 1. Business"),
        _paragraph(1),
        Block(type="heading", text="Item 1A. Risk Factors"),
        _paragraph(2),
    ]

    chunker = SECFilingChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    labels = {c.item_label for c in chunks}
    assert labels == {"Item 1", "Item 1A"}
    for c in chunks:
        assert c.part_label == "Part I"
        if c.item_label == "Item 1":
            assert c.section_path == "Part I > Item 1. Business"
        else:
            assert c.section_path == "Part I > Item 1A. Risk Factors"


def test_chunk_index_is_sequential_across_whole_filing():
    blocks = [
        Block(type="heading", text="Item 1. Business"),
        _paragraph(1),
        _paragraph(2),
        Block(type="heading", text="Item 1A. Risk Factors"),
        _paragraph(3),
    ]

    chunker = SECFilingChunker(target_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(blocks)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def _table(rows: list[list[str]]) -> Block:
    text = "\n".join(" | ".join(row) for row in rows)
    return Block(type="table", text=text, table_rows=rows)


def test_table_never_merges_with_surrounding_prose():
    table_rows = [["Segment", "Revenue"], ["Data Center", "193737"], ["Gaming", "16042"]]
    blocks = [
        Block(type="heading", text="Item 7. Management's Discussion"),
        _paragraph(1),
        _table(table_rows),
        _paragraph(2),
    ]

    chunker = SECFilingChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_data == table_rows
    assert "Data Center" in table_chunks[0].text

    text_chunks = [c for c in chunks if c.modality == "text"]
    assert all("Data Center" not in c.text for c in text_chunks)
    assert all("193737" not in c.text for c in text_chunks)


def test_table_gets_nearest_preceding_paragraph_as_title():
    table_rows = [["Segment", "Revenue"], ["Data Center", "193737"]]
    blocks = [
        Block(type="heading", text="Item 7. Management's Discussion"),
        Block(type="paragraph", text="Revenue by Reportable Segments"),
        _table(table_rows),
    ]

    chunker = SECFilingChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert table_chunks[0].text.startswith("Revenue by Reportable Segments")


def test_back_to_back_table_inherits_previous_table_title():
    # Second table has no prose before it at all — should inherit the first
    # table's title rather than being left with no context.
    blocks = [
        Block(type="heading", text="Item 15. Exhibits"),
        Block(type="paragraph", text="Exhibit Index"),
        _table([["Exhibit No.", "Description"], ["3.1", "Certificate of Incorporation"]]),
        _table([["10.10", "Employment Agreement"], ["10.11", "Offer Letter"]]),
    ]

    chunker = SECFilingChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 2
    assert table_chunks[0].text.startswith("Exhibit Index")
    assert table_chunks[1].text.startswith("Exhibit Index")


def test_table_title_does_not_leak_across_a_new_heading():
    blocks = [
        Block(type="heading", text="Item 7. Management's Discussion"),
        Block(type="paragraph", text="Revenue by Reportable Segments"),
        _table([["Segment", "Revenue"], ["Data Center", "193737"]]),
        Block(type="heading", text="Item 8. Financial Statements"),
        _table([["Line Item", "Amount"], ["Net income", "100000"]]),
    ]

    chunker = SECFilingChunker(target_tokens=900, overlap_tokens=150)
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 2
    assert table_chunks[0].text.startswith("Revenue by Reportable Segments")
    assert not table_chunks[1].text.startswith("Revenue by Reportable Segments")


def test_large_table_splits_by_row_group_not_mid_row():
    # Each row is long enough that a small target_tokens forces a split.
    rows = [[f"Line item {i}", "A" * 40, "B" * 40] for i in range(10)]
    blocks = [Block(type="heading", text="Item 8. Financial Statements"), _table(rows)]

    chunker = SECFilingChunker(target_tokens=20, overlap_tokens=0)  # ~80 chars/chunk
    chunks = chunker.chunk(blocks)

    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) > 1, "fixture should force a multi-group split"

    # every original row appears intact in exactly one chunk's table_data
    all_rows_in_chunks = [row for c in table_chunks for row in c.table_data]
    assert all_rows_in_chunks == rows

    # every chunk's item_label/heading still propagated onto table chunks too
    for c in table_chunks:
        assert c.item_label == "Item 8"
