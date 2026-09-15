from pathlib import Path

from financial_rag_agent.ingestion.parser import parse_filing_html

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k_fragment.html"


def test_toc_table_entries_are_not_detected_as_headings():
    blocks = parse_filing_html(FIXTURE)
    headings = [b.text for b in blocks if b.type == "heading"]
    assert headings == ["Item 1. Business", "Item 1A. Risk Factors"]


def test_tables_are_flattened_as_table_blocks():
    blocks = parse_filing_html(FIXTURE)
    table_blocks = [b for b in blocks if b.type == "table"]
    assert any("Data Center" in b.text and "47250" in b.text for b in table_blocks)


def test_document_order_is_preserved():
    blocks = parse_filing_html(FIXTURE)
    types_in_order = [b.type for b in blocks]
    first_heading_index = types_in_order.index("heading")
    second_heading_index = types_in_order.index(
        "heading", first_heading_index + 1
    )
    assert second_heading_index > first_heading_index
    # at least one paragraph between the two headings
    assert "paragraph" in types_in_order[first_heading_index + 1 : second_heading_index]
