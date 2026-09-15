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
