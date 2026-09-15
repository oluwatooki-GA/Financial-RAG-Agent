from financial_rag_agent.retrieval.citations import split_sentences


def test_splits_on_sentence_boundaries():
    text = "Revenue grew significantly. Costs also increased. We expect continued growth."
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == [
        "Revenue grew significantly.",
        "Costs also increased.",
        "We expect continued growth.",
    ]


def test_does_not_split_on_decimals_or_abbreviations():
    text = (
        "Revenue from Data Center computing grew 59% in the U.S. market. "
        "Other commitments were $3.4 billion as of Jan. 25, 2026."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "U.S. market" in sentences[0].text
    assert "$3.4 billion" in sentences[1].text


def test_char_offsets_round_trip_to_original_text():
    text = "Total revenue was $215,938 million. Operating income grew 58%."
    sentences = split_sentences(text)
    for s in sentences:
        assert text[s.char_start : s.char_end] == s.text
