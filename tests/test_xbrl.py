from financial_rag_agent.tools.xbrl import _dedupe_facts


def test_dedupe_keeps_most_recently_filed_entry_per_period():
    # Mirrors the real duplication found against NVIDIA's actual XBRL data:
    # FY2025's revenue appears once as the primary figure (filed with the
    # FY2025 10-K) and again as a prior-year comparative (filed later, with
    # the FY2026 10-K).
    facts = [
        {"end": "2025-01-26", "val": 130497000000, "filed": "2025-02-26", "fy": 2025, "fp": "FY", "form": "10-K"},
        {"end": "2025-01-26", "val": 130497000000, "filed": "2026-02-25", "fy": 2026, "fp": "FY", "form": "10-K"},
        {"end": "2026-01-25", "val": 215938000000, "filed": "2026-02-25", "fy": 2026, "fp": "FY", "form": "10-K"},
    ]

    result = _dedupe_facts(facts, unit="USD")

    assert len(result) == 2
    by_end_date = {f.end_date: f for f in result}
    assert by_end_date["2025-01-26"].filed_date == "2026-02-25"
    assert by_end_date["2025-01-26"].fiscal_year == 2026
    assert by_end_date["2026-01-25"].value == 215938000000


def test_dedupe_sorts_oldest_to_newest():
    facts = [
        {"end": "2026-01-25", "val": 2, "filed": "2026-02-25", "fy": 2026, "fp": "FY", "form": "10-K"},
        {"end": "2024-01-28", "val": 0, "filed": "2024-02-26", "fy": 2024, "fp": "FY", "form": "10-K"},
        {"end": "2025-01-26", "val": 1, "filed": "2025-02-26", "fy": 2025, "fp": "FY", "form": "10-K"},
    ]

    result = _dedupe_facts(facts, unit="USD")

    assert [f.end_date for f in result] == ["2024-01-28", "2025-01-26", "2026-01-25"]


def test_dedupe_no_duplicates_passes_through():
    facts = [{"end": "2026-01-25", "val": 5, "filed": "2026-02-25", "fy": 2026, "fp": "FY", "form": "10-K"}]
    result = _dedupe_facts(facts, unit="USD")
    assert len(result) == 1
    assert result[0].unit == "USD"
