from financial_rag_agent.documents.pdf_parser import _is_real_table


def test_genuine_financial_table_passes():
    rows = [
        ["", "Group", "Group"],
        ["In thousands of Naira", "Dec-2025", "Dec-2024"],
        ["Share capital", "18,275,115", "17,069,475"],
        ["Share premium", "500,604,865", "329,229,161"],
    ]
    assert _is_real_table(rows) is True


def test_table_of_contents_false_positive_is_rejected():
    # Real shape observed live: a grid-detected TOC page where almost
    # every row crams its whole text into one cell, the rest empty.
    rows = [
        ["Contents\n01 02 03\nStrategic\nNotice Introduction", "", "", "", ""],
        ["18 GTCO as an Investment 33 Leading with Technology", "", "", "", ""],
        ["", "", "", "07 08\nFinancial Others", ""],
    ]
    assert _is_real_table(rows) is False


def test_empty_rows_is_not_a_table():
    assert _is_real_table([]) is False


def test_mostly_single_cell_with_some_real_rows_still_rejected():
    # A bare majority of rows must actually be multi-cell, not just one.
    rows = [
        ["one cell only", "", ""],
        ["also one cell", "", ""],
        ["Revenue", "100", "200"],
    ]
    assert _is_real_table(rows) is False
