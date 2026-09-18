from financial_rag_agent.tools.ngx_corporate_actions import NGXCorporateAction, _parse_actions

_RAW_FIXTURE = [
    {
        "company": "Guaranty Trust Holding Company Plc",
        "company_symbol": "GTCO",
        "year": "2024",
        "dividend_share": "N2.70",
        "bonus": "Nil",
        "closure_of_register": "29th April 2024",
        "agm_date": "9th May 2024",
        "payment_date": "9th May 2024",
    },
    {
        "company": "Cutix Plc",
        "company_symbol": "CUTIX",
        "year": "2024",
        "dividend_share": "N0.15",
        "bonus": "1 for 1",
        "closure_of_register": "19th August 2024",
        "agm_date": "30th August 2024",
        "payment_date": "4th September 2024",
    },
]


def test_parses_all_records_when_no_symbol_filter():
    actions = _parse_actions(_RAW_FIXTURE, company_symbol=None)

    assert len(actions) == 2
    assert isinstance(actions[0], NGXCorporateAction)
    assert actions[0].year == 2024


def test_filters_to_one_company_symbol():
    actions = _parse_actions(_RAW_FIXTURE, company_symbol="gtco")  # lowercase on purpose

    assert len(actions) == 1
    assert actions[0].company_symbol == "GTCO"
    assert actions[0].dividend_share == "N2.70"


def test_unknown_symbol_returns_empty():
    assert _parse_actions(_RAW_FIXTURE, company_symbol="NOPE") == []
