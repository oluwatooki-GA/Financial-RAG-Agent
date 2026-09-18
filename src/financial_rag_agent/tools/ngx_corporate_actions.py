from dataclasses import dataclass

import requests

_BY_YEAR_URL = "https://ngxgroup.com/wp-json/corporate-actions/v1/by-year/{year}"

# ngxgroup.com serves Brotli-compressed responses; requests' urllib3 will
# silently hand back raw compressed bytes as "text" if a brotli decoder
# isn't installed, which looks like garbage, not an error. Restricting to
# gzip/deflate (which requests always supports) avoids that trap entirely.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (financial-rag-agent; +https://github.com)",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass
class NGXCorporateAction:
    company: str
    company_symbol: str
    year: int
    dividend_share: str
    bonus: str
    closure_of_register: str
    agm_date: str
    payment_date: str


def _parse_actions(raw: list[dict], company_symbol: str | None) -> list[NGXCorporateAction]:
    """Maps NGX's raw JSON records to NGXCorporateAction, optionally
    filtered to one ticker. Network-free so this logic is directly
    unit-testable against a fixture."""
    actions = [
        NGXCorporateAction(
            company=item["company"],
            company_symbol=item["company_symbol"],
            year=int(item["year"]),
            dividend_share=item["dividend_share"],
            bonus=item["bonus"],
            closure_of_register=item["closure_of_register"],
            agm_date=item["agm_date"],
            payment_date=item["payment_date"],
        )
        for item in raw
    ]
    if company_symbol:
        actions = [a for a in actions if a.company_symbol.upper() == company_symbol.upper()]
    return actions


def get_corporate_actions(year: int, company_symbol: str | None = None) -> list[NGXCorporateAction]:
    """Real NGX corporate-actions data (dividends, bonus issues, AGM and
    payment dates) for a given year, optionally filtered to one company
    by ticker symbol. Verified live against ngxgroup.com's public
    WordPress REST API (not a documented/versioned API like SEC's XBRL,
    but genuinely real structured JSON, not a scrape of rendered HTML) —
    confirmed real 2024 dividend data for GTCO, Custodian Investment,
    Seplat Energy, and others.

    This does NOT cover annual reports / full financial statements: NGX's
    corporate-disclosures page renders its document listing via
    client-side JavaScript from a source that couldn't be found by
    fetching its HTML directly (see discovery/ngx_source.py, still an
    honest NotImplementedError for that document-discovery case)."""
    resp = requests.get(_BY_YEAR_URL.format(year=year), headers=_HEADERS, timeout=20)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    return _parse_actions(resp.json(), company_symbol)
