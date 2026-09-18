from dataclasses import dataclass

import requests

from financial_rag_agent.core.config import get_settings

COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:0>10}/{taxonomy}/{concept}.json"


@dataclass
class XBRLFact:
    concept: str
    end_date: str
    value: float
    unit: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed_date: str


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_user_agent}


def _dedupe_facts(facts: list[dict], unit: str) -> list[XBRLFact]:
    """The same reporting period's value often appears multiple times
    across filings (once as the primary figure, again later as a prior-
    year comparative in the next filing). Keeps only the most recently
    filed entry for each end_date — network-free so this logic is
    directly unit-testable."""
    best_by_end_date: dict[str, dict] = {}
    for fact in facts:
        end_date = fact["end"]
        existing = best_by_end_date.get(end_date)
        if existing is None or fact["filed"] > existing["filed"]:
            best_by_end_date[end_date] = fact

    return [
        XBRLFact(
            concept=fact.get("concept", ""),
            end_date=fact["end"],
            value=fact["val"],
            unit=unit,
            fiscal_year=fact.get("fy"),
            fiscal_period=fact.get("fp"),
            form=fact.get("form", ""),
            filed_date=fact["filed"],
        )
        for fact in sorted(best_by_end_date.values(), key=lambda f: f["end"])
    ]


def get_company_concept(cik: str, concept: str, taxonomy: str = "us-gaap") -> list[XBRLFact]:
    """Real, exactly-as-reported structured figures for one XBRL concept
    (e.g. "Revenues", "NetIncomeLoss") straight from SEC's XBRL API — no
    HTML table parsing involved. Returns one fact per reporting period,
    deduped across filings, sorted oldest to newest."""
    url = COMPANY_CONCEPT_URL.format(cik=cik, taxonomy=taxonomy, concept=concept)
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results: list[XBRLFact] = []
    for unit, facts in data.get("units", {}).items():
        for fact in facts:
            fact.setdefault("concept", concept)
        results.extend(_dedupe_facts(facts, unit))

    return results
