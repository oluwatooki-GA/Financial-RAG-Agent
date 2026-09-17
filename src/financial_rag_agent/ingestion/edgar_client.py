from dataclasses import dataclass
from pathlib import Path

import requests

from financial_rag_agent.core.config import get_settings

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{document}"
RAW_DATA_DIR = Path("data/raw")


@dataclass
class FilingRef:
    cik: str
    company_name: str
    ticker: str | None
    sic: str | None
    accession_number: str
    form_type: str
    filing_date: str
    period_of_report: str | None
    primary_document: str
    source_url: str


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_user_agent}


def get_latest_10k(cik: str) -> FilingRef:
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":
            accession_number = recent["accessionNumber"][i]
            accession_no_dashes = accession_number.replace("-", "")
            primary_document = recent["primaryDocument"][i]
            source_url = ARCHIVE_URL.format(
                cik_int=int(cik),
                accession_no_dashes=accession_no_dashes,
                document=primary_document,
            )
            tickers = data.get("tickers") or []
            return FilingRef(
                cik=cik,
                company_name=data["name"],
                ticker=tickers[0] if tickers else None,
                sic=data.get("sic"),
                accession_number=accession_number,
                form_type=form,
                filing_date=recent["filingDate"][i],
                period_of_report=recent["reportDate"][i] or None,
                primary_document=primary_document,
                source_url=source_url,
            )

    raise ValueError(f"No 10-K filing found for CIK {cik}")


def fetch_filing_html(filing: FilingRef) -> Path:
    dest_dir = RAW_DATA_DIR / filing.accession_number
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filing.primary_document

    if dest_path.exists():
        return dest_path

    resp = requests.get(filing.source_url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return dest_path
