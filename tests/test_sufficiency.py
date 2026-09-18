import pytest
from sqlmodel import Session, SQLModel, create_engine

from financial_rag_agent.core.models import Company, Filing
from financial_rag_agent.retrieval import sufficiency as sufficiency_module
from financial_rag_agent.retrieval.citations import CitationSentence
from financial_rag_agent.retrieval.sufficiency import _resolve_filing_id, check_retrieval_sufficiency
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk


@pytest.fixture
def session(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        monkeypatch.setattr(sufficiency_module, "get_session", lambda: Session(engine))
        yield s


def _make_filing(session, company, status: str) -> Filing:
    filing = Filing(
        company_id=company.id,
        source_url="https://example.com/report.pdf",
        primary_document_filename="report.pdf",
        ingestion_status=status,
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)
    return filing


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=__import__("uuid").uuid4(),
        score=0.5,
        text="some chunk text",
        item_label=None,
        item_heading=None,
        filing_accession_number="acc-1",
        citation_sentences=[CitationSentence(text="a sentence", char_start=0, char_end=10, score=score)],
    )


def test_resolve_filing_id_returns_none_none_when_no_company_given():
    assert _resolve_filing_id(None, None) == (None, None)


def test_resolve_filing_id_reports_unknown_company(session):
    filing_id, reason = _resolve_filing_id("Nonexistent Co", None)
    assert filing_id is None
    assert "not in the knowledge base" in reason


def test_resolve_filing_id_reports_company_with_no_complete_filing(session):
    company = Company(name="GTCO")
    session.add(company)
    session.commit()
    session.refresh(company)
    _make_filing(session, company, status="downloaded")

    filing_id, reason = _resolve_filing_id("GTCO", None)

    assert filing_id is None
    assert "no fully-ingested document" in reason


def test_resolve_filing_id_finds_complete_filing(session):
    company = Company(name="NVIDIA CORP", cik="0001045810")
    session.add(company)
    session.commit()
    session.refresh(company)
    complete_filing = _make_filing(session, company, status="complete")

    filing_id, reason = _resolve_filing_id(None, "0001045810")

    assert reason is None
    assert filing_id == complete_filing.id


def test_check_sufficiency_insufficient_when_company_not_in_kb(monkeypatch):
    monkeypatch.setattr(sufficiency_module, "_resolve_filing_id", lambda name, cik: (None, "GTCO is not in the knowledge base"))

    result = check_retrieval_sufficiency("What was GTCO's profit?", company_name="GTCO")

    assert result.sufficient is False
    assert "not in the knowledge base" in result.reason
    assert result.best_score is None


def test_check_sufficiency_insufficient_when_no_results(monkeypatch):
    monkeypatch.setattr(sufficiency_module, "_resolve_filing_id", lambda name, cik: (None, None))
    monkeypatch.setattr(sufficiency_module, "search", lambda query, k, filing_id: [])

    result = check_retrieval_sufficiency("some query")

    assert result.sufficient is False
    assert result.reason == "no retrieval results"


def test_check_sufficiency_sufficient_when_score_above_threshold(monkeypatch):
    monkeypatch.setattr(sufficiency_module, "_resolve_filing_id", lambda name, cik: (None, None))
    monkeypatch.setattr(sufficiency_module, "search", lambda query, k, filing_id: [_chunk(score=0.83)])

    result = check_retrieval_sufficiency("What was NVIDIA revenue?")

    assert result.sufficient is True
    assert result.best_score == 0.83


def test_check_sufficiency_insufficient_when_score_below_threshold(monkeypatch):
    monkeypatch.setattr(sufficiency_module, "_resolve_filing_id", lambda name, cik: (None, None))
    monkeypatch.setattr(sufficiency_module, "search", lambda query, k, filing_id: [_chunk(score=0.45)])

    result = check_retrieval_sufficiency("some off-topic query")

    assert result.sufficient is False
    assert result.best_score == 0.45
    assert "is below threshold" in result.reason
