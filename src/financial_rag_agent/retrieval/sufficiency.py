from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel import select

from financial_rag_agent.core import Company, Filing, get_session
from financial_rag_agent.core.config import get_settings
from financial_rag_agent.retrieval.service import search
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk


@dataclass
class SufficiencyResult:
    sufficient: bool
    reason: str
    best_score: float | None = None
    results: list[RetrievedChunk] = field(default_factory=list)


def _resolve_filing_id(company_name: str | None, cik: str | None) -> tuple[UUID | None, str | None]:
    """Returns (filing_id, insufficiency_reason).

    No company given at all -> (None, None): caller runs an unscoped,
    whole-KB check. A company given but not usably in the KB yet ->
    (None, reason): caller should stop here, before ever computing a
    score - see check_retrieval_sufficiency's docstring for why.
    """
    if not company_name and not cik:
        return None, None

    with get_session() as session:
        if cik:
            company = session.exec(select(Company).where(Company.cik == cik)).first()
        else:
            company = session.exec(select(Company).where(Company.name == company_name)).first()

        if company is None:
            return None, f"{company_name or cik} is not in the knowledge base"

        filing = session.exec(
            select(Filing)
            .where(Filing.company_id == company.id, Filing.ingestion_status == "complete")
            .order_by(Filing.created_at.desc())
        ).first()

        if filing is None:
            return None, f"{company_name or cik} is known but has no fully-ingested document yet"

        return filing.id, None


def _best_citation_score(results: list[RetrievedChunk]) -> float:
    scores = [cs.score for r in results for cs in r.citation_sentences]
    return max(scores, default=0.0)


def check_retrieval_sufficiency(
    query: str, company_name: str | None = None, cik: str | None = None, k: int = 5
) -> SufficiencyResult:
    """Decides whether the persistent KB already has good enough evidence
    for `query`, or whether runtime document discovery should run instead.

    Company/filing existence is checked FIRST, before any score, and is
    the real defense against a wrong-company answer - not the score
    threshold. Measured live: an off-topic query against the *right*
    company's filing scored 0.455 (real cosine similarity between the
    query and the best-matching sentence, via retrieval/citations.py).
    A query about a company entirely absent from the KB, searched
    unscoped, still scored 0.736 - shared financial vocabulary alone
    drives embedding similarity up regardless of which company it's
    actually about. A score threshold cannot reliably tell "right
    company, off-topic" apart from "wrong company"; only checking that
    the company actually has a fully-ingested filing can. That check
    happens here, before search() is ever called with company context.

    Once scoped to the right company's filing, retrieval_sufficiency_
    min_score (settings) is the topic-relevance signal, calibrated
    against the same two real measurements (0.834 on-topic vs. 0.455
    off-topic-same-company).

    An unscoped check (no company given) still runs when the caller has
    no company context at all, but inherits the limitation above -
    documented, not hidden.
    """
    filing_id, reason = _resolve_filing_id(company_name, cik)
    if reason is not None:
        return SufficiencyResult(sufficient=False, reason=reason)

    results = search(query, k=k, filing_id=filing_id)
    if not results:
        return SufficiencyResult(sufficient=False, reason="no retrieval results")

    best_score = _best_citation_score(results)
    threshold = get_settings().retrieval_sufficiency_min_score
    sufficient = best_score >= threshold
    verdict = "meets" if sufficient else "is below"
    return SufficiencyResult(
        sufficient=sufficient,
        reason=f"best supporting-sentence similarity {best_score:.3f} {verdict} threshold {threshold:.3f}",
        best_score=best_score,
        results=results,
    )
