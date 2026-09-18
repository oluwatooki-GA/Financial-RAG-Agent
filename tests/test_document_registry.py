import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from financial_rag_agent.core.models import Company, Filing
from financial_rag_agent.documents.hashing import compute_file_hash
from financial_rag_agent.documents.registry import find_document_by_hash, get_or_register_document


@pytest.fixture
def session():
    # Real ORM/constraint behavior (unique file_hash, FK, etc.) without
    # touching the actual Postgres dev DB — Filing/Company have no
    # pgvector-specific columns, so SQLite is a faithful stand-in here.
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def company(session):
    company = Company(name="GTCO", country="Nigeria", exchange="NGX", currency="NGN")
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def test_compute_file_hash_matches_known_sha256():
    assert (
        compute_file_hash(b"hello world")
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_different_content_hashes_differently():
    assert compute_file_hash(b"document a") != compute_file_hash(b"document b")


def test_new_document_is_registered(session, company):
    document, created = get_or_register_document(
        session,
        company_id=company.id,
        file_hash="hash-a",
        source_url="https://ngxgroup.com/reports/gtco-2025.pdf",
        source_type="ngx",
        document_type="annual_report",
        primary_document_filename="gtco-2025.pdf",
        fiscal_year=2025,
    )

    assert created is True
    assert document.file_hash == "hash-a"
    assert document.company_id == company.id
    assert find_document_by_hash(session, "hash-a") is not None


def test_duplicate_hash_from_a_different_url_reuses_the_same_row(session, company):
    first, first_created = get_or_register_document(
        session,
        company_id=company.id,
        file_hash="shared-hash",
        source_url="https://ngxgroup.com/reports/gtco-2025.pdf",
        source_type="ngx",
        primary_document_filename="gtco-2025.pdf",
    )
    second, second_created = get_or_register_document(
        session,
        company_id=company.id,
        file_hash="shared-hash",
        source_url="https://gtco.com/investor-relations/gtco-2025.pdf",  # different URL, same bytes
        source_type="company_ir",
        primary_document_filename="gtco-2025-annual-report.pdf",
    )

    assert first_created is True
    assert second_created is False
    assert second.id == first.id
    # the second discovery must not have overwritten the first registration
    assert second.source_url == first.source_url

    all_documents = session.exec(select(Filing)).all()
    assert len(all_documents) == 1


def test_unrelated_documents_get_separate_rows(session, company):
    _, created_a = get_or_register_document(
        session,
        company_id=company.id,
        file_hash="hash-a",
        source_url="https://example.com/a.pdf",
        source_type="ngx",
        primary_document_filename="a.pdf",
    )
    _, created_b = get_or_register_document(
        session,
        company_id=company.id,
        file_hash="hash-b",
        source_url="https://example.com/b.pdf",
        source_type="ngx",
        primary_document_filename="b.pdf",
    )

    assert created_a is True
    assert created_b is True
    assert len(session.exec(select(Filing)).all()) == 2


def test_find_document_by_hash_returns_none_when_absent(session):
    assert find_document_by_hash(session, "no-such-hash") is None
