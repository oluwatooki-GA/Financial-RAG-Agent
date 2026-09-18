import _bootstrap  # noqa: F401

from financial_rag_agent.documents.service import DiscoveryError, discover_and_register_document

COMPANY = "Zenith Bank"  # a real NGX-listed company we haven't discovered yet this session

if __name__ == "__main__":
    print(f"Running the full Phase 4 discovery pipeline for: {COMPANY!r}")
    print("Stages: discover_documents() -> download_pdf() -> hash -> get_or_register_document()")
    print("(NGX itself has no working discovery source yet, so this will fall through to a")
    print(" real web search -- see discovery/ngx_source.py for why.)\n")

    try:
        document, created = discover_and_register_document(COMPANY)
    except DiscoveryError as exc:
        raise SystemExit(f"DiscoveryError: {exc}")

    print(f"created={created}")
    print(f"  source_type:      {document.source_type}")
    print(f"  source_url:       {document.source_url}")
    print(f"  file_hash:        {document.file_hash}")
    print(f"  ingestion_status: {document.ingestion_status}  (not 'complete' -- no PDF chunker built yet)")
    print(f"  local_raw_path:   {document.local_raw_path}")
    print("\nRun this script again: it should print created=False with the SAME id --")
    print("that's real content-hash dedup, not a fresh download every time.")
