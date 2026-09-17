import _bootstrap  # noqa: F401
from financial_rag_agent.ingestion.service import ingest_filing

NVIDIA_CIK = "0001045810"

if __name__ == "__main__":
    filing = ingest_filing(NVIDIA_CIK)
    print(f"Ingested filing {filing.accession_number}: {filing.chunk_count} chunks, status={filing.ingestion_status}")
