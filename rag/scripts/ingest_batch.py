import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingest import ingest_document, infer_collection


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = RAG_DIR / "documents" / "rbnz" / "manifest.json"
DEFAULT_DOCUMENTS_DIR = RAG_DIR / "documents"


@dataclass(frozen=True)
class IngestJob:
    file_path: Path
    collection: str
    title: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest local RBNZ PDFs into pgvector.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Scraper manifest path. Used first if it exists.",
    )
    parser.add_argument(
        "--documents-dir",
        default=str(DEFAULT_DOCUMENTS_DIR),
        help="Fallback directory to scan recursively for PDFs.",
    )
    parser.add_argument("--collection", default=None, help="Optional collection filter.")
    parser.add_argument("--source", default="rbnz", help="Document source name.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of PDFs to ingest.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned ingest jobs without indexing.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    documents_dir = Path(args.documents_dir)
    jobs = discover_ingest_jobs(
        manifest_path=manifest_path,
        documents_dir=documents_dir,
        collection_filter=args.collection,
    )

    if args.limit is not None:
        jobs = jobs[: args.limit]

    if not jobs:
        print("No PDFs found to ingest.")
        return

    print(f"Found {len(jobs)} PDF(s) to ingest.")
    for job in jobs:
        print(f"- {job.collection} | {job.title or job.file_path.stem} | {job.file_path}")

    if args.dry_run:
        print("Dry run complete. No documents were indexed.")
        return

    results = []
    for job in jobs:
        result = ingest_document(
            job.file_path,
            source=args.source,
            collection=job.collection,
            title=job.title,
        )
        results.append(result)
        print(result)

    ingested = sum(1 for result in results if result.get("status") == "ingested")
    skipped = sum(1 for result in results if result.get("status") == "skipped")
    print(f"Batch ingest complete. Ingested: {ingested}. Skipped: {skipped}.")


def discover_ingest_jobs(
    *,
    manifest_path: Path,
    documents_dir: Path,
    collection_filter: str | None,
) -> list[IngestJob]:
    if manifest_path.exists():
        return jobs_from_manifest(manifest_path, collection_filter=collection_filter)
    return jobs_from_directory(documents_dir, collection_filter=collection_filter)


def jobs_from_manifest(manifest_path: Path, *, collection_filter: str | None) -> list[IngestJob]:
    with manifest_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    jobs = []
    for record in records:
        file_path = Path(record["file_path"])
        collection = record["collection"]
        if collection_filter and collection != collection_filter:
            continue
        if not file_path.exists():
            continue
        jobs.append(
            IngestJob(
                file_path=file_path,
                collection=collection,
                title=record.get("title"),
            )
        )

    return jobs


def jobs_from_directory(documents_dir: Path, *, collection_filter: str | None) -> list[IngestJob]:
    jobs = []
    for file_path in sorted(documents_dir.rglob("*.pdf")):
        collection = infer_collection(file_path)
        if collection_filter and collection != collection_filter:
            continue
        jobs.append(
            IngestJob(
                file_path=file_path,
                collection=collection,
                title=file_path.stem,
            )
        )

    return jobs


if __name__ == "__main__":
    main()
