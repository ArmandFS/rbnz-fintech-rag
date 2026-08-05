import argparse
import sys
from pathlib import Path

# Add the rag/ package root to sys.path so this script can be run directly
# (e.g. `venv/bin/python scripts/ingest_local.py ...`) without installing the
# project as a package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingest import ingest_document


def main() -> None:
    # CLI wrapper around ingest_document(): reads one PDF, chunks it, embeds
    # each chunk, and stores it in pgvector. See ingest.py for the pipeline.
    parser = argparse.ArgumentParser(description="Ingest one local RBNZ PDF into pgvector.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    parser.add_argument("--source", default="rbnz", help="Document source name.")
    parser.add_argument("--collection", default=None, help="Collection, such as mps or financial_stability.")
    parser.add_argument("--title", default=None, help="Optional clean title for the document.")
    args = parser.parse_args()

    result = ingest_document(
        args.pdf_path,
        source=args.source,
        collection=args.collection,
        title=args.title,
    )
    print(result)


if __name__ == "__main__":
    main()
