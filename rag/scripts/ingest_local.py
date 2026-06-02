import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

#import the get_settings function to load configuration parameters, and the ingest_document function to perform the actual ingestion of the PDF document
from ingest import ingest_document

#ingest one local PDF document into the pgvector database, with optional parameters for source, collection, and title. This can be used to quickly add a new document to the RAG system without needing to set up a full ingestion pipeline.
def main() -> None:
    #set up command-line argument parsing to allow the user to specify the PDF file path and optional metadata parameters
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
