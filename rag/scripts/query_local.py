import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval import retrieve_and_answer, retrieve_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local RAG index.")
    parser.add_argument("query", help="Question to ask.")
    parser.add_argument("--collection", default=None, help="Optional collection filter.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument("--chunks-only", action="store_true", help="Print retrieved chunks without LLM answer.")
    args = parser.parse_args()

    if args.chunks_only:
        pprint(retrieve_chunks(args.query, collection=args.collection, top_k=args.top_k))
    else:
        pprint(retrieve_and_answer(args.query, collection=args.collection, top_k=args.top_k))


if __name__ == "__main__":
    main()
