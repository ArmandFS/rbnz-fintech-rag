import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval import retrieve_chunks


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_collection: str


EVAL_CASES: list[EvalCase] = [
    EvalCase("What is the Official Cash Rate outlook?", "mps"),
    EvalCase("What does the RBNZ say about inflation expectations?", "mps"),
    EvalCase("How is the export sector performing, particularly dairy and meat prices?", "mps"),
    EvalCase("Is the economic recovery broadening?", "mps"),
    EvalCase("What financial stability risks does the RBNZ identify?", "financial_stability"),
    EvalCase("How resilient is the financial system amid heightened global risk?", "financial_stability"),
    EvalCase("What are the loan-to-value ratio restrictions for housing?", "financial_stability"),
    EvalCase("How do domestic regulators coordinate with Trans-Tasman counterparts on risk assessment?", "financial_stability"),
    EvalCase("What has the Reserve Bank done to address cyber-attacks?", "annual_report"),
    EvalCase("What is the Reserve Bank's Future of Money work programme?", "annual_report"),
    EvalCase("What refurbishment work is happening at the Reserve Bank's Wellington premises?", "annual_report"),
]


def evaluate(cases: list[EvalCase], *, top_k: int) -> list[dict]:
    results = []
    for case in cases:
        chunks = retrieve_chunks(case.query, top_k=top_k)
        collections = [chunk["collection"] for chunk in chunks]
        hit_rank = next(
            (rank for rank, collection in enumerate(collections, start=1) if collection == case.expected_collection),
            None,
        )
        results.append(
            {
                "query": case.query,
                "expected_collection": case.expected_collection,
                "hit_rank": hit_rank,
                "top_collection": collections[0] if collections else None,
                "top_similarity": chunks[0]["similarity"] if chunks else None,
            }
        )
    return results


def print_report(results: list[dict]) -> None:
    hits = sum(1 for result in results if result["hit_rank"] is not None)
    top1_hits = sum(1 for result in results if result["hit_rank"] == 1)

    for result in results:
        status = f"hit @ rank {result['hit_rank']}" if result["hit_rank"] else "MISS"
        print(f"[{status}] {result['query']}")
        print(
            f"    expected: {result['expected_collection']} | "
            f"top result: {result['top_collection']} "
            f"(similarity {result['top_similarity']:.3f})"
            if result["top_similarity"] is not None
            else f"    expected: {result['expected_collection']} | top result: none"
        )

    total = len(results)
    print(f"\nHit@top_k: {hits}/{total}")
    print(f"Hit@1: {top1_hits}/{total}")


def main() -> None:
    results = evaluate(EVAL_CASES, top_k=5)
    print_report(results)


if __name__ == "__main__":
    main()