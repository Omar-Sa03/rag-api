"""
RAG API Evaluation Pipeline
============================
MLOps evaluation script that benchmarks the RAG API against a golden Q&A
dataset (eval_dataset.json). Measures keyword coverage, answer quality, and
latency — then writes a structured JSON report to evaluation/reports/.

Usage:
    python evaluation/eval_pipeline.py [--host HOST] [--port PORT] [--mode MODE]

Examples:
    python evaluation/eval_pipeline.py
    python evaluation/eval_pipeline.py --host localhost --port 8000 --mode hybrid
    python evaluation/eval_pipeline.py --report-dir evaluation/reports
"""

import argparse
import json
import sys
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
DEFAULT_REPORT_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def keyword_coverage_score(answer: str, expected_keywords: list[str]) -> float:
    """
    Fraction of expected keywords present in the answer (case-insensitive).
    Returns a float in [0.0, 1.0].
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(matched / len(expected_keywords), 4)


def answer_length_score(answer: str) -> float:
    """
    Penalises very short or empty answers. Returns 0.0–1.0.
    Answers >= 50 words score 1.0; proportional below that.
    """
    words = len(answer.split())
    return min(1.0, round(words / 50, 4))


def is_answer_grounded(answer: str) -> bool:
    """
    Heuristic: answers that admit no knowledge are NOT grounded.
    """
    low_confidence_phrases = [
        "no relevant context",
        "i don't know",
        "i do not know",
        "cannot find",
        "not found in",
        "no information",
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in low_confidence_phrases)


def composite_score(keyword_score: float, length_score: float, grounded: bool) -> float:
    """Weighted composite of individual scores."""
    grounded_bonus = 1.0 if grounded else 0.0
    return round(
        0.60 * keyword_score +   # Primary: does the answer cover expected concepts?
        0.20 * length_score +    # Secondary: is it a substantive answer?
        0.20 * grounded_bonus,   # Tertiary: does the LLM claim to have context?
        4,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def query_api(
    base_url: str,
    question: str,
    mode: str = "hybrid",
    n_results: int = 5,
    rerank: bool = True,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    POST a query to the RAG API and return the parsed response dict.
    Raises on HTTP errors or connection issues.
    """
    payload = {
        "q": question,
        "mode": mode,
        "n_results": n_results,
        "rerank": rerank,
        "include_scores": True,
    }
    response = requests.post(
        f"{base_url}/v1/query",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    base_url: str,
    mode: str,
    n_results: int,
    rerank: bool,
    report_dir: Path,
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    """
    Run the full evaluation pipeline and return the report dict.
    """
    print(f"\n{'='*60}")
    print("  RAG API — MLOps Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"  Target  : {base_url}")
    print(f"  Mode    : {mode}  |  n_results={n_results}  |  rerank={rerank}")
    print(f"  Dataset : {dataset_path}")
    print(f"{'='*60}\n")

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    latencies: list[float] = []
    errors: list[str] = []

    for i, item in enumerate(dataset, start=1):
        qid = item["id"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        domain = item.get("domain", "unknown")
        difficulty = item.get("difficulty", "unknown")

        print(f"[{i:02d}/{len(dataset):02d}] {qid}  ({domain} / {difficulty})")
        print(f"       Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        try:
            t0 = time.perf_counter()
            api_response = query_api(base_url, question, mode=mode, n_results=n_results, rerank=rerank)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            answer = api_response.get("answer", "")
            total_results = api_response.get("total_results", 0)

            # Score
            kw_score = keyword_coverage_score(answer, expected_keywords)
            len_score = answer_length_score(answer)
            grounded = is_answer_grounded(answer)
            c_score = composite_score(kw_score, len_score, grounded)

            latencies.append(latency_ms)

            print(f"       ✓ latency={latency_ms}ms  |  composite={c_score:.2f}  |  "
                  f"keywords={kw_score:.2f}  |  grounded={grounded}  |  sources={total_results}")

            results.append({
                "id": qid,
                "domain": domain,
                "difficulty": difficulty,
                "question": question,
                "answer_snippet": answer[:200] + ("..." if len(answer) > 200 else ""),
                "total_sources_returned": total_results,
                "scores": {
                    "keyword_coverage": kw_score,
                    "answer_length": len_score,
                    "grounded": grounded,
                    "composite": c_score,
                },
                "latency_ms": latency_ms,
                "error": None,
            })

        except requests.exceptions.ConnectionError:
            msg = f"Connection refused — is the API running at {base_url}?"
            print(f"       ✗ ERROR: {msg}")
            errors.append(f"{qid}: {msg}")
            results.append({
                "id": qid, "domain": domain, "difficulty": difficulty,
                "question": question, "error": msg,
                "scores": {"composite": 0.0}, "latency_ms": None,
            })

        except Exception as exc:
            msg = str(exc)
            print(f"       ✗ ERROR: {msg}")
            errors.append(f"{qid}: {msg}")
            results.append({
                "id": qid, "domain": domain, "difficulty": difficulty,
                "question": question, "error": msg,
                "scores": {"composite": 0.0}, "latency_ms": None,
            })

        print()

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    successful = [r for r in results if r.get("error") is None]
    composite_scores = [r["scores"]["composite"] for r in successful]
    kw_scores = [r["scores"]["keyword_coverage"] for r in successful]

    # Per-domain averages
    domain_scores: dict[str, list[float]] = {}
    for r in successful:
        domain_scores.setdefault(r["domain"], []).append(r["scores"]["composite"])
    domain_avg = {d: round(statistics.mean(v), 4) for d, v in domain_scores.items()}

    # Per-difficulty averages
    diff_scores: dict[str, list[float]] = {}
    for r in successful:
        diff_scores.setdefault(r["difficulty"], []).append(r["scores"]["composite"])
    difficulty_avg = {d: round(statistics.mean(v), 4) for d, v in diff_scores.items()}

    summary = {
        "total_questions": len(dataset),
        "successful": len(successful),
        "failed": len(errors),
        "avg_composite_score": round(statistics.mean(composite_scores), 4) if composite_scores else 0.0,
        "avg_keyword_coverage": round(statistics.mean(kw_scores), 4) if kw_scores else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20
                         else max(latencies), 2) if latencies else None,
            "min": round(min(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "by_domain": domain_avg,
        "by_difficulty": difficulty_avg,
    }

    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": base_url,
            "mode": mode,
            "n_results": n_results,
            "rerank": rerank,
            "dataset": str(dataset_path),
        },
        "summary": summary,
        "errors": errors,
        "results": results,
    }

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_{report['run_id']}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions   : {summary['total_questions']}  (✓ {summary['successful']}  ✗ {summary['failed']})")
    print(f"  Composite   : {summary['avg_composite_score']:.2%}")
    print(f"  Keywords    : {summary['avg_keyword_coverage']:.2%}")
    lm = summary["latency_ms"]
    if lm["mean"]:
        print(f"  Latency     : mean={lm['mean']}ms  median={lm['median']}ms  max={lm['max']}ms")
    print()
    print("  By domain:")
    for domain, score in summary["by_domain"].items():
        bar = "█" * int(score * 20)
        print(f"    {domain:<22} {score:.2%}  {bar}")
    print()
    print("  By difficulty:")
    for diff, score in summary["by_difficulty"].items():
        bar = "█" * int(score * 20)
        print(f"    {diff:<10} {score:.2%}  {bar}")
    print()
    print(f"  Report saved → {report_path}")
    print(f"{'='*60}\n")

    return report


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG API MLOps Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", default=8000, type=int, help="API port (default: 8000)")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["vector", "bm25", "hybrid"],
        help="Search mode (default: hybrid)",
    )
    parser.add_argument("--n-results", default=5, type=int, help="n_results per query (default: 5)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory to write JSON reports (default: evaluation/reports/)",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to eval_dataset.json (default: evaluation/eval_dataset.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    report = run_evaluation(
        base_url=base_url,
        mode=args.mode,
        n_results=args.n_results,
        rerank=not args.no_rerank,
        report_dir=Path(args.report_dir),
        dataset_path=Path(args.dataset),
    )

    # Exit with non-zero if majority of questions failed
    failed_fraction = report["summary"]["failed"] / report["summary"]["total_questions"]
    sys.exit(1 if failed_fraction > 0.5 else 0)
