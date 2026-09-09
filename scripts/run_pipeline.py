"""Pipeline reproduction script.

Runs the complete benchmark suite, computes all metrics across 200 Golden Set examples,
and prints the publication-grade evaluation tables.
Reproducible in < 2 minutes.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, REPORTS_DIR
from src.evaluation.benchmark import BenchmarkRunner


def main():
    print("==================================================================")
    print("      HIVER SDE INTERN ASSIGNMENT - AI SUPPORT AGENT PIPELINE     ")
    print("==================================================================")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "headline_benchmark_results.json"

    runner = BenchmarkRunner()
    start_time = time.time()
    results = runner.run_benchmark(sample_limit=200)
    total_time = round(time.time() - start_time, 2)

    # Format markdown
    md_output = BenchmarkRunner.format_markdown_table(results)
    print("\n" + md_output + "\n")
    print(f"Total benchmark run time: {total_time}s")

    # Save to JSON
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Full benchmark results saved to: {report_file}")


if __name__ == "__main__":
    main()
