from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commurenew_agent.evaluation import evaluate_result_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scheme results with resident personas")
    parser.add_argument("--result", required=True, help="Path to output/result_*.json")
    parser.add_argument("--survey-csv", default=None, help="Path to inputs/serveyData.csv (optional)")
    parser.add_argument("--workers", type=int, default=10, help="Thread workers (default: 10)")
    parser.add_argument("--model", default="gpt-5.2", help="Evaluation model")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    csv_path = evaluate_result_json(
        result_json_path=Path(args.result),
        survey_csv_path=args.survey_csv,
        output_dir=Path(args.output_dir),
        model=args.model,
        max_workers=args.workers,
    )
    print(f"Saved evaluation summary: {csv_path}")


if __name__ == "__main__":
    main()
