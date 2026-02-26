from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commurenew_agent.evaluation import evaluate_result_json, export_scores_from_detail_json, plot_eval_radar_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scheme results with resident personas")
    parser.add_argument("--result", default="output/result_20260224_210002.json", help="Path to output/result_*.json")
    parser.add_argument("--survey-csv", default="inputs/serveyData.csv", help="Path to inputs/serveyData.csv (optional)")
    parser.add_argument("--workers", type=int, default=10, help="Thread workers (default: 10)")
    parser.add_argument("--model", default="gpt-5.2", help="Evaluation model")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--summary-csv", default="output/eval_result_20260226_140000.csv", help="Optional existing eval_result_*.csv to render radar directly")
    parser.add_argument("--detail-json", default="output/eval_detail_20260226_140000.json", help="Optional existing eval_detail_*.json to export detail+summary CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if args.summary_csv:
        radar = plot_eval_radar_from_csv(Path(args.summary_csv), output_dir=Path(args.output_dir))
        print(f"Saved evaluation radar chart: {radar}")
        return

    if args.detail_json:
        detail_csv, summary_csv = export_scores_from_detail_json(
            detail_json_path=Path(args.detail_json),
            output_dir=Path(args.output_dir),
        )
        print(f"Saved detailed scores csv: {detail_csv}")
        print(f"Saved evaluation summary: {summary_csv}")
        radar = plot_eval_radar_from_csv(Path(summary_csv), output_dir=Path(args.output_dir))
        print(f"Saved evaluation radar chart: {radar}")
        return

    csv_path = evaluate_result_json(
        result_json_path=Path(args.result),
        survey_csv_path=args.survey_csv,
        output_dir=Path(args.output_dir),
        model=args.model,
        max_workers=args.workers,
    )
    print(f"Saved evaluation summary: {csv_path}")
    radar = plot_eval_radar_from_csv(Path(csv_path), output_dir=Path(args.output_dir))
    print(f"Saved evaluation radar chart: {radar}")


if __name__ == "__main__":
    main()
