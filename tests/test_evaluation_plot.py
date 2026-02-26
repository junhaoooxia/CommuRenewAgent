from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import importlib.util

from commurenew_agent.evaluation import plot_eval_radar_from_csv


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
class EvaluationRadarPlotTest(unittest.TestCase):
    def test_plot_eval_radar_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "eval_result_20260225_230313.csv"
            with summary.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "result_file",
                        "scheme_index",
                        "scheme_name",
                        "respondent_count",
                        "建筑与居住条件_均分",
                        "交通与基础设施_均分",
                        "公共空间与绿化_均分",
                        "环境与管理设施_均分",
                        "综合满意度_均分",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "result_file": "result_x.json",
                        "scheme_index": 1,
                        "scheme_name": "方案A",
                        "respondent_count": 2,
                        "建筑与居住条件_均分": 4.0,
                        "交通与基础设施_均分": 3.5,
                        "公共空间与绿化_均分": 4.2,
                        "环境与管理设施_均分": 3.8,
                        "综合满意度_均分": 4.1,
                    }
                )
            out = plot_eval_radar_from_csv(summary, output_dir=root)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
