"""Tests for the rerunnable publication exporter."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicationExportTest(unittest.TestCase):
    def test_rerun_discovers_new_problem_and_ignores_partial_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "demo"
            out = root / "publication"
            create = subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "make_demo_runs.py"),
                 "--out", str(demo)], capture_output=True, text=True)
            self.assertEqual(create.returncode, 0, create.stderr)
            (demo / "still-writing.json").write_text('{"judge_model":')

            self._build(demo, out)
            manifests = sorted(out.glob("*/included_results.json"))
            self.assertTrue(manifests)
            before = json.loads(manifests[0].read_text())
            self.assertTrue((manifests[0].parent / "figure_problem_matrix.pdf").is_file())
            self.assertTrue((manifests[0].parent / "table_summary.pdf").is_file())
            self.assertTrue((manifests[0].parent / "table_evaluators.pdf").is_file())
            self.assertTrue((manifests[0].parent / "table_problems.pdf").is_file())
            with (manifests[0].parent / "table_summary.csv").open(newline="") as handle:
                columns = set(next(csv.DictReader(handle)).keys())
            self.assertTrue({
                "category_correctness", "category_performance", "category_code_quality",
                "category_algorithm", "category_petsc", "cost_usd_per_run",
                "category_correctness_sd", "category_performance_sd",
                "category_code_quality_sd", "category_algorithm_sd", "category_petsc_sd",
                "total_tokens_per_run", "model_calls_per_run", "tool_calls_per_run",
                "peak_context_tokens", "execution_time_sec",
            }.issubset(columns))
            with (manifests[0].parent / "table_evaluators.csv").open(newline="") as handle:
                evaluator_rows = list(csv.DictReader(handle))
            self.assertTrue(evaluator_rows)
            self.assertTrue({"evaluator", "evaluator_type", "mean",
                             "sd_across_judge_passes"}.issubset(evaluator_rows[0]))

            docs = sorted(demo.glob("*.json"))
            source = next(path for path in docs if path.name != "still-writing.json")
            doc = json.loads(source.read_text())
            extra = dict(doc["results"][0])
            extra["problem_name"] = "Incremental_problem"
            extra["problem_id"] = "incremental"
            doc["results"].append(extra)
            source.write_text(json.dumps(doc))

            self._build(demo, out)
            after = json.loads(manifests[0].read_text())
            self.assertGreaterEqual(after["n_problems"], before["n_problems"])
            self.assertIn("Incremental_problem", {
                problem
                for path in out.glob("*/included_results.json")
                for problem in json.loads(path.read_text())["problems"]
            })

    def _build(self, source: Path, out: Path) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "build_publication.py"),
             "--dir", str(source), "--out", str(out)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
