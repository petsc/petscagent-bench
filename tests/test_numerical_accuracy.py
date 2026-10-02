"""Tests for per-test-case numerical accuracy semantics."""

import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluators.metrics.numerical_accuracy import NumericalAccuracyMetric


def load(name):
    return json.loads((ROOT / "data" / name).read_text())


def score(problem, execution_result, config=None):
    metric = NumericalAccuracyMetric(config=config or {"tolerance": 1e-3})
    return asyncio.run(metric.evaluate("", problem, execution_result))


class UpperBoundTest(unittest.TestCase):
    def test_smaller_divergence_than_the_bound_still_scores_full_marks(self):
        problem = load("NS2D.json")
        # The first value is the one NS2D result in the archive.
        for output in ("1.403321903126198e-13", "1e-12", "0", "1.0e-9"):
            with self.subTest(output=output):
                result = score(problem, {"stdout": output})
                self.assertTrue(result.passed)
                self.assertEqual(result.normalized_score, 1.0)

    def test_exceeding_the_bound_fails(self):
        problem = load("NS2D.json")
        for output in ("1e-8", "-1e-8", "-1e300", "1e309"):
            with self.subTest(output=output):
                result = score(problem, {"stdout": output})
                self.assertFalse(result.passed)
                self.assertLess(result.normalized_score, 0.01)

    def test_a_solver_monitor_line_is_not_read_as_the_answer(self):
        # Three archived NS2D runs end here, and the leading 0 used to be
        # taken as a divergence of zero and scored full marks.
        result = score(load("NS2D.json"), {
            "stdout": "Solving\n    0 SNES Function norm 7.108533604056e-02"
        })
        self.assertFalse(result.passed)
        self.assertEqual(result.normalized_score, 0.0)


class MatchTest(unittest.TestCase):
    def test_scalar_reference_is_scored(self):
        problem = {"test_cases": [{"expected_output": 1.0}]}
        result = score(problem, {"stdout": "1.0"})
        self.assertTrue(result.passed)
        self.assertEqual(result.normalized_score, 1.0)

    def test_exact_answer_passes(self):
        result = score(load("Rosenbrock.json"), {"stdout": "1.0\n1.0"})
        self.assertTrue(result.passed)
        self.assertEqual(result.normalized_score, 1.0)

    def test_wrong_answer_fails(self):
        result = score(load("Rosenbrock.json"), {"stdout": "0.0\n0.0"})
        self.assertFalse(result.passed)

    def test_darcy_reference_matches_an_independent_solve(self):
        # 5.1910e-03 comes from two discretisations, a vertex centred finite
        # difference and a cell centred finite volume, which agree to 2e-5
        # relative at N = 1024.
        problem = load("Darcyflow.json")
        self.assertAlmostEqual(
            problem["test_cases"][0]["expected_output"][0], 5.1910e-03, places=7
        )
        # A coarse but correct solve reads slightly low and must still pass.
        self.assertTrue(score(problem, {"stdout": "5.1659e-03"}).passed)
        self.assertFalse(score(problem, {"stdout": "2.77e-03"}).passed)


class MissingReferenceTest(unittest.TestCase):
    def test_absent_reference_is_reported_as_not_applicable(self):
        problem = {"test_cases": [{"args": ""}]}
        result = score(problem, {"stdout": "1.0"})
        self.assertIsNone(result.normalized_score)
        self.assertIsNone(result.passed)

    def test_missing_output_scores_zero(self):
        result = score(load("Rosenbrock.json"), {})
        self.assertEqual(result.normalized_score, 0.0)
        self.assertFalse(result.passed)


class ConvergenceTest(unittest.TestCase):
    def _run(self, coarse, fine):
        return score(load("GradShafranov.json"), {
            "cases": [
                {"index": 0, "stdout": coarse},
                {"index": 1, "stdout": fine},
            ],
        })

    def test_reference_second_order_refinement_scores_as_normal_cases(self):
        result = self._run("9.007956e-06", "2.254066e-06")
        self.assertEqual(result.metadata["test_cases"][0]["score"], 1.0)
        self.assertEqual(result.metadata["test_cases"][1]["score"], 1.0)

    def test_first_order_refinement_is_penalised_by_the_fine_case(self):
        result = self._run("9.007956e-06", "4.503978e-06")
        self.assertFalse(result.metadata["test_cases"][1]["passed"])

    def test_a_constant_output_cannot_game_the_reference_errors(self):
        for constant in ("0.0", "1e-300"):
            with self.subTest(constant=constant):
                result = self._run(constant, constant)
                self.assertLess(result.normalized_score, 0.7)

    def test_unrun_cases_fail_instead_of_disappearing_from_the_score(self):
        # Only case 0 ran, so every other declared case must score zero.
        result = score(load("GradShafranov.json"), {"stdout": "9.007956e-06"})
        self.assertEqual(result.metadata["cases_scored"], 6)
        self.assertFalse(result.passed)
        self.assertLess(result.normalized_score, 0.2)

    def test_partial_per_case_results_fail_missing_cases(self):
        result = score(load("GradShafranov.json"), {
            "cases": [
                {"index": 0, "stdout": "9.007956e-06"},
                {"index": 1, "stdout": "2.254066e-06"},
            ],
        })
        self.assertFalse(result.passed)
        self.assertEqual(result.metadata["cases_scored"], 6)
        self.assertLess(result.normalized_score, 0.5)

if __name__ == "__main__":
    unittest.main()
