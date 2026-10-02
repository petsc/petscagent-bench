"""Guards on the benchmark problem definitions in data/.

A missing reference value scores None for numerical accuracy, which
aggregation turns into 0.0 rather than excluding, so the problem silently
forfeits the full correctness weight.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = sorted((ROOT / "data").glob("*.json"))

COMPARISONS = {"match", "upper_bound"}


def _significant_digits(value: float) -> str:
    """The mantissa digits of a number, independent of how it is written."""
    return f"{abs(float(value)):.12e}".split("e")[0].replace(".", "").rstrip("0")


class BenchmarkDataTest(unittest.TestCase):
    def test_data_directory_is_not_empty(self):
        self.assertTrue(DATA, "no problem definitions found in data/")

    def test_every_test_case_has_a_reference_value(self):
        for path in DATA:
            problem = json.loads(path.read_text())
            test_cases = problem.get("test_cases")
            self.assertTrue(test_cases, f"{path.name} declares no test cases")
            for idx, case in enumerate(test_cases):
                with self.subTest(problem=path.name, case=idx):
                    expected = case.get("expected_output")
                    self.assertIsInstance(
                        expected, list,
                        f"{path.name} case {idx} expected_output must be a list",
                    )
                    self.assertTrue(
                        expected,
                        f"{path.name} case {idx} expected_output is empty",
                    )
                    for value in expected:
                        self.assertIsInstance(value, (int, float))

    def test_declared_comparisons_and_tolerances_are_valid(self):
        for path in DATA:
            problem = json.loads(path.read_text())
            for idx, case in enumerate(problem["test_cases"]):
                with self.subTest(problem=path.name, case=idx):
                    comparison = case.get("comparison", "match")
                    self.assertIn(comparison, COMPARISONS)
                    tolerance = case.get("tolerance", 1e-3)
                    self.assertIsInstance(tolerance, (int, float))
                    self.assertGreaterEqual(tolerance, 0.0)

    def test_descriptions_do_not_leak_their_own_reference_values(self):
        """``problem_description`` is the text sent to the agent under test,
        so a reference value quoted there can simply be printed back."""
        for path in DATA:
            problem = json.loads(path.read_text())
            # Dropping decimal points catches a value however it is written.
            description = problem["problem_description"].replace(".", "")
            for idx, case in enumerate(problem["test_cases"]):
                for value in case["expected_output"]:
                    digits = _significant_digits(value)
                    if len(digits) < 4:
                        continue  # too round to be distinctive
                    with self.subTest(problem=path.name, case=idx, value=value):
                        self.assertNotIn(
                            digits, description,
                            f"{path.name} quotes the reference value {value} in "
                            f"the problem description, which is sent to the agent",
                        )

if __name__ == "__main__":
    unittest.main()
