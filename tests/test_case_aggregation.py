"""Regression tests for case-first execution aggregation."""

import asyncio
import unittest

from src.evaluators.gates.execution_gate import ExecutionGate
from src.evaluators.gates.memory_safety_gate import MemorySafetyGate
from src.evaluators.metrics.execution_time import ExecutionTimeMetric


class CaseAggregationTests(unittest.TestCase):
    def test_single_canonical_case_passes_execution_and_runtime_checks(self):
        execution_result = {
            "compiles": True,
            "cases": [{
                "index": 0,
                "runs": True,
                "stdout": "1.0",
                "stderr": "",
                "execution_time_sec": 0.3,
                "valgrind_output": None,
            }],
        }

        execution = asyncio.run(
            ExecutionGate({}).evaluate("", {}, execution_result)
        )
        runtime = asyncio.run(
            ExecutionTimeMetric({}).evaluate("", {}, execution_result)
        )

        self.assertTrue(execution.passed)
        self.assertTrue(runtime.passed)
        self.assertEqual(runtime.raw_value, 0.3)

    def test_later_slow_case_affects_runtime_score(self):
        metric = ExecutionTimeMetric({
            "excellent_time_sec": 1.0,
            "good_time_sec": 2.0,
            "acceptable_time_sec": 4.0,
            "max_time_sec": 10.0,
        })
        result = asyncio.run(metric.evaluate("", {}, {
            "cases": [
                {"index": 0, "runs": True, "execution_time_sec": 0.1},
                {"index": 1, "runs": True, "execution_time_sec": 20.0},
            ],
        }))
        self.assertFalse(result.passed)
        self.assertLess(result.normalized_score, 0.6)
        self.assertAlmostEqual(result.metadata["actual_time_sec"], 20.1)

    def test_later_execution_failure_fails_gate(self):
        gate = ExecutionGate({})
        result = asyncio.run(gate.evaluate("", {}, {"cases": [
            {"index": 0, "runs": True, "stderr": ""},
            {"index": 1, "runs": False, "stderr": "aborted"},
        ]}))
        self.assertFalse(result.passed)
        self.assertEqual(result.metadata["failed_case_indices"], [1])

    def test_later_memory_failure_fails_gate(self):
        gate = MemorySafetyGate({})
        result = asyncio.run(gate.evaluate("", {}, {"cases": [
            {"index": 0, "valgrind_output": "ERROR SUMMARY: 0 errors"},
            {"index": 1, "valgrind_output": "Invalid read of size 8"},
        ]}))
        self.assertFalse(result.passed)
        self.assertTrue(result.metadata["test_cases"][0]["passed"])
        self.assertFalse(result.metadata["test_cases"][1]["passed"])


class NothingRanTests(unittest.TestCase):
    """Compilation failure leaves no cases. A program that never ran must
    not read as an infinitely fast one, which is what a scalar runtime
    defaulted to 0.0 used to buy it."""

    FAILED_COMPILE = {
        "compiles": False, "runs": False, "stdout": "", "stderr": "",
        "cases": [], "execution_time_sec": 0.0,
    }

    def test_runtime_scores_zero_rather_than_excellent(self):
        metric = ExecutionTimeMetric({
            "excellent_time_sec": 0.5, "good_time_sec": 1.0,
            "acceptable_time_sec": 2.0, "max_time_sec": 60.0,
        })
        result = asyncio.run(metric.evaluate("", {}, self.FAILED_COMPILE))
        self.assertEqual(result.normalized_score, 0.0)
        self.assertFalse(result.passed)

    def test_execution_gate_fails(self):
        result = asyncio.run(ExecutionGate({}).evaluate("", {}, self.FAILED_COMPILE))
        self.assertFalse(result.passed)

    def test_memory_gate_abstains_instead_of_passing(self):
        result = asyncio.run(MemorySafetyGate({}).evaluate("", {}, self.FAILED_COMPILE))
        self.assertIsNone(result.passed)


if __name__ == "__main__":
    unittest.main()
