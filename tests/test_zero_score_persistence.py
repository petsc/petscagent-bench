"""Regression tests for preserving legitimate zero evaluator scores."""

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluators.base import EvaluationResult, EvaluatorType
from src.metrics.aggregation import MetricsAggregator


class DetailedBreakdownTests(unittest.TestCase):
    def test_quality_zero_is_not_replaced_by_missing_normalized_score(self):
        result = EvaluationResult(
            evaluator_name="error_handling",
            evaluator_type=EvaluatorType.QUALITY,
            quality_score=0.0,
            confidence=1.0,
        )

        breakdown = MetricsAggregator().get_detailed_breakdown([result])

        self.assertEqual(breakdown["quality"][0]["score"], 0.0)

    def test_normalized_zero_is_not_replaced_by_raw_value(self):
        result = EvaluationResult(
            evaluator_name="execution_time",
            evaluator_type=EvaluatorType.METRIC,
            raw_value=12.5,
            normalized_score=0.0,
            confidence=1.0,
        )

        breakdown = MetricsAggregator().get_detailed_breakdown([result])

        self.assertEqual(breakdown["metrics"][0]["score"], 0.0)


class StoredEvaluationDetailsTests(unittest.TestCase):
    def test_green_agent_persists_zero_quality_and_metric_scores(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        class Pipeline:
            async def evaluate(self, code, problem, execution_result):
                return [
                    EvaluationResult(
                        evaluator_name="error_handling",
                        evaluator_type=EvaluatorType.QUALITY,
                        passed=False,
                        quality_score=0.0,
                        confidence=1.0,
                    ),
                    EvaluationResult(
                        evaluator_name="execution_time",
                        evaluator_type=EvaluatorType.METRIC,
                        passed=False,
                        raw_value=12.5,
                        normalized_score=0.0,
                        confidence=1.0,
                    ),
                ]

        agent = Agent.__new__(Agent)
        agent.evaluation_pipeline = Pipeline()
        agent.metrics_aggregator = MetricsAggregator()

        benchmark = BenchmarkResult(
            problem_name="zero-score",
            problem_id="zero-score",
            runs=True,
            compiles=True,
        )

        asyncio.run(
            agent._evaluate_code(
                benchmark,
                {},
                [{"server_name": "zero.c", "source": "int main(void) { return 0; }"}],
            )
        )

        scores = {row["name"]: row["score"] for row in benchmark.evaluation_details}
        self.assertEqual(scores["error_handling"], 0.0)
        self.assertEqual(scores["execution_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
