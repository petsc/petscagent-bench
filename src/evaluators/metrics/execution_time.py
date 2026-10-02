import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class ExecutionTimeMetric(Evaluator):
    """Measures execution time performance.
    
    This metric evaluates performance based on absolute execution time.
    Faster execution yields higher scores.
    """
    
    @property
    def name(self) -> str:
        return "execution_time"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.METRIC
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Measure execution time performance.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification
            execution_result: Must contain a ``cases`` list whose entries
                provide ``execution_time_sec``
        
        Returns:
            EvaluationResult with raw_value=time, normalized_score based on performance tiers
        """
        start_time = time.time()
        
        # Runtime is measured per case. Nothing to time means nothing ran,
        # which must not be read as an infinitely fast program.
        if execution_result is None or not execution_result.get('cases'):
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                raw_value=None,
                normalized_score=0.0,
                confidence=1.0,
                feedback="No execution time available",
                metadata={},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Performance tiers (configurable)
        excellent_time = self.config.get('excellent_time_sec', 1.0) if self.config else 1.0
        good_time = self.config.get('good_time_sec', 5.0) if self.config else 5.0
        acceptable_time = self.config.get('acceptable_time_sec', 15.0) if self.config else 15.0
        max_time = self.config.get('max_time_sec', 60.0) if self.config else 60.0
        
        details = []
        for case in execution_result['cases']:
            runtime = case.get('execution_time_sec')
            score, tier = self._score_runtime(
                runtime, excellent_time, good_time, acceptable_time, max_time
            )
            details.append({
                'index': case.get('index', len(details)),
                'runtime_sec': runtime,
                'score': score,
                'performance_tier': tier,
                'passed': runtime is not None and runtime <= max_time,
            })

        normalized_score = sum(case['score'] for case in details) / len(details)
        passed = all(case['passed'] for case in details)
        # Total wall time is the one aggregate that means something across
        # cases of different sizes. A mean over them does not.
        runtimes = [c['runtime_sec'] for c in details if c['runtime_sec'] is not None]
        actual_time = sum(runtimes) if runtimes else None
        performance_tier = min(details, key=lambda case: case['score'])['performance_tier']
        feedback = (
            f"Mean per-case performance score: {normalized_score:.3f} "
            f"across {len(details)} case(s)"
        )

        print(feedback)
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=passed,
            raw_value=actual_time,
            normalized_score=normalized_score,
            confidence=1.0,
            feedback=feedback,
            metadata={
                'actual_time_sec': actual_time,
                'performance_tier': performance_tier,
                'excellent_time_sec': excellent_time,
                'good_time_sec': good_time,
                'acceptable_time_sec': acceptable_time,
                'max_time_sec': max_time,
                'test_cases': details,
                'memory_mb': execution_result.get('memory_mb'),
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )

    @staticmethod
    def _score_runtime(actual_time, excellent, good, acceptable, maximum):
        """Return a normalized score and tier for one invocation."""
        if actual_time is None:
            return 0.0, "unavailable"
        if actual_time <= excellent:
            return 1.0, "excellent"
        if actual_time <= good:
            return 0.8 + 0.2 * (good - actual_time) / (good - excellent), "good"
        if actual_time <= acceptable:
            return 0.6 + 0.2 * (acceptable - actual_time) / (acceptable - good), "acceptable"
        if actual_time <= maximum:
            return 0.2 + 0.4 * (maximum - actual_time) / (maximum - acceptable), "poor"
        return max(0.0, 0.2 * maximum / actual_time), "very poor"
