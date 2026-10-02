"""Numerical accuracy metric - measures error against reference solution."""

import time
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import Evaluator, EvaluatorType, EvaluationResult


_NUMERIC_LINE = re.compile(r'^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$')


def _has_reference(test_case: Dict[str, Any]) -> bool:
    """Whether a test case carries a usable reference value."""
    expected = test_case.get('expected_output')
    if isinstance(expected, (int, float, np.number)):
        return True
    return isinstance(expected, (list, tuple, np.ndarray)) and len(expected) > 0


class NumericalAccuracyMetric(Evaluator):
    """Measures numerical accuracy against reference solution.
    
    Each test case declares a ``comparison``, either ``match`` (the default,
    relative L2 distance to the reference) or ``upper_bound`` (the reference
    is a ceiling and only overshoot counts), and a ``tolerance`` that falls
    back to the configured default.
    """
    
    @property
    def name(self) -> str:
        return "numerical_accuracy"
    
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
        """Compute numerical accuracy across every test case with a reference.
        
        Args:
            code: The generated code (not used directly)
            problem: Must contain 'test_cases' with 'expected_output'
            execution_result: Must contain 'cases', one entry per test case,
                or a lone 'stdout' for a single-case caller
        
        Returns:
            EvaluationResult with raw_value=mean error and normalized_score
            the mean of the per-case scores
        """
        start_time = time.time()
        
        test_cases = problem.get('test_cases') or []
        if not test_cases:
            return self._no_reference(
                start_time, "No test cases available for comparison"
            )
        
        if not any(_has_reference(tc) for tc in test_cases):
            return self._no_reference(
                start_time, "No expected output in test cases"
            )
        
        outputs = self._case_outputs(execution_result, len(test_cases))
        if outputs is None:
            return self._no_output(start_time)

        default_tolerance = (
            self.config.get('tolerance', 1e-3) if self.config else 1e-3
        )

        scores: List[float] = []
        errors: List[float] = []
        details: List[Dict[str, Any]] = []
        for idx, test_case in enumerate(test_cases):
            if not _has_reference(test_case):
                continue

            comparison = test_case.get('comparison', 'match')
            tolerance = float(test_case.get('tolerance', default_tolerance))

            if outputs[idx] is None:
                scores.append(0.0)
                details.append({
                    'index': idx,
                    'comparison': comparison,
                    'tolerance': tolerance,
                    'passed': False,
                    'score': 0.0,
                    'error_message': 'No execution output available',
                })
                continue

            try:
                solution, reference = self._parse_output(
                    outputs[idx], test_case['expected_output']
                )
                error = self._error(solution, reference, comparison)
            except Exception as e:
                scores.append(0.0)
                details.append({
                    'index': idx,
                    'comparison': comparison,
                    'tolerance': tolerance,
                    'passed': False,
                    'score': 0.0,
                    'error_message': str(e),
                })
                continue

            score = self._score(error, tolerance)
            scores.append(score)
            errors.append(error)
            details.append({
                'index': idx,
                'comparison': comparison,
                'tolerance': tolerance,
                'error': error,
                'score': score,
                'passed': bool(error <= tolerance),
            })

        normalized_score = float(np.mean(scores))
        raw_value = float(np.mean(errors)) if errors else None
        passed = all(d['passed'] for d in details)

        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=passed,
            raw_value=raw_value,
            normalized_score=normalized_score,
            confidence=1.0,
            feedback=self._feedback(details, normalized_score),
            metadata={
                'test_cases': details,
                'cases_scored': len(details),
                'reference_available': True,
                'output_available': True,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )

    def _no_reference(self, start_time: float, feedback: str) -> EvaluationResult:
        """Result for a problem with nothing to compare against."""
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=None,
            raw_value=None,
            normalized_score=None,
            confidence=None,
            feedback=feedback,
            metadata={'reference_available': False},
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )

    def _no_output(self, start_time: float) -> EvaluationResult:
        """Result for a reference that has no program output to score."""
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=False,
            raw_value=None,
            normalized_score=0.0,
            confidence=1.0,
            feedback="No execution output available",
            metadata={'reference_available': True, 'output_available': False},
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )

    @staticmethod
    def _score(error: float, tolerance: float) -> float:
        """Full credit inside the tolerance, exponential decay outside it."""
        if error <= tolerance:
            return 1.0
        if tolerance <= 0.0:
            return float(np.exp(-error))
        return float(np.exp(-(error - tolerance) / tolerance))

    @staticmethod
    def _error(
        solution: np.ndarray, reference: np.ndarray, comparison: str
    ) -> float:
        """Error between a solution and its reference under one comparison."""
        if comparison == 'upper_bound':
            scale = np.maximum(np.abs(reference), 1e-300)
            with np.errstate(over='ignore'):
                overshoot = (np.abs(solution) - np.abs(reference)) / scale
            return float(max(0.0, np.max(overshoot)))

        if comparison != 'match':
            raise ValueError(f"Unknown comparison: {comparison}")

        error = np.linalg.norm(solution - reference)
        ref_norm = np.linalg.norm(reference)
        if ref_norm > 1e-14:  # Avoid division by zero
            return float(error / ref_norm)
        return float(error)

    @staticmethod
    def _feedback(
        details: List[Dict[str, Any]],
        score: float,
    ) -> str:
        """One line per scored test case."""
        parts = []
        for d in details:
            if 'error_message' in d:
                parts.append(f"case {d['index']}: {d['error_message']}")
            else:
                verdict = "ok" if d['passed'] else "off"
                parts.append(
                    f"case {d['index']} ({d['comparison']}): error = "
                    f"{d['error']:.2e}, tolerance {d['tolerance']:.2e} [{verdict}]"
                )
        return f"Accuracy {score:.3f}. " + "; ".join(parts)

    @staticmethod
    def _case_outputs(
        execution_result: Optional[Dict[str, Any]], n_cases: int
    ) -> Optional[List[Optional[str]]]:
        """Align program output with test cases, one entry per case.

        Uses ``cases`` when the caller has them. A lone ``stdout``, which is
        how single-case callers and tests supply output, goes to case 0.
        """
        if execution_result is None:
            return None

        per_case = execution_result.get('cases')
        if per_case is not None:
            outputs: List[Optional[str]] = [None] * n_cases
            for entry in per_case:
                idx = entry.get('index')
                if isinstance(idx, int) and 0 <= idx < n_cases:
                    outputs[idx] = entry.get('stdout')
            return outputs

        if 'stdout' not in execution_result:
            return None
        outputs = [None] * n_cases
        outputs[0] = execution_result['stdout']
        return outputs

    def _parse_output(
        self, stdout: str, expected_output: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract solution values from program output.
        
        Args:
            stdout: Program output containing solution (last N lines should
                contain N numbers)
            expected_output: Expected output (array of N numbers)
        
        Returns:
            (solution_values, reference_values)
        """
        # Convert expected_output to numpy array
        if isinstance(expected_output, (int, float, np.number)):
            reference_values = np.array([expected_output], dtype=float)
        elif isinstance(expected_output, (list, tuple, np.ndarray)):
            reference_values = np.asarray(expected_output, dtype=float)
        else:
            raise ValueError(f"Unsupported expected_output type: {type(expected_output)}")

        if not np.all(np.isfinite(reference_values)):
            raise ValueError("Expected output contains a non-finite value")
        
        # Every problem asks for lines holding nothing but the value, so
        # discard anything else. Without this a PETSc monitor line such as
        # "0 SNES Function norm 7.1e-02" is read as the answer 0.
        lines = [l for l in stdout.splitlines() if _NUMERIC_LINE.match(l)]
        n_expected = len(reference_values)
        last_n_lines = lines[-n_expected:] if len(lines) >= n_expected else lines
        
        solution_values = []
        for line in last_n_lines:
            # Extract numbers from each line
            numbers = self._extract_numbers(line)
            if len(numbers) > 0:
                # Take the first number from each line (or could take last, depending on format)
                solution_values.append(numbers[0])
        
        solution_values = np.array(solution_values, dtype=float)

        if not np.all(np.isfinite(solution_values)):
            raise ValueError("Program output contains a non-finite value")
        
        if len(solution_values) == 0:
            raise ValueError("No numerical values found in output")
        
        # Ensure we have the right number of values
        if len(solution_values) != len(reference_values):
            raise ValueError(
                f"Number of output values ({len(solution_values)}) does not match "
                f"expected output length ({len(reference_values)})"
            )
        
        return solution_values, reference_values
    
    def _extract_numbers(self, text: str) -> np.ndarray:
        """Extract floating point numbers from text.
        
        Args:
            text: Text containing numbers
        
        Returns:
            Array of extracted numbers
        """
        # Pattern to match floating point numbers (including scientific notation)
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, text)
        
        if not matches:
            return np.array([])
        
        return np.array([float(m) for m in matches])
