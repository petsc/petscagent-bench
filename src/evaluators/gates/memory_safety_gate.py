"""Memory safety gate - checks for memory leaks and errors."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class MemorySafetyGate(Evaluator):
    """Checks for memory safety issues using valgrind or similar tools.
    
    This gate checks for:
    - Memory leaks
    - Invalid memory access
    - Use of uninitialized values
    """
    
    @property
    def name(self) -> str:
        return "memory_safety"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.GATE
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Check memory safety.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification (not used for memory check)
            execution_result: Must contain a ``cases`` list whose entries
                may provide ``valgrind_output`` and ``stderr``
        
        Returns:
            EvaluationResult with passed=True if no memory issues found
        """
        start_time = time.time()
        
        if execution_result is None:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,  # Can't determine without execution
                feedback="No execution result provided - memory safety check skipped",
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        cases = execution_result.get('cases') or []
        if not cases:
            # Nothing ran, so there is no memory behaviour to judge. A None
            # verdict is neutral in aggregation rather than a free pass.
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,
                feedback="No executed cases - memory safety check skipped",
                metadata={'valgrind_available': False, 'test_cases': []},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        details = []
        for case in cases:
            output = case.get('valgrind_output')
            stderr = case.get('stderr', '')
            if output is None:
                # Documented fallback when the server cannot instrument.
                safe = not self._stderr_has_memory_issue(stderr)
                method = 'stderr'
            else:
                safe = self._parse_valgrind_output(output)
                method = 'valgrind'
            details.append({
                'index': case.get('index'),
                'passed': safe,
                'method': method,
            })

        memory_safe = all(case['passed'] for case in details)
        used_valgrind = all(case['method'] == 'valgrind' for case in details)
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=memory_safe,
            confidence=1.0 if used_valgrind else 0.7,
            feedback=(
                "All test cases passed memory-safety checks"
                if memory_safe else "Memory safety issues detected in one or more test cases"
            ),
            metadata={
                'valgrind_available': used_valgrind,
                'test_cases': details,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    @staticmethod
    def _stderr_has_memory_issue(stderr: str) -> bool:
        stderr = stderr.lower()
        return any(issue in stderr for issue in (
            'memory leak', 'invalid read', 'invalid write', 'segmentation fault'
        ))
    
    def _parse_valgrind_output(self, output: str) -> bool:
        """Parse valgrind output to determine if memory is safe.
        
        Args:
            output: Valgrind output text
        
        Returns:
            True if no memory issues, False otherwise
        """
        # Look for the summary line
        # Example: "ERROR SUMMARY: 0 errors from 0 contexts"
        if 'ERROR SUMMARY: 0 errors' in output:
            return True
        
        # Look for specific issues
        memory_issues = [
            'definitely lost',
            'indirectly lost',
            'possibly lost',
            'Invalid read',
            'Invalid write',
            'Use of uninitialised',
        ]
        
        for issue in memory_issues:
            if issue in output:
                return False
        
        # If we can't determine clearly, be conservative
        return 'All heap blocks were freed' in output or 'no leaks are possible' in output
