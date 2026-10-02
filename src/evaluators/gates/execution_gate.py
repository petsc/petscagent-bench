"""Execution gate - checks if code runs without crashes."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class ExecutionGate(Evaluator):
    """Checks if code executes without runtime errors.
    
    This is a hard gate - code MUST run successfully to be considered valid.
    """
    
    @property
    def name(self) -> str:
        return "execution"
    
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
        """Check execution status.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification (not used for execution check)
            execution_result: Must contain a ``cases`` list whose entries
                provide ``runs`` and ``stderr``
        
        Returns:
            EvaluationResult with passed=True if executed successfully
        """
        start_time = time.time()
        
        if execution_result is None:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                feedback="No execution result provided",
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # A run is described by its cases. No cases means nothing was run,
        # which the compilation gate has already reported.
        cases = execution_result.get('cases') or []
        failed_cases = [case for case in cases if not case.get('runs', False)]
        runs = bool(cases) and not failed_cases
        stderr = "\n".join(
            f"case {case.get('index')}: {case.get('stderr', '')}"
            for case in failed_cases
        )

        # Check for common runtime issues
        has_segfault = 'segmentation fault' in stderr.lower() or 'sigsegv' in stderr.lower()
        has_assertion = 'assertion' in stderr.lower()
        has_abort = 'abort' in stderr.lower()
        
        if runs:
            feedback = "Code executed successfully"
        elif not cases:
            feedback = "Executable was never run"
        else:
            error_indicators = []
            if has_segfault:
                error_indicators.append("segmentation fault")
            if has_assertion:
                error_indicators.append("assertion failure")
            if has_abort:
                error_indicators.append("aborted")
            
            if error_indicators:
                feedback = f"Execution failed: {', '.join(error_indicators)}"
            else:
                error_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
                feedback = f"Execution failed: {error_preview or 'Unknown error'}"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=runs,
            confidence=1.0,  # Deterministic check
            feedback=feedback,
            metadata={
                'stderr': stderr,
                'has_segfault': has_segfault,
                'has_assertion': has_assertion,
                'has_abort': has_abort,
                'failed_case_indices': [case.get('index') for case in failed_cases],
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )
