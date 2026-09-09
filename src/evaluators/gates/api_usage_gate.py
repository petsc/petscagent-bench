"""API usage gate - checks basic PETSc API requirements."""

import re
import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class APIUsageGate(Evaluator):
    """Checks for basic PETSc API requirements.
    
    Verifies that code includes:
    - PetscInitialize() call
    - PetscFinalize() call
    - Basic PETSc includes
    """
    
    @property
    def name(self) -> str:
        return "api_usage"
    
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
        """Check basic PETSc API usage.
        
        Args:
            code: The generated code to analyze
            problem: Problem specification (not used)
            execution_result: Not needed for static check
        
        Returns:
            EvaluationResult with passed=True if basic API requirements met
        """
        start_time = time.time()
        
        stripped = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        stripped = re.sub(r"//[^\n]*", "", stripped)
        init_match = re.search(r"\bPetscInitialize\s*\(", stripped)
        finalize_match = re.search(r"\bPetscFinalize\s*\(", stripped)

        checks = {
            'has_petsc_initialize': init_match is not None,
            'has_petsc_finalize': finalize_match is not None,
            'has_petsc_include': re.search(
                r'^\s*#\s*include\s*[<"]petsc[^>"]*\.h[>"]',
                stripped,
                flags=re.MULTILINE | re.IGNORECASE,
            ) is not None,
            'initialization_precedes_finalization': (
                init_match is not None
                and finalize_match is not None
                and init_match.start() < finalize_match.start()
            ),
        }
        
        all_passed = all(checks.values())
        passed_count = sum(checks.values())
        total_count = len(checks)
        
        # Generate feedback
        if all_passed:
            feedback = "All basic PETSc API requirements met"
        else:
            missing = [name.replace('has_', '').replace('_', ' ') for name, passed in checks.items() if not passed]
            feedback = f"Missing {len(missing)} requirement(s): {', '.join(missing)}"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=all_passed,
            confidence=1.0,  # Deterministic check
            feedback=feedback,
            metadata={
                **checks,
                'passed_count': passed_count,
                'total_count': total_count,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )
