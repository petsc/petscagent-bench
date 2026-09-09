"""Deterministic PETSc error-handling quality evaluator."""

import re
import time
from typing import Any, Dict, Optional

from ...base import Evaluator, EvaluatorType, EvaluationResult


class ErrorHandlingQuality(Evaluator):
    """Evaluate PETSc error propagation, preferring the modern API.

    ``PetscCall`` and ``PetscCallMPI`` are the current PETSc conventions.
    ``CHKERRQ`` still propagates errors, but is legacy syntax.  The previous
    implementation counted only ``CHKERRQ`` and therefore penalized modern
    code while rewarding legacy code.
    """

    _CALL_RE = re.compile(
        r"\b((?:Petsc|Vec|Mat|KSP|SNES|TS|DM|Tao|PC|IS|AO|MPI_)[A-Za-z0-9_]*)\s*\("
    )
    _NON_API_CALLS = {
        "PetscCall", "PetscCallMPI", "PetscCallAbort", "PetscCallMPIAbort",
        "PetscFunctionBegin",
        "PetscFunctionBeginUser", "PetscFunctionReturn",
    }

    @classmethod
    def _counts(cls, code: str) -> tuple[int, int, int]:
        """Return estimated API calls, modern wrappers, and legacy wrappers."""
        # Remove comments so examples in comments do not affect the score.
        stripped = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        stripped = re.sub(r"//[^\n]*", "", stripped)
        names = [m.group(1) for m in cls._CALL_RE.finditer(stripped)]
        api_calls = sum(name not in cls._NON_API_CALLS for name in names)
        modern = len(re.findall(r"\bPetscCall(?:MPI)?(?:Abort)?\s*\(", stripped))
        legacy = len(re.findall(r"\bCHKERRQ\s*\(", stripped))
        return api_calls, modern, legacy
    
    @property
    def name(self) -> str:
        return "error_handling"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate error handling."""
        start_time = time.time()
        
        petsc_calls, petsc_call_count, chkerrq_count = self._counts(code)
        checked_calls = min(petsc_calls, petsc_call_count + chkerrq_count)

        if petsc_calls > 0:
            coverage = checked_calls / petsc_calls
            modern_fraction = (
                petsc_call_count / (petsc_call_count + chkerrq_count)
                if petsc_call_count + chkerrq_count else 0.0
            )
            # Complete modern coverage earns 1.0. Complete legacy coverage
            # earns 0.8: it is safe error propagation, but not current style.
            score = coverage * (0.8 + 0.2 * modern_fraction)
            if coverage >= 0.8 and modern_fraction >= 0.8:
                feedback = "Good modern PETSc error propagation"
            elif coverage >= 0.8:
                feedback = "Errors are propagated, primarily with legacy CHKERRQ"
            elif checked_calls:
                feedback = "Only some PETSc calls use an error-propagation wrapper"
            else:
                feedback = "PETSc calls do not use PetscCall/PetscCallMPI or CHKERRQ"
        else:
            score = 0.5
            feedback = "No PETSc calls detected"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=score >= 0.7,
            quality_score=score,
            confidence=1.0,  # Deterministic
            feedback=feedback,
            metadata={
                'petsc_call_count': petsc_call_count,
                'chkerrq_count': chkerrq_count,
                'petsc_calls_estimate': petsc_calls,
                'checked_call_ratio': checked_calls / petsc_calls if petsc_calls else 0.0,
                'modern_wrapper_fraction': (
                    petsc_call_count / (petsc_call_count + chkerrq_count)
                    if petsc_call_count + chkerrq_count else 0.0
                ),
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )
