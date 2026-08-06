"""PETSc best practices quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class BestPracticesResponse(BaseModel):
    """Structured response for best practices evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    uses_command_line_options: bool
    uses_viewers: bool
    configurable: bool
    practices_followed: list[str] = []


class PETScBestPracticesQuality(Evaluator):
    """Evaluates adherence to PETSc best practices.
    
    Checks for:
    - Use of command-line options (-ksp_type, -pc_type, etc.)
    - PetscViewer usage for output
    - Runtime configurability
    - Proper use of PETSc logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        llm_model = config.get('llm_model', 'gpt-4o-mini') if config else 'gpt-4o-mini'
        llm_temp = config.get('llm_temperature', 0.3) if config else 0.3
        llm_api_base_url = config.get('llm_api_base_url') if config else None
        llm_max_tokens = config.get('llm_max_tokens') if config else None
        self.llm = LLMClient(model=llm_model, temperature=llm_temp, api_base_url=llm_api_base_url, max_tokens=llm_max_tokens)
    
    @property
    def name(self) -> str:
        return "petsc_best_practices"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    @property
    def evaluation_method(self) -> str:
        return f"llm_{self.llm.model}"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate PETSc best practices."""
        start_time = time.time()
        
        prompt = f"""Evaluate PETSc best practices in this code.

Code:
```c
{code[:2000]}
```

Check for PETSc best practices:
1. Command-line options: Uses -ksp_type, -pc_type, etc.?
2. PetscViewer: Uses viewers for output?
3. Configurability: Can be configured at runtime?
4. PetscOptionsSetValue or similar for flexibility?
5. Proper use of PETSc data structures?

Provide:
- score: Best practices adherence (0-10)
- confidence: Assessment confidence (0-1)
- feedback: Explanation
- uses_command_line_options: Supports runtime options? (true/false)
- uses_viewers: Uses PetscViewer? (true/false)
- configurable: Runtime configurable? (true/false)
- practices_followed: List of best practices observed

Return as JSON.
"""
        
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=BestPracticesResponse
            )
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=response.score >= 7.0,
                quality_score=response.score / 10.0,
                confidence=response.confidence,
                feedback=response.feedback,
                metadata={
                    'raw_score': response.score,
                    'uses_command_line_options': response.uses_command_line_options,
                    'uses_viewers': response.uses_viewers,
                    'configurable': response.configurable,
                    'practices_followed': response.practices_followed,
                },
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,
                quality_score=None,
                confidence=0.0,
                feedback=f"LLM evaluation failed: {str(e)}",
                metadata={'error': str(e)},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )