"""Solver choice quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class SolverChoiceResponse(BaseModel):
    """Structured response for solver choice evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    solver_identified: str  # e.g., "GMRES", "CG", "Newton"
    appropriate_for_problem: bool
    suggestions: list[str] = []


class SolverChoiceQuality(Evaluator):
    """Evaluates whether the PETSc solver choice is appropriate.
    
    This evaluator uses LLM to assess:
    - Is the KSP/SNES/TS solver type appropriate?
    - Does it match the problem characteristics?
    - Are there better solver options?
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
        return "solver_choice"
    
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
        """Evaluate solver choice.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification
            execution_result: Optional execution results
        
        Returns:
            EvaluationResult with quality_score (0-1)
        """
        start_time = time.time()
        
        problem_desc = problem.get('problem_description', '')
        
        prompt = f"""Evaluate the PETSc solver choice for this problem.

Problem:
{problem_desc}

Generated Code:
```c
{code[:2000]}
```

Assess the solver selection:
1. What solver type is being used? (KSP, SNES, TS, etc.)
2. Is this solver appropriate for the problem?
3. For linear problems: Is the KSP type suitable (GMRES, CG, etc.)?
4. For nonlinear: Is SNES appropriate?
5. Are there better alternatives?

Provide:
- score: Solver choice quality (0-10)
- confidence: Assessment confidence (0-1)
- feedback: Explanation
- solver_identified: Main solver type identified in code
- appropriate_for_problem: Is this solver suitable? (true/false)
- suggestions: A list of better solver options (if any)

Return as JSON.
"""
        
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=SolverChoiceResponse
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
                    'solver_identified': response.solver_identified,
                    'appropriate_for_problem': response.appropriate_for_problem,
                    'suggestions': response.suggestions,
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