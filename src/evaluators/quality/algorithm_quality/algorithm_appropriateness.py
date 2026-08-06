"""Algorithm appropriateness quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class AlgorithmResponse(BaseModel):
    """Structured response for algorithm evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    approach_suitable: bool
    better_alternatives: list[str] = []


class AlgorithmAppropriatenessQuality(Evaluator):
    """Evaluates whether the overall algorithmic approach is appropriate.
    
    This evaluator uses LLM to judge:
    - Is the chosen approach suitable for the problem?
    - Are there obvious better alternatives?
    - Does the solution strategy make sense?
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
        return "algorithm_appropriateness"
    
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
        """Evaluate algorithm appropriateness.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification with description
            execution_result: Optional execution results
        
        Returns:
            EvaluationResult with quality_score (0-1)
        """
        start_time = time.time()
        
        problem_desc = problem.get('problem_description', '')
        
        prompt = f"""Evaluate the algorithmic approach for solving this problem.

Problem:
{problem_desc}

Generated Code:
```c
{code[:2000]}  # Truncate if very long
```

Assess:
1. Is the overall approach suitable for this problem?
2. Does the algorithm choice make sense?
3. Are there obviously better alternatives?
4. Is this a reasonable solution strategy?

Provide:
- score: Algorithm appropriateness (0-10)
- confidence: Assessment confidence (0-1)
- feedback: Explanation of score
- approach_suitable: Is this approach appropriate? (true/false)
- better_alternatives: List of potentially better approaches (if any)

Return as JSON.
"""
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=AlgorithmResponse
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
                    'approach_suitable': response.approach_suitable,
                    'better_alternatives': response.better_alternatives,
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