"""Code style quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class CodeStyleResponse(BaseModel):
    """Structured response for code style evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    follows_conventions: bool
    issues: list[str] = []


class CodeStyleQuality(Evaluator):
    """Evaluates adherence to PETSc and C coding conventions.
    
    Checks:
    - PETSc naming conventions
    - C coding standards
    - Formatting consistency
    - Style guide compliance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.use_llm = config.get('use_llm', True) if config else True
        
        if self.use_llm:
            llm_model = config.get('llm_model', 'gpt-4o-mini') if config else 'gpt-4o-mini'
            llm_temp = config.get('llm_temperature', 0.3) if config else 0.3
            llm_api_base_url = config.get('llm_api_base_url') if config else None
            llm_max_tokens = config.get('llm_max_tokens') if config else None
            self.llm = LLMClient(model=llm_model, temperature=llm_temp, api_base_url=llm_api_base_url, max_tokens=llm_max_tokens)
    
    @property
    def name(self) -> str:
        return "code_style"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    @property
    def evaluation_method(self) -> str:
        return f"llm_{self.llm.model}" if self.use_llm else "static_analysis"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate code style.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification
            execution_result: Not used for style check
        
        Returns:
            EvaluationResult with quality_score (0-1)
        """
        start_time = time.time()
        
        if self.use_llm:
            result = await self._evaluate_with_llm(code)
        else:
            result = self._evaluate_with_static_analysis(code)
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=result['score'] >= 0.7,
            quality_score=result['score'],
            confidence=result['confidence'],
            feedback=result['feedback'],
            metadata=result.get('metadata', {}),
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _evaluate_with_llm(self, code: str) -> Dict[str, Any]:
        """Evaluate code style using LLM."""
        prompt = f"""Evaluate the coding style of this PETSc C code.

Code:
```c
{code}
```

Assess adherence to PETSc and C conventions:
1. PETSc naming (PetscErrorCode, Vec, Mat, etc.)
2. Function naming (PascalCase for PETSc functions)
3. Variable naming (camelCase or snake_case consistently)
4. Brace placement and indentation
5. Spacing and formatting

Provide:
- score: Overall style quality (0-10)
- confidence: Assessment confidence (0-1)
- feedback: Brief explanation
- follows_conventions: Does it generally follow PETSc/C conventions? (true/false)
- issues: List of specific style issues found (2-4 items)

Return as JSON.
"""
        
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=CodeStyleResponse
            )
            return {
                'score': response.score / 10.0,
                'confidence': response.confidence,
                'feedback': response.feedback,
                'metadata': {
                    'raw_score': response.score,
                    'follows_conventions': response.follows_conventions,
                    'issues': response.issues,
                }
            }
        except Exception as e:
            return {
                'score': 0.5,
                'confidence': 0.0,
                'feedback': f"LLM evaluation failed: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _evaluate_with_static_analysis(self, code: str) -> Dict[str, Any]:
        """Evaluate code style using static checks."""
        score = 0.5
        issues = []
        
        # Check for PETSc-style naming
        if 'PetscErrorCode' in code:
            score += 0.15
        else:
            issues.append("Missing PetscErrorCode return type")
        
        # Check for proper error handling macro
        if 'CHKERRQ' in code:
            score += 0.15
        else:
            issues.append("Missing CHKERRQ error handling")
        
        # Check for consistent indentation (spaces)
        lines = code.split('\n')
        tab_lines = sum(1 for line in lines if '\t' in line)
        if tab_lines == 0:
            score += 0.1
        else:
            issues.append("Uses tabs instead of spaces")
        
        # Check brace style (opening brace on same line for functions)
        import re
        functions = re.findall(r'(\w+)\s*\([^)]*\)\s*{', code)
        if functions:
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'confidence': 0.6,
            'feedback': f"Found {len(issues)} style issues" if issues else "Basic style check passed",
            'metadata': {
                'issues': issues,
                'method': 'static_checks'
            }
        }