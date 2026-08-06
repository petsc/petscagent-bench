"""Documentation quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class DocumentationResponse(BaseModel):
    """Structured response for documentation evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    has_function_docs: bool
    has_inline_comments: bool
    clarity: str  # "excellent", "good", "fair", "poor"


class DocumentationQuality(Evaluator):
    """Evaluates quality and helpfulness of code documentation.
    
    Assesses:
    - Function/file header comments
    - Inline comments explaining logic
    - Clarity and usefulness of comments
    - Documentation of complex sections
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
        return "documentation"
    
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
        """Evaluate documentation quality.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification
            execution_result: Not used for documentation check
        
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
        """Evaluate documentation using LLM."""
        prompt = f"""Evaluate the documentation quality of this PETSc C code.

Code:
```c
{code}
```

Assess:
1. Function documentation: Are functions documented?
2. Inline comments: Are complex sections explained?
3. Comment clarity: Are comments helpful and clear?
4. Completeness: Is enough documented for understanding?

Provide:
- score: Overall documentation quality (0-10)
- confidence: Assessment confidence (0-1)
- feedback: Brief explanation
- has_function_docs: Are there function-level comments? (true/false)
- has_inline_comments: Are there inline explanatory comments? (true/false)
- clarity: Comment clarity level ("excellent", "good", "fair", "poor")

Return as JSON.
"""
        
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=DocumentationResponse
            )
            return {
                'score': response.score / 10.0,
                'confidence': response.confidence,
                'feedback': response.feedback,
                'metadata': {
                    'raw_score': response.score,
                    'has_function_docs': response.has_function_docs,
                    'has_inline_comments': response.has_inline_comments,
                    'clarity': response.clarity,
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
        """Evaluate documentation using static analysis."""
        score = 0.0
        
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        
        # Count different types of comments
        single_line_comments = sum(1 for line in lines if '//' in line)
        multi_line_start = sum(1 for line in lines if '/*' in line)
        
        total_comment_lines = single_line_comments + multi_line_start
        
        # Calculate comment ratio
        comment_ratio = total_comment_lines / max(total_lines, 1)
        
        # Score based on comment density
        if comment_ratio > 0.2:  # >20% comments
            score = 0.9
        elif comment_ratio > 0.15:
            score = 0.8
        elif comment_ratio > 0.1:
            score = 0.7
        elif comment_ratio > 0.05:
            score = 0.5
        else:
            score = 0.3
        
        # Check for function documentation (/* or /** before functions)
        import re
        has_function_docs = bool(re.search(r'/\*.*?\*/\s*\w+\s*\(', code, re.DOTALL))
        if has_function_docs:
            score += 0.1
        
        score = min(1.0, score)
        
        return {
            'score': score,
            'confidence': 0.6,
            'feedback': f"Comment ratio: {comment_ratio:.1%}",
            'metadata': {
                'comment_ratio': comment_ratio,
                'single_line_comments': single_line_comments,
                'multi_line_blocks': multi_line_start,
                'has_function_docs': has_function_docs,
            }
        }