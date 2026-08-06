"""Readability quality evaluator."""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

from ...base import Evaluator, EvaluatorType, EvaluationResult
from src.util.llm_client import LLMClient


class ReadabilityResponse(BaseModel):
    """Structured response for readability evaluation."""
    score: float  # 0-10
    confidence: float  # 0-1
    feedback: str
    strengths: list[str] = []
    weaknesses: list[str] = []


class ReadabilityQuality(Evaluator):
    """Evaluates code readability using LLM or static analysis.
    
    Assesses:
    - Variable naming clarity
    - Code organization and structure
    - Logical flow
    - Ease of understanding
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
        return "readability"
    
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
        """Evaluate code readability.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification (for context)
            execution_result: Not used for readability
        
        Returns:
            EvaluationResult with quality_score (0-1)
        """
        start_time = time.time()
        
        if self.use_llm:
            result = await self._evaluate_with_llm(code, problem)
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
    
    async def _evaluate_with_llm(self, code: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate readability using LLM."""
        prompt = f"""Evaluate the readability of this PETSc C code.

Problem Context:
{problem.get('problem_description', 'N/A')[:500]}

Code:
```c
{code}
```

Rate the readability on a scale of 0-10, considering:
1. Variable naming: Are names clear and meaningful?
2. Code structure: Is the code well-organized?
3. Logical flow: Is it easy to follow the logic?
4. Comments: Are there helpful comments where needed?

Provide:
- score: Overall readability score (0-10)
- confidence: How confident are you in this assessment? (0-1)
- feedback: Brief explanation of the score
- strengths: List of readability strengths (2-3 items)
- weaknesses: List of readability issues (2-3 items)

Return as JSON.
"""
        
        try:
            response = await self.llm.structured_completion(
                prompt=prompt,
                response_model=ReadabilityResponse
            )
            return {
                'score': response.score / 10.0,  # Normalize to 0-1
                'confidence': response.confidence,
                'feedback': response.feedback,
                'metadata': {
                    'raw_score': response.score,
                    'strengths': response.strengths,
                    'weaknesses': response.weaknesses,
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
        """Evaluate readability using static heuristics."""
        score = 0.5  # Start at neutral
        feedback_items = []
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Check average line length
        if non_empty_lines:
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            if avg_line_length < 80:
                score += 0.1
                feedback_items.append("Good: reasonable line length")
            elif avg_line_length > 120:
                score -= 0.1
                feedback_items.append("Issue: very long lines")
        
        # Check for comments
        comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
        comment_ratio = comment_lines / max(len(non_empty_lines), 1)
        if comment_ratio > 0.1:
            score += 0.15
            feedback_items.append("Good: includes comments")
        else:
            score -= 0.1
            feedback_items.append("Issue: lacks comments")
        
        # Check for meaningful variable names (heuristic: length > 2)
        import re
        variables = re.findall(r'\b[a-z_][a-z0-9_]*\b', code.lower())
        meaningful = sum(1 for v in variables if len(v) > 2 and v not in ['int', 'for', 'if'])
        if variables and meaningful / len(variables) > 0.6:
            score += 0.15
            feedback_items.append("Good: descriptive variable names")
        
        # Check indentation consistency
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if indents and all(i % 2 == 0 or i % 4 == 0 for i in indents):
            score += 0.1
            feedback_items.append("Good: consistent indentation")
        
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        return {
            'score': score,
            'confidence': 0.6,  # Static analysis is less confident
            'feedback': '; '.join(feedback_items) if feedback_items else "Basic readability check",
            'metadata': {
                'avg_line_length': avg_line_length if non_empty_lines else 0,
                'comment_ratio': comment_ratio,
                'method': 'static_heuristics'
            }
        }