"""Base classes for the evaluation system.

This module defines the core abstractions for the multi-tier evaluation framework:

1. EvaluatorType: Enum defining three types of evaluation
   - GATE: Binary pass/fail checks (must pass to continue)
   - METRIC: Continuous measurements (e.g., execution time, accuracy)
   - QUALITY: Qualitative assessments (e.g., code style, algorithm choice)

2. EvaluationResult: Unified result structure supporting all evaluator types
   - Flexible schema accommodating different evaluation outputs
   - Tracking metadata (confidence, execution time, evaluation method)
   - Human-readable feedback for debugging and reporting

3. Evaluator: Abstract base class for all evaluators
   - Defines required interface for custom evaluators
   - Supports both deterministic and LLM-based evaluation
   - Enables consistent integration with the evaluation pipeline

The design allows for:
- Easy addition of new evaluators without modifying core pipeline
- Consistent result format for aggregation and reporting
- Flexible evaluation methods (static analysis, execution, LLM)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EvaluatorType(Enum):
    """Type of evaluation being performed."""
    
    GATE = "gate"           # Binary checks (must pass)
    METRIC = "metric"       # Continuous measurements
    QUALITY = "quality"     # Quality assessments


@dataclass
class EvaluationResult:
    """Result from a single evaluator.
    
    This unified result structure supports all three evaluator types,
    providing flexibility while maintaining consistency:
    
    **For GATE evaluators:**
    - Set `passed` to True/False
    - Provide `feedback` explaining why it passed/failed
    - Set `confidence` to 1.0 for deterministic checks
    
    **For METRIC evaluators:**
    - Set `raw_value` to the actual measurement (e.g., 2.5 seconds, 1e-8 error)
    - Set `normalized_score` to a 0-1 normalized value for aggregation
    - Include units and context in `metadata`
    
    **For QUALITY evaluators:**
    - Set `quality_score` to a 0-1 assessment score
    - Provide detailed `feedback` with reasoning
    - Set `confidence` to indicate LLM certainty (if applicable)
    
    Attributes:
        evaluator_name: Unique identifier for the evaluator
        evaluator_type: Type of evaluation (GATE, METRIC, or QUALITY)
        passed: For gates - whether the check passed
        raw_value: For metrics - the actual measured value
        normalized_score: For metrics - normalized 0-1 score
        quality_score: For quality - assessed quality (0-1)
        confidence: How confident the evaluator is (0-1, especially for LLM)
        feedback: Human-readable explanation of the result
        metadata: Additional data specific to the evaluator
        evaluation_method: How evaluation was performed (e.g., "deterministic", "llm_gpt4")
        execution_time_ms: Time taken to perform the evaluation
    """
    evaluator_name: str
    evaluator_type: EvaluatorType
    
    # For GATE evaluators
    passed: Optional[bool] = None
    
    # For METRIC evaluators
    raw_value: Optional[float] = None  # Actual measurement (e.g., 2.5 seconds, 1e-8 error)
    normalized_score: Optional[float] = None  # 0-1 normalized score
    
    # For QUALITY evaluators
    quality_score: Optional[float] = None  # 0-1 score
    
    # Common fields
    confidence: Optional[float] = None  # How confident (0-1), especially for LLM
    feedback: str = ""  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    
    # Tracking
    evaluation_method: str = "deterministic"  # deterministic, llm_gpt4, manual, etc.
    execution_time_ms: float = 0.0


class Evaluator(ABC):
    """Base class for all evaluators.
    
    All custom evaluators must inherit from this class and implement:
    - `name`: Unique identifier for the evaluator
    - `evaluator_type`: The type of evaluation (GATE, METRIC, or QUALITY)
    - `evaluate()`: The core evaluation logic
    
    Optional properties:
    - `evaluation_method`: Description of how evaluation is performed
    
    The base class provides:
    - Configuration management
    - String representation for debugging
    - Consistent interface for the evaluation pipeline
    
    Example:
        ```python
        class MyGate(Evaluator):
            @property
            def name(self) -> str:
                return "my_check"
            
            @property
            def evaluator_type(self) -> EvaluatorType:
                return EvaluatorType.GATE
            
            async def evaluate(self, code, problem, execution_result):
                # Evaluation logic here
                return EvaluationResult(
                    evaluator_name=self.name,
                    evaluator_type=self.evaluator_type,
                    passed=True,
                    feedback="Check passed"
                )
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with optional configuration.
        
        Args:
            config: Optional configuration dictionary containing evaluator-specific
                   settings (e.g., thresholds, LLM parameters, etc.)
        """
        self.config = config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        pass
    
    @property
    @abstractmethod
    def evaluator_type(self) -> EvaluatorType:
        """Type of evaluation this performs."""
        pass
    
    @property
    def evaluation_method(self) -> str:
        """How evaluation is performed (deterministic, llm, manual, etc.)."""
        return "deterministic"
    
    @abstractmethod
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate the generated code.
        
        Args:
            code: The generated code to evaluate
            problem: The problem specification/description
            execution_result: Results from compilation/execution (if available)
                Expected keys:
                - compiles: bool
                - compile_errors: str
                - stdout: str
                - cases: list of per-invocation dictionaries containing
                  index, runs, stdout, stderr, execution_time_sec, and
                  valgrind_output
                - memory_mb: float
        
        Returns:
            EvaluationResult with appropriate fields filled
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.evaluator_type.value})"
