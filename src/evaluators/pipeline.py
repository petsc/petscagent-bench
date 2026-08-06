"""Evaluation pipeline orchestrator.

This module implements the three-phase evaluation pipeline:

Phase 1: GATES (Critical Pass/Fail)
  - All gates must pass for evaluation to continue
  - Run in parallel for efficiency
  - Examples: compilation, execution, memory safety
  - If any gate fails, code receives FAIL tier immediately

Phase 2: METRICS (Quantitative Measurements)
  - Deterministic measurements of code properties
  - Run in parallel (all are fast and deterministic)
  - Examples: execution time, numerical accuracy
  - Results normalized to 0-1 for aggregation

Phase 3: QUALITY (Qualitative Assessments)
  - Code quality and algorithm appropriateness
  - Mix of deterministic and LLM-based evaluators
  - LLM calls are rate-limited to avoid API throttling
  - Examples: readability, algorithm choice, PETSc best practices

The pipeline supports:
- Configurable evaluator sets (enable/disable phases)
- Parallel execution where safe
- Rate limiting for LLM-based evaluators
- Graceful degradation on evaluator failures
- Dynamic evaluator registration
"""

import asyncio
from typing import List, Dict, Any, Optional

from .base import Evaluator, EvaluatorType, EvaluationResult

# Import all evaluators
from .gates import (
    CompilationGate,
    ExecutionGate,
    MemorySafetyGate,
    APIUsageGate,
)
from .metrics import (
    NumericalAccuracyMetric,
    ExecutionTimeMetric,
)
from .quality.code_quality import (
    ReadabilityQuality,
    CodeStyleQuality,
    DocumentationQuality,
)
from .quality.algorithm_quality import (
    AlgorithmAppropriatenessQuality,
    SolverChoiceQuality,
)
from .quality.petsc_quality import (
    PETScBestPracticesQuality,
    ErrorHandlingQuality,
    ParallelAwarenessQuality,
)


class EvaluationPipeline:
    """Orchestrates multiple evaluators in a structured pipeline.
    
    The pipeline manages three types of evaluators:
    
    1. **Gates (Critical Checks):**
       - Binary pass/fail checks that must succeed
       - Examples: CompilationGate, ExecutionGate, MemorySafetyGate
       - If any gate fails, evaluation stops with FAIL tier
       - Executed in parallel for speed
    
    2. **Metrics (Measurements):**
       - Quantitative measurements of code properties
       - Examples: NumericalAccuracyMetric, ExecutionTimeMetric
       - Results are normalized to 0-1 for aggregation
       - Executed in parallel (all deterministic)
    
    3. **Quality (Assessments):**
       - Qualitative code quality evaluations
       - Mix of deterministic and LLM-based evaluators
       - Examples: ReadabilityQuality, AlgorithmAppropriatenessQuality
       - LLM-based evaluators are rate-limited
    
    Configuration:
        The pipeline behavior is controlled via config dictionary:
        - evaluation.enable_gates: Enable/disable gate phase
        - evaluation.enable_metrics: Enable/disable metrics phase
        - evaluation.enable_quality: Enable/disable quality phase
        - evaluation.parallel_evaluation: Run evaluators in parallel
        - evaluation.llm.model: LLM model for quality evaluations
        - evaluation.llm.api_base_url: LLM API base URL
        - evaluation.llm.max_concurrent_calls: Rate limit for LLM calls
    
    Attributes:
        config: Configuration dictionary
        gates: List of gate evaluators
        metrics: List of metric evaluators
        quality: List of quality evaluators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model: str = None, api_base_url: str = None):
        """Initialize evaluation pipeline.
        
        Args:
            config: Configuration dictionary loaded from YAML/JSON.
                   If None, uses default configuration.
            model: Agent LLM model
            api_base_url: Optinal API base URL
        """
        self.config = config or {}
        self.gates: List[Evaluator] = []
        self.metrics: List[Evaluator] = []
        self.quality: List[Evaluator] = []
        self.model = model
        self.api_base_url = api_base_url
        self._setup_evaluators()
    
    def _get_eval_config(self, key: str, default: Any = None) -> Any:
        """Get value from evaluation config section.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        return self.config.get('evaluation', {}).get(key, default)
    
    def _get_llm_config(self, key: str, default: Any = None) -> Any:
        """Get value from LLM config section.
        
        Args:
            key: LLM configuration key to retrieve
            default: Default value if key not found
        
        Returns:
            LLM configuration value or default
        """
        return self.config.get('evaluation', {}).get('llm', {}).get(key, default)
    
    def _setup_evaluators(self):
        """Initialize evaluators based on configuration.
        
        This method instantiates all evaluators according to the config.
        Evaluators are organized into three lists by type:
        - gates: Critical binary checks
        - metrics: Quantitative measurements
        - quality: Qualitative assessments
        
        LLM-based quality evaluators receive LLM configuration
        (model, temperature, etc.) for consistent behavior.
        """
        
        # Gates (always enabled - these are critical)
        # Gates are fundamental checks that code must pass
        if self._get_eval_config('enable_gates', True):
            self.gates = [
                CompilationGate(self.config.get('compilation', {})),
                ExecutionGate(self.config.get('execution', {})),
                MemorySafetyGate(self.config.get('memory_safety', {})),
                APIUsageGate(self.config.get('api_usage', {})),
            ]
        
        # Metrics (deterministic measurements)
        if self._get_eval_config('enable_metrics', True):
            self.metrics = [
                NumericalAccuracyMetric(self.config.get('numerical_accuracy', {})),
                ExecutionTimeMetric(self.config.get('execution_time', {})),
            ]
        
        # Quality evaluators
        if self._get_eval_config('enable_quality', True):
            llm_config = {
                'llm_model': self.model,
                'llm_api_base_url': self.api_base_url,
                'llm_temperature': self._get_llm_config('temperature', 0.3),
                'llm_max_tokens': self._get_llm_config('max_tokens', None),
                'max_concurrent_calls': self._get_llm_config('max_concurrent_calls', 3),
            }
            
            self.quality = [
                # Code quality (can use LLM or static analysis)
                ReadabilityQuality({**llm_config, **self.config.get('readability', {})}),
                CodeStyleQuality({**llm_config, **self.config.get('code_style', {})}),
                DocumentationQuality({**llm_config, **self.config.get('documentation', {})}),
                
                # Algorithm quality (LLM-based)
                AlgorithmAppropriatenessQuality({**llm_config, **self.config.get('algorithm_appropriateness', {})}),
                SolverChoiceQuality({**llm_config, **self.config.get('solver_choice', {})}),
                
                # PETSc quality (mixed)
                PETScBestPracticesQuality({**llm_config, **self.config.get('petsc_best', {})}),
                ErrorHandlingQuality(self.config.get('error_handling', {})),  # Deterministic
                ParallelAwarenessQuality(self.config.get('parallel_awareness', {})),  # Deterministic
            ]
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Run all evaluators and return results.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification
            execution_result: Results from compilation/execution
        
        Returns:
            List of evaluation results from all evaluators
        """
        all_results: List[EvaluationResult] = []
        parallel = self._get_eval_config('parallel_evaluation', True)
        
        # Phase 1: Gates (must all pass to continue)
        print("Phase 1: Running gate evaluators...")
        if parallel:
            gate_results = await asyncio.gather(*[
                gate.evaluate(code, problem, execution_result)
                for gate in self.gates
            ])
        else:
            gate_results = []
            for gate in self.gates:
                result = await gate.evaluate(code, problem, execution_result)
                gate_results.append(result)
        
        all_results.extend(gate_results)
        
        # Check if all gates passed
        gates_passed = all(
            r.passed for r in gate_results 
            if r.passed is not None
        )
        
        if not gates_passed:
            print("Gate(s) failed. Skipping remaining evaluation.")
            return all_results
        
        # Phase 2: Metrics (parallel - all deterministic)
        print("Phase 2: Running metric evaluators...")
        if parallel:
            metric_results = await asyncio.gather(*[
                metric.evaluate(code, problem, execution_result)
                for metric in self.metrics
            ])
        else:
            metric_results = []
            for metric in self.metrics:
                result = await metric.evaluate(code, problem, execution_result)
                metric_results.append(result)
        all_results.extend(metric_results)
        
        # Phase 3: Quality (rate-limited for LLM-based)
        print("Phase 3: Running quality evaluators...")
        
        # Separate LLM-based from deterministic
        llm_quality = [q for q in self.quality if 'llm' in q.evaluation_method]
        non_llm_quality = [q for q in self.quality if 'llm' not in q.evaluation_method]
        # Non-LLM quality (can run in parallel, fast)
        if non_llm_quality:
            if parallel:
                non_llm_results = await asyncio.gather(*[
                    q.evaluate(code, problem, execution_result)
                    for q in non_llm_quality
                ])
            else:
                non_llm_results = []
                for q in non_llm_quality:
                    result = await q.evaluate(code, problem, execution_result)
                    non_llm_results.append(result)
            
            all_results.extend(non_llm_results)
        
        # LLM quality (rate-limited)
        if llm_quality:
            max_concurrent = self._get_llm_config('max_concurrent_calls', 3)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def rate_limited_eval(evaluator: Evaluator) -> EvaluationResult:
                async with semaphore:
                    return await evaluator.evaluate(code, problem, execution_result)
            
            llm_results = await asyncio.gather(*[
                rate_limited_eval(q) for q in llm_quality
            ], return_exceptions=True)
            
            # Filter out exceptions
            for result in llm_results:
                if isinstance(result, Exception):
                    print(f"LLM evaluation error: {result}")
                else:
                    all_results.append(result)
        
        print(f"Evaluation complete: {len(all_results)} evaluators ran")
        return all_results
    
    def add_evaluator(self, evaluator: Evaluator):
        """Add a custom evaluator to the pipeline.
        
        This method allows dynamic addition of evaluators at runtime,
        useful for:
        - Problem-specific evaluators
        - Experimental evaluators
        - Plugin-style extensions
        
        The evaluator is automatically added to the correct list based
        on its evaluator_type property.
        
        Args:
            evaluator: Custom evaluator instance (must inherit from Evaluator)
        
        Example:
            >>> pipeline = EvaluationPipeline()
            >>> custom_gate = MyCustomGate()
            >>> pipeline.add_evaluator(custom_gate)
        """
        if evaluator.evaluator_type == EvaluatorType.GATE:
            self.gates.append(evaluator)
        elif evaluator.evaluator_type == EvaluatorType.METRIC:
            self.metrics.append(evaluator)
        elif evaluator.evaluator_type == EvaluatorType.QUALITY:
            self.quality.append(evaluator)
    
    def get_evaluator_count(self) -> Dict[str, int]:
        """Get count of evaluators by type.
        
        Useful for diagnostics and logging to verify pipeline configuration.
        
        Returns:
            Dictionary with counts:
            - 'gates': Number of gate evaluators
            - 'metrics': Number of metric evaluators
            - 'quality': Number of quality evaluators
            - 'total': Total number of all evaluators
        """
        return {
            'gates': len(self.gates),
            'metrics': len(self.metrics),
            'quality': len(self.quality),
            'total': len(self.gates) + len(self.metrics) + len(self.quality),
        }
