# PETSc Code Evaluation System - Implementation Summary

## Overview

This document describes the implementation of the PETSc code evaluation system as wired into the Green Agent.

- Evaluation is performed by **14 evaluators** (when all phases are enabled):
  - Gates: 4
  - Metrics: 2
  - Quality: 8
- Evaluator outputs are aggregated into category scores and a composite score (0–100), then mapped to tiers.

> Note: Some evaluators depend on fields that may not exist in all datasets (e.g., numerical accuracy requires `test_cases.expected_output`). Where data is missing, those evaluators may be skipped and/or contribute 0 to the final score.

## Architecture

### Three-Tier Evaluation Model

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: GATES (Must Pass)                                │
│  ├── Compilation          (deterministic)                  │
│  ├── Execution            (deterministic)                  │
│  ├── Memory Safety        (deterministic)                  │
│  └── API Usage            (deterministic)                  │
│                                                             │
│  Phase 2: METRICS (Measurements)                           │
│  ├── Numerical Accuracy   (deterministic)                  │
│  └── Execution Time       (deterministic)                  │
│                                                             │
│  Phase 3: QUALITY (Assessments)                            │
│  ├── Code Quality         (LLM or static)                  │
│  │   ├── Readability                                       │
│  │   ├── Code Style                                        │
│  │   └── Documentation                                     │
│  ├── Algorithm Quality    (LLM)                            │
│  │   ├── Algorithm Appropriateness                         │
│  │   └── Solver Choice                                     │
│  └── PETSc Quality        (mixed)                          │
│      ├── Best Practices   (LLM)                            │
│      ├── Error Handling   (deterministic)                  │
│      └── Parallel Aware   (deterministic)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   METRICS AGGREGATOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Category Scores (0-100):                                  │
│  ├── Correctness  (35% weight)                            │
│  ├── Performance  (15% weight)                            │
│  ├── Code Quality (15% weight)                            │
│  ├── Algorithm    (15% weight)                            │
│  ├── PETSc Usage  (10% weight)                            │
│  └── Semantic     (10% weight)                            │
│                                                             │
│  Composite Score = Σ(category × weight)                   │
│                                                             │
│  Tier Assignment:                                          │
│  ├── GOLD   (≥85)                                         │
│  ├── SILVER (≥70)                                         │
│  ├── BRONZE (≥50)                                         │
│  └── FAIL   (<50 or gates failed)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Complete Metrics List (14 Total)

### 1. Gates (4) - Binary Pass/Fail

| Metric | What | Method | Critical |
|--------|------|--------|----------|
| **Compilation** | Code compiles successfully | Deterministic | ✅ |
| **Execution** | Runs without crash | Deterministic | ✅ |
| **Memory Safety** | Memory-safety signals (valgrind output if provided, otherwise stderr heuristics) | Deterministic | ✅ |
| **API Usage** | `PetscInitialize`/`PetscFinalize` present and PETSc include detected | Deterministic (static analysis) | ✅ |

**If ANY gate fails → Overall score = 0 (FAIL)**

### 2. Metrics (2) - Continuous Measurements

| Metric | Raw Value | Normalized Score | Method |
|--------|-----------|------------------|--------|
| **Numerical Accuracy** | Error norm | exp(-error/tol) | Deterministic (requires `problem.test_cases[0].expected_output`) |
| **Execution Time** | Seconds | tiered piecewise-linear score (see `execution_time` config) | Deterministic |

### 3. Quality (8) - Subjective Assessments

#### Code Quality (3)
| Metric | Assessment | Default Method | Configurable |
|--------|------------|----------------|---------------|
| **Readability** | Variable names, structure | LLM | ✅ Static option |
| **Code Style** | PETSc/C conventions | LLM | ✅ Static option |
| **Documentation** | Comments, clarity | LLM | ✅ Static option |

#### Algorithm Quality (2)
| Metric | Assessment | Method |
|--------|------------|--------|
| **Algorithm Appropriateness** | Suitable approach | LLM |
| **Solver Choice** | KSP/SNES type | LLM |

#### PETSc Quality (3)
| Metric | Assessment | Method |
|--------|------------|--------|
| **Best Practices** | CLI options, viewers | LLM |
| **Error Handling** | PETSc error handling patterns | Deterministic |
| **Parallel Awareness** | MPI-aware code | Deterministic |

## Implementation Structure

```
petscagent_bench/
├── src/
│   ├── evaluators/
│   │   ├── __init__.py                # Exports
│   │   ├── base.py                    # Base classes, enums
│   │   ├── pipeline.py                # Orchestration
│   │   ├── README.md                  # Documentation
│   │   ├── gates/                     # 4 gate evaluators
│   │   │   ├── __init__.py
│   │   │   ├── compilation_gate.py
│   │   │   ├── execution_gate.py
│   │   │   ├── memory_safety_gate.py
│   │   │   └── api_usage_gate.py
│   │   ├── metrics/                   # 2 metric evaluators
│   │   │   ├── __init__.py
│   │   │   ├── numerical_accuracy.py
│   │   │   └── execution_time.py
│   │   └── quality/                   # 8 quality evaluators
│   │       ├── __init__.py
│   │       ├── code_quality/          # 3 evaluators
│   │       │   ├── __init__.py
│   │       │   ├── readability.py
│   │       │   ├── code_style.py
│   │       │   └── documentation.py
│   │       ├── algorithm_quality/     # 2 evaluators
│   │       │   ├── __init__.py
│   │       │   ├── algorithm_appropriateness.py
│   │       │   └── solver_choice.py
│   │       └── petsc_quality/         # 3 evaluators
│   │           ├── __init__.py
│   │           ├── best_practices.py
│   │           ├── error_handling.py
│   │           └── parallel_awareness.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── types.py                   # Data structures
│   │   └── aggregation.py             # Scoring logic
│   ├── util/
│   │   └── llm_client.py              # LiteLLM-based client
│   └── green_agent/
│       └── agent.py                   # ✅ INTEGRATED
├── config/
│   └── green_agent_config.yaml        # Evaluation + scoring + Green LLM config
```

## Key Design Decisions

### 1. Why 3 Types (Gates/Metrics/Quality)?

**Semantic types instead of method-based classification:**

- **Gates**: Must-pass requirements (compilation, execution, memory safety, API usage)
- **Metrics**: Objective measurements (time, error)
- **Quality**: Subjective assessments (readability, appropriateness)

This separates **WHAT** we evaluate from **HOW** we evaluate it.

### 2. Why Both Deterministic and LLM?

| Use Deterministic When | Use LLM When |
|------------------------|---------------|
| ✅ Objective facts (compiles? runs?) | ✅ Subjective quality (readable?) |
| ✅ Measurable (time, error) | ✅ Semantic understanding (correct approach?) |
| ✅ Ground truth available | ✅ No reference solution |
| ✅ Fast & free | ✅ Complex reasoning needed |

**Rule**: If you can measure it objectively, don't use LLM.

### 3. Scoring Formula

```python
if not all_gates_passed:
    score = 0
else:
    score = (
        0.35 × correctness +      # Numerical accuracy
        0.15 × performance +       # Execution time
        0.15 × code_quality +      # Readability, style, docs
        0.15 × algorithm +         # Algorithm, solver
        0.10 × petsc +             # Best practices
        0.10 × semantic            # NOTE: no evaluators currently map to `semantic`, so it defaults to 0 unless you change weights or add evaluators
    )
```

## Full Green Agent Integration

The evaluation system is **fully integrated** into the Green Agent's benchmarking pipeline:

### Enhanced BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    problem_name: str
    problem_id: str
    runs: bool
    time_used_sec: float
    compiles: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    cli_args: Optional[str] = None
    
    # Evaluation fields (NEW)
    composite_score: Optional[float] = None  # 0-100
    tier: Optional[str] = None  # GOLD/SILVER/BRONZE/FAIL
    category_scores: Optional[Dict[str, float]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_details: Optional[List[Dict[str, Any]]] = None
```

### Agent initialization (Green Agent)

In `src/green_agent/agent.py`, the Green Agent initializes the evaluation system like this:

```python
# Initialize evaluation system with config from file or defaults
# (default path: config/green_agent_config.yaml)
eval_config = load_green_agent_config()
self.eval_config = eval_config
self.evaluation_pipeline = EvaluationPipeline(eval_config, self.model, self.api_base_url)
self.metrics_aggregator = MetricsAggregator(eval_config)

print(
    f"@@@ Green agent: ✅ Evaluation system initialized with "
    f"{self.evaluation_pipeline.get_evaluator_count()['total']} evaluators"
)
```

### Evaluation workflow (high level)

In `src/green_agent/agent.py`, for each problem the Green Agent:

1. Requests code from the Purple Agent (optionally using cache)
2. Uploads files to the MCP server
3. Compiles and runs the code
4. Runs the evaluation pipeline and aggregates scores
5. Emits per-problem artifacts and final summary artifacts

### Evaluation method (code)

The evaluation method in `src/green_agent/agent.py` builds an `execution_result` dict, runs the pipeline, then aggregates:

```python
execution_result = {
    'compiles': benchmark_result.compiles,
    'runs': benchmark_result.runs,
    'stdout': benchmark_result.stdout or '',
    'stderr': benchmark_result.stderr or '',
    'execution_time_sec': benchmark_result.time_used_sec,
    'memory_mb': None,  # TODO: Add memory tracking if available
}

# Run evaluation pipeline


eval_results = await self.evaluation_pipeline.evaluate(
    code=generated_codes[0],
    problem=problem_data,
    execution_result=execution_result,
)

# Aggregate results
aggregated = self.metrics_aggregator.aggregate(eval_results)

# Update benchmark result
benchmark_result.composite_score = aggregated.composite_score
benchmark_result.tier = aggregated.overall_tier
benchmark_result.category_scores = {
    'correctness': aggregated.category_scores.correctness,
    'performance': aggregated.category_scores.performance,
    'code_quality': aggregated.category_scores.code_quality,
    'algorithm': aggregated.category_scores.algorithm,
    'petsc': aggregated.category_scores.petsc,
}
```

## Configuration System

### YAML configuration (`config/green_agent_config.yaml`)

The repo’s default config contains additional comments and evaluator-specific sections; below is an abridged excerpt of the keys the evaluation system consumes:

```yaml
evaluation:
  enable_gates: true
  enable_metrics: true
  enable_quality: true

  llm:
    model: "openai/gpt52"
    api_base_url: "https://apps-dev.inside.anl.gov/argoapi/v1"  # set to null to use provider default
    temperature: 0
    max_concurrent_calls: 3

  parallel_evaluation: true

scoring:
  weights:
    correctness: 0.35
    performance: 0.15
    code_quality: 0.15
    algorithm: 0.15
    petsc: 0.10
    semantic: 0.10

  tiers:
    gold: 85
    silver: 70
    bronze: 50

# Metric configurations
numerical_accuracy:
  error_tolerance: 1.0e-3
  error_threshold: 1.0e-3

execution_time:
  excellent_time_sec: 1.0
  good_time_sec: 5.0
  acceptable_time_sec: 15.0
  max_time_sec: 60.0
  max_slowdown_factor: 2.0
```

### Configuration loading

The Green Agent loads configuration via `load_green_agent_config()` in `src/green_agent/agent.py`.
The default path is:

- `config/green_agent_config.yaml`

YAML and JSON are supported (format detected by file extension).

## Output Format

### Console Output During Evaluation

```
@@@ Green agent: ✅ Evaluation system initialized with 14 evaluators
[1/3] Running Advection_PDE...
@@@ Green agent: ✅ Loaded cached response for Advection_PDE
@@@ Green agent: Compile and run the code...
@@@ Green agent: Evaluating generated code...
Phase 1: Running gate evaluators...
Phase 2: Running metric evaluators...
Phase 3: Running quality evaluators...
Evaluation complete: 14 evaluators ran
@@@ Green agent: ✅ Evaluation complete: Score=87.5, Tier=GOLD
```

### Text Report (evaluation_report.txt)

```
================================================================================
EVALUATION REPORT
================================================================================

Total Problems: 3
Successful Executions: 3
Failed Executions: 0
Average Execution Time: 2.45s

Average Composite Score: 76.3/100

Tier Distribution:
  🥇 GOLD:   1 (33.3%)
  🥈 SILVER: 1 (33.3%)
  🥉 BRONZE: 1 (33.3%)
  ❌ FAIL:   0 (0.0%)

================================================================================
PER-PROBLEM RESULTS
================================================================================

🥇 Advection_PDE (Score: 87.5/100)
   Correctness: 92.0, Performance: 85.0, Code Quality: 78.0

🥈 Robertson_ODE (Score: 73.2/100)
   Correctness: 80.0, Performance: 70.0, Code Quality: 68.0

🥉 Rosenbrock_banana_function (Score: 68.1/100)
   Correctness: 75.0, Performance: 65.0, Code Quality: 60.0
```

### JSON output (`output/<model>-judge-<judge>-run<N>.json`)

The Green Agent writes a JSON file with this top-level structure:

```json
{
  "agent": "<purple_id>",
  "summary": { /* ... */ },
  "results": [ /* ... */ ]
}
```

Notes:

- `agent` is populated from the `<purple_id>` tag passed to the Green Agent.
- Each entry in `results` contains execution fields plus evaluation fields.

### Detailed evaluation report (`evaluation_detailed_report.json`)

This report is emitted as a **task artifact** (not written to `output/` by default). It contains:

- summary statistics
- per-problem tier/composite score/category scores

## Purple Agent Caching System

The Green Agent includes a **caching system** to avoid redundant Purple Agent calls:

```python
# In src/green_agent/agent.py
# Caching is controlled by the `use_cache` flag.
self.use_cache = use_cache
self.cache_dir = Path("./purple_agent_cache")
self.cache_dir.mkdir(exist_ok=True)

def _get_cache_path(self, problem_name: str) -> Path:
    """Get the cache file path for a given problem."""
    safe_name = re.sub(r'[^\w\-_]', '_', problem_name)
    return self.cache_dir / f"{safe_name}.pkl"

def _load_cached_response(self, problem_name: str):
    """Load cached purple agent response if it exists."""
    cache_path = self._get_cache_path(problem_name)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def _save_cached_response(self, problem_name: str, response):
    """Save purple agent response to cache."""
    cache_path = self._get_cache_path(problem_name)
    with open(cache_path, 'wb') as f:
        pickle.dump(response, f)
```

**Benefits:**
- 🚀 Faster re-evaluation during development
- 💰 No redundant Purple Agent calls (cost savings)
- 🔄 Consistent results for testing evaluation changes
- 📁 Stored in `./purple_agent_cache/` as `.pkl` files

## Performance & cost

Actual runtime and token cost depend heavily on:

- number of problems
- which evaluators are enabled (`evaluation.enable_*`)
- which quality evaluators use LLM vs static analysis (`use_llm` flags)
- the configured model/provider (`evaluation.llm.model`, `evaluation.llm.api_base_url`)

General guidance:

- **Gates + metrics only**: typically fast (no LLM calls).
- **Quality with static analysis**: still fast (no LLM calls).
- **Quality with LLM**: can be slow and can consume significant tokens; consider rate limiting with `evaluation.llm.max_concurrent_calls`.

### Optimization Strategies

1. **Development/Testing**: 
   ```yaml
   enable_quality: false  # Skip LLM evaluations
   ```

2. **Hybrid approach**:

   The following `use_llm` switches are implemented in the code-quality evaluators and will switch those evaluators to static heuristics:

   ```yaml
   readability:
     use_llm: false  # Use static analysis
   code_style:
     use_llm: false
   documentation:
     use_llm: false
   ```

## Usage Examples

### Running the full benchmark locally

```bash
uv run main.py launch
```

### Quick test without LLM quality evaluators

Edit `config/green_agent_config.yaml`:

```yaml
evaluation:
  enable_quality: false  # Fast mode
```

### Standalone evaluation

```python
from src.evaluators import EvaluationPipeline
from src.metrics import MetricsAggregator
from src.green_agent.agent import load_green_agent_config

# Initialize
config = load_green_agent_config()

# The pipeline needs a model/api_base_url for LLM-based quality evaluators
pipeline = EvaluationPipeline(config, model="openai/gpt52", api_base_url=None)
aggregator = MetricsAggregator(config)

# Prepare execution result (keys used by metrics/gates)
execution_result = {
    'compiles': True,
    'runs': True,
    'stdout': '...',
    'stderr': '',
    'execution_time_sec': 2.5,
    'memory_mb': None,
}

# Evaluate
results = await pipeline.evaluate(code, problem_data, execution_result)
metrics = aggregator.aggregate(results)

print(f"Score: {metrics.composite_score:.1f}/100")
print(f"Tier: {metrics.overall_tier}")
```

Notes:

- `NumericalAccuracyMetric` only runs when `problem_data` includes `test_cases[0].expected_output`.

### Custom Evaluator

```python
from src.evaluators.base import Evaluator, EvaluatorType, EvaluationResult

class CustomQualityEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "custom_metric"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    async def evaluate(self, code, problem, execution_result):
        # Your evaluation logic here
        score = 0.85  # 0-1 scale
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            quality_score=score,
            confidence=0.9,
            feedback="Custom evaluation passed",
            evaluation_method="custom"
        )

# Add to pipeline
pipeline.add_evaluator(CustomQualityEvaluator())
```

## Files in Repository (Complete Implementation)

### Evaluators (14 files)

```
src/evaluators/
├── __init__.py                          # ✅ Exports
├── base.py                              # ✅ Base classes, EvaluatorType enum
├── pipeline.py                          # ✅ Orchestration
├── README.md                            # ✅ Documentation
├── gates/                               # ✅ 4 gate evaluators
│   ├── __init__.py
│   ├── compilation_gate.py
│   ├── execution_gate.py
│   ├── memory_safety_gate.py
│   └── api_usage_gate.py
├── metrics/                             # ✅ 2 metric evaluators
│   ├── __init__.py
│   ├── numerical_accuracy.py
│   └── execution_time.py
└── quality/                             # ✅ 8 quality evaluators
    ├── __init__.py
    ├── code_quality/
    │   ├── __init__.py
    │   ├── readability.py
    │   ├── code_style.py
    │   └── documentation.py
    ├── algorithm_quality/
    │   ├── __init__.py
    │   ├── algorithm_appropriateness.py
    │   └── solver_choice.py
    └── petsc_quality/
        ├── __init__.py
        ├── best_practices.py
        ├── error_handling.py
        └── parallel_awareness.py
```

### Supporting Infrastructure

```
src/metrics/
├── __init__.py                          # ✅ Exports
├── types.py                             # ✅ AggregatedMetrics, CategoryScores
└── aggregation.py                       # ✅ MetricsAggregator

src/util/
└── llm_client.py                        # ✅ LiteLLM-based client

src/green_agent/
└── agent.py                             # ✅ FULLY INTEGRATED

config/
└── green_agent_config.yaml              # ✅ Evaluation + scoring + Green LLM config
```

### Outputs

**Written to disk**:

```
output/
└── <model>-judge-<judge>-run<N>.json
```

**Emitted as task artifacts** (via `TaskUpdater.add_artifact`):

- `<model>-judge-<judge>-run<N>.json`
- `evaluation_report.txt`
- `evaluation_detailed_report.json`
- `benchmark_result_<problem_name>.json`

**Cached responses**:

```
purple_agent_cache/
└── *.pkl
```

## Status

- ✅ Implemented with 14 evaluators (when all phases are enabled)
- ✅ Integrated into the Green Agent benchmarking pipeline
- ✅ Configurable via `config/green_agent_config.yaml`
- ✅ Emits summary results to disk (`output/<model>-judge-<judge>-run<N>.json`) and additional reports as task artifacts
- ✅ Supports caching of Purple Agent responses (`purple_agent_cache/`)
