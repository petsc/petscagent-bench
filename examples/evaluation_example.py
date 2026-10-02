"""Example of using the evaluation system."""

import asyncio
import json
from pathlib import Path

from src.evaluators import EvaluationPipeline
from src.metrics import MetricsAggregator


async def evaluate_code_example():
    """Example: Evaluate a piece of generated PETSc code."""
    
    # Sample problem
    problem = {
        'problem_id': 'test_001',
        'problem_name': 'Poisson 2D',
        'problem_description': 'Solve the 2D Poisson equation with Dirichlet boundary conditions',
        'reference_solution': [0.1, 0.2, 0.3, 0.4],  # Example values
        'baseline_time_sec': 0.5,
    }
    
    # Sample generated code
    code = """
#include <petsc.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    Vec x, b;
    Mat A;
    KSP ksp;
    
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    
    // Create vectors and matrix
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, 100); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &b); CHKERRQ(ierr);
    
    // Setup and solve
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
    
    // Cleanup
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}
"""
    
    # Sample execution result
    execution_result = {
        'compiles': True,
        'compile_errors': '',
        'stdout': '0.1\n0.2\n0.3\n0.4',
        'cases': [{
            'index': 0,
            'runs': True,
            'stdout': '0.1\n0.2\n0.3\n0.4',
            'stderr': '',
            'execution_time_sec': 0.3,
            'valgrind_output': None,
        }],
        'memory_mb': 50.0,
    }
    
    # Create evaluation configuration
    config = {
        'evaluation': {
            'enable_gates': True,
            'enable_metrics': True,
            'enable_quality': True,
            'llm': {
                'model': 'gpt-4o-mini',
                'temperature': 0.3,
                'max_concurrent_calls': 3,
            },
            'parallel_evaluation': True,
        },
        'scoring': {
            'weights': {
                'correctness': 0.35,
                'performance': 0.15,
                'code_quality': 0.15,
                'algorithm': 0.15,
                'petsc': 0.10,
                'semantic': 0.10,
            },
            'tiers': {
                'gold': 85,
                'silver': 70,
                'bronze': 50,
            },
        },
    }
    
    pipeline = EvaluationPipeline(config)
    
    print(f"Evaluation pipeline initialized with {pipeline.get_evaluator_count()['total']} evaluators")
    print("="*60)
    
    # Run evaluation
    print("Running evaluation...\n")
    results = await pipeline.evaluate(code, problem, execution_result)
    
    print(f"\nEvaluation complete: {len(results)} results")
    print("="*60)
    
    # Aggregate results
    aggregator = MetricsAggregator(config)
    aggregated = aggregator.aggregate(results)
    
    # Print summary
    print(aggregated.get_summary_string())
    
    # Print individual results
    print("\nIndividual Evaluation Results:")
    print("="*60)
    
    for result in results:
        status = "✓" if result.passed else "✗" if result.passed is False else "○"
        score_str = ""
        if result.quality_score is not None:
            score_str = f"Quality: {result.quality_score:.2f}"
        elif result.normalized_score is not None:
            score_str = f"Score: {result.normalized_score:.2f}"
        elif result.raw_value is not None:
            score_str = f"Value: {result.raw_value:.2e}"
        
        print(f"{status} {result.evaluator_name:30s} {score_str:20s} | {result.feedback[:50]}")
    
    # Save detailed results
    output_dir = Path("output/evaluation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_results = {
        'problem': problem,
        'aggregated_metrics': aggregated.to_dict(),
        'individual_results': [
            {
                'name': r.evaluator_name,
                'type': r.evaluator_type.value,
                'method': r.evaluation_method,
                'passed': r.passed,
                'score': r.quality_score or r.normalized_score,
                'raw_value': r.raw_value,
                'confidence': r.confidence,
                'feedback': r.feedback,
                'metadata': r.metadata,
            }
            for r in results
        ],
    }
    
    output_file = output_dir / f"evaluation_{problem['problem_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return aggregated


if __name__ == "__main__":
    # Run the example
    result = asyncio.run(evaluate_code_example())
    print(f"\nFinal Tier: {result.overall_tier}")
    print(f"Final Score: {result.composite_score:.1f}/100")
