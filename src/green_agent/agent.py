"""Green Agent - Assessment manager and evaluation coordinator.

The Green Agent is responsible for orchestrating the complete benchmark workflow:
1. Loading test problems from the dataset
2. Distributing problems to the Purple Agent (code generator)
3. Collecting generated code
4. Compiling and executing code via MCP tools
5. Running comprehensive evaluation pipeline
6. Aggregating results and generating reports

Key features:
- Caching of Purple Agent responses for faster development iteration
- Comprehensive evaluation using gates, metrics, and quality assessments
- Detailed per-problem and aggregate reporting
- Support for both JSON and YAML configuration
"""

import os
import json
import time
import re
import pickle
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    TaskState,
    TextPart,
    SendMessageSuccessResponse,
)
from a2a.utils import get_message_text, new_agent_text_message, get_text_parts, get_file_parts
from src.util.a2a_comm import send_message
from pathlib import Path

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import dotenv

dotenv.load_dotenv()

import petscmcp
from petsc_compile_run_mcp_client import PetscCompileRunMCPClient

# Import evaluation system
from src.evaluators import EvaluationPipeline
from src.metrics import MetricsAggregator


def read_from_json(path):
    """Read all test problems from JSONL files in a directory.

    Each file should contain one JSON object per line, with fields:
    - problem_name: Unique identifier for the problem
    - problem_id: Numeric or string ID
    - problem_description: Natural language problem specification

    Args:
        path: Path to directory containing JSONL files

    Returns:
        List of problem dictionaries

    Raises:
        RuntimeError: If directory does not exist
    """
    if not os.path.isdir(path):
        raise RuntimeError(f"Directory {path} does not exist")

    data = []
    for file in Path(path).iterdir():
        if not os.path.isfile(file):
            continue
        with open(file, "r", encoding="utf-8") as fd:
            data.append(json.loads(fd.read().strip()))
    return data


@dataclass
class BenchmarkResult:
    """Container for a single problem's benchmark results.

    This dataclass stores both execution results and evaluation metrics
    for a single problem, providing a complete record of the assessment.

    Execution Results:
        problem_name: Human-readable problem identifier
        problem_id: Unique problem ID
        compiles: Whether the code compiled successfully
        runs: Whether the code executed without errors
        time_used_sec: Total time for generation + compilation + execution
        stdout: Program standard output
        stderr: Program standard error
        cli_args: Command-line arguments used for execution

    Evaluation Results:
        composite_score: Overall score 0-100 (weighted average of categories)
        tier: Performance tier (GOLD/SILVER/BRONZE/FAIL)
        category_scores: Scores by category (correctness, performance, etc.)
        evaluation_summary: High-level evaluation statistics
        evaluation_details: Detailed results from each evaluator
    """
    problem_name: str
    problem_id: str
    runs: bool
    time_used_sec: float
    compiles: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    cli_args: Optional[str] = None
    # Compilation fields
    compile_stdout: Optional[str] = None
    compile_stderr: Optional[str] = None
    # Evaluation fields
    composite_score: Optional[float] = None  # 0-100
    tier: Optional[str] = None  # GOLD/SILVER/BRONZE/FAIL
    category_scores: Optional[Dict[str, float]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_details: Optional[List[Dict[str, Any]]] = None


class Agent:
    """
    This class represents a green agent that manages assessment and evaluation of test tasks.

    The agent distributes test tasks to participant agents, collects their responses, and reports the results.
    """
    def __init__(self, config: Dict[str, Any], purple_agent_url, mcp_server_url, max_num_prob=None, use_cache=False, green_id=None, purple_id=None):
        self.config = config
        self.llm_config = config.get("evaluation", {}).get("llm", {})
        self.model = self.llm_config.get("model")
        self.api_base_url = self.llm_config.get("api_base_url")
        self.purple_agent_url = purple_agent_url
        self.mcp_client = PetscCompileRunMCPClient(mcp_server_url)
        self.max_num_prob = max_num_prob
        self.metrics = {}
        self.use_cache = use_cache
        self.green_id = green_id
        self.purple_id = purple_id
        # Create cache directory
        self.cache_dir = Path("./purple_agent_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize evaluation system with config
        self.evaluation_pipeline = EvaluationPipeline(config, self.model, self.api_base_url)
        self.metrics_aggregator = MetricsAggregator(config)
        print(f"@@@ Green agent: ✅ Evaluation system initialized with {self.evaluation_pipeline.get_evaluator_count()['total']} evaluators")

    def _get_cache_path(self, problem_name: str) -> Path:
        """Get the cache file path for a given problem.

        Sanitizes the problem name to create a valid filename.

        Args:
            problem_name: Original problem name (may contain special chars)

        Returns:
            Path object for the cache file
        """
        # Sanitize problem name for filename (replace non-alphanumeric with _)
        safe_name = re.sub(r'[^\w\-_]', '_', problem_name)
        return self.cache_dir / f"{safe_name}.pkl"

    def _load_cached_response(self, problem_name: str) -> Optional[Any]:
        """Load cached purple agent response if it exists."""
        cache_path = self._get_cache_path(problem_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"@@@ Green agent: ✅ Loaded cached response for {problem_name}")
                return cached_data
            except Exception as e:
                print(f"@@@ Green agent: ❌ Failed to load cache for {problem_name}: {e}")
                return None
        return None

    def _save_cached_response(self, problem_name: str, response: Any) -> None:
        """Save purple agent response to cache."""
        cache_path = self._get_cache_path(problem_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(response, f)
            print(f"@@@ Green agent: 💾 Cached response for {problem_name}")
        except Exception as e:
            print(f"@@@ Green agent: ❌ Failed to save cache for {problem_name}: {e}")

    async def _create_files_on_server(self, pname: str, file_list: List[Any], generated_codes: List[bytes]) -> str:
        """Upload generated files to MCP server.

        Args:
            pname: Project name prefix for generated files
            file_list: List of file parts from purple agent response
            generated_codes: List to append generated code bytes to

        Raises:
            RuntimeError: If file creation fails

        Returns:
        String of dependency file names separated by spaces
        """
        dep_list = []
        for f in file_list:
            generated_codes.append(f.bytes)
            parts = f.name.split('.')
            ext = parts[-1]
            # Use a base name to avoid overwriting multiple files of the same type
            if ext == "c":
                f.name = f"{pname}.c"
            elif ext == "cu":
                f.name = f"{pname}cu.cu"
                dep_list.append(f.name)
            if ext == "cpp" and len(parts) > 2 and parts[-2] == "kokkos":
                f.name = f"{pname}kok.kokkos.cpp"
                dep_list.append(f.name)
            created = await self.mcp_client.create_file_from_string(
                filename=f.name, file_contents=str(f.bytes)
                )
            if not created:
                raise RuntimeError(
                    'MCP tool create_file_from_string() returned false indicating the file was not created'
                )
        return " ".join(dep_list)

    async def _compile_code(self, br: BenchmarkResult, pname: str, dep_list: str) -> None:
        """Compile the generated code.
        Args:
            br: BenchmarkResult to update with compilation results
            pname: Problem/executable name
        """
        try:
            br.compile_stdout = await self.mcp_client.make(executable=pname, dependencies=dep_list)
            br.compile_stderr = self.mcp_client.response.stderr
            br.compiles = True
        except petscmcp.MCPDynamicClientReturnCode as e:
            br.compile_stdout = e.stdout
            br.compile_stderr = e.stderr
            br.compiles = False
            br.runs = False
            raise
        except petscmcp.MCPDynamicClientException as e:
            br.compile_stdout = ''
            br.compile_stderr = 'Error condition in accessing MCP server'
            br.compiles = False
            br.runs = False

    async def _run_executable(self, br: BenchmarkResult, pname: str, cli_args: str) -> None:
        """Run the compiled executable.
        Args:
            br: BenchmarkResult to update with execution results
            pname: Problem/executable name
            cli_args: Command line arguments for execution
        """
        try:
            br.stdout = await self.mcp_client.run_executable(executable=pname, args=cli_args)
            br.stderr = ""
            br.runs = True
        except petscmcp.MCPDynamicClientReturnCode as e:
            br.stdout = e.stdout
            br.stderr = e.stderr
            br.runs = False
            raise
        except petscmcp.MCPDynamicClientException as e:
            br.compile_stdout = ''
            br.compile_stderr = 'Error condition in accessing MCP server'
            br.compiles = False
            br.runs = False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Green agent implementation - manages assessment and evaluation.

        This Green agent distributes test tasks to participant agents and collects their response. No environment interaction or multiple steps for now.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use send_message(message, url) to call participant agents.
        """
        results: List[BenchmarkResult] = []
        summary: Dict[str, Any] = {
            "total": 0,
            "runs_count": 0,
            "failure_count": 0,
            "avg_time_sec": None,
            "avg_composite_score": None,
            "tier_distribution": {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAIL": 0},
        }

        # input_text = get_message_text(message)
        data_file_path = Path("./data")
        test_data = read_from_json(data_file_path)
        limit = self.max_num_prob or len(test_data)
        mcp_initialized = False

        for idx, data in enumerate(test_data[:limit], start=1):
            pname = data["problem_name"]
            pid = data["problem_id"]
            pdesc = data["problem_description"]

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[{idx}/{len(test_data)}] Running {pname}..."),
            )

            br = BenchmarkResult(
                problem_name=pname,
                problem_id=pid,
                runs=False,
                time_used_sec=0.0,
                compiles=False,
            )
            generated_codes = []

            try:
                # Try to load from cache first
                purple_agent_response = None
                if self.use_cache:
                    purple_agent_response = self._load_cached_response(pname)
                # If no cache, call the purple agent
                if purple_agent_response is None:
                    print(
                        f"@@@ Green agent: Sending message to purple agent... -->\n{pdesc}"
                    )
                    timestamp_started = time.time()
                    purple_agent_response = await send_message(
                        self.purple_agent_url,
                        pdesc,
                        context_id=pname,
                    )
                    br.time_used_sec = time.time() - timestamp_started
                else:
                    print(f"@@@ Green agent: Using cached response for {pname}")

                # Cache the response
                if self.use_cache:
                    self._save_cached_response(pname, purple_agent_response)

                res_root = purple_agent_response.root
                if not isinstance(res_root, SendMessageSuccessResponse):
                    raise ValueError(f"Expected SendMessageSuccessResponse, got {type(res_root).__name__}")
                res_result = res_root.result
                if not isinstance(res_result, Message):
                    raise ValueError(f"Expected Message, got {type(res_result).__name__}")
                text_list = get_text_parts(res_result.parts)
                file_list = get_file_parts(res_result.parts)
                if len(text_list) != 1:
                    raise ValueError(f"Expected exactly one text part from purple agent, got {len(text_list)}")
                # Parse response to find code
                _PATTERN = re.compile(
                    r"^Code generation successful[^\n]*\n"
                    r"nsize:\s*(?P<nsize>[^\n]+)\n"
                    r"cli_args:\s*(?P<cli_args>[^\n]+)\n",
                    re.DOTALL,
                )
                m = _PATTERN.search(text_list[0])
                if not m:
                    raise ValueError(
                        "Could not parse purple agent response. Probably failed to generate the code."
                    )
                # nsize = m.group("nsize")
                cli_args = m.group("cli_args")
                br.cli_args = cli_args
                print(
                    f"@@@ Green agent: Compile and run the code generated by purple agent..."
                )
                if not mcp_initialized:
                    await self.mcp_client.initialize()
                    mcp_initialized = True
                # Upload files to server
                dep_list = await self._create_files_on_server(pname, file_list, generated_codes)
                # Compile the code
                await self._compile_code(br, pname, dep_list)
                # Run the executable (only if compilation succeeded)
                if br.compiles:
                    await self._run_executable(br, pname, cli_args)

                # Run evaluation system
                print(f"@@@ Green agent: Evaluating generated code...")
                await self._evaluate_code(br, data, generated_codes)
                # Update rolling summary
                if br.runs:
                    summary["runs_count"] += 1
                else:
                    summary["failure_count"] += 1
                # Update evaluation summary
                if br.tier:
                    summary["tier_distribution"][br.tier] += 1
                # Optional: per-case artifact (useful for debugging)
                await updater.add_artifact(
                    name=f"benchmark_result_{pname}.json",
                    parts=[TextPart(text=json.dumps(asdict(br), indent=2))],
                )

            except Exception as e:
                # Log error, mark as failed, continue to next problem
                print(f"@@@ Green agent: ❌ Problem {pname} failed: {type(e).__name__}: {e}")
                br.tier = "FAIL"
                br.composite_score = 0.0
                br.evaluation_summary = {'error': str(e)}
                summary["failure_count"] += 1
                summary["tier_distribution"]["FAIL"] += 1

            finally:
                summary["total"] += 1
                results.append(br)

        if mcp_initialized:
            await self.mcp_client.finalize()
            mcp_initialized = False

        # Final summary artifact
        times = [r.time_used_sec for r in results]
        summary["avg_time_sec"] = (sum(times) / len(times)) if times else None

        # Calculate average evaluation score
        scores = [r.composite_score for r in results if r.composite_score is not None]
        summary["avg_composite_score"] = (sum(scores) / len(scores)) if scores else None

        # Save output to file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        local_path = output_dir / "benchmark_summary.json"
        json_data = {
            "agent": self.purple_id,
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        local_path.write_text(json.dumps(json_data, indent=2))
        await updater.add_artifact(
            name="benchmark_summary.json",
            parts=[TextPart(text=json.dumps(json_data, indent=2))],
            metadata=summary,
        )

        # Create evaluation summary report
        await self._create_evaluation_report(results, summary, updater)

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(
                f"Done. {summary['runs_count']}/{summary['total']} succeeded. "
                f"Avg score: {summary.get('avg_composite_score', 0):.1f}/100"
            ),
        )

    async def _evaluate_code(
        self,
        benchmark_result: BenchmarkResult,
        problem_data: Dict[str, Any],
        generated_codes: List[str],
    ) -> None:
        """Run evaluation pipeline on generated codes.

        Args:
            benchmark_result: BenchmarkResult to update with evaluation metrics
            problem_data: Original problem specification
            generated_codes: The generated codes
        """
        try:
            # Guard against empty generated_codes
            if not generated_codes:
                raise ValueError("No generated code to evaluate")

            # Prepare execution result for evaluators
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
                code=generated_codes[0],  # Focus on the main file for now
                problem=problem_data,
                execution_result=execution_result
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
            benchmark_result.evaluation_summary = {
                'total_evaluators': aggregated.total_evaluators,
                'passed_evaluators': aggregated.passed_evaluators,
                'failed_evaluators': aggregated.failed_evaluators,
                'all_gates_passed': aggregated.all_gates_passed,
                'gates_passed': aggregated.gates_passed,
                'gates_total': aggregated.gates_total,
            }
            # Store detailed evaluation results
            benchmark_result.evaluation_details = [
                {
                    'name': r.evaluator_name,
                    'type': r.evaluator_type.value,
                    'method': r.evaluation_method,
                    'passed': r.passed,
                    'score': r.quality_score or r.normalized_score,
                    'raw_value': r.raw_value,
                    'confidence': r.confidence,
                    'feedback': r.feedback,
                }
                for r in eval_results
            ]
            print(f"@@@ Green agent: ✅ Evaluation complete: Score={aggregated.composite_score:.1f}, Tier={aggregated.overall_tier}")
        except Exception as e:
            print(f"@@@ Green agent: ❌ Evaluation failed: {e}")
            raise # let ourter handler catch it

    async def _create_evaluation_report(
        self,
        results: List[BenchmarkResult],
        summary: Dict[str, Any],
        updater: TaskUpdater
    ) -> None:
        """Create a comprehensive evaluation report.

        Args:
            results: All benchmark results
            summary: Summary statistics
            updater: TaskUpdater for creating artifacts
        """
        report_lines = [
            "=" * 80,
            "EVALUATION REPORT",
            "=" * 80,
            "",
            f"Total Problems: {summary['total']}",
            f"Successful Executions: {summary['runs_count']}",
            f"Failed Executions: {summary['failure_count']}",
            f"Average Execution Time: {summary['avg_time_sec']:.2f}s",
            "",
            f"Average Composite Score: {summary['avg_composite_score']:.1f}/100",
            "",
            "Tier Distribution:",
            f"  🥇 GOLD:   {summary['tier_distribution']['GOLD']} ({summary['tier_distribution']['GOLD']/summary['total']*100:.1f}%)",
            f"  🥈 SILVER: {summary['tier_distribution']['SILVER']} ({summary['tier_distribution']['SILVER']/summary['total']*100:.1f}%)",
            f"  🥉 BRONZE: {summary['tier_distribution']['BRONZE']} ({summary['tier_distribution']['BRONZE']/summary['total']*100:.1f}%)",
            f"  ❌ FAIL:   {summary['tier_distribution']['FAIL']} ({summary['tier_distribution']['FAIL']/summary['total']*100:.1f}%)",
            "",
            "=" * 80,
            "PER-PROBLEM RESULTS",
            "=" * 80,
            "",
        ]

        for r in results:
            tier_emoji = {
                'GOLD': '🥇',
                'SILVER': '🥈',
                'BRONZE': '🥉',
                'FAIL': '❌'
            }.get(r.tier or 'FAIL', '❓')

            report_lines.append(f"{tier_emoji} {r.problem_name} (Score: {r.composite_score:.1f}/100)")
            if r.category_scores:
                report_lines.append(f"   Correctness: {r.category_scores['correctness']:.1f}, "
                                  f"Performance: {r.category_scores['performance']:.1f}, "
                                  f"Code Quality: {r.category_scores['code_quality']:.1f}")
            report_lines.append("")

        report_text = "\n".join(report_lines)
        print(report_text)
        # Save as artifact
        await updater.add_artifact(
            name="evaluation_report.txt",
            parts=[TextPart(text=report_text)],
        )

        # Also save detailed JSON
        detailed_report = {
            'summary': summary,
            'per_problem_scores': [
                {
                    'problem_name': r.problem_name,
                    'problem_id': r.problem_id,
                    'tier': r.tier,
                    'composite_score': r.composite_score,
                    'category_scores': r.category_scores,
                    'evaluation_summary': r.evaluation_summary,
                }
                for r in results
            ]
        }

        await updater.add_artifact(
            name="evaluation_detailed_report.json",
            parts=[TextPart(text=json.dumps(detailed_report, indent=2))],
        )



