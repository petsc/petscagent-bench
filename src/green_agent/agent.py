"""Green Agent - Assessment manager and evaluation coordinator.

The Green Agent is responsible for orchestrating the complete benchmark workflow:
1. Loading test problems from the dataset
2. Distributing problems to the Purple Agent (code generator)
3. Collecting generated code
4. Compiling and executing code via MCP tools
5. Running comprehensive evaluation pipeline
6. Aggregating results and generating reports

Key features:
- Replay of a recorded run's submissions for faster development iteration
- Comprehensive evaluation using gates, metrics, and quality assessments
- Detailed per-problem and aggregate reporting
- Support for both JSON and YAML configuration
"""

import os
import json
import time
import re
import fnmatch
import hashlib
import math
import numbers
import statistics
from collections import Counter
from a2a.server.tasks import TaskUpdater
from src.util.a2a_v1 import (
    Message, TaskState, StreamResponse, get_data, get_text_parts, get_file_parts,
    new_agent_parts_message, new_agent_text_message, new_data_part, new_raw_part,
    new_text_part, protobuf_size,
)
from src.util.a2a_comm import send_message
from src.util.telemetry import (
    PURPLE_TELEMETRY_FIELDS,
    PURPLE_TELEMETRY_INTEGER_FIELDS,
    PURPLE_TELEMETRY_MODEL_FIELD,
    PURPLE_TELEMETRY_SCHEMA,
)
from pathlib import Path

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import dotenv

dotenv.load_dotenv()

import petscmcp
from petsc_compile_run_mcp_client import PetscCompileRunMCPClient

# Import evaluation system
from src.evaluators import EvaluationPipeline
from src.metrics import MetricsAggregator


def _extract_purple_telemetry(parts: List[Any]) -> Optional[Dict[str, Any]]:
    """Return optional implementation-reported Purple Agent telemetry.

    Telemetry is deliberately carried in an A2A data Part so agents implemented
    with Claude Code, LangGraph, or any other framework can expose the same
    optional contract without the Green Agent depending on that framework.
    """
    for part in parts:
        if not part.HasField("data"):
            continue
        data = get_data(part)
        version = data.get("schema_version")
        if version != PURPLE_TELEMETRY_SCHEMA:
            # An agent that reports nothing and an agent whose schema has
            # moved on both yield zero telemetry, so name the second case.
            if (
                isinstance(version, str)
                and version.startswith("petscagent.telemetry.")
            ):
                print(
                    f"@@@ Green agent: ⚠️ Ignoring data Part with telemetry schema "
                    f"{version!r}; this benchmark reads {PURPLE_TELEMETRY_SCHEMA!r}"
                )
            continue
        telemetry = {"schema_version": PURPLE_TELEMETRY_SCHEMA}
        for field in PURPLE_TELEMETRY_FIELDS:
            metric = data.get(field)
            # Optional telemetry must never invalidate a usable solution.
            # Ignore malformed, negative, or non-finite values rather than
            # trusting arbitrary agent output.
            if (
                not isinstance(metric, numbers.Real)
                or isinstance(metric, bool)
                or not math.isfinite(metric)
                or metric < 0
            ):
                continue
            if field in PURPLE_TELEMETRY_INTEGER_FIELDS:
                # Counts must be whole. Truncating 1200.7 to 1200 would turn a
                # reporting bug into a number that looks trustworthy.
                if metric != int(metric):
                    continue
                metric = int(metric)
            telemetry[field] = metric
        # Optional self-reported model name (a string, not an aggregated metric)
        # that the agent wants recorded for this run; used for output naming.
        model = data.get(PURPLE_TELEMETRY_MODEL_FIELD)
        if isinstance(model, str) and model.strip():
            telemetry[PURPLE_TELEMETRY_MODEL_FIELD] = model.strip()
        return telemetry
    return None


def _a2a_payload_bytes(payload: Any) -> int:
    """Measure a serialized A2A protobuf payload.

    Requests and responses both go through this one function so the reported
    request and response byte counts are directly comparable. Measuring must
    never fail a usable solution, so an unmeasurable payload reports 0.
    """
    try:
        return protobuf_size(payload)
    except Exception as e:
        print(f"@@@ Green agent: ⚠️ Could not measure A2A payload size: {e}")
        return 0


def _purple_efficiency_summary(
    results: List["BenchmarkResult"], efficiency_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Aggregate measured and optional agent-declared Purple efficiency.

    Replayed cases are excluded from both halves. A replayed response carries
    the telemetry of an earlier run, so folding it in would report work that
    this run never performed. Per-problem records keep their telemetry
    regardless.
    """
    live = [r for r in results if not r.purple_response_replayed]
    times = [r.purple_wall_time_sec for r in live if r.purple_wall_time_sec is not None]
    first_response_times = [
        r.purple_time_to_first_response_sec
        for r in live
        if r.purple_time_to_first_response_sec is not None
    ]
    reported = [r.purple_telemetry for r in live if r.purple_telemetry is not None]
    scores = [r.efficiency_score for r in live if r.efficiency_score is not None]
    efficiency_config = efficiency_config or {}
    declared = {}
    for field in PURPLE_TELEMETRY_FIELDS:
        values = [item[field] for item in reported if item.get(field) is not None]
        declared[field] = {
            "reported_cases": len(values),
            "total": sum(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return {
        "benchmark_measured": {
            "request_count": sum(r.purple_request_count for r in live),
            "total_request_bytes": sum(r.purple_request_bytes for r in live),
            "total_response_bytes": sum(r.purple_response_bytes for r in live),
            "median_wall_time_sec": statistics.median(times) if times else None,
            "median_time_to_first_response_sec": (
                statistics.median(first_response_times) if first_response_times else None
            ),
            "response_event_count": sum(r.purple_response_event_count for r in live),
            "average_efficiency_score": sum(scores) / len(scores) if scores else None,
            "replayed_cases": sum(bool(r.purple_response_replayed) for r in results),
        },
        "agent_declared": declared,
        "score_budgets": {
            "time_budget_sec": float(efficiency_config.get("time_budget_sec", 300)),
            "response_bytes_budget": int(
                efficiency_config.get("response_bytes_budget", 100000)
            ),
        },
        "telemetry_reported_cases": len(reported),
        "live_cases": len(live),
        "total_cases": len(results),
    }


def _calculate_efficiency_score(
    result: "BenchmarkResult", efficiency_config: Dict[str, Any]
) -> Optional[float]:
    """Score Purple resource efficiency independently of solution quality."""
    if result.purple_response_replayed:
        return None

    evaluation = result.evaluation_summary or {}
    successful = bool(result.runs and evaluation.get("all_gates_passed", False))
    if not successful:
        return 0.0

    elapsed = result.purple_wall_time_sec
    response_bytes = result.purple_response_bytes
    time_budget = float(efficiency_config.get("time_budget_sec", 300))
    byte_budget = int(efficiency_config.get("response_bytes_budget", 100000))
    if not elapsed or elapsed <= 0 or response_bytes <= 0:
        return None
    if time_budget <= 0 or byte_budget <= 0:
        raise ValueError("Purple efficiency budgets must be positive")

    time_score = min(1.0, time_budget / elapsed)
    byte_score = min(1.0, byte_budget / response_bytes)
    return round(100.0 * math.sqrt(time_score * byte_score), 2)


def _reported_model(results):
    """The model a Purple Agent self-reported for this run, or None.

    Only live results are considered. A replayed response carries the
    telemetry of an earlier run, so the model string it holds names that run
    rather than this one. This is the same reason replayed results are
    excluded from _purple_efficiency_summary and _calculate_efficiency_score.
    A rescore names its run from the replay file instead, which the launcher
    reads before the task is sent.
    """
    for result in results:
        if result.purple_response_replayed or not result.purple_telemetry:
            continue
        model = result.purple_telemetry.get(PURPLE_TELEMETRY_MODEL_FIELD)
        if model:
            return model
    return None


def _slug(name):
    """Filename-safe short form of a model identifier."""
    return re.sub(r"[^a-z0-9]+", "", (name or "unknown").lower().split("/")[-1])


def _name_slug(name):
    """Filename-safe form that KEEPS hyphens, so a composite label such as
    ``pdesim-<model>-c<N>`` stays readable in the output filename instead of
    collapsing to ``pdesim<model>c<N>``. Other punctuation is dropped."""
    return re.sub(r"[^a-z0-9-]+", "", (name or "unknown").lower().split("/")[-1]).strip("-") or "unknown"


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

    # Sorted because iterdir() returns filesystem order, which made the
    # problem order, and so any count-based limit, differ between machines.
    data = []
    for file in sorted(Path(path).iterdir()):
        if not os.path.isfile(file):
            continue
        with open(file, "r", encoding="utf-8") as fd:
            problem = json.loads(fd.read().strip())
            problem["source_file"] = file.name  # for the problems listing
            data.append(problem)
    return data


def select_problems(test_data, spec):
    """Narrow `test_data` to the comma-separated terms in `spec`.

    Terms match problem_name ignoring case. A term holding a wildcard is a
    glob matched against the whole name; any other term matches a substring.

    Args:
        test_data: Problem dicts from read_from_json.
        spec: Comma-separated terms, or None to keep everything.

    Returns:
        The matching problems in test_data order, each one once.

    Raises:
        ValueError: If `spec` holds no term, or a term matches no problem.
            Either would otherwise silently change which problems run.
    """
    if not spec:
        return test_data

    terms = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not terms:
        # Separators only. Returning everything here would turn a typo into a
        # full run of the benchmark.
        raise ValueError(f"no problem names in: {spec!r}")

    def matches(term, name):
        if any(c in term for c in "*?["):
            return fnmatch.fnmatch(name, term)
        return term in name

    unmatched = [
        term
        for term in terms
        if not any(matches(term, d["problem_name"].lower()) for d in test_data)
    ]
    if unmatched:
        available = ", ".join(sorted(d["problem_name"] for d in test_data))
        raise ValueError(
            f"no problem matches: {', '.join(unmatched)}\navailable: {available}"
        )

    return [
        d
        for d in test_data
        if any(matches(term, d["problem_name"].lower()) for term in terms)
    ]


@dataclass
class TestCaseResult:
    """Record of one executable invocation."""

    index: int
    args: str
    nsize: int
    runs: bool
    stdout: str = ""
    stderr: str = ""
    execution_time_sec: Optional[float] = None
    valgrind_output: Optional[str] = None


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
        purple_wall_time_sec: Wall-clock time spent waiting for the Purple Agent
        stdout: Program standard output
        stderr: Program standard error
        cli_args: Command-line arguments used for execution
        cases: Canonical result records for every executable invocation. The
            scalar execution fields are retained as a compatibility view.

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
    compiles: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    cli_args: Optional[str] = None
    cases: List[TestCaseResult] = field(default_factory=list)
    requested_nsize: Optional[int] = None
    actual_nsize: Optional[int] = None
    execution_time_sec: Optional[float] = None  # Code execution time only
    total_execution_time_sec: Optional[float] = None
    valgrind_output: Optional[str] = None
    generated_sources: Optional[List[Dict[str, str]]] = None
    # Token cost of the purple agent's code-generation call (metadata only,
    # not part of any score). prompt=input, completion=output, cached=prompt
    # tokens served from cache (0 when caching is not in use).
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    # Framework-independent Purple Agent efficiency. Boundary fields are
    # measured by Green; internal telemetry is optional and agent-declared.
    purple_wall_time_sec: Optional[float] = None
    purple_time_to_first_response_sec: Optional[float] = None
    purple_request_count: int = 0
    purple_request_bytes: int = 0
    purple_response_bytes: int = 0
    purple_response_event_count: int = 0
    purple_response_replayed: bool = False
    purple_telemetry: Optional[Dict[str, Any]] = None
    efficiency_score: Optional[float] = None
    # Compilation fields
    compile_stdout: Optional[str] = None
    compile_stderr: Optional[str] = None
    # Evaluation fields
    composite_score: Optional[float] = None  # 0-100
    tier: Optional[str] = None  # GOLD/SILVER/BRONZE/FAIL
    category_scores: Optional[Dict[str, float]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_details: Optional[List[Dict[str, Any]]] = None
    scored_at: Optional[str] = None     # UTC ISO 8601, one value per pass
    scored_by: Optional[str] = None     # the judge model that scored it


def _derive_summary(results: List["BenchmarkResult"]) -> Dict[str, Any]:
    """Count the run's outcomes from the list that gets written."""
    counts = Counter(r.tier for r in results)
    tiers = {t: counts.get(t, 0) for t in ("GOLD", "SILVER", "BRONZE", "FAIL")}
    runs_count = sum(1 for r in results if r.runs)
    return {
        "total": len(results),
        "runs_count": runs_count,
        "failure_count": len(results) - runs_count,
        "avg_purple_wall_time_sec": None,
        "avg_composite_score": None,
        "tier_distribution": tiers,
    }


class Agent:
    """
    This class represents a green agent that manages assessment and evaluation of test tasks.

    The agent distributes test tasks to participant agents, collects their responses, and reports the results.
    """
    def __init__(self, config: Dict[str, Any], purple_agent_url, mcp_server_url, max_num_prob=None, green_id=None, purple_id=None, purple_model=None, replay_path=None, problems=None):
        self.config = config
        self.llm_config = config.get("evaluation", {}).get("llm", {})
        self.model = self.llm_config.get("model")
        self.api_base_url = self.llm_config.get("api_base_url")
        self.purple_agent_url = purple_agent_url
        self.mcp_client = PetscCompileRunMCPClient(mcp_server_url)
        self.max_num_prob = max_num_prob
        # Comma-separated terms narrowing the problem set, None for all of it.
        self.problems = problems
        if replay_path and problems:
            # A rescore must cover the whole recorded set. Scoring a subset
            # would write a file holding only those problems, and replay
            # reads submissions from the file rather than the tree, so the
            # ones left out could never be rescored again.
            raise ValueError(
                "--problems cannot be combined with --replay: a rescore "
                "always sweeps the whole recorded run"
            )
        # A rescore writes beside the run it replays rather than on top of it,
        # so both passes survive.
        self.output_dir = Path(replay_path).parent if replay_path else Path("output")
        self.metrics = {}
        self.green_id = green_id
        self.purple_id = purple_id
        self.purple_model = purple_model
        # Submissions recorded by an earlier run, keyed by problem name. When
        # set, the Purple Agent is never called: a rescore then varies only
        # the judge, so a score difference is attributable to it.
        self.replay_index = None
        if replay_path:
            record = json.loads(Path(replay_path).read_text())
            self.replay_index = {
                r["problem_name"]: r for r in record.get("results", [])
            }
            print(
                f"@@@ Green agent: ✅ Replaying {len(self.replay_index)} recorded "
                f"submissions from {replay_path}"
            )

        # Initialize evaluation system with config
        self.evaluation_pipeline = EvaluationPipeline(config, self.model, self.api_base_url)
        self.metrics_aggregator = MetricsAggregator(config)
        print(f"@@@ Green agent: ✅ Evaluation system initialized with {self.evaluation_pipeline.get_evaluator_count()['total']} evaluators")

    def _replay_response(self, problem_name: str) -> Any:
        """Rebuild the Purple Agent response recorded for a problem.

        The recorded sources are replayed under their original filenames, so
        the same normalization and the same compile and run path apply as on
        the run that produced them.

        Raises:
            ValueError: If the replay file has no usable submission.
        """
        record = self.replay_index.get(problem_name)
        if record is None:
            raise ValueError(f"Replay file has no record for {problem_name}")
        sources = record.get("generated_sources")
        nsize = record.get("requested_nsize")
        cli_args = record.get("cli_args")
        if not sources:
            # The purple produced nothing and a rescore cannot discover
            # otherwise, so re-raise the record's own diagnosis rather than
            # one about the replay machinery.
            raise ValueError(
                (record.get("evaluation_summary") or {}).get("error")
                or f"Replay record for {problem_name} has no sources"
            )
        # An empty cli_args is a valid submission, so only an absent one is an
        # error. Every test case supplies its own args, which means the value
        # replayed here is a fallback that has to parse rather than a value the
        # executions depend on.
        if nsize is None or cli_args is None:
            raise ValueError(
                f"Replay record for {problem_name} has no nsize or cli_args"
            )
        parts = [
            new_text_part(
                f"Code generation successful ✅\nnsize: {nsize}\ncli_args: {cli_args}\n"
            )
        ]
        # Carry the original telemetry so the record states what generated the
        # code. It stays out of this run's live aggregates because a replayed
        # result is flagged as not freshly generated.
        telemetry = record.get("purple_telemetry")
        if telemetry:
            parts.append(new_data_part(telemetry))
        for source in sources:
            parts.append(new_raw_part(
                source["source"].encode("utf-8"),
                filename=source["original_name"], media_type="text/plain",
            ))
        return StreamResponse(
            message=new_agent_parts_message(parts, context_id=problem_name)
        )

    async def _create_files_on_server(self, pname: str, file_list: List[Any], generated_sources: List[Dict[str, str]]) -> str:
        """Upload generated files to MCP server.

        Args:
            pname: Project name prefix for generated files
            file_list: List of file parts from purple agent response
            generated_sources: List to append source records to

        Raises:
            RuntimeError: If file creation fails

        Returns:
        String of dependency file names separated by spaces
        """
        dep_list = []
        used_names = set()
        for f in file_list:
            source = f.raw.decode("utf-8")
            original_name = f.filename
            safe_name = Path(original_name).name
            if not safe_name or safe_name in {".", ".."}:
                raise ValueError(f"Invalid generated filename: {original_name!r}")
            parts = safe_name.split('.')
            ext = parts[-1]
            server_name = safe_name
            # Use a base name to avoid overwriting multiple files of the same type
            if ext == "c":
                server_name = f"{pname}.c"
            elif ext == "cu":
                server_name = f"{pname}cu.cu"
                dep_list.append(server_name)
            if ext == "cpp" and len(parts) > 2 and parts[-2] == "kokkos":
                server_name = f"{pname}kok.kokkos.cpp"
                dep_list.append(server_name)
            if server_name in used_names:
                raise ValueError(f"Duplicate generated filename after normalization: {server_name}")
            used_names.add(server_name)
            generated_sources.append({
                "original_name": original_name,
                "server_name": server_name,
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "source": source,
            })
            created = await self.mcp_client.create_file_from_string(
                filename=server_name, file_contents=source
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

    async def _run_executable(self, br: BenchmarkResult, pname: str, nsize: int, cli_args: str,
                              valgrind: bool = True) -> None:
        """Run the compiled executable.
        Args:
            br: BenchmarkResult to update with execution results
            pname: Problem/executable name
            cli_args: Command line arguments for execution
            valgrind: Whether this run may also be instrumented
        """
        try:
            t0 = time.time()
            br.stdout = await self.mcp_client.run_executable(
                executable=pname, nsize=nsize, args=cli_args
            )
            br.execution_time_sec = time.time() - t0
            br.actual_nsize = nsize
            br.stderr = ""
            br.runs = True

            if valgrind and self.config.get("memory_safety", {}).get("use_valgrind", False):
                try:
                    await self.mcp_client.run_executable(
                        executable=pname, nsize=nsize, args=cli_args, valgrind=True
                    )
                    response = getattr(self.mcp_client, "response", None)
                    br.valgrind_output = getattr(response, "stderr", None)
                except petscmcp.MCPDynamicClientReturnCode as e:
                    # Valgrind reports are useful even when the instrumented
                    # process exits nonzero; the memory gate parses the text.
                    br.valgrind_output = e.stderr
                except petscmcp.MCPDynamicClientException:
                    # Preserve the normal run. The memory gate will use its
                    # documented stderr fallback when instrumentation is not
                    # available on the server.
                    br.valgrind_output = None
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

    async def _run_test_cases(
        self,
        br: BenchmarkResult,
        pname: str,
        problem: Dict[str, Any],
        nsize: int,
        cli_args: str,
    ) -> None:
        """Run every case and derive legacy scalar fields from case 0."""
        test_cases = problem.get("test_cases") or [{}]
        results: List[TestCaseResult] = []
        for idx, test_case in enumerate(test_cases):
            case_args = test_case.get("args", cli_args)
            case_nsize = int(test_case.get("nsize", nsize))
            print(f"@@@ Green agent: test case {idx} (nsize {case_nsize}) {case_args}")
            case_br = BenchmarkResult(
                problem_name=br.problem_name,
                problem_id=br.problem_id,
                runs=False,
                compiles=br.compiles,
            )
            try:
                await self._run_executable(
                    case_br, pname, case_nsize, case_args, valgrind=True
                )
            except petscmcp.MCPDynamicClientReturnCode:
                # A failed case is still a result. Continue so the execution
                # gate can report all failures instead of losing prior cases.
                pass
            results.append(TestCaseResult(
                index=idx,
                args=case_args,
                nsize=case_nsize,
                runs=case_br.runs,
                stdout=case_br.stdout or "",
                stderr=case_br.stderr or "",
                execution_time_sec=case_br.execution_time_sec,
                valgrind_output=case_br.valgrind_output,
            ))

        br.cases = results
        first = results[0]
        br.stdout = first.stdout
        br.stderr = first.stderr
        br.cli_args = first.args
        br.actual_nsize = first.nsize
        br.runs = all(case.runs for case in results)
        # The scalar field keeps its old meaning, the runtime of the first
        # invocation. Cases of different sizes have no meaningful mean.
        br.execution_time_sec = first.execution_time_sec
        runtimes = [
            case.execution_time_sec for case in results
            if case.execution_time_sec is not None
        ]
        br.total_execution_time_sec = sum(runtimes) if runtimes else None
        reports = [
            f"test case {case.index}:\n{case.valgrind_output}"
            for case in results if case.valgrind_output is not None
        ]
        br.valgrind_output = "\n".join(reports) if reports else None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Green agent implementation - manages assessment and evaluation.

        This Green agent distributes test tasks to participant agents and collects their response. No environment interaction or multiple steps for now.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use send_message(message, url) to call participant agents.
        """
        results: List[BenchmarkResult] = []

        # input_text = get_message_text(message)
        data_file_path = Path("./data")
        all_data = read_from_json(data_file_path)
        test_data = select_problems(all_data, self.problems)
        limit = self.max_num_prob or len(test_data)
        selected = test_data[:limit]
        if self.problems:
            print(
                f"@@@ Green agent: Running {len(selected)} problem(s) matching "
                f"'{self.problems}': {', '.join(d['problem_name'] for d in selected)}"
            )
        if self.replay_index is not None:
            recorded = set(self.replay_index)
            present = {d["problem_name"] for d in selected}
            if recorded != present:
                # Neither direction can be scored honestly. A recorded problem
                # missing from data/ has no specification to grade against, and
                # a problem only in data/ has no submission to replay and would
                # otherwise be recorded as a FAIL at zero.
                raise ValueError(
                    "replay file and data/ disagree, refusing to rescore a "
                    "partial set. Only in the replay file: "
                    f"{', '.join(sorted(recorded - present)) or 'none'}. "
                    "Only in data/: "
                    f"{', '.join(sorted(present - recorded)) or 'none'}"
                )

        mcp_initialized = False
        # Taken once, before the loop, so the whole pass shares one value
        # rather than each record holding the moment it happened to finish.
        pass_stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for idx, data in enumerate(selected, start=1):
            pname = data["problem_name"]
            pid = data["problem_id"]
            pdesc = data["problem_description"]

            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                new_agent_text_message(f"[{idx}/{len(selected)}] Running {pname}..."),
            )

            br = BenchmarkResult(
                problem_name=pname,
                problem_id=pid,
                runs=False,
                compiles=False,
                # Stamped here rather than after a successful evaluation, so a
                # problem that fails below is still attributed to this pass
                # instead of leaving the field null.
                scored_at=pass_stamp,
                scored_by=self.model,
            )
            generated_sources = []

            try:
                # Replay a recorded submission when one was supplied
                purple_agent_response = None
                if self.replay_index is not None:
                    # Flagged before the replay, so a record with nothing to
                    # replay still counts as replayed and stays out of this
                    # pass's aggregates over live purple calls.
                    br.purple_response_replayed = True
                    purple_agent_response = self._replay_response(pname)
                # Otherwise ask the purple agent to generate one
                if purple_agent_response is None:
                    print(
                        f"@@@ Green agent: Sending message to purple agent... -->\n{pdesc}"
                    )
                    def record_boundary_metrics(metrics, br=br):
                        br.purple_request_count = metrics.request_count
                        br.purple_request_bytes = metrics.request_bytes
                        br.purple_response_bytes = metrics.response_bytes
                        br.purple_response_event_count = metrics.response_event_count
                        br.purple_time_to_first_response_sec = (
                            metrics.time_to_first_response_sec
                        )
                        br.purple_wall_time_sec = metrics.wall_time_sec

                    purple_agent_response = await send_message(
                        self.purple_agent_url,
                        pdesc,
                        context_id=pname,
                        on_metrics=record_boundary_metrics,
                    )
                else:
                    print(f"@@@ Green agent: Using replayed response for {pname}")

                if not isinstance(purple_agent_response, StreamResponse):
                    raise ValueError(f"Expected StreamResponse, got {type(purple_agent_response).__name__}")
                if not purple_agent_response.HasField("message"):
                    raise ValueError("Expected a Message response from Purple Agent")
                res_result = purple_agent_response.message
                text_list = get_text_parts(res_result.parts)
                file_list = get_file_parts(res_result.parts)
                br.purple_telemetry = _extract_purple_telemetry(res_result.parts)
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
                try:
                    nsize = int(m.group("nsize").strip())
                except ValueError as exc:
                    raise ValueError("Purple agent nsize must be an integer") from exc
                max_nsize = int(self.config.get("execution", {}).get("max_nsize", 64))
                if not 1 <= nsize <= max_nsize:
                    raise ValueError(f"Purple agent nsize must be between 1 and {max_nsize}")
                cli_args = m.group("cli_args")
                br.cli_args = cli_args
                br.requested_nsize = nsize
                # Token usage is optional, so that agents which do not report
                # it still parse correctly.
                tok = re.search(r"prompt_tokens:\s*(\d+)\s*\ncompletion_tokens:\s*(\d+)", text_list[0])
                if tok:
                    br.prompt_tokens = int(tok.group(1))
                    br.completion_tokens = int(tok.group(2))
                    br.total_tokens = br.prompt_tokens + br.completion_tokens
                    for field, pat in (("total_tokens", r"total_tokens:\s*(\d+)"),
                                       ("cached_tokens", r"cached_tokens:\s*(\d+)")):
                        m2 = re.search(pat, text_list[0])
                        if m2:
                            setattr(br, field, int(m2.group(1)))
                # Prefer the versioned telemetry artifact when present, while
                # retaining legacy text token fields during migration.
                if br.purple_telemetry:
                    for target, source in (
                        ("prompt_tokens", "input_tokens"),
                        ("completion_tokens", "output_tokens"),
                        ("total_tokens", "total_tokens"),
                        ("cached_tokens", "cached_tokens"),
                    ):
                        # Token fields are validated as whole numbers by
                        # _extract_purple_telemetry, so no conversion here.
                        value = br.purple_telemetry.get(source)
                        if value is not None:
                            setattr(br, target, value)
                print(
                    f"@@@ Green agent: Compile and run the code generated by purple agent..."
                )
                if not mcp_initialized:
                    await self.mcp_client.initialize()
                    mcp_initialized = True
                # Upload files to server
                dep_list = await self._create_files_on_server(pname, file_list, generated_sources)
                br.generated_sources = generated_sources
                # Compile the code
                await self._compile_code(br, pname, dep_list)
                # Run the executable (only if compilation succeeded)
                if br.compiles:
                    await self._run_test_cases(br, pname, data, nsize, cli_args)

                # Run evaluation system
                print(f"@@@ Green agent: Evaluating generated code...")
                await self._evaluate_code(br, data, generated_sources)
                br.efficiency_score = _calculate_efficiency_score(
                    br, self.config.get("scoring", {}).get("efficiency", {})
                )
                # Optional: per-case artifact (useful for debugging)
                await updater.add_artifact(
                    name=f"benchmark_result_{pname}.json",
                    parts=[new_text_part(json.dumps(asdict(br), indent=2))],
                )

            except Exception as e:
                # Log error, mark as failed, continue to next problem
                print(f"@@@ Green agent: ❌ Problem {pname} failed: {type(e).__name__}: {e}")
                br.tier = "FAIL"
                br.composite_score = 0.0
                br.evaluation_summary = {'error': str(e)}
                br.efficiency_score = _calculate_efficiency_score(
                    br, self.config.get("scoring", {}).get("efficiency", {})
                )

            finally:
                results.append(br)

        if mcp_initialized:
            await self.mcp_client.finalize()
            mcp_initialized = False

        summary = _derive_summary(results)

        # Final summary artifact
        times = [
            r.purple_wall_time_sec
            for r in results
            if not r.purple_response_replayed and r.purple_wall_time_sec is not None
        ]
        summary["avg_purple_wall_time_sec"] = (
            sum(times) / len(times) if times else None
        )

        # Calculate average evaluation score
        scores = [r.composite_score for r in results if r.composite_score is not None]
        summary["avg_composite_score"] = (sum(scores) / len(scores)) if scores else None

        # Token cost of code generation across the suite (metadata only).
        # prompt=input, completion=output, cached=prompt tokens served from cache.
        live_results = [r for r in results if not r.purple_response_replayed]
        summary["total_prompt_tokens"] = sum(r.prompt_tokens or 0 for r in live_results)
        summary["total_completion_tokens"] = sum(r.completion_tokens or 0 for r in live_results)
        summary["total_tokens"] = sum(r.total_tokens or 0 for r in live_results)
        summary["total_cached_tokens"] = sum(r.cached_tokens or 0 for r in live_results)
        summary["purple_efficiency"] = _purple_efficiency_summary(
            results, self.config.get("scoring", {}).get("efficiency", {})
        )

        # Save output as <purple_model>-judged-by-<green_model>-run<N>.json so that
        # repeated launches do not overwrite each other. Prefer the model the Purple
        # self-reported in its telemetry (so a composite agent can label itself, e.g.
        # "pdesim-<model>-c<N>"), falling back to the purple_model task tag.
        reported_model = _reported_model(results)
        effective_model = reported_model or self.purple_model
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # _name_slug keeps hyphens so a composite label stays readable.
        model_slug = _name_slug(effective_model)
        judge_slug = _name_slug(self.model)
        prefix = f"{model_slug}-judged-by-{judge_slug}"
        # Take the next free index by creating its directory, which is atomic
        # and so settles a race between two tasks finishing at once. The loser
        # sees FileExistsError and takes the next index rather than losing a
        # whole run's results to the winner's aggregate.
        while True:
            # Count every artifact a run leaves behind, or an index whose JSON
            # was deleted but whose tree survived would be reused forever.
            # "sources" is the pre-rename name, still present in output/.
            used = [
                int(m.group(1))
                for p in [
                    *output_dir.glob(f"{prefix}-run*.json"),
                    *(output_dir / "runs").glob(f"{prefix}-run*"),
                    *(output_dir / "sources").glob(f"{prefix}-run*"),
                ]
                if (m := re.search(r"-run(\d+)(?:\.json)?$", p.name))
            ]
            run_index = max(used, default=0) + 1
            run_dir = output_dir / "runs" / f"{prefix}-run{run_index}"
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                # The rescan now counts this directory, so the index rises and
                # the loop cannot spin.
                continue

        local_path, json_data = self._write_aggregate(
            results, summary, pass_stamp,
            output_dir, run_dir, prefix, run_index, reported_model,
        )
        print(f"@@@ Green agent: Saved results to {local_path}")
        await updater.add_artifact(
            name=local_path.name,
            parts=[new_text_part(json.dumps(json_data, indent=2))],
            metadata=summary,
        )

        # Create evaluation summary report
        await self._create_evaluation_report(results, summary, updater)

        avg = summary["avg_composite_score"]
        await updater.update_status(
            TaskState.TASK_STATE_COMPLETED,
            new_agent_text_message(
                f"Done. {summary['runs_count']}/{summary['total']} succeeded. "
                f"Avg score: {f'{avg:.1f}/100' if avg is not None else 'n/a'}"
            ),
        )

    def _write_aggregate(
        self, results, summary, pass_stamp,
        output_dir, run_dir, prefix, run_index, reported_model,
    ):
        """Write the run's aggregate JSON and its run tree.

        The run index and directory are decided by the caller, because
        allocating them is what settles a race between two runs finishing at
        once.

        Returns:
            (path, json_data) for the caller to report and attach.
        """
        local_path = output_dir / f"{prefix}-run{run_index}.json"
        # Built once so the aggregate and the per-problem files below hold the
        # same data. They are indented differently, being at different depths.
        records = [asdict(r) for r in results]
        json_data = {
            "agent": self.purple_id,
            # purple_model is the tag the run was launched with, so the record
            # always states what was configured. reported_model is what the
            # agent said about itself, which names the file but is not a
            # substitute for the configured value.
            "purple_model": self.purple_model,
            "reported_model": reported_model,
            "judge_model": self.model,
            "run_index": run_index,
            # Null for a full run. Never set on a rescore, which always
            # sweeps the whole recorded set.
            "problem_filter": self.problems,
            "scored_at": pass_stamp,
            "summary": summary,
            "results": records,
        }
        local_path.write_text(json.dumps(json_data, indent=2))
        source_manifest = []
        for result, record in zip(results, records):
            problem_dir = run_dir / _slug(result.problem_name)
            problem_dir.mkdir(exist_ok=True)
            (problem_dir / "result.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            for source_record in result.generated_sources or []:
                source_path = problem_dir / source_record["server_name"]
                source_path.write_text(source_record["source"], encoding="utf-8")
                source_manifest.append({
                    "problem_name": result.problem_name,
                    "filename": str(source_path.relative_to(run_dir)),
                    "sha256": source_record["sha256"],
                })
        (run_dir / "manifest.json").write_text(
            json.dumps(source_manifest, indent=2), encoding="utf-8"
        )
        return local_path, json_data

    async def _evaluate_code(
        self,
        benchmark_result: BenchmarkResult,
        problem_data: Dict[str, Any],
        generated_sources: List[Dict[str, str]],
    ) -> None:
        """Run evaluation pipeline on generated codes.

        Args:
            benchmark_result: BenchmarkResult to update with evaluation metrics
            problem_data: Original problem specification
            generated_sources: Generated source records
        """
        try:
            if not generated_sources:
                raise ValueError("No generated code to evaluate")

            code = "\n\n".join(
                f"/* FILE: {item['server_name']} */\n{item['source']}"
                for item in generated_sources
            )

            # Prepare execution result for evaluators. Execution is described
            # by its cases. The scalar fields stay on BenchmarkResult for the
            # stored output, and are deliberately not duplicated here.
            execution_result = {
                'compiles': benchmark_result.compiles,
                'stdout': benchmark_result.stdout or '',
                'cases': [asdict(case) for case in benchmark_result.cases],
                'memory_mb': None,  # TODO: Add memory tracking if available
            }
            # Run evaluation pipeline
            eval_results = await self.evaluation_pipeline.evaluate(
                code=code,
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
        def pct(tier):
            # A run with nothing in it is a report to write, not a crash.
            total = summary["total"]
            return f"{summary['tier_distribution'][tier] / total * 100:.1f}%" if total else "n/a"

        report_lines = [
            "=" * 80,
            "EVALUATION REPORT",
            "=" * 80,
            "",
            f"Total Problems: {summary['total']}",
            f"Successful Executions: {summary['runs_count']}",
            f"Failed Executions: {summary['failure_count']}",
            (
                f"Average Purple Agent Time: {summary['avg_purple_wall_time_sec']:.2f}s"
                if summary["avg_purple_wall_time_sec"] is not None
                else "Average Purple Agent Time: n/a"
            ),
            "",
            (
                f"Average Composite Score: {summary['avg_composite_score']:.1f}/100"
                if summary.get("avg_composite_score") is not None
                else "Average Composite Score: n/a"
            ),
            "",
            "Tier Distribution:",
            *(
                f"  {emoji} {name + ':':<7} {summary['tier_distribution'][name]} ({pct(name)})"
                for emoji, name in (
                    ("🥇", "GOLD"), ("🥈", "SILVER"), ("🥉", "BRONZE"), ("❌", "FAIL"),
                )
            ),
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
            parts=[new_text_part(report_text)],
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
            parts=[new_text_part(json.dumps(detailed_report, indent=2))],
        )
