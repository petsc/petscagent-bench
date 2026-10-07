import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).parents[1]


sys.path.insert(0, str(ROOT))

from src.evaluators.quality.petsc_quality.error_handling import ErrorHandlingQuality


class FakeMCPClient:
    def __init__(self):
        self.calls = []
        self.response = SimpleNamespace(stderr="ERROR SUMMARY: 0 errors from 0 contexts")

    async def run_executable(self, **kwargs):
        self.calls.append(kwargs)
        return "program output"

    async def create_file_from_string(self, **kwargs):
        self.calls.append(kwargs)
        return True


class FakeFile:
    def __init__(self, name, source):
        self.filename = name
        self.raw = source.encode("utf-8")


class CodeFixTests(unittest.TestCase):
    def test_error_handling_prefers_petsccall(self):
        evaluator = ErrorHandlingQuality()
        modern = """
        PetscErrorCode f(Vec x) {
          PetscCall(VecSet(x, 1.0));
          PetscCall(VecAssemblyBegin(x));
          PetscCall(VecAssemblyEnd(x));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
        """
        legacy = """
        PetscErrorCode f(Vec x) {
          ierr = VecSet(x, 1.0);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
          return 0;
        }
        """
        unchecked = """
        PetscErrorCode f(Vec x) {
          VecSet(x, 1.0);
          VecAssemblyBegin(x);
          VecAssemblyEnd(x);
          return 0;
        }
        """

        modern_result = asyncio.run(evaluator.evaluate(modern, {}))
        legacy_result = asyncio.run(evaluator.evaluate(legacy, {}))
        unchecked_result = asyncio.run(evaluator.evaluate(unchecked, {}))

        self.assertEqual(modern_result.quality_score, 1.0)
        self.assertEqual(legacy_result.quality_score, 0.8)
        self.assertEqual(unchecked_result.quality_score, 0.0)
        self.assertEqual(modern_result.metadata["petsc_call_count"], 3)
        self.assertEqual(legacy_result.metadata["chkerrq_count"], 3)

    def test_error_handling_ignores_comments_and_wrapper_names_as_api_calls(self):
        counts = ErrorHandlingQuality._counts(
            "// CHKERRQ(ierr)\nPetscCall(VecSet(x, 0)); /* VecDestroy(&x); */"
        )
        self.assertEqual(counts, (1, 1, 0))

    def test_llm_evaluators_do_not_truncate_source(self):
        for path in (
            "src/evaluators/quality/algorithm_quality/algorithm_appropriateness.py",
            "src/evaluators/quality/algorithm_quality/solver_choice.py",
            "src/evaluators/quality/petsc_quality/best_practices.py",
        ):
            text = (ROOT / path).read_text()
            self.assertNotIn("code[:2000]", text)
            self.assertIn("{code}", text)

    def test_hybrid_methods_are_implemented(self):
        style = (ROOT / "src/evaluators/quality/code_quality/code_style.py").read_text()
        solver = (ROOT / "src/evaluators/quality/algorithm_quality/solver_choice.py").read_text()
        practices = (ROOT / "src/evaluators/quality/petsc_quality/best_practices.py").read_text()
        self.assertIn("+static_analysis", style)
        self.assertIn("static_score", style)
        self.assertIn("+heuristic", solver)
        self.assertIn("heuristic_score", solver)
        self.assertIn("+patterns", practices)
        self.assertIn("pattern_score", practices)

    def test_api_gate_checks_order_and_ignores_comments(self):
        text = (ROOT / "src/evaluators/gates/api_usage_gate.py").read_text()
        self.assertIn("initialization_precedes_finalization", text)
        self.assertIn("re.DOTALL", text)

    def test_configuration_preserves_artifact_defaults(self):
        text = (ROOT / "config/green_agent_config.yaml").read_text()
        for expected in (
            'model: "anthropic/claudeopus46"',
            'api_base_url: "https://apps.inside.anl.gov/argoapi"',
            "tolerance: 1.0e-3",
            "excellent_time_sec: 0.5",
            "good_time_sec: 1.0",
            "acceptable_time_sec: 2.0",
            "max_time_sec: 4.0",
            "max_nsize: 64",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("use_valgrind: true", text)

    def test_agent_forwards_rank_and_archives_sources(self):
        text = (ROOT / "src/green_agent/agent.py").read_text()
        self.assertIn("executable=pname, nsize=nsize, args=cli_args", text)
        self.assertIn('"sha256": hashlib.sha256', text)
        self.assertIn('"source": source', text)
        self.assertIn("code=code", text)

    def test_rank_forwarding_and_valgrind_behavior(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        agent = Agent.__new__(Agent)
        agent.config = {"memory_safety": {"use_valgrind": True}}
        agent.mcp_client = FakeMCPClient()
        result = BenchmarkResult("parallel", "p1", False, True)

        asyncio.run(agent._run_executable(result, "parallel", 3, "-ksp_type cg"))

        self.assertTrue(result.runs)
        self.assertEqual(result.requested_nsize, None)
        self.assertEqual(result.actual_nsize, 3)
        self.assertEqual(agent.mcp_client.calls[0]["nsize"], 3)
        self.assertEqual(agent.mcp_client.calls[1]["nsize"], 3)
        self.assertTrue(agent.mcp_client.calls[1]["valgrind"])

    def test_purple_telemetry_is_optional_structured_a2a_data(self):
        from unittest import mock
        from a2a.helpers.proto_helpers import new_data_part
        from src.green_agent.agent import _extract_purple_telemetry

        telemetry = {
            "schema_version": "petscagent.telemetry.v1",
            "model_calls": 3,
            "input_tokens": 1200,
            "cost_usd": 0.04,
        }
        self.assertEqual(
            _extract_purple_telemetry([new_data_part(telemetry)]), telemetry
        )
        self.assertIsNone(_extract_purple_telemetry([]))
        with mock.patch("builtins.print") as print_mock:
            self.assertIsNone(
                _extract_purple_telemetry([
                    new_data_part({"schema_version": "petscagent.solution.v1"})
                ])
            )
        print_mock.assert_not_called()
        with mock.patch("builtins.print") as print_mock:
            self.assertIsNone(
                _extract_purple_telemetry([
                    new_data_part({"schema_version": "petscagent.telemetry.v2"})
                ])
            )
        print_mock.assert_called_once()
        self.assertEqual(
            _extract_purple_telemetry([new_data_part({
                "schema_version": "petscagent.telemetry.v1",
                "model_calls": "unknown",
                "tool_calls": -1,
            })]),
            {"schema_version": "petscagent.telemetry.v1"},
        )

    def test_purple_telemetry_counts_must_be_whole_numbers(self):
        from a2a.helpers.proto_helpers import new_data_part
        from src.green_agent.agent import _extract_purple_telemetry

        # A fractional count is a reporting bug, so it is dropped rather than
        # truncated into a number that looks trustworthy. Dollar cost keeps
        # its precision.
        extracted = _extract_purple_telemetry([new_data_part({
            "schema_version": "petscagent.telemetry.v1",
            "input_tokens": 1200.7,
            "output_tokens": 600.0,
            "cost_usd": 0.51,
        })])
        self.assertNotIn("input_tokens", extracted)
        self.assertEqual(extracted["output_tokens"], 600)
        self.assertIsInstance(extracted["output_tokens"], int)
        self.assertEqual(extracted["cost_usd"], 0.51)

    def test_purple_telemetry_schema_is_shared_with_the_reference_agent(self):
        # A drift between producer and consumer looks exactly like an agent
        # that reports nothing, so both must read the same constant.
        from src.util.telemetry import PURPLE_TELEMETRY_SCHEMA
        from src.green_agent import agent as green_agent

        self.assertIs(green_agent.PURPLE_TELEMETRY_SCHEMA, PURPLE_TELEMETRY_SCHEMA)
        purple_source = (
            Path(__file__).resolve().parents[1]
            / "src" / "purple_agent" / "petsc_agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn("PURPLE_TELEMETRY_SCHEMA", purple_source)
        self.assertNotIn(f'"{PURPLE_TELEMETRY_SCHEMA}"', purple_source)

    def test_reference_agent_reports_context_and_known_litellm_cost(self):
        from unittest import mock
        from src.purple_agent.petsc_agent import _build_usage_telemetry

        response = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            cache_read_input_tokens=20,
        ))
        with mock.patch("litellm.completion_cost", return_value=0.0125):
            telemetry = _build_usage_telemetry(response, "provider/model")

        self.assertEqual(telemetry["peak_context_tokens"], 150)
        self.assertEqual(telemetry["cost_usd"], 0.0125)

    def test_purple_efficiency_summary_separates_measured_and_declared(self):
        from src.green_agent.agent import BenchmarkResult, _purple_efficiency_summary

        live = BenchmarkResult(
            "live", "p1", True, True,
            purple_wall_time_sec=2.0,
            purple_time_to_first_response_sec=0.5,
            purple_request_count=1,
            purple_request_bytes=100,
            purple_response_bytes=500,
            purple_response_event_count=3,
            purple_telemetry={
                "schema_version": "petscagent.telemetry.v1",
                "model_calls": 4,
                "cost_usd": 0.2,
            },
        )
        cached = BenchmarkResult(
            "cached", "p2", True, True,
            purple_response_replayed=True,
        )

        summary = _purple_efficiency_summary([live, cached])

        self.assertEqual(summary["benchmark_measured"]["request_count"], 1)
        self.assertEqual(summary["benchmark_measured"]["total_request_bytes"], 100)
        self.assertEqual(summary["benchmark_measured"]["median_wall_time_sec"], 2.0)
        self.assertEqual(
            summary["benchmark_measured"]["median_time_to_first_response_sec"], 0.5
        )
        self.assertEqual(summary["benchmark_measured"]["response_event_count"], 3)
        self.assertEqual(summary["benchmark_measured"]["replayed_cases"], 1)
        self.assertEqual(summary["agent_declared"]["model_calls"]["total"], 4)
        self.assertEqual(summary["agent_declared"]["cost_usd"]["reported_cases"], 1)
        self.assertIsNone(summary["agent_declared"]["tool_calls"]["total"])
        self.assertEqual(summary["live_cases"], 1)
        self.assertEqual(summary["total_cases"], 2)

    def test_efficiency_score_is_separate_and_budget_based(self):
        from src.green_agent.agent import BenchmarkResult, _calculate_efficiency_score

        config = {"time_budget_sec": 10, "response_bytes_budget": 1000}
        successful = BenchmarkResult(
            "success", "p1", True, True,
            purple_wall_time_sec=20,
            purple_response_bytes=4000,
            evaluation_summary={"all_gates_passed": True},
        )
        failed = BenchmarkResult(
            "failure", "p2", False, False,
            purple_wall_time_sec=1,
            purple_response_bytes=10,
            evaluation_summary={"all_gates_passed": False},
        )
        cached = BenchmarkResult(
            "cached", "p3", True, True,
            purple_response_replayed=True,
            evaluation_summary={"all_gates_passed": True},
        )

        self.assertEqual(_calculate_efficiency_score(successful, config), 35.36)
        self.assertEqual(_calculate_efficiency_score(failed, config), 0.0)
        self.assertIsNone(_calculate_efficiency_score(cached, config))

    def test_self_reported_model_is_an_identity_not_a_counted_metric(self):
        from a2a.helpers.proto_helpers import new_data_part
        from src.green_agent.agent import _extract_purple_telemetry

        extracted = _extract_purple_telemetry([new_data_part({
            "schema_version": "petscagent.telemetry.v1",
            "model": "  pdesim-gpt5-c3  ",
            "model_calls": 2,
        })])
        self.assertEqual(extracted["model"], "pdesim-gpt5-c3")

        # An identity is a string, so the whole-number rule that governs the
        # counted fields does not apply to it, and a non-string is dropped.
        for bad in (5, 1.5, None, "", "   "):
            self.assertNotIn("model", _extract_purple_telemetry([new_data_part({
                "schema_version": "petscagent.telemetry.v1",
                "model": bad,
            })]))

    def test_reported_model_ignores_cached_telemetry(self):
        from src.green_agent.agent import BenchmarkResult, _reported_model

        # A cached response replays an earlier run's telemetry, so the model it
        # names is the one that filled the cache, not the one under test. The
        # cache key is built from the configured purple_model tag, so a
        # reconfigured agent hits the same cache and would otherwise write its
        # results under the label it replaced.
        cached = BenchmarkResult(
            "cached", "p1", True, True,
            purple_response_replayed=True,
            purple_telemetry={
                "schema_version": "petscagent.telemetry.v1",
                "model": "pdesim-gpt5-c1",
            },
        )
        live = BenchmarkResult(
            "live", "p2", True, True,
            purple_telemetry={
                "schema_version": "petscagent.telemetry.v1",
                "model": "pdesim-gpt5-c2",
            },
        )

        self.assertIsNone(_reported_model([]))
        self.assertIsNone(_reported_model([cached]))
        self.assertEqual(_reported_model([cached, live]), "pdesim-gpt5-c2")

    def test_cached_telemetry_is_kept_per_problem_but_not_aggregated(self):
        from src.green_agent.agent import BenchmarkResult, _purple_efficiency_summary

        # A cached response replays an earlier run's telemetry. Counting it
        # would report generation work this run never performed.
        cached = BenchmarkResult(
            "cached", "p1", True, True,
            purple_response_replayed=True,
            purple_telemetry={
                "schema_version": "petscagent.telemetry.v1",
                "model_calls": 7,
            },
        )

        summary = _purple_efficiency_summary([cached])

        self.assertIsNotNone(cached.purple_telemetry)
        self.assertEqual(summary["telemetry_reported_cases"], 0)
        self.assertIsNone(summary["agent_declared"]["model_calls"]["total"])
        self.assertIsNone(summary["benchmark_measured"]["median_wall_time_sec"])
        self.assertEqual(summary["benchmark_measured"]["replayed_cases"], 1)
        self.assertEqual(summary["live_cases"], 0)

    def test_request_and_response_bytes_are_measured_identically(self):
        from unittest import mock
        from a2a.types.a2a_pb2 import StreamResponse
        from src.green_agent.agent import _a2a_payload_bytes
        from src.util.a2a_comm import send_message
        from src.util.a2a_v1 import new_agent_text_message

        # The request is assembled inside send_message, so the on_request hook
        # is what lets the caller size the same payload that goes on the wire.
        captured = {}

        class FakeClient:
            def __init__(self, interceptor):
                self.interceptor = interceptor

            async def send_message(self, request):
                captured["sent"] = request
                await self.interceptor.before(SimpleNamespace(input=request))
                response = StreamResponse(message=new_agent_text_message("done"))
                await self.interceptor.after(SimpleNamespace(result=response))
                yield response

        class FakeFactory:
            def __init__(self, *a, **kw):
                pass

            def create(self, card, interceptors):
                return FakeClient(interceptors[0])

        class FakeResolver:
            def __init__(self, *a, **kw):
                pass

            async def get_agent_card(self):
                return object()

        with mock.patch("src.util.a2a_comm.ClientFactory", FakeFactory), \
             mock.patch("src.util.a2a_comm.A2ACardResolver", FakeResolver):
            observed = []
            asyncio.run(send_message(
                "http://example.invalid",
                "describe a PETSc problem",
                on_metrics=observed.append,
            ))

        self.assertEqual(
            observed[0].request_bytes, _a2a_payload_bytes(captured["sent"])
        )
        self.assertEqual(observed[0].request_count, 1)
        self.assertEqual(observed[0].response_event_count, 1)
        self.assertGreater(observed[0].response_bytes, 0)
        self.assertIsNotNone(observed[0].time_to_first_response_sec)
        # The whole protobuf request is measured, not just the description.
        self.assertGreater(observed[0].request_bytes, len("describe a PETSc problem"))

    def test_request_callback_is_not_called_when_discovery_fails(self):
        from unittest import mock
        from src.util.a2a_comm import send_message

        class FailingResolver:
            def __init__(self, *a, **kw):
                pass

            async def get_agent_card(self):
                raise RuntimeError("discovery failed")

        callbacks = []
        with mock.patch("src.util.a2a_comm.A2ACardResolver", FailingResolver):
            with self.assertRaisesRegex(RuntimeError, "discovery failed"):
                asyncio.run(send_message(
                    "http://example.invalid",
                    "describe a PETSc problem",
                    on_metrics=callbacks.append,
                ))

        self.assertEqual(callbacks, [])

    def test_replay_rebuilds_a_recorded_submission(self):
        from src.green_agent.agent import Agent
        from src.util.a2a_v1 import get_text_parts

        agent = Agent.__new__(Agent)
        agent.replay_index = {
            "problem": {
                "problem_name": "problem",
                "requested_nsize": 2,
                # An empty cli_args is a valid submission, so replay must
                # accept it rather than treat it as a missing field.
                "cli_args": "",
                "generated_sources": [
                    {"original_name": "main.c", "source": "int main(void){return 0;}"}
                ],
            }
        }
        response = agent._replay_response("problem")
        parts = response.message.parts
        self.assertEqual(len(parts), 2)
        self.assertIn("nsize: 2", "".join(get_text_parts(parts)))

    def test_replay_rejects_a_record_it_cannot_rebuild(self):
        from src.green_agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.replay_index = {
            "no_sources": {"requested_nsize": 1, "cli_args": "", "generated_sources": []},
            "no_nsize": {
                "cli_args": "",
                "generated_sources": [{"original_name": "a.c", "source": "x"}],
            },
        }
        for name in ("missing", "no_sources", "no_nsize"):
            with self.assertRaises(ValueError):
                agent._replay_response(name)

    def test_a_rescore_writes_beside_the_run_it_replays(self):
        import json
        import tempfile
        from pathlib import Path
        from src.green_agent.agent import Agent

        cfg = {"evaluation": {"llm": {"model": "none/none"}}}

        def build(**kw):
            return Agent(config=cfg, purple_agent_url="http://purple",
                         mcp_server_url="http://mcp", **kw)

        with tempfile.TemporaryDirectory() as tmp:
            recorded = Path(tmp) / "recorded" / "anything.json"
            recorded.parent.mkdir()
            recorded.write_text(json.dumps(
                {"reported_model": "pdesim-Claude-Opus4.6-c1", "results": []}))

            # A plain run still writes to the live output tree, and names its
            # variant from this pass rather than from a replayed one.
            plain = build()
            self.assertEqual(plain.output_dir, Path("output"))
            self.assertIsNone(plain.replay_variant)

            # A rescore lands in the directory it replays, and takes the
            # variant from the record rather than from the filename, so a
            # file named by hand still scores the code it carries.
            rescore = build(replay_path=str(recorded), pass_index=1)
            self.assertEqual(rescore.output_dir, recorded.parent)
            self.assertEqual(rescore.replay_variant, "pdesim-claude-opus46-c1")

    def test_select_problems_matches_by_substring_and_glob(self):
        from src.green_agent.agent import select_problems

        data = [
            {"problem_name": "Robertson_ODE"},
            {"problem_name": "NS2D_FV_Implicit"},
            {"problem_name": "DarcyFlow2D_Steady"},
        ]
        def names(spec):
            return [d["problem_name"] for d in select_problems(data, spec)]

        # A bare term is a substring, so the full name need not be typed.
        self.assertEqual(names("darcy"), ["DarcyFlow2D_Steady"])
        self.assertEqual(names("ROBERTSON"), ["Robertson_ODE"])
        # A wildcard makes the term a whole-name glob.
        self.assertEqual(names("ns2d*"), ["NS2D_FV_Implicit"])
        self.assertEqual(names("*implicit"), ["NS2D_FV_Implicit"])
        # Data order, not the order the terms were given.
        self.assertEqual(
            names("darcy,robertson"), ["Robertson_ODE", "DarcyFlow2D_Steady"]
        )
        # Two terms hitting one problem still yield it once.
        self.assertEqual(names("darcy,flow2d"), ["DarcyFlow2D_Steady"])
        self.assertEqual(len(select_problems(data, None)), 3)
        self.assertEqual(len(select_problems(data, "")), 3)
        # Separators with no term is a typo, not a request for everything.
        with self.assertRaises(ValueError):
            select_problems(data, " , ")

    def test_select_problems_rejects_a_term_that_matches_nothing(self):
        from src.green_agent.agent import select_problems

        data = [{"problem_name": "Robertson_ODE"}, {"problem_name": "Advection_PDE"}]
        # A typo must stop the run rather than quietly shrink it, so this
        # raises even though the other term is good.
        with self.assertRaises(ValueError) as caught:
            select_problems(data, "robertson,darcey")
        message = str(caught.exception)
        self.assertIn("darcey", message)
        self.assertNotIn("robertson,", message)
        # The message names the alternatives.
        self.assertIn("Advection_PDE", message)

    def test_a2a_payload_measurement_never_fails_a_solution(self):
        from src.green_agent.agent import _a2a_payload_bytes

        class Unmeasurable:
            def model_dump_json(self, **kwargs):
                raise RuntimeError("boom")

        self.assertEqual(_a2a_payload_bytes(Unmeasurable()), 0)
        self.assertEqual(_a2a_payload_bytes(object()), 0)

    def test_uploaded_source_is_preserved_and_hashed(self):
        from src.green_agent.agent import Agent

        source = "#include <petscvec.h>\nint main(void) { return 0; }\n"
        agent = Agent.__new__(Agent)
        agent.mcp_client = FakeMCPClient()
        records = []

        dependencies = asyncio.run(
            agent._create_files_on_server(
                "example", [FakeFile("answer.c", source)], records
            )
        )

        self.assertEqual(dependencies, "")
        self.assertEqual(records[0]["source"], source)
        self.assertEqual(records[0]["server_name"], "example.c")
        self.assertEqual(
            records[0]["sha256"], hashlib.sha256(source.encode()).hexdigest()
        )
        self.assertEqual(agent.mcp_client.calls[0]["file_contents"], source)

    def test_auxiliary_filename_cannot_escape_source_directory(self):
        from src.green_agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.mcp_client = FakeMCPClient()
        records = []
        asyncio.run(
            agent._create_files_on_server(
                "example", [FakeFile("../../helper.h", "#pragma once\n")], records
            )
        )
        self.assertEqual(records[0]["server_name"], "helper.h")
        self.assertEqual(agent.mcp_client.calls[0]["filename"], "helper.h")


class FakeUpdater:
    """Collects what the agent reports instead of sending it anywhere."""

    def __init__(self):
        self.artifacts = []
        self.statuses = []

    async def add_artifact(self, name=None, parts=None, metadata=None):
        self.artifacts.append((name, parts, metadata))

    async def update_status(self, state=None, message=None):
        self.statuses.append((state, message))


class ARescoreScoresTheWholeRecordedRunTest(unittest.TestCase):
    """A rescore has to cover exactly the set the replayed file holds.

    Replay reads submissions from the aggregate rather than from the tree, so
    a pass that wrote back fewer problems than it read would leave the ones it
    left out with no copy of their code anywhere, and they could never be
    rescored again.
    """

    def test_narrowing_a_rescore_is_refused_up_front(self):
        import json
        import tempfile
        from pathlib import Path
        from src.green_agent.agent import Agent

        with tempfile.TemporaryDirectory() as tmp:
            recorded = Path(tmp) / "recorded.json"
            recorded.write_text(json.dumps({"results": []}))
            with self.assertRaises(ValueError) as caught:
                Agent(config={"evaluation": {"llm": {"model": "none/none"}}},
                      purple_agent_url="http://purple",
                      mcp_server_url="http://mcp",
                      replay_path=str(recorded), problems="alpha")
        self.assertIn("--problems", str(caught.exception))

    def run_against(self, recorded_names, dataset_names):
        import asyncio
        from unittest import mock
        from src.green_agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.replay_index = {n: {} for n in recorded_names}
        agent.problems = None
        agent.max_num_prob = None
        dataset = [{"problem_name": n, "problem_id": n} for n in dataset_names]
        with mock.patch("src.green_agent.agent.read_from_json",
                        return_value=dataset):
            with self.assertRaises(ValueError) as caught:
                asyncio.run(agent.run(None, None))
        return str(caught.exception)

    def test_a_disagreement_in_either_direction_aborts_before_scoring(self):
        # A recorded problem gone from data/ has no specification left to
        # grade against. A problem only in data/ has no submission to replay
        # and would otherwise be written down as a FAIL at zero.
        message = self.run_against(["alpha", "beta"], ["beta", "gamma"])
        self.assertIn("Only in the replay file: alpha", message)
        self.assertIn("Only in data/: gamma", message)


class DerivedSummaryTest(unittest.TestCase):
    """The counters must agree with the results list they describe."""

    def summarize(self, *specs):
        from src.green_agent.agent import BenchmarkResult, _derive_summary

        return _derive_summary([
            BenchmarkResult(problem_name=n, problem_id=n, runs=runs,
                            compiles=True, tier=tier)
            for n, runs, tier in specs
        ])

    def test_every_record_is_counted(self):
        s = self.summarize(("a", True, "GOLD"), ("b", True, "SILVER"),
                           ("c", False, "FAIL"))
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["runs_count"], 2)
        self.assertEqual(s["failure_count"], 1)


class WhatALaterPassMayOverwriteTest(unittest.TestCase):
    """Numbering, and what a second pass is allowed to touch.

    Every loss guarded against here would be a silent one.
    """

    def test_a_rescore_has_to_be_numbered_and_a_live_run_cannot_be(self):
        from src.green_agent.agent import Agent

        def build(**kw):
            return Agent(config={"evaluation": {"llm": {"model": "none/none"}}},
                         purple_agent_url="http://purple",
                         mcp_server_url="http://mcp", **kw)

        # Unnumbered, a rescore would land on the live aggregate and destroy
        # the run it is replaying.
        with self.assertRaises(ValueError):
            build(replay_path="recorded.json")
        # A number on a live run would claim a slot the run does not own.
        with self.assertRaises(ValueError):
            build(pass_index=1)

    def write(self, out, source, judge, replaying, pass_index=None):
        """Write one pass, naming the aggregate the way `run` would."""
        from src.green_agent.agent import Agent, BenchmarkResult, _derive_summary

        agent = Agent.__new__(Agent)
        agent.purple_id, agent.purple_model, agent.model = "p", "tag", judge
        agent.problems = None
        agent.replay_index = {"alpha": {}} if replaying else None
        results = [BenchmarkResult(
            problem_name="alpha", problem_id="a", runs=True, compiles=True,
            composite_score=1.0, tier="GOLD", scored_at="t", scored_by=judge,
            generated_sources=[{"original_name": "a.c", "server_name": "a.c",
                                "source": source, "sha256": "x"}],
        )]
        suffix = f"-s{pass_index}" if pass_index else ""
        agent._write_aggregate(
            results, _derive_summary(results), "t",
            out / f"tag-judged-by-{judge}{suffix}.json",
            out / "code" / "tag", out / "scores" / "tag", pass_index, None,
        )

    def test_a_rescore_scores_the_code_it_replays(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out, "/* generated */", "a", replaying=False)
            code = out / "code" / "tag" / "alpha" / "a.c"

            # A rescore replays an earlier generation, so writing it would
            # put old code back over whatever the tree holds now. It leaves
            # the tree alone, and the two judges' scores sit side by side.
            self.write(out, "/* replayed */", "b", replaying=True, pass_index=1)
            self.assertEqual(code.read_text(), "/* generated */")
            self.assertEqual(
                sorted(p.name for p in (out / "scores" / "tag" / "alpha").iterdir()),
                ["judged-by-a.json", "judged-by-b.json"],
            )

            # Regenerating overwrites the code tree, the score tree and the
            # live aggregate, which is what a live run is allowed to do. It
            # does not touch the slot the rescore was given.
            self.write(out, "/* regenerated */", "a", replaying=False)
            self.assertEqual(code.read_text(), "/* regenerated */")
            latest = json.loads(
                (out / "scores" / "tag" / "alpha" / "judged-by-a.json").read_text()
            )
            self.assertEqual(latest["generated_sources"][0]["source"],
                             "/* regenerated */")
            self.assertEqual(
                sorted(p.name for p in out.glob("*.json")),
                ["tag-judged-by-a.json", "tag-judged-by-b-s1.json"],
            )
            kept = json.loads((out / "tag-judged-by-b-s1.json").read_text())
            self.assertEqual(
                kept["results"][0]["generated_sources"][0]["source"],
                "/* replayed */",
            )


class TheCodeTreeHoldsTheLatestGenerationTest(unittest.TestCase):
    """What a later run is allowed to do to the tree it finds.

    Code and scores live in separate trees, so a regeneration may overwrite
    the code freely. The manifest still has to describe everything on disk,
    not just what the latest run touched.
    """

    def test_a_later_run_adds_to_the_manifest_instead_of_replacing_it(self):
        import json
        import tempfile
        from pathlib import Path
        from src.green_agent.agent import Agent, BenchmarkResult

        def result(name):
            return BenchmarkResult(
                problem_name=name, problem_id=name, runs=True, compiles=True,
                generated_sources=[{"original_name": f"{name}.c",
                                    "server_name": f"{name}.c",
                                    "source": "/* c */", "sha256": "x"}],
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent.__new__(Agent)
            tree = Path(tmp) / "code" / "tag"
            agent._write_submissions([result("alpha")], tree)
            # Rebuilding the manifest from this run alone would leave alpha's
            # code on disk with nothing describing it.
            agent._write_submissions([result("beta")], tree)

            listed = json.loads((tree / "manifest.json").read_text())
            self.assertEqual(
                sorted(entry["problem_name"] for entry in listed),
                ["alpha", "beta"],
            )


class EvaluationReportTest(unittest.TestCase):
    """The report must describe what it is given rather than crash on it."""

    def report_text(self, results, summary):
        from src.green_agent.agent import Agent

        agent = Agent.__new__(Agent)
        updater = FakeUpdater()
        asyncio.run(agent._create_evaluation_report(results, summary, updater))
        return "\n".join(
            p.text for name, parts, _ in updater.artifacts
            if name == "evaluation_report.txt" for p in parts
        )

    def test_an_empty_run_reports_rather_than_dividing_by_zero(self):
        from src.green_agent.agent import _derive_summary

        text = self.report_text([], _derive_summary([]))
        self.assertIn("Total Problems: 0", text)
        self.assertIn("Average Composite Score: n/a", text)
        self.assertIn("GOLD:   0 (n/a)", text)


if __name__ == "__main__":
    unittest.main()
