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
        self.assertIn('output_dir / "runs" / f"{prefix}-run{run_index}"', text)
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

    def test_a_rescore_writes_back_to_the_directory_it_replays(self):
        import json
        import tempfile
        from pathlib import Path
        from src.green_agent.agent import Agent

        cfg = {"evaluation": {"llm": {"model": "none/none"}}}

        def build(**kw):
            return Agent(config=cfg, purple_agent_url="http://purple",
                         mcp_server_url="http://mcp", **kw)

        with tempfile.TemporaryDirectory() as tmp:
            recorded = Path(tmp) / "recorded" / "pdesim-judged-by-judge-run3.json"
            recorded.parent.mkdir()
            recorded.write_text(json.dumps({"results": []}))

            # A plain run still writes to the live output tree, allocating an
            # index rather than reusing one.
            plain = build()
            self.assertEqual(plain.output_dir, Path("output"))
            self.assertIsNone(plain.replay_run_index)

            # A rescore updates the set it replays rather than landing beside
            # unrelated runs in output/, and keeps that run's index so the
            # record it rescores is replaced instead of duplicated.
            rescore = build(replay_path=str(recorded))
            self.assertEqual(rescore.output_dir, recorded.parent)
            self.assertEqual(rescore.replay_run_index, 3)

            # A file named by hand carries no index, so the allocator runs.
            plain_name = recorded.parent / "recorded.json"
            plain_name.write_text(json.dumps({"results": []}))
            self.assertIsNone(build(replay_path=str(plain_name)).replay_run_index)

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


def a_record(name, **kw):
    """One entry of a recorded aggregate, as a dict read back from JSON."""
    record = {
        "problem_name": name,
        "problem_id": name,
        "runs": True,
        "compiles": True,
        "composite_score": 80.0,
        "tier": "SILVER",
        "generated_sources": [
            {"original_name": "main.c", "server_name": f"{name}.c",
             "source": f"/* {name} */", "sha256": "abc"}
        ],
    }
    record.update(kw)
    return record


class NarrowedReplayKeepsEveryProblemTest(unittest.TestCase):
    """A rescore of some problems must not delete the records of the rest.

    A rescore writes back over the run it replays. Rebuilding the aggregate
    from only the problems that ran drops the others' generated_sources, and
    replay reads sources from the aggregate rather than from the tree, so those
    submissions could never be rescored again.
    """

    def build_agent(self, replay_records):
        from src.green_agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.replay_index = {r["problem_name"]: r for r in replay_records}
        return agent

    def test_a_record_survives_the_round_trip_through_a_dict(self):
        from dataclasses import asdict
        from src.green_agent.agent import (
            BenchmarkResult, TestCaseResult, _benchmark_result_from_record,
        )

        original = BenchmarkResult(
            problem_name="p", problem_id="1", runs=True, compiles=True,
            composite_score=77.5, tier="SILVER",
            category_scores={"correctness": 90.0},
            generated_sources=[{"original_name": "a.c", "source": "x"}],
            cases=[TestCaseResult(index=0, args="-n 1", nsize=2, runs=True,
                                  stdout="out", execution_time_sec=1.5)],
            scored_at="2026-01-01T00:00:00+00:00", scored_by="judge",
        )
        rebuilt = _benchmark_result_from_record(asdict(original))
        self.assertEqual(asdict(rebuilt), asdict(original))
        # cases must come back as dataclasses, not the dicts they serialized to.
        self.assertIsInstance(rebuilt.cases[0], TestCaseResult)

    def test_a_record_from_another_schema_still_rebuilds(self):
        from src.green_agent.agent import _benchmark_result_from_record

        # A key this version has dropped must not raise, and a record missing
        # every optional key must still produce a result. Either one refusing
        # would make an older file unreplayable, which is the bug being fixed.
        older = _benchmark_result_from_record(
            {"problem_name": "p", "problem_id": "1", "runs": False,
             "compiles": False, "a_field_we_removed": 1}
        )
        self.assertEqual(older.problem_name, "p")
        self.assertEqual(older.cases, [])
        self.assertIsNone(older.scored_at)

    def test_a_field_this_version_does_not_know_is_written_back_out(self):
        from src.green_agent.agent import (
            _benchmark_result_from_record, _record_to_dict,
        )

        # Real output files carry purple_response_from_cache, retired when the
        # purple cache became replay. Dropping a retired field is harmless, but
        # nothing here can tell it from a field a newer checkout wrote, and
        # losing that is the data loss a carried record exists to prevent.
        original = {"problem_name": "p", "problem_id": "1", "runs": True,
                    "compiles": True, "purple_response_from_cache": False,
                    "something_newer": {"nested": 1}}
        out = _record_to_dict(_benchmark_result_from_record(original))
        for key, value in original.items():
            self.assertEqual(out[key], value)
        # The holding field is an implementation detail and never reaches disk.
        self.assertNotIn("unrecognized", out)

    def test_the_merge_keeps_the_problems_it_did_not_evaluate(self):
        from src.green_agent.agent import BenchmarkResult

        agent = self.build_agent([a_record("alpha"), a_record("beta"), a_record("gamma")])
        fresh = BenchmarkResult(problem_name="beta", problem_id="beta",
                                runs=True, compiles=True, composite_score=91.0)

        merged, carried = agent._merge_replay_records([fresh])

        self.assertEqual([r.problem_name for r in merged], ["alpha", "beta", "gamma"])
        self.assertEqual(carried, ["alpha", "gamma"])
        # The freshly scored record wins its slot, in place.
        self.assertIs(merged[1], fresh)
        # And the point of the whole exercise: the other two keep their code.
        for r in (merged[0], merged[2]):
            self.assertTrue(r.generated_sources)
            self.assertEqual(r.composite_score, 80.0)

    def test_a_full_pass_carries_nothing(self):
        from src.green_agent.agent import BenchmarkResult

        agent = self.build_agent([a_record("alpha"), a_record("beta")])
        fresh = [
            BenchmarkResult(problem_name=n, problem_id=n, runs=True, compiles=True)
            for n in ("alpha", "beta")
        ]
        merged, carried = agent._merge_replay_records(fresh)
        self.assertEqual(carried, [])
        self.assertEqual([r.problem_name for r in merged], ["alpha", "beta"])

    def test_a_plain_run_is_untouched(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        agent = Agent.__new__(Agent)
        agent.replay_index = None
        fresh = [BenchmarkResult(problem_name="p", problem_id="1",
                                 runs=True, compiles=True)]
        merged, carried = agent._merge_replay_records(fresh)
        self.assertEqual(carried, [])
        self.assertEqual([r.problem_name for r in merged], ["p"])

    def test_a_problem_absent_from_the_replay_file_is_still_kept(self):
        from src.green_agent.agent import BenchmarkResult

        agent = self.build_agent([a_record("alpha")])
        fresh = BenchmarkResult(problem_name="new_problem", problem_id="2",
                                runs=True, compiles=True)
        merged, carried = agent._merge_replay_records([fresh])
        # It has no slot in the replay file's order, so it goes at the end
        # rather than being dropped.
        self.assertEqual([r.problem_name for r in merged], ["alpha", "new_problem"])
        self.assertEqual(carried, ["alpha"])


class CarriedRecordsAreNotThisPassesWorkTest(unittest.TestCase):
    """A carried record was generated by an earlier run.

    Every "did this run do the work" filter keys off purple_response_replayed,
    and a record carried from a first-generation file has that flag false. The
    aggregates about generation must therefore be computed over this pass's
    records, not the merged list.
    """

    def a_live_looking_record(self):
        from src.green_agent.agent import _benchmark_result_from_record

        return _benchmark_result_from_record(a_record(
            "carried",
            purple_response_replayed=False,
            purple_wall_time_sec=42.0,
            purple_request_count=3,
            purple_request_bytes=1000,
            prompt_tokens=500, completion_tokens=250, total_tokens=750,
            purple_telemetry={"model": "some-other-purple"},
        ))

    def test_a_carried_record_does_not_rename_the_output_file(self):
        from src.green_agent.agent import _reported_model

        # _reported_model names the output file. A carried record satisfying
        # it would make a rescore write beside the file it meant to overwrite,
        # leaving two files under two variant tags at the same run index.
        self.assertEqual(
            _reported_model([self.a_live_looking_record()]), "some-other-purple"
        )
        # Which is why it must be called over this pass's records only. An
        # empty pass reports nothing, so the configured tag names the file.
        self.assertIsNone(_reported_model([]))

    def test_a_carried_record_is_excluded_from_measured_effort(self):
        from src.green_agent.agent import _purple_efficiency_summary

        carried = self.a_live_looking_record()
        # Over this pass's records, which is the empty set for a rescore that
        # evaluated nothing live.
        measured = _purple_efficiency_summary([])["benchmark_measured"]
        self.assertEqual(measured["request_count"], 0)
        self.assertIsNone(measured["median_wall_time_sec"])
        # Had it been folded in, this run would report an earlier run's effort.
        billed = _purple_efficiency_summary([carried])["benchmark_measured"]
        self.assertEqual(billed["request_count"], 3)


class DerivedSummaryTest(unittest.TestCase):
    """Counters come from the records, so a merged file counts all of them."""

    def results(self, *specs):
        from src.green_agent.agent import BenchmarkResult

        return [
            BenchmarkResult(problem_name=n, problem_id=n, runs=runs,
                            compiles=True, tier=tier)
            for n, runs, tier in specs
        ]

    def test_the_counters_describe_the_whole_merged_list(self):
        from src.green_agent.agent import _derive_summary

        summary = _derive_summary(self.results(
            ("a", True, "GOLD"), ("b", True, "SILVER"), ("c", False, "FAIL"),
        ))
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["runs_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(sum(summary["tier_distribution"].values()), 3)

    def test_an_unknown_tier_is_counted_rather_than_crashing(self):
        from src.green_agent.agent import _derive_summary

        # A carried record can hold a null tier, or one from a schema this
        # version does not know. Indexing the distribution with it would raise.
        summary = _derive_summary(self.results(
            ("a", True, "GOLD"), ("b", True, None), ("c", True, "PLATINUM"),
        ))
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["tier_distribution"]["GOLD"], 1)
        self.assertEqual(summary["tier_unknown"], 2)

    def test_an_empty_run_counts_to_zero(self):
        from src.green_agent.agent import _derive_summary

        summary = _derive_summary([])
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["avg_composite_score"])


class TheWrittenFileDescribesTheMergeTest(unittest.TestCase):
    """What lands on disk after a narrowed rescore.

    Drives the real write path, so the aggregate, the source tree and the
    manifest the dashboard reads are all checked together.
    """

    def write(self, tmp, results, carried_over, problems):
        import json
        from pathlib import Path
        from src.green_agent.agent import Agent, _derive_summary

        agent = Agent.__new__(Agent)
        agent.purple_id, agent.purple_model, agent.model = "purple", "tag", "judge"
        agent.problems = problems
        out = Path(tmp)
        run_dir = out / "runs" / "tag-judged-by-judge-run3"
        run_dir.mkdir(parents=True)
        path, _ = agent._write_aggregate(
            results, _derive_summary(results), carried_over,
            "2026-10-06T12:00:00+00:00", out, run_dir,
            "tag-judged-by-judge", 3, None,
        )
        return path, json.loads(path.read_text()), run_dir

    def test_a_narrowed_rescore_writes_the_whole_set_back(self):
        import json
        import tempfile
        from src.green_agent.agent import (
            BenchmarkResult, _benchmark_result_from_record,
        )

        fresh = BenchmarkResult(
            problem_name="beta", problem_id="beta", runs=True, compiles=True,
            composite_score=91.0, tier="GOLD",
            scored_at="2026-10-06T12:00:00+00:00", scored_by="judge",
            generated_sources=[{"original_name": "b.c", "server_name": "b.c",
                                "source": "/* new beta */", "sha256": "bbb"}],
        )
        results = [
            _benchmark_result_from_record(a_record("alpha")),
            fresh,
            _benchmark_result_from_record(a_record("gamma")),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path, doc, run_dir = self.write(tmp, results, ["alpha", "gamma"], "beta")

            # The file keeps the run index it replaced, so the rescore
            # overwrites that run rather than landing beside it.
            self.assertEqual(path.name, "tag-judged-by-judge-run3.json")
            self.assertEqual(len(doc["results"]), 3)
            self.assertEqual([r["problem_name"] for r in doc["results"]],
                             ["alpha", "beta", "gamma"])
            # A mixed file says so at the top level, rather than leaving a
            # reader to diff the per-record stamps.
            self.assertEqual(doc["carried_over"], ["alpha", "gamma"])
            self.assertEqual(doc["problem_filter"], "beta")
            self.assertEqual(doc["summary"]["total"], 3)
            self.assertEqual(doc["scored_at"], "2026-10-06T12:00:00+00:00")

            by_name = {r["problem_name"]: r for r in doc["results"]}
            # The property the bug destroyed: the carried records still hold
            # the sources, so they can be replayed again.
            for name in ("alpha", "gamma"):
                self.assertTrue(by_name[name]["generated_sources"])
                self.assertIsNone(by_name[name]["scored_at"])
            self.assertEqual(by_name["beta"]["scored_at"],
                             "2026-10-06T12:00:00+00:00")

            # And the manifest the dashboard reads lists every problem's code,
            # not just the one that was rescored.
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(
                sorted(m["problem_name"] for m in manifest),
                ["alpha", "beta", "gamma"],
            )
            for entry in manifest:
                self.assertTrue((run_dir / entry["filename"]).is_file())

    def test_a_full_run_says_it_carried_nothing(self):
        import tempfile
        from src.green_agent.agent import BenchmarkResult

        r = BenchmarkResult(problem_name="p", problem_id="1", runs=True,
                            compiles=True, tier="GOLD", composite_score=90.0)
        with tempfile.TemporaryDirectory() as tmp:
            _, doc, _ = self.write(tmp, [r], [], None)
            self.assertIsNone(doc["carried_over"])
            self.assertIsNone(doc["problem_filter"])

    def test_a_field_this_version_retired_survives_the_write(self):
        import tempfile
        from src.green_agent.agent import _benchmark_result_from_record

        # Real files carry purple_response_from_cache. A rescore must write it
        # back rather than strip it from every record it did not evaluate.
        carried = _benchmark_result_from_record(
            a_record("alpha", purple_response_from_cache=False)
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, doc, _ = self.write(tmp, [carried], ["alpha"], "beta")
            self.assertIs(doc["results"][0]["purple_response_from_cache"], False)
            self.assertNotIn("unrecognized", doc["results"][0])


class ReportSurvivesAMixedFileTest(unittest.TestCase):
    """The report must describe a merged file rather than crash on it."""

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

    def test_a_record_with_no_score_does_not_break_the_report(self):
        from src.green_agent.agent import BenchmarkResult, _derive_summary

        # A carried record from a file where evaluation never finished.
        unscored = BenchmarkResult(problem_name="carried", problem_id="1",
                                   runs=True, compiles=True,
                                   composite_score=None, tier=None)
        scored = BenchmarkResult(problem_name="fresh", problem_id="2",
                                 runs=True, compiles=True,
                                 composite_score=90.0, tier="GOLD",
                                 scored_at="2026-10-06T00:00:00+00:00")
        results = [unscored, scored]
        summary = _derive_summary(results)
        summary["avg_composite_score"] = 90.0
        text = self.report_text(results, summary)
        self.assertIn("carried (Score: n/a)", text)
        self.assertIn("fresh (Score: 90.0/100)", text)
        # The stamp is what makes a mixed file legible.
        self.assertIn("scored 2026-10-06T00:00:00+00:00", text)

    def test_partial_category_scores_do_not_break_the_report(self):
        from src.green_agent.agent import BenchmarkResult, _derive_summary

        r = BenchmarkResult(problem_name="p", problem_id="1", runs=True,
                            compiles=True, composite_score=50.0, tier="BRONZE",
                            category_scores={"correctness": 50.0})
        text = self.report_text([r], _derive_summary([r]))
        self.assertIn("Correctness: 50.0", text)
        self.assertNotIn("Performance:", text)


if __name__ == "__main__":
    unittest.main()
