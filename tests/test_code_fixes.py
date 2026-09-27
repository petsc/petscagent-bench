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
        self.name = name
        self.bytes = source


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
            "error_tolerance: 1.0e-3",
            "error_threshold: 1.0e-3",
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
        self.assertIn('output_dir / "sources" / local_path.stem', text)
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
        from a2a.types import DataPart
        from src.green_agent.agent import _extract_purple_telemetry

        telemetry = {
            "schema_version": "petscagent.telemetry.v1",
            "model_calls": 3,
            "input_tokens": 1200,
            "cost_usd": 0.04,
        }
        self.assertEqual(
            _extract_purple_telemetry([DataPart(data=telemetry)]), telemetry
        )
        self.assertIsNone(_extract_purple_telemetry([]))
        with mock.patch("builtins.print") as print_mock:
            self.assertIsNone(
                _extract_purple_telemetry([
                    DataPart(data={"schema_version": "petscagent.solution.v1"})
                ])
            )
        print_mock.assert_not_called()
        with mock.patch("builtins.print") as print_mock:
            self.assertIsNone(
                _extract_purple_telemetry([
                    DataPart(data={"schema_version": "petscagent.telemetry.v2"})
                ])
            )
        print_mock.assert_called_once()
        self.assertEqual(
            _extract_purple_telemetry([DataPart(data={
                "schema_version": "petscagent.telemetry.v1",
                "model_calls": "unknown",
                "tool_calls": -1,
            })]),
            {"schema_version": "petscagent.telemetry.v1"},
        )

    def test_purple_telemetry_counts_must_be_whole_numbers(self):
        from a2a.types import DataPart
        from src.green_agent.agent import _extract_purple_telemetry

        # A fractional count is a reporting bug, so it is dropped rather than
        # truncated into a number that looks trustworthy. Dollar cost keeps
        # its precision.
        extracted = _extract_purple_telemetry([DataPart(data={
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

    def test_purple_efficiency_summary_separates_measured_and_declared(self):
        from src.green_agent.agent import BenchmarkResult, _purple_efficiency_summary

        live = BenchmarkResult(
            "live", "p1", True, True,
            purple_wall_time_sec=2.0,
            purple_request_count=1,
            purple_request_bytes=100,
            purple_response_bytes=500,
            purple_telemetry={
                "schema_version": "petscagent.telemetry.v1",
                "model_calls": 4,
                "cost_usd": 0.2,
            },
        )
        cached = BenchmarkResult(
            "cached", "p2", True, True,
            purple_response_from_cache=True,
        )

        summary = _purple_efficiency_summary([live, cached])

        self.assertEqual(summary["benchmark_measured"]["request_count"], 1)
        self.assertEqual(summary["benchmark_measured"]["total_request_bytes"], 100)
        self.assertEqual(summary["benchmark_measured"]["median_wall_time_sec"], 2.0)
        self.assertEqual(summary["benchmark_measured"]["cached_cases"], 1)
        self.assertEqual(summary["agent_declared"]["model_calls"]["total"], 4)
        self.assertEqual(summary["agent_declared"]["cost_usd"]["reported_cases"], 1)
        self.assertIsNone(summary["agent_declared"]["tool_calls"]["total"])
        self.assertEqual(summary["live_cases"], 1)
        self.assertEqual(summary["total_cases"], 2)

    def test_cached_telemetry_is_kept_per_problem_but_not_aggregated(self):
        from src.green_agent.agent import BenchmarkResult, _purple_efficiency_summary

        # A cached response replays an earlier run's telemetry. Counting it
        # would report generation work this run never performed.
        cached = BenchmarkResult(
            "cached", "p1", True, True,
            purple_response_from_cache=True,
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
        self.assertEqual(summary["benchmark_measured"]["cached_cases"], 1)
        self.assertEqual(summary["live_cases"], 0)

    def test_request_and_response_bytes_are_measured_identically(self):
        from unittest import mock
        from src.green_agent.agent import _a2a_payload_bytes
        from src.util.a2a_comm import send_message

        # The request is assembled inside send_message, so the on_request hook
        # is what lets the caller size the same payload that goes on the wire.
        captured = {}

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def send_message(self, request):
                captured["sent"] = request
                return request

        class FakeResolver:
            def __init__(self, *a, **kw):
                pass

            async def get_agent_card(self):
                return object()

        with mock.patch("src.util.a2a_comm.A2AClient", FakeClient), \
             mock.patch("src.util.a2a_comm.A2ACardResolver", FakeResolver):
            sizes = []
            asyncio.run(send_message(
                "http://example.invalid",
                "describe a PETSc problem",
                on_request=lambda req: sizes.append(_a2a_payload_bytes(req)),
            ))

        self.assertEqual(sizes, [_a2a_payload_bytes(captured["sent"])])
        # The whole JSON-RPC envelope is measured, not just the description.
        self.assertGreater(sizes[0], len("describe a PETSc problem"))

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
                    on_request=lambda req: callbacks.append(req),
                ))

        self.assertEqual(callbacks, [])

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


if __name__ == "__main__":
    unittest.main()
