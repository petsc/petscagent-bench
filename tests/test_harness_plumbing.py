"""Tests for the harness that runs each declared test case."""

import asyncio
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name):
    return json.loads((ROOT / "data" / name).read_text())


class HarnessPlumbingTest(unittest.TestCase):
    def test_multi_case_runs_fill_the_case_list_and_the_scalar_view(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        problem = load("GradShafranov.json")
        calls = []

        class StubClient:
            async def run_executable(self, executable, nsize, args, valgrind=False):
                calls.append((nsize, args))
                return f"{1e-5 / (len(calls) ** 2):.6e}"

        agent = Agent.__new__(Agent)
        agent.config = {}
        agent.mcp_client = StubClient()

        br = BenchmarkResult(
            problem_name="g", problem_id="1", runs=False, compiles=True
        )
        asyncio.run(agent._run_test_cases(br, "g", problem, 1, "-ts_type euler"))

        self.assertEqual(len(br.cases), len(problem["test_cases"]))
        self.assertEqual(len(calls), len(problem["test_cases"]))
        # The agent's request runs, with each case's own arguments appended
        # last so that PETSc's last-wins resolves any key the case names.
        self.assertEqual(
            calls[1][1], "-ts_type euler " + problem["test_cases"][1]["args"]
        )
        self.assertEqual(calls[2][0], 4)  # case 2 declares nsize 4
        self.assertEqual(
            br.cases[0].executed_args,
            "-ts_type euler " + problem["test_cases"][0]["args"],
        )
        self.assertEqual(
            br.cases[0].declared_args, problem["test_cases"][0]["args"]
        )
        self.assertEqual(br.stdout, br.cases[0].stdout)
        self.assertTrue(br.runs)
        json.dumps(asdict(br))  # must still serialise into the output file

    def test_single_case_problems_use_the_declared_test_parameters(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        calls = []

        class StubClient:
            async def run_executable(self, executable, nsize, args, valgrind=False):
                calls.append((nsize, args))
                return "1.0"

        agent = Agent.__new__(Agent)
        agent.config = {}
        agent.mcp_client = StubClient()

        br = BenchmarkResult(
            problem_name="r", problem_id="1", runs=False, compiles=True
        )
        asyncio.run(
            agent._run_test_cases(br, "r", load("vecmpi.json"), 2, "-agent_choice")
        )
        self.assertEqual(calls, [(3, "-agent_choice -N 10")])
        self.assertEqual(br.cases[0].executed_args, "-agent_choice -N 10")
        self.assertEqual(br.cases[0].declared_args, "-N 10")
        self.assertEqual(br.actual_nsize, 3)
        self.assertEqual(len(br.cases), 1)

    def test_a_case_declaring_no_args_runs_the_agent_request_alone(self):
        from src.green_agent.agent import Agent, BenchmarkResult

        calls = []

        class StubClient:
            async def run_executable(self, executable, nsize, args, valgrind=False):
                calls.append((nsize, args, valgrind))
                return "1.0"

        agent = Agent.__new__(Agent)
        agent.config = {"memory_safety": {"use_valgrind": True}}
        agent.mcp_client = StubClient()
        br = BenchmarkResult("p", "1", False, True)
        asyncio.run(agent._run_test_cases(
            br, "p", {"test_cases": [{}, {}]}, 2, "-agent_choice"
        ))

        normal_calls = [call for call in calls if not call[2]]
        valgrind_calls = [call for call in calls if call[2]]
        self.assertEqual([call[1] for call in normal_calls], ["-agent_choice"] * 2)
        self.assertEqual(len(valgrind_calls), 2)

if __name__ == "__main__":
    unittest.main()
