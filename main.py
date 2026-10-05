"""CLI entry point for the PETSc Agent Benchmark system.

This module provides the command-line interface for running the agentified
petscagent-bench framework. It supports four main commands:
- green: Start the assessment manager agent (Green Agent)
- purple: Start the target agent being tested (Purple Agent)
- launch: Run the complete evaluation workflow
- problems: List the benchmark problems, or preview a selection

The system uses the A2A (Agent-to-Agent) protocol and MCP (Model Context Protocol)
for inter-agent communication and tool access.
"""

import typer
import asyncio
from pathlib import Path

from src.green_agent.server import start_green_agent
from src.green_agent.agent import read_from_json, select_problems
from src.purple_agent.petsc_agent import start_purple_agent
from src.launcher import launch_evaluation

# Initialize Typer application with descriptive help text
app = typer.Typer(help="Agentified petscagent-bench - PETSc coding agent assessment framework")


@app.command()
def green():
    """Start the green agent (assessment manager).
    
    The Green Agent is responsible for:
    - Loading test problems from the benchmark dataset
    - Distributing problems to the Purple Agent
    - Collecting and evaluating generated code
    - Running the evaluation pipeline (gates, metrics, quality checks)
    - Generating comprehensive assessment reports
    
    The agent runs on http://localhost:9001 by default.
    """
    start_green_agent()


@app.command()
def purple():
    """Start the purple agent (target being tested).
    
    The Purple Agent is the code generation agent under evaluation:
    - Receives problem descriptions via A2A protocol
    - Generates PETSc C/C++ code using an LLM
    - Returns generated code along with CLI arguments
    - Operates in isolation from evaluation logic
    
    The agent runs on http://localhost:9002 by default.
    """
    start_purple_agent()


@app.command()
def launch(
    purple_url: str = typer.Option(
        None, help="Evaluate an already-running agent at this URL instead of "
                   "starting the built-in purple."),
    replay: str = typer.Option(
        None, help="Rescore the submissions recorded in a previous run's "
                   "output JSON instead of generating new ones."),
    problems: str = typer.Option(
        None, help="Evaluate only the problems matching these comma-separated "
                   "terms, e.g. 'darcy,robertson'. A term matches part of a "
                   "name, ignoring case; one with a wildcard is a glob. See "
                   "the 'problems' command."),
):
    """Launch the complete evaluation workflow.
    
    This command orchestrates the full benchmark process:
    1. Starts the Green Agent (assessment manager)
    2. Starts the Purple Agent (code generator)
    3. Starts the MCP server (for compilation/execution tools)
    4. Initiates the evaluation process
    5. Collects results and generates reports
    6. Cleanly shuts down all components
    
    Results are saved to the 'output/' directory.
    
    Prerequisites:
    - PETSc must be installed and PETSC_DIR/PETSC_ARCH set in .env
    - API keys for LLM providers must be configured in .env
    """
    # Resolved here so a mistyped term fails now rather than as a failed task
    # after the three servers have come up.
    if problems:
        try:
            select_problems(read_from_json(Path("./data")), problems)
        except (ValueError, RuntimeError) as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

    asyncio.run(launch_evaluation(purple_url=purple_url, replay=replay, problems=problems))


@app.command()
def problems(
    match: str = typer.Argument(
        None, help="Comma-separated terms, matched as for --problems. Omit "
                   "to list every problem."),
):
    """List the benchmark problems, or preview what a selection would run.

    Reads the dataset directly, so no agent or server starts.

        main.py problems           every problem
        main.py problems darcy     what --problems darcy selects
    """
    all_problems = read_from_json(Path("./data"))
    try:
        selected = select_problems(all_problems, match)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    width = max((len(p["problem_name"]) for p in selected), default=4)
    typer.echo(f"{'NAME':<{width}}  ID  CASES  FILE")
    # Dataset order, which is the order a run evaluates them in.
    for p in selected:
        cases = len(p.get("test_cases") or [])
        typer.echo(
            f"{p['problem_name']:<{width}}  {p['problem_id']:>2}  "
            f"{cases:>5}  {p.get('source_file', '')}"
        )

    if match:
        typer.echo(
            f"\n{len(selected)} of {len(all_problems)} problems match: {match}"
        )
    else:
        typer.echo(f"\n{len(selected)} problems")


if __name__ == "__main__":
    app()
