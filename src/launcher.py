"""Evaluation launcher module - orchestrates the complete benchmark workflow.

This module is responsible for:
- Starting all required agents (Green, Purple) and servers (MCP)
- Coordinating inter-process communication
- Managing the evaluation lifecycle
- Ensuring clean shutdown of all components

The launcher uses multiprocessing to run agents in separate processes,
allowing them to communicate via HTTP using the A2A protocol.
"""

import multiprocessing
import asyncio
import json
import mcp
from src.green_agent.server import start_green_agent, load_green_agent_config
from src.purple_agent.petsc_agent import start_purple_agent, load_purple_agent_config
from src.util.a2a_comm import wait_agent_ready, send_message
import os
import dotenv

# Load environment variables before importing server code
# This ensures PETSC_DIR, PETSC_ARCH, and API keys are available
dotenv.load_dotenv()
from petsc_compile_run_mcp_server import main as start_mcp_server


async def wait_port_open(host, port, timeout=60.0, interval=0.5):
    """Poll until a TCP port accepts connections, or the timeout elapses."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(interval)
    return False


def run_green_agent(agent_llm, api_base_url=None):
    """Execute the Green Agent in a separate process.
    
    This wrapper function is needed for multiprocessing.Process,
    which requires a synchronous entry point. The function creates
    a new asyncio event loop and runs the async green agent server.
    """
    asyncio.run(start_green_agent(agent_llm=agent_llm, api_base_url=api_base_url))


def run_purple_agent(agent_llm, api_base_url=None):
    """Execute the Purple Agent in a separate process.
    
    Starts the Purple Agent with a specific LLM configuration.
    The LLM model can be changed here to test different models.
    
    Currently configured to use: openai/gpt-5.2
    Other options: gemini/gemini-2.5-flash, openai/gpt-4o, etc.
    """
    asyncio.run(start_purple_agent(agent_llm=agent_llm, api_base_url=api_base_url))
    # asyncio.run(start_purple_agent(agent_llm="openai/google-claude-45-opus")) # test AskSage


async def launch_evaluation(purple_url=None, replay=None, problems=None, pass_index=None,
                            output=None):
    """Main launcher function - initiates and coordinates the evaluation process.
    
    This function orchestrates the complete benchmark workflow:
    
    1. Process Initialization:
       - Spawns the Green Agent process (assessment manager)
       - Spawns the Purple Agent process (code generator under test)
       - Spawns the MCP server process (PETSc compilation/execution tools)
    
    2. Health Checks:
       - Waits for each agent to become ready (HTTP health check)
       - Ensures all components are operational before proceeding
    
    3. Task Execution:
       - Sends the evaluation task to the Green Agent
       - Green Agent autonomously manages the evaluation workflow
       - Waits for evaluation to complete
    
    4. Cleanup:
       - Terminates all spawned processes
       - Ensures clean shutdown
    
    The Green Agent writes the results itself, to the directory `output` names
    or, failing that, to the one its own config sets.

    Args:
        purple_url: Evaluate this already-running agent instead of starting the
            built-in purple. Its lifetime belongs to the caller. Such an agent
            names its own output file by self-reporting a model in its
            telemetry, otherwise the run is filed as "unknown".
        replay: A previous run's output JSON. Its recorded submissions are
            scored again and no purple agent is started.
        problems: Comma-separated terms, selecting the problems to evaluate.
        pass_index: Which numbered aggregate the rescore writes. Required with
            `replay` and rejected without it.
        output: Where the results go, overriding the Green Agent's config.

    Raises:
        AssertionError: If any agent fails to become ready within timeout
        Exception: If communication or execution errors occur
    """
    # Define service endpoints
    green_url = "http://localhost:9001"    # Green Agent A2A server
    mcp_server_url = "http://localhost:8080/mcp"  # MCP tools server
    green_id = "019bb856-c8bf-7390-8c4f-bced52276932" # AgentBeats ID
    purple_id = ""

    replaying = replay is not None
    external_purple = purple_url is not None and not replaying
    purple_url = purple_url or "http://localhost:9002"
    # Empty for an external agent, which names itself by self-reporting a model.
    purple_model = ""
    if replaying:
        # Name the rescored run for the model that generated the submissions,
        # not for whatever purple happens to be configured now.
        recorded = json.loads(open(replay).read())
        purple_model = recorded.get("reported_model") or recorded.get("purple_model", "")

    green_cfg = load_green_agent_config()
    green_llm_cfg = green_cfg.get('evaluation', {}).get('llm', {})
    green_model = green_llm_cfg.get('model', 'openai/gpt-4o-mini')
    green_api_base_url = green_llm_cfg.get('api_base_url')

    if not external_purple and not replaying:
        purple_cfg = load_purple_agent_config()
        purple_llm_cfg = purple_cfg.get('llm')
        purple_model = purple_llm_cfg.get('model', 'openai/gpt-4o-mini')
        purple_api_base_url = purple_llm_cfg.get('api_base_url')

    # Everything we start goes in here so the finally below can reap it. These
    # are non-daemon children, so a failure that skips cleanup does not just
    # leak ports: the interpreter blocks forever joining them at exit.
    started = []
    try:
        # Step 1: Start Green Agent (assessment manager)
        print("Launching green agent...")
        p_green = multiprocessing.Process(target=run_green_agent, args=(green_model, green_api_base_url))
        p_green.start()
        started.append(p_green)
        assert await wait_agent_ready(green_url), "Green agent not ready in time"
        print("Green agent is ready.")

        # Step 2: Start Purple Agent (code generator being tested)
        if replaying:
            print(f"Replaying recorded submissions from {replay}; purple not started.")
        elif external_purple:
            print(f"Using external purple agent at {purple_url}...")
            assert await wait_agent_ready(purple_url), "external purple agent not ready in time"
            print("External purple agent is ready.")
        else:
            print("Launching purple agent...")
            p_purple = multiprocessing.Process(target=run_purple_agent, args=(purple_model, purple_api_base_url))
            p_purple.start()
            started.append(p_purple)
            assert await wait_agent_ready(purple_url), "purple agent not ready in time"
            print("purple agent is ready.")

        # Step 3: Start MCP server (provides PETSc compilation/execution tools)
        print("Launching MCP server for green agent...")
        petsc_mcp_server = multiprocessing.Process(target=start_mcp_server)
        petsc_mcp_server.start()
        started.append(petsc_mcp_server)
        # Wait for the port to accept connections. Without this the first compile
        # can reach the server before it is listening and fail with a connection
        # error, which is then recorded as a gate failure. Previously this was
        # masked by the time the purple agent spent generating code, so it only
        # surfaced when submissions were served from cache.
        assert await wait_port_open("localhost", 8080), "MCP server not ready in time"
        print("PETSc MCP server is ready.")

        # Step 4: Send evaluation task to Green Agent
        print("Sending task description to green agent...")
        replay_block = f"""Instead of calling the purple agent, replay the submissions recorded in
<replay>
{replay}
</replay>
Write this rescore to numbered slot
<pass>
{pass_index}
</pass>
""" if replaying else ""
        problems_block = f"""Evaluate only the problems matching
<problems>
{problems}
</problems>
""" if problems else ""
        output_block = f"""Write the results to
<output>
{output}
</output>
""" if output else ""
        task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<purple_agent_url>
{purple_url}/
</purple_agent_url>
You can use MCP tools from:
<mcp_server_url>
{mcp_server_url}/
</mcp_server_url>
Green agent's AgentBeats ID is
<green_id>
{green_id}
</green_id>
Purple agent's AgentBeats ID is
<purple_id>
{purple_id}
</purple_id>
Purple agent's LLM model is
<purple_model>
{purple_model}
</purple_model>
{replay_block}{problems_block}{output_block}    """
        print("Task description:")
        print(task_text)
        print("Sending...")

        # Send message and wait for completion
        # The Green Agent will autonomously manage the entire evaluation workflow
        response = await send_message(green_url, task_text)
        print("Evaluation complete.")
    finally:
        # Step 5: Cleanup - terminate the processes we started, youngest first.
        # An external purple was never added to `started`, so it survives.
        print("Terminating agents...")
        for p in reversed(started):
            p.terminate()
            p.join()
        print("Agents terminated.")
