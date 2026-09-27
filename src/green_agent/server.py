import argparse
import json
import uvicorn
import tomllib
from pathlib import Path
from typing import Any, Dict

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from src.util.a2a_v1 import AgentCard, AgentCapabilities, AgentInterface, AgentSkill
from src.green_agent.executor import GreenAgentExecutor
from loguru import logger


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def load_green_agent_config(config_path: str = "config/green_agent_config.yaml") -> Dict[str, Any]:
    """Load evaluation configuration from file or use defaults.

    Supports both JSON and YAML formats. Format is auto-detected by file extension.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                # Detect format by extension
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            print(f"@@@ Green agent: ✅ Loaded evaluation config from {config_path}")
            return config_data
        except Exception as e:
            print(f"@@@ Green agent: Failed to load config from {config_path}: {e}")
            print(f"@@@ Green agent: Using default evaluation configuration")
    else:
        print(f"@@@ Green agent: Config file {config_path} not found, using defaults")

    # Fall back to default configuration
    return {
        'evaluation': {
            'enable_gates': True,
            'enable_metrics': True,
            'enable_quality': True,
            'llm': {
                'model': 'gemini/gemini-3-flash-preview',
                'api_base_url': None,
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


def start_green_agent(
    host: str = "localhost",
    port: int = 9001,
    card_url: str | None = None,
    agent_llm: str | None = None,
    api_base_url: str | None = None,
    config_path: str = "config/green_agent_config.yaml",
):
    """Start the Green Agent A2A HTTP server.

    Loads the agent card configuration from ``green_agent.toml``, ensures the
    required ``url`` field is populated (from ``card_url`` or
    ``http://{host}:{port}``), wires up a :class:`~a2a.server.request_handlers.DefaultRequestHandler`
    with :class:`~petscagent_bench.src.green_agent.executor.GreenAgentExecutor`
    and an :class:`~a2a.server.tasks.InMemoryTaskStore`, then starts a Starlette
    app via Uvicorn.

    Args:
        host: Interface to bind the server to.
        port: Port to bind the server to.
        card_url: External URL to advertise in the agent card. If empty,
            ``http://{host}:{port}`` is used.
        agent_llm: Optional LLM model name (overrides config file value).
        api_base_url: Optional LLM API base URL (overrides config file value).
        config_path: Path to the Green Agent config file (YAML/JSON).

    Returns:
        None
    """
    logger.info("Starting green agent...")
    agent_card_dict = load_agent_card_toml("green_agent")
    agent_card_dict["url"] = card_url or f"http://{host}:{port}" # complete all required card fields

    # Load config and apply CLI overrides
    config = load_green_agent_config(config_path)
    if agent_llm:
        config["evaluation"]["llm"]["model"] = agent_llm
    if api_base_url:
        config["evaluation"]["llm"]["api_base_url"] = api_base_url

    agent_card = AgentCard(
        name=agent_card_dict["name"],
        description=agent_card_dict["description"],
        version=agent_card_dict["version"],
        documentation_url=agent_card_dict.get("documentation_url", ""),
        supported_interfaces=[AgentInterface(
            url=agent_card_dict["url"],
            protocol_binding="JSONRPC",
            protocol_version=agent_card_dict.get("protocol_version", "1.0"),
        )],
        default_input_modes=agent_card_dict.get("defaultInputModes", []),
        default_output_modes=agent_card_dict.get("defaultOutputModes", []),
        capabilities=AgentCapabilities(
            streaming=agent_card_dict.get("capabilities", {}).get("streaming", False),
        ),
        skills=[AgentSkill(**skill) for skill in agent_card_dict.get("skills", [])],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(config),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = Starlette(routes=[
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(request_handler, rpc_url="/"),
    ])
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the green agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument("--agent-llm", type=str, help="LLM model to use (overrides config), e.g. gemini/gemini-2.5-flash, openai/gpt-4o")
    parser.add_argument("--api-base-url", type=str, help="Optional LLM API base URL (overrides config)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/green_agent_config.yaml",
        help="Path to Green Agent config file (YAML/JSON)",
    )
    args = parser.parse_args()

    start_green_agent(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        agent_llm=args.agent_llm,
        api_base_url=args.api_base_url,
        config_path=args.config,
    )
