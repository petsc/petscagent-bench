"""Purple Agent implementation - the target code generation agent being tested.

The Purple Agent is responsible for generating PETSc C/C++ code from natural
language problem descriptions. It operates as an A2A-compliant agent that:

1. Receives problem descriptions via the A2A protocol
2. Uses an LLM to generate PETSc code
3. Returns generated code files along with CLI arguments
4. Maintains conversation context for multi-turn interactions

The agent is isolated from the evaluation logic and is the subject of testing
by the Green Agent through the petscagent-bench framework.
"""

import argparse
import uvicorn
import dotenv
import os
import json
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_parts_message
from a2a.types import TextPart, FilePart, FileWithBytes
import litellm
from litellm import completion
from pydantic import BaseModel
from loguru import logger
from typing import Any, Dict
from pathlib import Path

dotenv.load_dotenv()

# System prompt that defines the code generation contract
# This ensures the LLM produces output in a structured, parseable format
SYSTEM_CODE_CONTRACT = (
    "You are a code-generation agent.\n"
    "Return ONLY a single raw JSON object. No markdown, no backticks, no code blocks, no explanation outside the JSON.\n"
    "Top-level JSON keys MUST be exactly: 'codes', 'nsize', 'cli_args' (no additional keys).\n\n"
    "Rules:\n"
    "- 'codes': a list of objects with 'filename' and 'code'. Code must be valid C/C++/CUDA.\n"
    "- 'nsize': the number of MPI processes (use 1 for sequential).\n"
    "- 'cli_args': command line arguments string.\n"
    "- First file in 'codes' is the main file.\n"
    "- Any explanations MUST be inside C block comments /* ... */ within the code strings.\n"
)

def load_purple_agent_config(config_path: str = "config/purple_agent_config.yaml") -> Dict[str, Any]:
    """Load purple agent configuration from file or use defaults.

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

            print(f"@@@ Purple agent: ✅ Loaded agent config from {config_path}")
            return config_data
        except Exception as e:
            print(f"@@@ Purple agent: ⚠️ Failed to load config from {config_path}: {e}")
            print(f"@@@ Purple agent: Using default evaluation configuration")
    else:
        print(f"@@@ Purple agent: Config file {config_path} not found, using defaults")

    # Fall back to default configuration
    return {
        'llm': {
            'model': 'gemini/gemini-3-flash-preview',
            'api_base_url': None,
            'temperature': 0.3,
            'max_concurrent_calls': 3,
        },
    }

def prepare_purple_agent_card(url):
    """Create an A2A agent card for the Purple Agent.

    The agent card is a metadata descriptor that advertises the agent's
    capabilities, skills, and communication modes to potential clients.

    Args:
        url: The base URL where this agent is accessible

    Returns:
        AgentCard object describing the Purple Agent's capabilities
    """
    skill = AgentSkill(
        id="petsc_code_generation",
        name="PETSc Code Generation",
        description="Generates PETSc C/C++/CUDA code from natural language problem descriptions. "
                    "Returns structured JSON with source files, MPI process count, and CLI arguments.",
        tags=["purple agent", "code generation", "PETSc", "HPC"],
        examples=["Write a PETSc program that solves the Robertson ODE system using TS."],
    )
    card = AgentCard(
        name="purple_agent",
        description="PETSc code generation agent for petscagent-bench. "
                    "Receives scientific computing problem descriptions via A2A and "
                    "uses an LLM to produce compilable PETSc C/C++ source code.",
        url=url,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/octet-stream"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class PetscAgentExecutor(AgentExecutor):
    """Executor class that handles code generation requests.

    This class implements the AgentExecutor interface from the A2A framework.
    It manages:
    - LLM interactions for code generation
    - Conversation context tracking
    - Response formatting and validation
    - Error handling and reporting

    Attributes:
        model: LLM model identifier (e.g., "openai/gpt-4o", "gemini/gemini-2.5-flash")
        ctx_id_to_messages: Dict mapping context IDs to conversation histories
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor with an agent configuration.

        Args:
            config: Purple agent config dict (typically loaded from config/purple_agent_config.yaml).
        """
        self.config = config
        self.llm_config = config.get("llm")
        self.model = self.llm_config.get("model")
        self.temperature = float(self.llm_config.get("temperature"))
        self.api_base_url = self.llm_config.get("api_base_url")
        self.max_tokens = self.llm_config.get("max_tokens")

        # Track conversation history per context for multi-turn interactions
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a code generation request.

        This method is called by the A2A framework when a message is received.
        It processes the request through the following steps:

        1. Extract user input from the request context
        2. Initialize or retrieve conversation history
        3. Call the LLM to generate PETSc code
        4. Parse and validate the LLM response
        5. Format the response as A2A message parts (text + files)
        6. Enqueue the response event

        Args:
            context: Request context containing user input and metadata
            event_queue: Queue for sending response events back to the client

        Note:
            The method handles both success and error cases, always returning
            a properly formatted A2A response.
        """
        user_input = context.get_user_input()

        # Initialize conversation history for new contexts
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {
                    "role": "system",
                    "content": SYSTEM_CODE_CONTRACT,
                }
            ]
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        class CodeFile(BaseModel):
            filename: str
            code: str

        class ProblemResponse(BaseModel):
            codes: list[CodeFile]
            nsize: int    # e.g. 1 for sequential codes
            cli_args: str # e.g. "-ts_type beuler -ts_monitor"

        try:
            # Generate PETSc code using the LLM with JSON schema
            completion_kwargs = {
                'messages': messages,
                'model': self.model,
                'temperature': self.temperature,
                'response_format': ProblemResponse,
                'timeout': 300,
            }
            if self.max_tokens:
                completion_kwargs['max_tokens'] = int(self.max_tokens)
            litellm.ssl_verify = False
            if self.api_base_url:
                completion_kwargs['api_base'] = self.api_base_url
                is_asksage_endpoint = self.api_base_url.startswith('https://api.asksage.anl.gov')
                is_argo_endpoint = 'inside.anl.gov/argoapi' in self.api_base_url
                if is_asksage_endpoint:
                    litellm.ssl_verify = os.environ["ASKSAGE_SSL_CERT_FILE"]
                    completion_kwargs['api_key'] = os.environ["ASKSAGE_API_KEY"]
                elif is_argo_endpoint:
                    # Argo authenticates with the caller's ANL domain username
                    completion_kwargs['api_key'] = os.environ["ARGO_API_KEY"]
            response = completion(**completion_kwargs)
            # Log how much of the token budget was actually used, so the
            # configured max_tokens can be tuned against real measurements.
            try:
                used = response.usage.completion_tokens
                print(f"@@@ Purple agent: completion_tokens={used} (max_tokens={completion_kwargs.get('max_tokens')})")
            except Exception:
                pass
            # Extract the generated content from LLM response
            content = response.choices[0].message.content
            # logger.info(f"Raw LLM response (first 500 chars): {repr(content[:500]) if isinstance(content, str) else type(content)}")
            if isinstance(content, str):
                # Remove markdown code block wrapper that some LLMs such as Claude add
                # This ensures we get clean JSON for parsing
                if content.startswith("```"):
                    content = content.split("```", 2)[1]
                    content = content.lstrip("json").strip()
                    # logger.info(f"After stripping markdown (first 500 chars): {repr(content[:500])}")
                data = json.loads(content)
            else:
                raise TypeError(f"Expected string content from response, got {type(content)}")
            # Parse the JSON response
            nsize = data["nsize"]
            cli_args = data["cli_args"]
            # Report token usage so the evaluator can record generation cost.
            # Appended after cli_args to keep the existing response format intact.
            usage_line = ""
            try:
                u = response.usage
                # cached_tokens is reported in different places by different
                # providers, so check both and fall back to 0.
                cached = getattr(u, "cache_read_input_tokens", None)
                if cached is None:
                    details = getattr(u, "prompt_tokens_details", None)
                    cached = getattr(details, "cached_tokens", None) if details else None
                usage_line = (
                    f"prompt_tokens: {u.prompt_tokens}\n"
                    f"completion_tokens: {u.completion_tokens}\n"
                    f"total_tokens: {getattr(u, 'total_tokens', u.prompt_tokens + u.completion_tokens)}\n"
                    f"cached_tokens: {cached or 0}\n"
                )
            except Exception:
                pass
            parts_list = [TextPart(text=f"Code generation successful ✅\nnsize: {nsize}\ncli_args: {cli_args}\n{usage_line}")]
            for entry in data["codes"]:
                # Create file object with code content
                fwb = FileWithBytes(
                    name=entry["filename"],
                    bytes=entry["code"].encode("utf-8"),
                    mime_type="text/plain"
                )
                parts_list.append(FilePart(file=fwb))
            # Send the successful response back to the client
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:
            # Handle any errors during code generation
            # import traceback
            # logger.error(f"Task failed: {e}\n{traceback.format_exc()}")
            print(f"@@@ Purple agent: ❌ Task failed with agent error: {type(e).__name__}: {e}")
            # Dump the raw LLM reply so parse failures are diagnosable. Without
            # this the content is lost and truncated, empty, or oddly wrapped
            # responses are indistinguishable from each other.
            self._dump_failed_response(locals().get("response"), locals().get("content"), e)
            # Return error message to the client
            parts_list = [TextPart(text=f"Code generation failed ❌\nerror: {e}\n")]
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )

    def _dump_failed_response(self, response, content, exc):
        """Record a failed LLM reply so the cause can be diagnosed.

        Prints a short summary and writes the full raw reply to
        ``failed_responses/``. The finish reason distinguishes the common
        causes: ``length`` means the reply was truncated by the token limit,
        while a complete reply that still fails to parse points at unexpected
        formatting.
        """
        try:
            finish_reason = None
            usage = None
            if response is not None:
                try:
                    finish_reason = response.choices[0].finish_reason
                    usage = getattr(response, "usage", None)
                except Exception:
                    pass

            print(f"@@@ Purple agent: finish_reason={finish_reason!r} usage={usage}")
            if content is None:
                print("@@@ Purple agent: no content was returned by the model")
            else:
                text = content if isinstance(content, str) else repr(content)
                print(f"@@@ Purple agent: raw reply is {len(text)} chars")
                print(f"@@@ Purple agent: head: {text[:300]!r}")
                print(f"@@@ Purple agent: tail: {text[-300:]!r}")
                if finish_reason == "length":
                    print("@@@ Purple agent: reply was TRUNCATED by the token limit")

            dump_dir = Path("failed_responses")
            dump_dir.mkdir(exist_ok=True)
            existing = len(list(dump_dir.glob("*.json")))
            path = dump_dir / f"failure-{existing + 1:03d}.json"
            path.write_text(json.dumps({
                "model": self.model,
                "api_base_url": self.api_base_url,
                "error": f"{type(exc).__name__}: {exc}",
                "finish_reason": finish_reason,
                "usage": str(usage) if usage is not None else None,
                "content": content if isinstance(content, str) else repr(content),
            }, indent=2))
            print(f"@@@ Purple agent: wrote raw reply to {path}")
        except Exception as dump_err:
            # Diagnostics must never mask the original failure
            print(f"@@@ Purple agent: could not dump failed response: {dump_err}")

    async def cancel(self, context, event_queue) -> None:
        """Cancel a running task (not implemented).

        This method is required by the AgentExecutor interface but is not
        currently implemented as code generation tasks are atomic and fast.

        Args:
            context: Request context
            event_queue: Event queue for sending cancellation acknowledgment

        Raises:
            NotImplementedError: Always, as cancellation is not supported
        """
        raise NotImplementedError

def start_purple_agent(
    host: str = "localhost",
    port: int = 9002,
    card_url: str | None = None,
    agent_llm: str | None = None,
    api_base_url: str | None = None,
    config_path: str = "config/purple_agent_config.yaml",
):
    """Start the Purple Agent A2A HTTP service.

    Args:
        host: Interface to bind the HTTP server to.
        port: Port to bind the HTTP server to.
        card_url: Optional explicit URL used to build/advertise the agent card.
            If not provided, a URL is constructed from `host` and `port`.
        agent_llm: Optional LLM model name (overrides config file value).
        api_base_url: Optional LLM API base URL (overrides config file value).
        config_path: Path to the Purple Agent config file (YAML/JSON).
    """
    logger.info("Starting purple agent...")
    card = prepare_purple_agent_card(card_url or f"http://{host}:{port}")

    # Load config and apply CLI overrides
    config = load_purple_agent_config(config_path)
    if agent_llm:
        config["llm"]["model"] = agent_llm
    if api_base_url:
        config["llm"]["api_base_url"] = api_base_url

    request_handler = DefaultRequestHandler(
        agent_executor=PetscAgentExecutor(config),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the purple agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument(
        "--config",
        type=str,
        default="config/purple_agent_config.yaml",
        help="Path to Purple Agent config file (YAML/JSON)",
    )
    parser.add_argument("--api-base-url", type=str, help="Optional LLM API base URL (overrides config)")
    parser.add_argument("--agent-llm", type=str, help="LLM model to use (overrides config), e.g. gemini/gemini-2.5-flash, openai/gpt-4o")
    args = parser.parse_args()

    start_purple_agent(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        agent_llm=args.agent_llm,
        api_base_url=args.api_base_url,
        config_path=args.config,
    )
