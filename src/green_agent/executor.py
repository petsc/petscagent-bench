from typing import Any, Dict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from src.util.a2a_comm import parse_tags
from src.green_agent.agent import Agent

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class GreenAgentExecutor(AgentExecutor):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor with an agent configuration.

        Args:
            config: Green agent config dict (typically loaded from config/green_agent_config.yaml).
        """
        self.config = config
        self.agents: dict[str, Agent] = {}  # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the context to get purple agent URL and other configurations
        print("@@@ Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        purple_agent_url = tags["purple_agent_url"]
        mcp_server_url = tags["mcp_server_url"]
        green_id = tags["green_id"]
        purple_id = tags["purple_id"]
        purple_model = tags.get("purple_model", "")
        # create a new task
        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        context_id = task.context_id
        agent = Agent(config=self.config, purple_agent_url=purple_agent_url, mcp_server_url=mcp_server_url, green_id=green_id, purple_id=purple_id, purple_model=purple_model)
        # for debugging
        # max_num_prob = 1
        # agent = Agent(purple_agent_url=purple_agent_url, mcp_server_url=mcp_server_url, max_num_prob=max_num_prob)
        self.agents[context_id] = agent
        agent = self.agents.get(context_id)
        updater = TaskUpdater(event_queue, task.id, context_id)

        print("@@@ Green agent: Start working...")
        await updater.start_work()
        try:
            print("@@@ Green agent: Starting code generation request...")

            res = await agent.run(context.message, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}", context_id=context_id, task_id=task.id
                )
            )

        print("@@@ Green agent: ✅ Code generation request complete.")
        await event_queue.enqueue_event(new_agent_text_message(f"Finished. ✅\n"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError
