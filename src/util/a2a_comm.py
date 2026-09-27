"""Utility module for Agent-to-Agent (A2A) communication.

This module provides helper functions for:
- Retrieving agent cards (agent metadata/capabilities)
- Checking agent health/readiness
- Sending messages between agents using the A2A protocol

The A2A protocol enables standardized communication between autonomous agents,
allowing them to discover capabilities and exchange messages reliably.
"""

import httpx
import asyncio
import uuid

import re
from typing import Dict

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from src.util.a2a_v1 import AgentCard, Message, Role, SendMessageRequest, StreamResponse, new_text_part


async def get_agent_card(url: str) -> AgentCard | None:
    """Retrieve the agent card from an A2A-compliant agent.

    The agent card contains metadata about the agent including:
    - Name and description
    - Supported skills and capabilities
    - Input/output modes
    - Version information

    Args:
        url: Base URL of the agent's A2A server (e.g., "http://localhost:9001")

    Returns:
        AgentCard object if successful, None if the agent is unreachable
        or doesn't provide a valid card
    """
    httpx_client = httpx.AsyncClient(timeout=30.0)
    try:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        card: AgentCard | None = await resolver.get_agent_card()
        return card
    finally:
        await httpx_client.aclose()


async def wait_agent_ready(url, timeout=10):
    """Wait for an agent to become ready by polling its agent card endpoint.

    This function is useful during startup to ensure an agent is fully
    initialized before attempting to send messages to it.

    Args:
        url: Base URL of the agent's A2A server
        timeout: Maximum time to wait in seconds (default: 10)

    Returns:
        True if agent became ready within timeout, False otherwise
    """
    # Wait until the A2A server is ready, check by getting the agent card
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                print(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def send_message(
    url, message, task_id=None, context_id=None, on_request=None
) -> StreamResponse:
    """Send a message to an A2A-compliant agent.

    This function handles the full A2A message protocol:
    1. Retrieves the agent card to verify capabilities
    2. Creates an A2A client with proper HTTP settings
    3. Constructs and sends a message with unique IDs
    4. Returns the agent's response

    Args:
        url: Base URL of the target agent's A2A server
        message: Text message to send to the agent
        task_id: Optional task identifier for message threading (default: None)
        context_id: Optional context identifier for maintaining conversation state (default: None)
        on_request: Optional callback invoked with the fully built
            SendMessageRequest just before it is sent. Callers use it to
            measure the outgoing payload, which is assembled here rather than
            by the caller (default: None)

    Returns:
        Final StreamResponse containing the agent's response

    Raises:
        Exception: If the agent is unreachable or returns an error
    """
    # Create HTTP client with extended timeout for long-running operations
    timeout = httpx.Timeout(connect=30.0, read=3000.0, write=30.0, pool=30.0)
    httpx_client = httpx.AsyncClient(timeout=timeout)
    try:
        # Retrieve agent card to get capabilities and validate endpoint
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        card = await resolver.get_agent_card()

        client = ClientFactory(ClientConfig(
            streaming=False, httpx_client=httpx_client
        )).create(card)

        # Generate unique message ID for tracking
        message_id = uuid.uuid4().hex

        # Construct message parameters with user role
        req = SendMessageRequest(
            message=Message(
                role=Role.ROLE_USER,
                parts=[new_text_part(message)],
                message_id=message_id,
                task_id=task_id or "",
                context_id=context_id or "",
            ),
        )
        if on_request is not None:
            on_request(req)

        response = None
        async for event in client.send_message(request=req):
            response = event
        if response is None:
            raise RuntimeError("Purple Agent returned no A2A response")
        return response
    finally:
        await httpx_client.aclose()

def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """Parse XML-style tags from a string and return their contents.

    This utility function extracts content between matching opening and closing tags.
    It's useful for parsing structured text responses from agents.

    Args:
        str_with_tags: String containing XML-style tags like "<tag>content</tag>"

    Returns:
        Dictionary mapping tag names to their content (stripped of whitespace)

    Example:
        >>> text = "<url>http://localhost:9002/</url><name>purple_agent</name>"
        >>> tags = parse_tags(text)
        >>> print(tags)
        {'url': 'http://localhost:9002/', 'name': 'purple_agent'}

    Note:
        - Tags must be properly matched (same tag name for open/close)
        - Nested tags with the same name are not supported
    """
    # Use regex to find all matching tag pairs
    # Pattern: <(tag_name)>content</tag_name>
    # re.DOTALL allows matching across newlines
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)

    # Convert list of tuples to dictionary, stripping whitespace from content
    return {tag: content.strip() for tag, content in tags}
