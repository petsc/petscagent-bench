"""Shared definition of the optional Purple Agent telemetry contract.

Telemetry is carried in an A2A DataPart so that agents implemented with Claude
Code, LangGraph, or any other framework can expose the same optional contract
without the Green Agent depending on that framework.

The schema version and the field list live here, rather than being repeated as
literals in the producer and the consumer, because a drift between the two
would look identical to an agent that simply reports nothing.
"""

PURPLE_TELEMETRY_SCHEMA = "petscagent.telemetry.v1"

# Fields the Green Agent aggregates. Every one of them is optional.
PURPLE_TELEMETRY_FIELDS = (
    "model_calls", "tool_calls",
    "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
    "peak_context_tokens", "cost_usd",
)

# Fields that count discrete events and must therefore arrive as whole
# numbers. A fractional count is a reporting bug in the agent, so the value is
# dropped rather than truncated into a plausible looking number.
PURPLE_TELEMETRY_INTEGER_FIELDS = frozenset({
    "model_calls", "tool_calls",
    "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
    "peak_context_tokens",
})
