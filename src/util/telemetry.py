"""Shared definition of the optional Purple Agent telemetry contract.

Telemetry is carried in an A2A data Part so that agents implemented with Claude
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

# Optional identity string the agent self-reports (NOT an aggregated metric): the
# model name to record for this run. The Green Agent uses it to name the output
# file, so a composite agent can encode its configuration too, e.g. a pde-sim
# binding reports "pdesim-<model>-c<N>". When absent, Green falls back to the
# purple_model task tag, then to "unknown".
PURPLE_TELEMETRY_MODEL_FIELD = "model"

# Fields that count discrete events and must therefore arrive as whole
# numbers. A fractional count is a reporting bug in the agent, so the value is
# dropped rather than truncated into a plausible looking number.
PURPLE_TELEMETRY_INTEGER_FIELDS = frozenset({
    "model_calls", "tool_calls",
    "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
    "peak_context_tokens",
})
