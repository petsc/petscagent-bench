"""Small, project-local conveniences for the protobuf-based A2A 1.x API."""

import uuid
from typing import Any, Sequence

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message as ProtobufMessage

from a2a.helpers.proto_helpers import (
    get_text_parts,
    new_data_part,
    new_message,
    new_raw_part,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.types.a2a_pb2 import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    TaskState,
)


def new_agent_text_message(text: str, **kwargs: Any) -> Message:
    return new_text_message(text, role=Role.ROLE_AGENT, **kwargs)


def new_agent_parts_message(parts: list[Part], **kwargs: Any) -> Message:
    return new_message(parts, role=Role.ROLE_AGENT, **kwargs)


def get_file_parts(parts: Sequence[Part]) -> list[Part]:
    return [part for part in parts if part.HasField("raw") or part.HasField("url")]


def get_data(part: Part) -> Any:
    return MessageToDict(part.data) if part.HasField("data") else None


def protobuf_size(payload: ProtobufMessage) -> int:
    """Return the exact protobuf wire size of an A2A payload."""
    return len(payload.SerializeToString())


__all__ = [
    "AgentCard", "AgentCapabilities", "AgentInterface", "AgentSkill",
    "Message", "Part", "Role", "SendMessageRequest", "StreamResponse",
    "TaskState", "get_data", "get_file_parts", "get_text_parts",
    "new_agent_parts_message", "new_agent_text_message", "new_data_part",
    "new_raw_part", "new_task_from_user_message", "new_text_part",
    "protobuf_size", "uuid",
]
