# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw visible-event evidence from a Pi coding-agent session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel
from pydantic import Field as PydanticField

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_VERSION = 3
PI_CONNECTOR_VERSION = "1"
IDENTITY_PREFIX = "pi"
CHECKPOINT_VERSION = 1

_CONTROL_ONLY_TYPES = frozenset(
    {
        "model_change",
        "thinking_level_change",
        "label",
        "session",
        "mode_change",
    }
)


class PiSessionInput(BaseModel):
    """One Pi session archive and the branch leaf whose history is captured."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One raw visible event with its graph position and full source record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any] = PydanticField(default_factory=dict)


class PiEventSource(Source):
    """Captured raw event evidence that can be cited back to one session event."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any] = PydanticField(default_factory=dict)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable ``pi:session:event`` identity for one visible event."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    identity = f"{IDENTITY_PREFIX}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"pi event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters"
        )
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse the selected branch and return its visible events from root to leaf."""

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("pi session content must be strict UTF-8") from error  # noqa: TRY003

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError("pi session line is not valid JSON") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError("pi session records must be JSON objects")  # noqa: TRY003
        records.append(record)

    if not records:
        raise ValueError("pi session archive is empty")  # noqa: TRY003

    header = records[0]
    if header.get("type") != "session" or header.get("version") != PI_SESSION_VERSION:
        raise ValueError("pi session header must declare type=session and version=3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("pi session header must declare a non-empty id")  # noqa: TRY003

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("pi session records must declare a non-empty id")  # noqa: TRY003
        if event_id in by_id:
            raise ValueError(f"pi session id {event_id!r} is duplicated")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("pi session parentId must be a string or null")  # noqa: TRY003
        if parent_id and parent_id not in by_id:
            raise ValueError(  # noqa: TRY003
                f"pi session parent {parent_id!r} is missing or appears after its child"
            )
        by_id[event_id] = record
        order.append(event_id)

    if not order:
        raise ValueError("pi session archive has no events")  # noqa: TRY003

    leaf_id = value.leaf_id if value.leaf_id is not None else order[-1]
    if leaf_id not in by_id:
        raise ValueError(f"pi session leaf {leaf_id!r} is missing")  # noqa: TRY003

    chain: list[str] = []
    cursor: str | None = leaf_id
    while cursor is not None:
        chain.append(cursor)
        cursor = by_id[cursor].get("parentId") or None
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        record = by_id[event_id]
        visible = _visible_content(record)
        if visible is None:
            continue
        role, body = visible
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=record.get("parentId") or None,
                event_type=str(record.get("type")),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def _visible_content(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return the role and body one record exposes, or ``None`` for control records."""

    record_type = record.get("type")
    if record_type == "compaction" or record_type == "branch_summary":
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return str(record_type), summary

    if record_type == "custom_message":
        body = _blocks_text(record.get("content"))
        return ("custom", body) if body else None

    if record_type != "message":
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None

    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        if not isinstance(command, str) or not command:
            return None
        rendered = f"$ {command}"
        if isinstance(output, str) and output:
            rendered = f"{rendered}\n{output}"
        return role, rendered

    if role == "toolResult":
        tool_name = message.get("toolName")
        call_id = message.get("toolCallId")
        body = _blocks_text(message.get("content"))
        if not body:
            return None
        return role, f"[tool_result {tool_name} {call_id}] {body}"

    body = _blocks_text(message.get("content"))
    return (role, body) if body else None


def _blocks_text(content: Any, /) -> str:
    """Render visible text and tool-call blocks, dropping thinking and images."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            value = block.get("text")
            if isinstance(value, str) and value:
                parts.append(value)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            rendered = json.dumps(
                arguments if arguments is not None else {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            parts.append(f"[tool_call {name} {call_id}] {rendered}")
    return "\n".join(parts)


class PiEventSourceAdapter:
    """Resolve and read raw Pi event evidence Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=dict(value.record),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=dict(source.record),
        )


class PiEventTextEvidenceProjection:
    """Expose one captured event as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)


def _canonical_record(record: dict[str, Any], /) -> bytes:
    """Render one raw record as stable bytes so rewriting can be detected."""

    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_digests(content: bytes, /) -> dict[str, str]:
    """Fingerprint every identified record in the archive, branch or not."""

    digests: dict[str, str] = {}
    for line in content.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            continue
        digests[event_id] = hashlib.sha256(_canonical_record(record)).hexdigest()
    return digests


def _imported_records(checkpoint: Any, /) -> dict[str, str]:
    """Read the previously imported ``event id -> record digest`` map."""

    if not isinstance(checkpoint, dict):
        return {}
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    return {
        key: value
        for key, value in events.items()
        if isinstance(key, str) and isinstance(value, str)
    }


class PiSessionConnector:
    """Import one Pi session branch incrementally through ``ConnectorLifecycle``."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self.path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=content, leaf_id=self.leaf_id))
        digests = _record_digests(content)
        imported = _imported_records(session.checkpoint)

        for event_id, digest in imported.items():
            current = digests.get(event_id)
            if current is not None and current != digest:
                raise ValueError(  # noqa: TRY003
                    f"pi session event {event_id!r} was rewritten after it was imported"
                )

        for event in events:
            if event.event_id in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)

        merged = dict(imported)
        for event in events:
            merged[event.event_id] = digests[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": CHECKPOINT_VERSION, "events": merged},
        )
