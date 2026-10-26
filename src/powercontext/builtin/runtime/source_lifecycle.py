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

"""Propagate Source availability through a per-Scope evidence dependency DAG.

Every Scope owns one DAG whose roots are Sources and whose derived nodes are Memory entries
and Topic Memories. A node is *available* only while its parents still support it under its
own ``all`` / ``any`` policy, so withdrawing a Source removes every derived body that loses
its backing and restores them when the Source comes back. Withdrawal never rewrites or deletes
a body: it only changes use eligibility, which keeps the historical citation readable for
audit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.sources.models import SourceRef

SourceLifecycleFamily = Literal["source", "memory", "topic"]
SourceLifecyclePolicy = Literal["all", "any"]
SourceLifecyclePurpose = Literal["generation", "audit"]

SOURCE_LIFECYCLE_PURPOSES: frozenset[str] = frozenset({"generation", "audit"})
MAX_LIFECYCLE_PARENTS = 64


class SourceLifecycleError(ValueError):
    """Report a lifecycle registration, event, or reference contract violation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class SourceLifecycleEvent(BaseModel):
    """One ordered availability change for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    reason: str
    revision: int
    sequence: int
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty, non-blank string")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence", mode="before")
    @classmethod
    def validate_positive_int(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be a positive integer")  # noqa: TRY003
        if value < 1:
            raise ValueError("must be a positive integer")  # noqa: TRY003
        return value

    @field_validator("available", mode="before")
    @classmethod
    def validate_available(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("must be a boolean")  # noqa: TRY003
        return value


class SourceLifecycleNode(BaseModel):
    """One persisted DAG node and its currently propagated availability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    family: SourceLifecycleFamily
    reference: dict[str, Any]
    parents: tuple[str, ...] = ()
    policy: SourceLifecyclePolicy = "all"
    available: bool = True
    revision: int = 1
    sequence: int = 0
    ordinal: int = 0


NodeLoader = Callable[[], Awaitable[tuple[SourceLifecycleNode, ...]]]
NodeSaver = Callable[[Sequence[SourceLifecycleNode]], Awaitable[None]]
EntryStateWriter = Callable[[MemoryCitation, bool], Awaitable[bool]]
ReferenceReader = Callable[[str, Mapping[str, Any]], Awaitable[bool]]


class ScopedSourceLifecycle:
    """Register one Scope's evidence DAG and propagate Source availability through it."""

    def __init__(
        self,
        scope_id: str,
        *,
        load: NodeLoader,
        save: NodeSaver,
        write_entry_state: EntryStateWriter,
        reference_exists: ReferenceReader,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self.scope_id = scope_id
        self._load = load
        self._save = save
        self._write_entry_state = write_entry_state
        self._reference_exists = reference_exists
        self._lock = asyncio.Lock() if lock is None else lock

    async def register_source(
        self,
        node_id: str,
        reference: SourceRef,
        /,
        *,
        revision: int = 1,
    ) -> None:
        """Register one Source root at a fixed lifecycle revision."""

        _require_node_id(node_id)
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("source-reference", "must be a SourceRef")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("source-revision", "must be a positive integer")
        if not await self._reference_exists("source", reference.model_dump(mode="json")):
            raise SourceLifecycleError("source-unreadable", node_id)
        node = SourceLifecycleNode(
            node_id=node_id,
            family="source",
            reference=reference.model_dump(mode="json"),
            revision=revision,
        )
        await self._register(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: SourceLifecyclePolicy,
    ) -> None:
        """Register one Memory entry whose body needs its parents' support."""

        _require_node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-reference", "must be a MemoryCitation")
        if not await self._reference_exists("memory", citation.model_dump(mode="json")):
            raise SourceLifecycleError("memory-unreadable", node_id)
        await self._register_derived(node_id, "memory", citation.model_dump(mode="json"), parents, policy)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: SourceLifecyclePolicy,
    ) -> None:
        """Register one Topic Memory whose body needs its parents' support."""

        _require_node_id(node_id)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("topic-reference", "must be an ArtifactRef")
        if not await self._reference_exists("topic", reference.model_dump(mode="json")):
            raise SourceLifecycleError("topic-unreadable", node_id)
        await self._register_derived(node_id, "topic", reference.model_dump(mode="json"), parents, policy)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event, propagate it, and return the sorted available node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event", "must be a SourceLifecycleEvent")
        async with self._lock:
            nodes = await self._nodes()
            node = nodes.get(event.node_id)
            if node is None:
                raise SourceLifecycleError("unknown-node", event.node_id)
            if node.family != "source":
                raise SourceLifecycleError("event-target", f"{event.node_id} is not a Source node")
            if event.revision != node.revision:
                raise SourceLifecycleError(
                    "revision-mismatch",
                    f"{event.node_id} expects revision {node.revision}, got {event.revision}",
                )
            if event.sequence < node.sequence:
                raise SourceLifecycleError(
                    "sequence-regression",
                    f"{event.node_id} already applied sequence {node.sequence}, got {event.sequence}",
                )
            if event.sequence == node.sequence:
                if event.available == node.available:
                    return sorted(name for name, item in nodes.items() if item.available)
                raise SourceLifecycleError(
                    "sequence-conflict",
                    f"{event.node_id} sequence {event.sequence} already recorded differently",
                )

            nodes[event.node_id] = node.model_copy(
                update={"available": event.available, "sequence": event.sequence}
            )
            propagated = _propagate(nodes)
            await self._sync_entries(nodes, propagated)
            await self._save(tuple(nodes.values()))
            return sorted(name for name, item in nodes.items() if item.available)

    async def resolve(self, node_id: str, /, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the original reference a node was registered with."""

        _require_node_id(node_id)
        if purpose not in SOURCE_LIFECYCLE_PURPOSES:
            raise SourceLifecycleError("unknown-purpose", str(purpose))
        async with self._lock:
            nodes = await self._nodes()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node", node_id)
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable", node_id)
        return _decode_reference(node)

    async def available_nodes(self) -> list[str]:
        """Return the sorted ids of every node currently eligible for generation."""

        async with self._lock:
            nodes = await self._nodes()
        return sorted(name for name, item in nodes.items() if item.available)

    async def inactive_topics(self) -> frozenset[str]:
        """Return the artifact ids of Topic Memories that lost their support."""

        async with self._lock:
            nodes = await self._nodes()
        return frozenset(
            str(node.reference["artifact_id"])
            for node in nodes.values()
            if node.family == "topic" and not node.available and "artifact_id" in node.reference
        )

    async def _register_derived(
        self,
        node_id: str,
        family: SourceLifecycleFamily,
        reference: dict[str, Any],
        parents: Sequence[str],
        policy: SourceLifecyclePolicy,
    ) -> None:
        if policy not in ("all", "any"):
            raise SourceLifecycleError("policy", "must be 'all' or 'any'")
        parent_ids = tuple(parents)
        if not parent_ids:
            raise SourceLifecycleError("parents", f"{node_id} needs at least one parent")
        if len(parent_ids) > MAX_LIFECYCLE_PARENTS:
            raise SourceLifecycleError("parents", f"{node_id} declares too many parents")
        if any(not isinstance(parent, str) or not parent.strip() for parent in parent_ids):
            raise SourceLifecycleError("parents", f"{node_id} has a blank parent id")
        if len(set(parent_ids)) != len(parent_ids):
            raise SourceLifecycleError("parents", f"{node_id} repeats a parent")
        if node_id in parent_ids:
            raise SourceLifecycleError("cycle", f"{node_id} cannot support itself")
        node = SourceLifecycleNode(
            node_id=node_id,
            family=family,
            reference=reference,
            parents=parent_ids,
            policy=policy,
        )
        await self._register(node)

    async def _register(self, node: SourceLifecycleNode, /) -> None:
        async with self._lock:
            nodes = await self._nodes()
            existing = nodes.get(node.node_id)
            if existing is not None:
                if _same_binding(existing, node):
                    return
                raise SourceLifecycleError("binding-conflict", node.node_id)
            for parent in node.parents:
                if parent not in nodes:
                    raise SourceLifecycleError("unknown-parent", f"{node.node_id} -> {parent}")
            nodes[node.node_id] = node.model_copy(update={"ordinal": _next_ordinal(nodes)})
            propagated = _propagate(nodes)
            await self._sync_entries(nodes, propagated)
            await self._save(tuple(nodes.values()))

    async def _nodes(self) -> dict[str, SourceLifecycleNode]:
        stored = await self._load()
        return {node.node_id: node for node in stored}

    async def _sync_entries(self, nodes: Mapping[str, SourceLifecycleNode], changed: set[str], /) -> None:
        for node_id in changed:
            node = nodes[node_id]
            if node.family != "memory":
                continue
            await self._write_entry_state(MemoryCitation.model_validate(node.reference), node.available)


def _propagate(nodes: dict[str, SourceLifecycleNode], /) -> set[str]:
    """Recompute every derived node in registration order; return the ones that changed."""

    active = {name for name, node in nodes.items() if node.family == "source" and node.available}
    changed: set[str] = set()
    for node in sorted(nodes.values(), key=lambda item: item.ordinal):
        if node.family == "source":
            continue
        support = [parent in active for parent in node.parents]
        available = all(support) if node.policy == "all" else any(support)
        if available:
            active.add(node.node_id)
        if available != node.available:
            nodes[node.node_id] = node.model_copy(update={"available": available})
            changed.add(node.node_id)
    return changed


def _same_binding(left: SourceLifecycleNode, right: SourceLifecycleNode, /) -> bool:
    return (
        left.family == right.family
        and left.reference == right.reference
        and left.parents == right.parents
        and left.policy == right.policy
        and left.revision == right.revision
    )


def _next_ordinal(nodes: Mapping[str, SourceLifecycleNode], /) -> int:
    return max((node.ordinal for node in nodes.values()), default=-1) + 1


def _decode_reference(node: SourceLifecycleNode, /) -> SourceRef | MemoryCitation | ArtifactRef:
    if node.family == "source":
        return SourceRef.model_validate(node.reference)
    if node.family == "memory":
        return MemoryCitation.model_validate(node.reference)
    return ArtifactRef.model_validate(node.reference)


def _require_node_id(node_id: object, /) -> None:
    if not isinstance(node_id, str) or not node_id or node_id != node_id.strip():
        raise SourceLifecycleError("node-id", "must be a non-empty trimmed string")
