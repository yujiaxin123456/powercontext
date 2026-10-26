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

"""Chunked, digest-addressed transport for prepared Context bodies."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
CONTEXT_PACK_CHUNK_BYTES = 4096

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PREFIX = "sha256:"


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta violates its wire contract."""


def _canonical(value: object, /) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _encode_items(items: tuple[bytes, ...], /) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(values: object, /) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("items must be a list")  # noqa: TRY003
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("each item must be a base64 string")  # noqa: TRY003
        try:
            decoded.append(base64.b64decode(value.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError("item is not valid base64") from error
    return tuple(decoded)


def _parse_json(data: bytes, /) -> object:
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ContextPackError("payload is not valid JSON") from error


def _require_digest(value: object, field: str, /) -> str:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX):
        raise ContextPackError(f"{field} must be a sha256 digest")  # noqa: TRY003
    body = value[len(_DIGEST_PREFIX) :]
    if len(body) != 64 or any(character not in "0123456789abcdef" for character in body):
        raise ContextPackError(f"{field} must be a sha256 digest")  # noqa: TRY003
    return value


class ContextPack:
    """One complete Context body carried as ordered byte chunks.

    ``byte_count`` is the size of the canonical wire form, so it includes the
    digits of ``byte_count`` itself. It is resolved to a fixed point before the
    pack is exposed, which keeps ``len(pack.to_bytes()) == pack.byte_count``.
    """

    __slots__ = ("items", "digest", "byte_count")

    def __init__(self, items: tuple[bytes, ...], digest: str, byte_count: int) -> None:
        self.items = tuple(items)
        self.digest = digest
        self.byte_count = byte_count

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes], /) -> ContextPack:
        chunked = tuple(bytes(item) for item in items)
        unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(chunked)}
        digest = _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()
        return cls(chunked, digest, _resolve_byte_count(unsigned, digest))

    @classmethod
    def from_content(cls, text: str, *, chunk_bytes: int = CONTEXT_PACK_CHUNK_BYTES) -> ContextPack:
        """Split UTF-8 text on character boundaries into chunks of at most ``chunk_bytes``."""

        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        chunks: list[bytes] = []
        current = bytearray()
        for character in text:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError(  # noqa: TRY003
                    "chunk_bytes is smaller than one UTF-8 character"
                )
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current = bytearray()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": CONTEXT_PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        payload = _parse_json(data)
        if not isinstance(payload, dict) or set(payload) != _PACK_FIELDS:
            raise ContextPackError("context pack payload has an unexpected shape")  # noqa: TRY003
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("context pack schema is not supported")  # noqa: TRY003
        digest = _require_digest(payload["digest"], "digest")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ContextPackError("byte_count must be a non-negative integer")  # noqa: TRY003
        rebuilt = cls.build(_decode_items(payload["items"]))
        if rebuilt.digest != digest:
            raise ContextPackError("context pack digest does not match its items")  # noqa: TRY003
        if rebuilt.byte_count != byte_count:
            raise ContextPackError("context pack byte_count does not match its payload")  # noqa: TRY003
        return rebuilt

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextPack):
            return NotImplemented
        return (
            self.items == other.items
            and self.digest == other.digest
            and self.byte_count == other.byte_count
        )

    def __hash__(self) -> int:
        return hash((self.items, self.digest, self.byte_count))

    def __repr__(self) -> str:
        return f"ContextPack(items={len(self.items)}, byte_count={self.byte_count})"


class ContextPackDelta:
    """One prefix-reusing delta that rebuilds a target pack from a known base."""

    __slots__ = ("base_digest", "target_digest", "prefix_items", "appended")

    def __init__(
        self,
        base_digest: str,
        target_digest: str,
        prefix_items: int,
        appended: tuple[bytes, ...] | list[bytes],
    ) -> None:
        self.base_digest = base_digest
        self.target_digest = target_digest
        self.prefix_items = prefix_items
        self.appended = tuple(bytes(item) for item in appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        prefix = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        payload = _parse_json(data)
        if not isinstance(payload, dict) or set(payload) != _DELTA_FIELDS:
            raise ContextPackError("context pack delta payload has an unexpected shape")  # noqa: TRY003
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema is not supported")  # noqa: TRY003
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        return cls(
            _require_digest(payload["base_digest"], "base_digest"),
            _require_digest(payload["target_digest"], "target_digest"),
            prefix_items,
            _decode_items(payload["appended"]),
        )

    def __repr__(self) -> str:
        return (
            f"ContextPackDelta(prefix_items={self.prefix_items}, appended={len(self.appended)}, "
            f"byte_count={self.byte_count})"
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from ``base`` and one delta, verifying both digests."""

    if base.digest != delta.base_digest:
        raise ContextPackError("delta base_digest does not match the supplied base pack")  # noqa: TRY003
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix_items exceeds the base pack")  # noqa: TRY003
    rebuilt = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack does not match the delta target_digest")  # noqa: TRY003
    return rebuilt


def _resolve_byte_count(unsigned: dict[str, Any], digest: str, /) -> int:
    count = 0
    for _ in range(8):
        size = len(_canonical({**unsigned, "digest": digest, "byte_count": count}))
        if size == count:
            return count
        count = size
    raise ContextPackError("byte_count did not converge")  # noqa: TRY003


__all__ = [
    "CONTEXT_PACK_CHUNK_BYTES",
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
