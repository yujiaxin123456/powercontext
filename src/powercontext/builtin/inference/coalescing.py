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

"""Coalesce concurrent embedding requests and reuse successful vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage


class _Entry:
    """One text a caller is waiting for, with the batch that claimed it."""

    __slots__ = ("text", "future", "batch")

    def __init__(self, text: str, future: asyncio.Future[EmbeddingVector]) -> None:
        self.text = text
        self.future = future
        self.batch: _Batch | None = None


class _Batch:
    """One underlying provider request and the texts it carries."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.usage: InferenceUsage | None = None


class CoalescingEmbeddingModel:
    """Share in-flight embedding work and cache successful vectors.

    Callers that ask for the same text inside one merge window wait on a single
    provider request. Cancelling a caller detaches only that caller: the shared
    request keeps running for everyone else and still populates the cache.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: OrderedDict[str, _Entry] = OrderedDict()
        self._inflight: dict[str, _Entry] = {}
        self._flush_scheduled = False

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        items = tuple(texts)
        if not items:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        loop = asyncio.get_running_loop()
        entries = tuple(self._entry_for(text, loop) for text in items)

        vectors: list[EmbeddingVector] = []
        for entry in entries:
            vectors.append(await asyncio.shield(entry.future))

        return EmbeddingResult(vectors=tuple(vectors), usage=self._usage_for(entries))

    def _entry_for(self, text: str, loop: asyncio.AbstractEventLoop) -> _Entry:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            resolved: asyncio.Future[EmbeddingVector] = loop.create_future()
            resolved.set_result(cached)
            return _Entry(text, resolved)

        shared = self._inflight.get(text) or self._pending.get(text)
        if shared is not None:
            return shared

        entry = _Entry(text, self._new_future(loop))
        self._pending[text] = entry
        self._schedule_flush()
        return entry

    @staticmethod
    def _new_future(loop: asyncio.AbstractEventLoop) -> asyncio.Future[EmbeddingVector]:
        future: asyncio.Future[EmbeddingVector] = loop.create_future()
        # Consume the exception so an abandoned waiter cannot log a spurious warning.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return future

    def _schedule_flush(self) -> None:
        if self._flush_scheduled:
            return
        self._flush_scheduled = True
        asyncio.get_running_loop().create_task(self._flush())

    async def _flush(self) -> None:
        try:
            await asyncio.sleep(self._coalesce_seconds)
            while self._pending:
                texts: list[str] = []
                while self._pending and len(texts) < self._max_batch_size:
                    text, entry = self._pending.popitem(last=False)
                    self._inflight[text] = entry
                    texts.append(text)
                batch = _Batch(tuple(texts))
                for text in texts:
                    self._inflight[text].batch = batch
                asyncio.get_running_loop().create_task(self._run_batch(batch))
        finally:
            self._flush_scheduled = False
            if self._pending:
                self._schedule_flush()

    async def _run_batch(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:
            self._fail_batch(batch, asyncio.CancelledError())
            raise
        except BaseException as error:  # noqa: BLE001 - forwarded verbatim to every waiter
            self._fail_batch(batch, error)
            return

        vectors = tuple(result.vectors)
        if len(vectors) != len(batch.texts):
            self._fail_batch(
                batch,
                ValueError(  # noqa: TRY003
                    f"provider returned {len(vectors)} vectors for {len(batch.texts)} texts"
                ),
            )
            return

        batch.usage = result.usage
        for text, vector in zip(batch.texts, vectors, strict=True):
            resolved = tuple(vector)
            entry = self._inflight.pop(text, None)
            if entry is None:
                continue
            self._remember(text, resolved)
            if not entry.future.done():
                entry.future.set_result(resolved)

    def _fail_batch(self, batch: _Batch, error: BaseException) -> None:
        for text in batch.texts:
            entry = self._inflight.pop(text, None)
            if entry is not None and not entry.future.done():
                entry.future.set_exception(error)

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _usage_for(entries: tuple[_Entry, ...]) -> InferenceUsage:
        seen: dict[int, _Batch] = {}
        for entry in entries:
            batch = entry.batch
            if batch is not None and batch.usage is not None:
                seen[id(batch)] = batch
        usages = [batch.usage for batch in seen.values() if batch.usage is not None]
        if not usages:
            return InferenceUsage(requests=0)
        inputs = [usage.input_tokens for usage in usages]
        outputs = [usage.output_tokens for usage in usages]
        return InferenceUsage(
            requests=sum(usage.requests for usage in usages),
            input_tokens=None if all(value is None for value in inputs) else sum(value or 0 for value in inputs),
            output_tokens=None if all(value is None for value in outputs) else sum(value or 0 for value in outputs),
        )


__all__ = ["CoalescingEmbeddingModel"]
