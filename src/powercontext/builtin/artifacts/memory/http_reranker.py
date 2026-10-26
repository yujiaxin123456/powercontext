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

"""Cross-encoder reranking through one batched HTTP scoring service."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class _ScoredRank(BaseModel):
    """One scored global candidate rank returned by the service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class _ScoringResponse(BaseModel):
    """The exact response envelope the scoring service must return."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: tuple[_ScoredRank, ...]


class HTTPMemoryReranker:
    """Score coarse candidates in bounded batches and select globally.

    The service sees the original one-based rank of every candidate, so splitting a
    pool across requests never renumbers the pool. Batches are scored in full before
    selection, which keeps the final order identical to scoring every candidate at once.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the globally best original ranks without changing hit identity."""

        _validate_bounds(len(candidates), limit)

        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            batch_ranks = tuple(range(offset + 1, offset + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(batch_ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            scores.update(_read_batch_scores(response, batch_ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected_ranks = tuple(rank for rank, _score in ordered[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests),
        )


def _read_batch_scores(response: httpx.Response, batch_ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("reranker response is not valid JSON") from error

    parsed = _ScoringResponse.model_validate(payload)
    expected = set(batch_ranks)
    observed: dict[int, float] = {}
    for entry in parsed.scores:
        if entry.rank not in expected:
            raise ValueError(f"reranker returned rank {entry.rank} outside the requested batch")  # noqa: TRY003
        if entry.rank in observed:
            raise ValueError(f"reranker returned rank {entry.rank} more than once")  # noqa: TRY003
        observed[entry.rank] = entry.score
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise ValueError(f"reranker did not score ranks {missing}")  # noqa: TRY003
    return observed


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
