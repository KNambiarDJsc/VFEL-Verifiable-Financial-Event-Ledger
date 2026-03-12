from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from data_ingestion.event_normalizer import LedgerEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Shard Assignment Result
# ──────────────────────────────────────────────

@dataclass
class ShardedEvent:
    """
    A LedgerEvent with its shard assignment resolved.
    This is the final output of the Phase 1 ingestion pipeline.
    """
    event: LedgerEvent
    shard_id: int
    shard_key: str          # The key used for routing (normalized)
    router_version: int     # For audit: which router config produced this assignment


# ──────────────────────────────────────────────
# Shard Router
# ──────────────────────────────────────────────

class ShardRouter:
    """
    Routes LedgerEvents to shards using consistent hashing.

    Shard count is fixed at construction — changing it invalidates
    all shard assignments (treat as a new ledger configuration).

    Default: 16 shards. For production at millions of events/sec,
    tune based on core count of the aggregation layer.

    Thread-safe: stateless after construction.
    """

    def __init__(self, num_shards: int = 16):
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if num_shards > 1024:
            raise ValueError("num_shards > 1024 is not supported")

        self.num_shards = num_shards
        self._version = 1  # Bump when routing logic changes — tracked in ShardedEvent

        logger.info("ShardRouter initialized: %d shards", num_shards)

    def route(self, event: LedgerEvent) -> ShardedEvent:
        """
        Assign a single LedgerEvent to a shard.
        Deterministic: same shard_key → same shard_id always.
        """
        shard_id = self._hash_to_shard(event.shard_key)

        return ShardedEvent(
            event=event,
            shard_id=shard_id,
            shard_key=event.shard_key,
            router_version=self._version,
        )

    def route_stream(self, events):
        """
        Generator: route a stream of LedgerEvents.
        Hot path — no logging per event.
        """
        for event in events:
            yield self.route(event)

    def shard_for_key(self, key: str) -> int:
        """
        Look up shard ID for a given key without a full event.
        Useful for query routing: "which shard has AAPL?"
        """
        return self._hash_to_shard(key)

    def describe(self) -> dict:
        """Returns router config — log this at startup for observability."""
        return {
            "num_shards": self.num_shards,
            "router_version": self._version,
            "algorithm": "sha256-modulo",
        }

    def _hash_to_shard(self, key: str) -> int:
        """
        Map a string key to a shard index [0, num_shards).

        Uses SHA256 for uniform distribution across arbitrary key spaces
        (symbol strings, hash prefixes, etc).

        SHA256 first 8 bytes → uint64 → modulo num_shards.
        This is stable and fast enough for the ingestion rate in Phase 1.
        Phase 3+ can upgrade to a virtual-node ring for elasticity.
        """
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        # Take first 8 bytes as big-endian uint64
        hash_int = int.from_bytes(digest[:8], byteorder="big")
        return hash_int % self.num_shards


# ──────────────────────────────────────────────
# Shard Affinity Inspector
# ──────────────────────────────────────────────

class ShardAffinityMap:
    """
    Tracks which symbols/keys are assigned to which shards.
    Built lazily as events flow through the router.

    Useful for:
    - Query routing ("send AAPL queries to shard 7")
    - Load balancing visibility ("shard 3 has 80% of volume")
    - Debugging replay distribution
    """

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self._shard_keys: dict[int, set[str]] = {i: set() for i in range(num_shards)}
        self._key_counts: dict[str, int] = {}

    def record(self, sharded_event: ShardedEvent):
        key = sharded_event.shard_key
        shard = sharded_event.shard_id
        self._shard_keys[shard].add(key)
        self._key_counts[key] = self._key_counts.get(key, 0) + 1

    def get_shard_distribution(self) -> dict[int, int]:
        """Returns event count per shard based on unique keys seen."""
        return {
            shard: sum(self._key_counts.get(k, 0) for k in keys)
            for shard, keys in self._shard_keys.items()
        }

    def get_keys_for_shard(self, shard_id: int) -> set[str]:
        return self._shard_keys.get(shard_id, set())

    def top_keys(self, n: int = 10) -> list[tuple[str, int]]:
        """Top N most frequent shard keys (symbols with most events)."""
        return sorted(self._key_counts.items(), key=lambda x: x[1], reverse=True)[:n]