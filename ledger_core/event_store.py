"""
ledger_core/event_store.py

Append-Only Event Store for VFEL.

Architecture:
- In-memory append-only store per shard (Phase 2)
- No deletes, no updates — ever. Ledger events are immutable facts.
- Per-shard chain: each StoredEvent links to the previous via prev_ledger_hash
- Indexed for O(1) lookup by: ledger_event_id, global_seq, shard+shard_seq
- Memory layout: flat list per shard + dict indexes for fast access

Persistence:
- Phase 2 keeps everything in memory (fast, simple, correct)
- Phase 4+ will add WAL (Write-Ahead Log) + memory-mapped shard files
- The interface is designed to be persistence-backend-agnostic

Replay:
- replay_shard(shard_key) → yields StoredEvents in shard_seq order
- replay_global(from_seq, to_seq) → yields StoredEvents in global_seq order
- Both are generators: O(1) memory regardless of store size

Integrity:
- append() verifies the chain linkage before writing
- verify_shard_chain() walks a shard and checks every hash link
- This is the fast local check — full cryptographic proofs are Phase 6
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional

from ledger_core.event_model import LedgerEvent, StoredEvent
from ledger_core.sequence_manager import SequenceManager

logger = logging.getLogger(__name__)

# Sentinel hash for the first event in any shard — no predecessor
GENESIS_HASH = "0" * 64


# ──────────────────────────────────────────────
# Store result types
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class AppendResult:
    """Result of a single event append operation."""
    success: bool
    stored_event: Optional[StoredEvent] = None
    error: Optional[str] = None
    duplicate: bool = False          # True if event_id already in store


@dataclass
class ShardStats:
    shard_key: str
    event_count: int
    first_stored_at_ns: Optional[int]
    last_stored_at_ns: Optional[int]
    head_hash: str                   # Hash of the latest event (chain tip)
    last_global_seq: int
    last_shard_seq: int


# ──────────────────────────────────────────────
# Shard Store (internal — one per shard)
# ──────────────────────────────────────────────

class _ShardStore:
    """
    Internal per-shard append-only storage.
    Not exposed directly — accessed through EventStore.
    """

    def __init__(self, shard_key: str):
        self.shard_key = shard_key
        self._events: list[StoredEvent] = []          # Dense list — index = shard_seq
        self._head_hash: str = GENESIS_HASH            # Hash of last appended event
        self._id_index: dict[str, int] = {}            # ledger_event_id → shard_seq
        self._global_seq_index: dict[int, int] = {}    # global_seq → shard_seq

    def append(self, stored_event: StoredEvent) -> None:
        """Append to this shard. Caller is responsible for sequence assignment."""
        expected_shard_seq = len(self._events)
        if stored_event.shard_sequence != expected_shard_seq:
            raise ValueError(
                f"Shard {self.shard_key}: expected shard_seq={expected_shard_seq}, "
                f"got {stored_event.shard_sequence}"
            )

        if stored_event.prev_ledger_hash != self._head_hash:
            raise ValueError(
                f"Shard {self.shard_key}: chain broken at shard_seq={expected_shard_seq}. "
                f"Expected prev={self._head_hash[:16]}..., "
                f"got {stored_event.prev_ledger_hash[:16]}..."
            )

        self._events.append(stored_event)
        self._head_hash = stored_event.stored_event_hash
        self._id_index[stored_event.event.ledger_event_id] = expected_shard_seq
        self._global_seq_index[stored_event.ledger_sequence] = expected_shard_seq

    def get_by_shard_seq(self, shard_seq: int) -> Optional[StoredEvent]:
        if 0 <= shard_seq < len(self._events):
            return self._events[shard_seq]
        return None

    def get_by_event_id(self, event_id: str) -> Optional[StoredEvent]:
        shard_seq = self._id_index.get(event_id)
        return self._events[shard_seq] if shard_seq is not None else None

    def get_by_global_seq(self, global_seq: int) -> Optional[StoredEvent]:
        shard_seq = self._global_seq_index.get(global_seq)
        return self._events[shard_seq] if shard_seq is not None else None

    def has_event_id(self, event_id: str) -> bool:
        return event_id in self._id_index

    def replay(self, from_shard_seq: int = 0) -> Generator[StoredEvent, None, None]:
        for event in self._events[from_shard_seq:]:
            yield event

    def head_hash(self) -> str:
        return self._head_hash

    def count(self) -> int:
        return len(self._events)

    def stats(self) -> ShardStats:
        count = len(self._events)
        return ShardStats(
            shard_key=self.shard_key,
            event_count=count,
            first_stored_at_ns=self._events[0].stored_at_ns if count > 0 else None,
            last_stored_at_ns=self._events[-1].stored_at_ns if count > 0 else None,
            head_hash=self._head_hash,
            last_global_seq=self._events[-1].ledger_sequence if count > 0 else -1,
            last_shard_seq=count - 1,
        )

    def verify_chain_integrity(self) -> tuple[bool, Optional[str]]:
        """
        Walk the shard's hash chain and verify every link.
        O(n) — use for audit/debug, not hot path.
        Returns (ok, error_message).
        """
        expected_prev = GENESIS_HASH

        for i, stored in enumerate(self._events):
            if stored.prev_ledger_hash != expected_prev:
                return False, (
                    f"Chain break at shard_seq={i}: "
                    f"expected prev={expected_prev[:16]}..., "
                    f"got {stored.prev_ledger_hash[:16]}..."
                )

            # Recompute hash to verify integrity
            recomputed = stored._compute_hash()
            if recomputed != stored.stored_event_hash:
                return False, (
                    f"Hash mismatch at shard_seq={i}: "
                    f"stored={stored.stored_event_hash[:16]}..., "
                    f"recomputed={recomputed[:16]}..."
                )

            expected_prev = stored.stored_event_hash

        return True, None


# ──────────────────────────────────────────────
# Event Store
# ──────────────────────────────────────────────

class EventStore:
    """
    VFEL's append-only event store.

    Single point of truth for all stored ledger events.
    All writes go through append() — read operations are non-mutating.

    Deduplication: events with the same ledger_event_id are rejected silently
    (idempotent append — safe for replay from JSONL).

    Thread safety: NOT thread-safe in Phase 2. Phase 4 adds per-shard locks.
    """

    def __init__(self, sequence_manager: Optional[SequenceManager] = None):
        self._seq_manager = sequence_manager or SequenceManager()
        self._shards: dict[str, _ShardStore] = {}          # shard_key → _ShardStore
        self._global_index: dict[int, tuple[str, int]] = {}  # global_seq → (shard_key, shard_seq)
        self._total_appended: int = 0
        self._total_duplicates: int = 0

    # ── Write ──────────────────────────────────────────────────────────

    def append(self, event: LedgerEvent) -> AppendResult:
        """
        Append a single LedgerEvent to the store.

        - Assigns sequence numbers via SequenceManager
        - Computes chain linkage (prev_ledger_hash)
        - Validates and writes to the appropriate shard store
        - Returns AppendResult (never raises on soft errors)
        """
        shard_key = event.shard_key

        # Deduplication check
        shard = self._shards.get(shard_key)
        if shard and shard.has_event_id(event.ledger_event_id):
            self._total_duplicates += 1
            logger.debug("Duplicate event skipped: %s", event.ledger_event_id[:16])
            return AppendResult(
                success=True,
                stored_event=shard.get_by_event_id(event.ledger_event_id),
                duplicate=True,
            )

        # Initialize shard store on first event
        if shard_key not in self._shards:
            self._shards[shard_key] = _ShardStore(shard_key)
            logger.debug("New shard initialized: %s", shard_key)

        shard = self._shards[shard_key]

        # Assign sequence numbers
        assignment = self._seq_manager.next(shard_key)

        # Get chain tip for this shard
        prev_hash = shard.head_hash()

        # Build stored event
        stored = StoredEvent(
            event=event,
            ledger_sequence=assignment.global_seq,
            shard_sequence=assignment.shard_seq,
            stored_at_ns=time.time_ns(),
            prev_ledger_hash=prev_hash,
            shard_key=shard_key,
        )

        try:
            shard.append(stored)
            self._global_index[assignment.global_seq] = (shard_key, assignment.shard_seq)
            self._total_appended += 1

            return AppendResult(success=True, stored_event=stored)

        except ValueError as exc:
            logger.error("Append failed for %s: %s", event.ledger_event_id[:16], exc)
            return AppendResult(success=False, error=str(exc))

    def append_batch(self, events) -> list[AppendResult]:
        """
        Append a batch of LedgerEvents.
        Returns one AppendResult per event in input order.
        """
        return [self.append(e) for e in events]

    # ── Read ───────────────────────────────────────────────────────────

    def get_by_event_id(self, event_id: str) -> Optional[StoredEvent]:
        """O(1) lookup by ledger_event_id. Scans known shards."""
        # Fast path: if we know the shard, go direct
        # Phase 4 will add a global event_id index; for now scan shards
        for shard in self._shards.values():
            result = shard.get_by_event_id(event_id)
            if result:
                return result
        return None

    def get_by_global_seq(self, global_seq: int) -> Optional[StoredEvent]:
        """O(1) lookup by global sequence number."""
        location = self._global_index.get(global_seq)
        if not location:
            return None
        shard_key, shard_seq = location
        return self._shards[shard_key].get_by_shard_seq(shard_seq)

    def get_shard_head(self, shard_key: str) -> Optional[StoredEvent]:
        """Get the most recently appended event for a shard."""
        shard = self._shards.get(shard_key)
        if not shard or shard.count() == 0:
            return None
        return shard.get_by_shard_seq(shard.count() - 1)

    # ── Replay ─────────────────────────────────────────────────────────

    def replay_shard(
        self,
        shard_key: str,
        from_shard_seq: int = 0,
    ) -> Generator[StoredEvent, None, None]:
        """
        Replay all events for a shard in order, starting from from_shard_seq.
        Generator: O(1) memory, correct for arbitrarily large shards.
        """
        shard = self._shards.get(shard_key)
        if not shard:
            logger.warning("replay_shard: unknown shard %s", shard_key)
            return
        yield from shard.replay(from_shard_seq=from_shard_seq)

    def replay_global(
        self,
        from_global_seq: int = 0,
        to_global_seq: Optional[int] = None,
    ) -> Generator[StoredEvent, None, None]:
        """
        Replay events in global sequence order.
        NOTE: This iterates the global_index dict — O(n) but ordered by seq.
        For production-scale replay, Phase 4 will use a sorted slab file.
        """
        end = to_global_seq if to_global_seq is not None else self._seq_manager.peek_global()
        for seq in range(from_global_seq, end + 1):
            event = self.get_by_global_seq(seq)
            if event:
                yield event

    def replay_all_shards(self) -> Generator[tuple[str, StoredEvent], None, None]:
        """
        Yield (shard_key, StoredEvent) for all shards, each in shard_seq order.
        Useful for Phase 3 Merkle tree reconstruction from scratch.
        """
        for shard_key in sorted(self._shards.keys()):
            for event in self._shards[shard_key].replay():
                yield shard_key, event

    # ── Integrity ──────────────────────────────────────────────────────

    def verify_shard_chain(self, shard_key: str) -> tuple[bool, Optional[str]]:
        """Verify hash chain integrity for a single shard. O(n)."""
        shard = self._shards.get(shard_key)
        if not shard:
            return False, f"Unknown shard: {shard_key}"
        return shard.verify_chain_integrity()

    def verify_all_chains(self) -> dict[str, tuple[bool, Optional[str]]]:
        """Verify all shard chains. Returns {shard_key: (ok, error)}."""
        return {
            shard_key: shard.verify_chain_integrity()
            for shard_key, shard in self._shards.items()
        }

    # ── Observability ──────────────────────────────────────────────────

    def shard_stats(self, shard_key: str) -> Optional[ShardStats]:
        shard = self._shards.get(shard_key)
        return shard.stats() if shard else None

    def all_shard_stats(self) -> dict[str, ShardStats]:
        return {k: v.stats() for k, v in self._shards.items()}

    def known_shards(self) -> list[str]:
        return list(self._shards.keys())

    def total_events(self) -> int:
        return self._total_appended

    def total_duplicates(self) -> int:
        return self._total_duplicates

    def describe(self) -> dict:
        return {
            "total_events":     self._total_appended,
            "total_duplicates": self._total_duplicates,
            "num_shards":       len(self._shards),
            "global_seq_head":  self._seq_manager.peek_global(),
            "sequence_state":   self._seq_manager.describe(),
        }