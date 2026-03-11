"""
ledger_core/sequence_manager.py

Sequence Manager for VFEL — monotonic sequence number assignment.

Architecture:
- Two sequence spaces:
    1. Global ledger sequence — unique across ALL shards, monotonically increasing
    2. Per-shard sequence — monotonically increasing within each shard, starts at 0

Why two sequences?
- Global sequence: total ordering across the ledger for DAG block construction (Phase 4)
- Shard sequence: dense sequential numbering for per-shard Merkle trees (Phase 3)
  and enables fast gap detection ("shard 7 is missing sequence 4442")

Threading model:
- This is intentionally NOT thread-safe in Phase 2.
- In Phase 4+ we'll introduce a lock-free atomic counter or a dedicated
  sequence service when we go multi-process.
- Single-threaded ingestion pipeline is the Phase 2 constraint.

Persistence:
- Sequences survive restarts via LedgerState (ledger_state.py).
- SequenceManager is initialized from persisted state on startup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Snapshot for persistence
# ──────────────────────────────────────────────

@dataclass
class SequenceSnapshot:
    """
    Serializable snapshot of all sequence counters.
    Written to LedgerState on flush/shutdown, loaded on startup.
    """
    global_sequence: int
    shard_sequences: dict[str, int]   # shard_key → last assigned shard_seq

    def to_dict(self) -> dict:
        return {
            "global_sequence":  self.global_sequence,
            "shard_sequences":  self.shard_sequences,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceSnapshot":
        return cls(
            global_sequence=d.get("global_sequence", -1),
            shard_sequences=d.get("shard_sequences", {}),
        )

    @classmethod
    def empty(cls) -> "SequenceSnapshot":
        return cls(global_sequence=-1, shard_sequences={})


# ──────────────────────────────────────────────
# Assigned sequence pair
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SequenceAssignment:
    """
    Result of assigning sequence numbers to an event.
    Immutable — passed directly into StoredEvent construction.
    """
    global_seq: int
    shard_seq: int
    shard_key: str


# ──────────────────────────────────────────────
# Sequence Manager
# ──────────────────────────────────────────────

class SequenceManager:
    """
    Assigns monotonic sequence numbers to ledger events.

    Usage:
        sm = SequenceManager()
        assignment = sm.next(shard_key="RELIANCE")
        # assignment.global_seq, assignment.shard_seq

    Initialization from snapshot (restart recovery):
        snapshot = SequenceSnapshot.from_dict(persisted_state)
        sm = SequenceManager.from_snapshot(snapshot)

    Stats/inspection:
        sm.get_shard_count("RELIANCE")  → int
        sm.snapshot()                   → SequenceSnapshot
    """

    def __init__(self):
        self._global_seq: int = -1                     # Next call returns 0
        self._shard_seqs: dict[str, int] = {}          # shard_key → last seq
        self._total_assigned: int = 0
        # Phase 4 note: replace with threading.Lock() when going concurrent
        self._lock = threading.Lock()

    @classmethod
    def from_snapshot(cls, snapshot: SequenceSnapshot) -> "SequenceManager":
        """Restore manager state from a persisted snapshot."""
        sm = cls()
        sm._global_seq = snapshot.global_sequence
        sm._shard_seqs = dict(snapshot.shard_sequences)
        logger.info(
            "SequenceManager restored: global_seq=%d, %d shards tracked",
            sm._global_seq, len(sm._shard_seqs)
        )
        return sm

    def next(self, shard_key: str) -> SequenceAssignment:
        """
        Atomically assign the next global + shard sequence for a given shard.
        Monotonic guarantee: returned values are always > all previous values.
        """
        with self._lock:
            self._global_seq += 1
            current_shard_seq = self._shard_seqs.get(shard_key, -1) + 1
            self._shard_seqs[shard_key] = current_shard_seq
            self._total_assigned += 1

            return SequenceAssignment(
                global_seq=self._global_seq,
                shard_seq=current_shard_seq,
                shard_key=shard_key,
            )

    def peek_global(self) -> int:
        """Current global sequence (last assigned). -1 if nothing assigned yet."""
        return self._global_seq

    def peek_shard(self, shard_key: str) -> int:
        """Current shard sequence for a given key. -1 if no events assigned."""
        return self._shard_seqs.get(shard_key, -1)

    def get_shard_count(self, shard_key: str) -> int:
        """Number of events assigned to this shard."""
        return self._shard_seqs.get(shard_key, -1) + 1

    def known_shards(self) -> list[str]:
        """All shard keys that have received at least one event."""
        return list(self._shard_seqs.keys())

    def total_assigned(self) -> int:
        return self._total_assigned

    def snapshot(self) -> SequenceSnapshot:
        """Capture current state for persistence."""
        with self._lock:
            return SequenceSnapshot(
                global_sequence=self._global_seq,
                shard_sequences=dict(self._shard_seqs),
            )

    def describe(self) -> dict:
        return {
            "global_seq":     self._global_seq,
            "total_assigned": self._total_assigned,
            "num_shards":     len(self._shard_seqs),
            "shard_summary":  {k: v for k, v in sorted(self._shard_seqs.items())},
        }