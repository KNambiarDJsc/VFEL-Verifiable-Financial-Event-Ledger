
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class EventClass(str, Enum):
    TRADE  = "TRADE"
    QUOTE  = "QUOTE"
    ORDER  = "ORDER"
    CANCEL = "CANCEL"
    STATUS = "STATUS"
    ANCHOR = "ANCHOR"
    UNKNOWN = "UNKNOWN"


class StorageStatus(str, Enum):
    PENDING  = "PENDING"   # Received, not yet written to store
    STORED   = "STORED"    # Written to append-only store
    SEALED   = "SEALED"    # Included in a Merkle tree (Phase 3)
    ANCHORED = "ANCHORED"  # Included in an external anchor (Phase 7)


@dataclass(frozen=True)
class LedgerEvent:

    # Identity
    ledger_event_id: str          # SHA256(source_file:line:content_hash) — replay-stable
    content_hash: str             # SHA256(canonical payload JSON) — crypto identity

    # Classification
    event_class: EventClass
    shard_key: str                # Symbol or HASH_prefix — determines shard affinity

    # Timing
    timestamp_ns: int             # Canonical nanosecond epoch
    ingested_at_ns: int           # Wall clock at normalization

    # Payload (immutable copy)
    payload: dict                 # Frozen-compatible: store as-is, never mutate

    # Source provenance
    source_file: str
    source_line: int
    source_byte_offset: int

    # VCP crypto fields preserved for cross-verification
    original_event_hash: Optional[str]
    original_prev_hash: Optional[str]
    original_signature: Optional[str]

    # Optional enrichment
    symbol: Optional[str] = None
    sequence: Optional[int] = None

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "ledger_event_id":    self.ledger_event_id,
            "content_hash":       self.content_hash,
            "event_class":        self.event_class.value,
            "shard_key":          self.shard_key,
            "timestamp_ns":       self.timestamp_ns,
            "ingested_at_ns":     self.ingested_at_ns,
            "source_file":        self.source_file,
            "source_line":        self.source_line,
            "source_byte_offset": self.source_byte_offset,
            "original_event_hash": self.original_event_hash,
            "original_prev_hash":  self.original_prev_hash,
            "original_signature":  self.original_signature,
            "symbol":             self.symbol,
            "sequence":           self.sequence,
            "payload":            self.payload,
        }

    @classmethod
    def from_wire_dict(cls, d: dict) -> "LedgerEvent":
        """Reconstruct from serialized form (e.g. loaded from disk)."""
        return cls(
            ledger_event_id=d["ledger_event_id"],
            content_hash=d["content_hash"],
            event_class=EventClass(d["event_class"]),
            shard_key=d["shard_key"],
            timestamp_ns=d["timestamp_ns"],
            ingested_at_ns=d["ingested_at_ns"],
            source_file=d["source_file"],
            source_line=d["source_line"],
            source_byte_offset=d["source_byte_offset"],
            original_event_hash=d.get("original_event_hash"),
            original_prev_hash=d.get("original_prev_hash"),
            original_signature=d.get("original_signature"),
            symbol=d.get("symbol"),
            sequence=d.get("sequence"),
            payload=d.get("payload", {}),
        )

    def compute_ledger_hash(self) -> str:
        canonical = json.dumps({
            "id":           self.ledger_event_id,
            "content_hash": self.content_hash,
            "shard_key":    self.shard_key,
            "timestamp_ns": self.timestamp_ns,
            "event_class":  self.event_class.value,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredEvent:

    event: LedgerEvent

    # Sequence numbers assigned at write time
    ledger_sequence: int          # Global monotonic — unique across all shards
    shard_sequence: int           # Per-shard monotonic — dense sequence within shard

    # Timing
    stored_at_ns: int

    # Chain linkage within shard
    prev_ledger_hash: str         # Hash of previous StoredEvent in this shard ("0"*64 for first)
    shard_key: str                # Denormalized for fast access

    # Derived: hash of this stored event (includes sequence + chain linkage)
    stored_event_hash: str = field(default="")

    def __post_init__(self):
        # Bypass frozen restriction for computed field
        if not self.stored_event_hash:
            object.__setattr__(self, "stored_event_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """
        Hash that covers: event identity + sequence positions + chain linkage.
        This is what goes into the Merkle tree as a leaf in Phase 3.
        """
        canonical = json.dumps({
            "ledger_event_id": self.event.ledger_event_id,
            "content_hash":    self.event.content_hash,
            "ledger_seq":      self.ledger_sequence,
            "shard_seq":       self.shard_sequence,
            "shard_key":       self.shard_key,
            "prev_hash":       self.prev_ledger_hash,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "event":              self.event.to_wire_dict(),
            "ledger_sequence":    self.ledger_sequence,
            "shard_sequence":     self.shard_sequence,
            "stored_at_ns":       self.stored_at_ns,
            "prev_ledger_hash":   self.prev_ledger_hash,
            "shard_key":          self.shard_key,
            "stored_event_hash":  self.stored_event_hash,
        }

    @classmethod
    def from_wire_dict(cls, d: dict) -> "StoredEvent":
        return cls(
            event=LedgerEvent.from_wire_dict(d["event"]),
            ledger_sequence=d["ledger_sequence"],
            shard_sequence=d["shard_sequence"],
            stored_at_ns=d["stored_at_ns"],
            prev_ledger_hash=d["prev_ledger_hash"],
            shard_key=d["shard_key"],
            stored_event_hash=d["stored_event_hash"],
        )