"""
data_ingestion/event_normalizer.py

Event Normalizer for VFEL — Verifiable Financial Event Ledger.

Responsibilities:
- Convert VCPEvent (VCP-schema) → LedgerEvent (VFEL internal schema)
- Resolve timestamps to a canonical nanosecond epoch int
- Assign a stable ledger_event_id (deterministic, content-addressed)
- Classify event types into VFEL's internal EventClass taxonomy
- Strip VCP-specific fields not needed in the ledger

This is the schema boundary. Everything above is "VCP world", everything
below is "VFEL world". Keeping this clean = easy to swap in other event
sources later (FIX, websocket feeds, Kafka topics, etc.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from data_ingestion.event_parser import VCPEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# VFEL Internal Event Schema
# ──────────────────────────────────────────────

class EventClass(str, Enum):
    """
    VFEL's internal event taxonomy.
    VCP event_type strings are mapped to these at normalization time.
    """
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    ORDER = "ORDER"
    CANCEL = "CANCEL"
    STATUS = "STATUS"
    ANCHOR = "ANCHOR"       # Cryptographic anchor events
    UNKNOWN = "UNKNOWN"     # Fallthrough — still ingested, just unclassified


# Maps VCP event_type strings (lowercased) → EventClass
_VCP_TYPE_MAP: dict[str, EventClass] = {
    "trade": EventClass.TRADE,
    "execution": EventClass.TRADE,
    "fill": EventClass.TRADE,
    "quote": EventClass.QUOTE,
    "bid": EventClass.QUOTE,
    "ask": EventClass.QUOTE,
    "order": EventClass.ORDER,
    "new_order": EventClass.ORDER,
    "order_new": EventClass.ORDER,
    "cancel": EventClass.CANCEL,
    "order_cancel": EventClass.CANCEL,
    "cancelled": EventClass.CANCEL,
    "status": EventClass.STATUS,
    "heartbeat": EventClass.STATUS,
    "anchor": EventClass.ANCHOR,
}


@dataclass
class LedgerEvent:
    """
    VFEL's canonical internal event representation.

    This is what flows through every layer of the system post-ingestion:
    Merkle trees, DAG blocks, proof generation, API responses.

    Immutable by convention — never mutate after creation.
    """

    # ── Identity ──────────────────────────────
    ledger_event_id: str        # Deterministic content-addressed ID (SHA256)
    shard_key: str              # Shard assignment (symbol or hash prefix)
    event_class: EventClass     # Normalized event type

    # ── Timing ────────────────────────────────
    timestamp_ns: int           # Canonical nanosecond epoch timestamp
    ingested_at_ns: int         # Wall clock at normalization (for latency tracking)

    # ── Source provenance ─────────────────────
    source_file: str
    source_line: int
    source_byte_offset: int

    # ── Crypto fields from VCP ─────────────────
    original_event_hash: Optional[str]   # Hash as declared in VCP event
    original_prev_hash: Optional[str]    # Chain linkage from VCP
    original_signature: Optional[str]    # Signature from VCP
    content_hash: str                    # VFEL's own hash of raw_payload

    # ── Raw payload (normalized copy) ─────────
    payload: dict[str, Any]

    # ── Derived fields ────────────────────────
    symbol: Optional[str] = None
    sequence: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize for storage / wire. Used by ledger_core event store."""
        return {
            "ledger_event_id": self.ledger_event_id,
            "shard_key": self.shard_key,
            "event_class": self.event_class.value,
            "timestamp_ns": self.timestamp_ns,
            "ingested_at_ns": self.ingested_at_ns,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "original_event_hash": self.original_event_hash,
            "original_prev_hash": self.original_prev_hash,
            "content_hash": self.content_hash,
            "symbol": self.symbol,
            "sequence": self.sequence,
            "payload": self.payload,
        }


# ──────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────

class EventNormalizer:
    """
    Converts VCPEvents into LedgerEvents.

    Stateless — safe to call concurrently from multiple threads/processes.
    All decisions are deterministic: same input → same LedgerEvent every time.

    normalize_stream() is the hot path for pipeline use.
    """

    def normalize(self, vcp_event: VCPEvent) -> LedgerEvent:
        """Convert a single VCPEvent to LedgerEvent."""
        meta = vcp_event.meta
        crypto = vcp_event.crypto

        # Resolve canonical timestamp
        timestamp_ns = _resolve_timestamp_ns(meta.timestamp_ns, meta.timestamp_ms)

        # Classify event type
        event_class = _classify_event_type(meta.event_type)

        # Determine shard key (symbol preferred; fall back to hash prefix)
        shard_key = _resolve_shard_key(meta.symbol, vcp_event.content_hash)

        # Compute deterministic ledger event ID
        # ID = SHA256(source_file + line_number + content_hash)
        # This ensures uniqueness across multi-file replays
        ledger_event_id = _compute_ledger_id(
            source_file=vcp_event.source_file,
            line_number=vcp_event.line_number,
            content_hash=vcp_event.content_hash,
        )

        return LedgerEvent(
            ledger_event_id=ledger_event_id,
            shard_key=shard_key,
            event_class=event_class,
            timestamp_ns=timestamp_ns,
            ingested_at_ns=time.time_ns(),
            source_file=vcp_event.source_file,
            source_line=vcp_event.line_number,
            source_byte_offset=vcp_event.byte_offset,
            original_event_hash=crypto.event_hash,
            original_prev_hash=crypto.prev_hash,
            original_signature=crypto.signature,
            content_hash=vcp_event.content_hash,
            payload=dict(vcp_event.raw_payload),  # Shallow copy — payload is treated as immutable
            symbol=meta.symbol,
            sequence=meta.sequence,
        )

    def normalize_stream(self, vcp_events):
        """Generator: normalize a stream of VCPEvents into LedgerEvents."""
        for vcp_event in vcp_events:
            try:
                yield self.normalize(vcp_event)
            except Exception as exc:
                logger.error(
                    "Normalization failed for line %d: %s",
                    vcp_event.line_number, exc
                )
                # Don't propagate — keep the pipeline moving


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _resolve_timestamp_ns(ts_ns: Optional[int], ts_ms: Optional[int]) -> int:
    """
    Return canonical nanosecond timestamp.
    Priority: explicit ns > ms converted to ns > wall clock (fallback for test data).
    """
    if ts_ns is not None:
        return ts_ns
    if ts_ms is not None:
        return ts_ms * 1_000_000  # ms → ns
    # Fallback: use ingestion time. Events without timestamps are ordered by arrival.
    logger.debug("No timestamp on event — using ingestion wall clock")
    return time.time_ns()


def _classify_event_type(event_type: Optional[str]) -> EventClass:
    """Map raw VCP event_type string to VFEL EventClass."""
    if not event_type:
        return EventClass.UNKNOWN
    return _VCP_TYPE_MAP.get(event_type.lower().strip(), EventClass.UNKNOWN)


def _resolve_shard_key(symbol: Optional[str], content_hash: str) -> str:
    """
    Determine shard key.
    Symbol-based sharding keeps all events for an instrument on one shard,
    which is essential for per-symbol sequence integrity.
    Falls back to first 8 chars of content hash for events without symbols.
    """
    if symbol:
        return symbol.upper().strip()
    return f"HASH_{content_hash[:8].upper()}"


def _compute_ledger_id(source_file: str, line_number: int, content_hash: str) -> str:
    """
    Deterministic ledger event ID.
    Stable across replays — same event always gets the same ID.
    """
    id_input = f"{source_file}:{line_number}:{content_hash}"
    return hashlib.sha256(id_input.encode("utf-8")).hexdigest()