"""
data_ingestion/event_parser.py

VCP Event Parser for VFEL — Verifiable Financial Event Ledger.

Responsibilities:
- Parse RawEvent dicts into structured VCPEvent objects
- Extract cryptographic fields (hashes, signatures, chains)
- Validate required fields without being brittle on optional ones
- Expose parse failures with enough context to debug the source dataset

Design: Pydantic-free in this layer — we want zero-cost parsing at ingestion
throughput. Validation is intentionally lenient (warn, don't reject) because
historical JSONL datasets may have schema drift across versions.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from data_ingestion.jsonl_loader import RawEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# VCP Event Schema (parsed representation)
# ──────────────────────────────────────────────

@dataclass
class VCPEventMeta:
    """
    Metadata extracted from a VCP event.
    These fields drive shard routing, sequencing, and integrity checks.
    """
    event_id: Optional[str]         # VCP: event_id or id
    event_type: Optional[str]       # e.g. TRADE, QUOTE, CANCEL
    symbol: Optional[str]           # Ticker / instrument — primary shard key
    timestamp_ns: Optional[int]     # Nanosecond epoch timestamp preferred
    timestamp_ms: Optional[int]     # Fallback millisecond timestamp
    sequence: Optional[int]         # VCP chain sequence number


@dataclass
class VCPCryptoFields:
    """
    Cryptographic fields from the VCP event format.
    Used for integrity verification during replay.
    """
    event_hash: Optional[str]       # SHA256 or BLAKE3 hash of event payload
    prev_hash: Optional[str]        # Chain linkage — hash of previous event
    signature: Optional[str]        # Ed25519 or similar signature
    public_key_id: Optional[str]    # Key ID reference


@dataclass
class VCPEvent:
    """
    Fully parsed VCP-format event.
    Preserves the raw payload for downstream normalization.
    Carries parse provenance for debugging.
    """
    # Source provenance
    source_file: str
    line_number: int
    byte_offset: int

    # Parsed fields
    meta: VCPEventMeta
    crypto: VCPCryptoFields

    # Raw payload preserved for normalization / hashing
    raw_payload: dict[str, Any]

    # Computed on parse: SHA256 of the raw JSON bytes for content-addressing
    content_hash: str = field(default="")

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = _hash_dict(self.raw_payload)


# ──────────────────────────────────────────────
# Parse result envelope
# ──────────────────────────────────────────────

@dataclass
class ParseResult:
    """
    Result of attempting to parse a RawEvent.
    Always returned — never raises. Failures carry error context.
    """
    success: bool
    event: Optional[VCPEvent] = None
    error: Optional[str] = None
    source_line: int = 0


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────

class EventParser:
    """
    Parses RawEvent objects into VCPEvent structures.

    Lenient by design:
    - Missing optional fields → None, not failure
    - Missing critical fields (no timestamp, no symbol) → logged warning, still parsed
    - Completely unparseable structure → ParseResult(success=False)

    Call parse() per event. Call parse_stream() for generator-based pipeline use.
    """

    # Field name aliases — VCP datasets may use different key names across versions
    _EVENT_ID_KEYS = ("event_id", "id", "eventId", "eid")
    _EVENT_TYPE_KEYS = ("event_type", "type", "eventType", "action")
    _SYMBOL_KEYS = ("symbol", "ticker", "instrument", "sym")
    _TS_NS_KEYS = ("timestamp_ns", "ts_ns", "timestampNs", "nano_ts")
    _TS_MS_KEYS = ("timestamp_ms", "ts_ms", "timestamp", "ts", "timestampMs")
    _SEQ_KEYS = ("sequence", "seq", "seqno", "seq_num")
    _HASH_KEYS = ("event_hash", "hash", "eventHash", "payload_hash")
    _PREV_HASH_KEYS = ("prev_hash", "prevHash", "previous_hash", "chain_prev")
    _SIG_KEYS = ("signature", "sig", "sign")
    _KEY_ID_KEYS = ("public_key_id", "key_id", "keyId", "kid")

    def parse(self, raw: RawEvent) -> ParseResult:
        """Parse a single RawEvent into a VCPEvent."""
        try:
            data = raw.data

            meta = VCPEventMeta(
                event_id=_extract(data, self._EVENT_ID_KEYS),
                event_type=_extract(data, self._EVENT_TYPE_KEYS),
                symbol=_extract(data, self._SYMBOL_KEYS),
                timestamp_ns=_extract_int(data, self._TS_NS_KEYS),
                timestamp_ms=_extract_int(data, self._TS_MS_KEYS),
                sequence=_extract_int(data, self._SEQ_KEYS),
            )

            crypto = VCPCryptoFields(
                event_hash=_extract(data, self._HASH_KEYS),
                prev_hash=_extract(data, self._PREV_HASH_KEYS),
                signature=_extract(data, self._SIG_KEYS),
                public_key_id=_extract(data, self._KEY_ID_KEYS),
            )

            # Warn on missing high-value fields — not a hard failure
            if not meta.symbol:
                logger.debug("Line %d: no symbol found — shard routing will use content hash", raw.line_number)
            if not meta.timestamp_ns and not meta.timestamp_ms:
                logger.debug("Line %d: no timestamp found — ordering may be unreliable", raw.line_number)

            event = VCPEvent(
                source_file=raw.source_file,
                line_number=raw.line_number,
                byte_offset=raw.byte_offset,
                meta=meta,
                crypto=crypto,
                raw_payload=data,
            )

            return ParseResult(success=True, event=event, source_line=raw.line_number)

        except Exception as exc:
            logger.error("Parse failure at line %d: %s", raw.line_number, exc)
            return ParseResult(success=False, error=str(exc), source_line=raw.line_number)

    def parse_stream(self, raw_events):
        """
        Generator: parse a stream of RawEvents.
        Yields only successful ParseResults by default.
        Failed parses are logged but not propagated.
        """
        for raw in raw_events:
            result = self.parse(raw)
            if result.success:
                yield result.event
            else:
                logger.warning("Dropped event at line %d: %s", result.source_line, result.error)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract(data: dict, keys: tuple) -> Optional[str]:
    """Try multiple field name aliases, return first match as string."""
    for key in keys:
        val = data.get(key)
        if val is not None:
            return str(val)
    return None


def _extract_int(data: dict, keys: tuple) -> Optional[int]:
    """Try multiple field name aliases, return first match as int."""
    for key in keys:
        val = data.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return None


def _hash_dict(data: dict) -> str:
    """
    Deterministic SHA256 of a dict via sorted-key JSON encoding.
    Used as content address for the raw payload.
    NOTE: Phase 5 (crypto layer) will replace this with canonical JSON + BLAKE3.
    """
    import json
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()