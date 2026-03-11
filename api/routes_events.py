"""
api/routes_events.py

Event Routes for VFEL API — Phase 8.

Endpoints:
    GET  /events/{event_id}                  Fetch a single stored event
    GET  /events/                            List events (paginated)
    GET  /events/shard/{shard_key}           List events in a shard
    GET  /events/global/{seq}               Fetch event by global sequence number
    GET  /events/{event_id}/block           Which block contains this event

Response models use dataclass-to-dict serialization (no Pydantic required,
but types are documented for FastAPI schema generation if Pydantic available).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_router(ctx):
    """
    Build and return the events APIRouter.
    ctx is the VFELContext dependency container.
    """
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    router = APIRouter()

    def _require_store():
        if not ctx.event_store:
            raise HTTPException(status_code=503, detail="EventStore not initialized")
        return ctx.event_store

    # ── GET /events/{event_id} ─────────────────────────────────────────

    @router.get("/{event_id}")
    async def get_event(event_id: str):
        """
        Fetch a single StoredEvent by its ledger_event_id.

        Returns the full stored event including:
        - Original event payload
        - Ledger sequence numbers (global + shard)
        - Chain hash linking to previous event in shard
        - Stored event hash (Merkle leaf hash input)
        """
        store = _require_store()
        stored = store.get_by_event_id(event_id)
        if not stored:
            raise HTTPException(
                status_code=404,
                detail={
                    "type":   "event-not-found",
                    "title":  "Event Not Found",
                    "detail": f"No event with id={event_id}",
                }
            )
        return _serialize_stored_event(stored)

    # ── GET /events/ ───────────────────────────────────────────────────

    @router.get("/")
    async def list_events(
        limit:  int = Query(default=50,  ge=1, le=1000),
        offset: int = Query(default=0,   ge=0),
        shard:  Optional[str] = Query(default=None),
    ):
        """
        List stored events, paginated by global sequence number.

        Query params:
            limit  — max events to return (1-1000, default 50)
            offset — global sequence to start from (default 0)
            shard  — filter by shard_key (optional)
        """
        store = _require_store()
        total = store.total_events()

        events = []
        count = 0
        for se in store.replay_global(offset, min(offset + limit - 1, total - 1)):
            if shard and se.shard_key != shard:
                continue
            events.append(_serialize_stored_event(se))
            count += 1
            if count >= limit:
                break

        return {
            "total":  total,
            "offset": offset,
            "limit":  limit,
            "shard":  shard,
            "count":  len(events),
            "events": events,
        }

    # ── GET /events/shard/{shard_key} ──────────────────────────────────

    @router.get("/shard/{shard_key}")
    async def list_shard_events(
        shard_key: str,
        limit:  int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0,  ge=0),
    ):
        """
        List events within a specific shard, ordered by shard_sequence.

        More efficient than global listing when querying a single symbol's events.
        """
        store = _require_store()
        shards = store.known_shards()
        if shard_key not in shards:
            raise HTTPException(
                status_code=404,
                detail=f"Shard not found: {shard_key}. Known shards: {sorted(shards)}"
            )

        shard_total = store.shard_event_count(shard_key)
        events = []
        end_seq = min(offset + limit - 1, shard_total - 1)

        for se in store.replay_shard(shard_key, offset, end_seq):
            events.append(_serialize_stored_event(se))

        return {
            "shard_key":   shard_key,
            "shard_total": shard_total,
            "offset":      offset,
            "limit":       limit,
            "count":       len(events),
            "events":      events,
        }

    # ── GET /events/global/{seq} ───────────────────────────────────────

    @router.get("/global/{seq}")
    async def get_event_by_global_seq(seq: int):
        """Fetch a single event by its global ledger sequence number."""
        store = _require_store()
        events = list(store.replay_global(seq, seq))
        if not events:
            raise HTTPException(
                status_code=404,
                detail=f"No event at global sequence {seq}"
            )
        return _serialize_stored_event(events[0])

    # ── GET /events/{event_id}/block ───────────────────────────────────

    @router.get("/{event_id}/block")
    async def get_event_block(event_id: str):
        """
        Find which DAG block contains this event.

        Returns block metadata including height, hash, parent count,
        and the event's position within the block's event range.
        """
        store = _require_store()
        stored = store.get_by_event_id(event_id)
        if not stored:
            raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

        if not ctx.block_store:
            return {"event_id": event_id, "block": None, "reason": "BlockStore not initialized"}

        gs = stored.ledger_sequence
        for bh, blk in ctx.block_store.all_blocks().items():
            er = blk.header.event_range
            if er.global_seq_start <= gs < er.global_seq_end:
                pos_in_block = gs - er.global_seq_start
                return {
                    "event_id":      event_id,
                    "global_seq":    gs,
                    "block_hash":    bh,
                    "block_height":  blk.height,
                    "pos_in_block":  pos_in_block,
                    "block_event_count": er.event_count,
                    "forest_root":   blk.header.forest_root,
                    "block_status":  blk.header.status.value,
                }

        return {"event_id": event_id, "block": "PENDING", "global_seq": gs}

    # ── GET /events/shards ─────────────────────────────────────────────

    @router.get("/meta/shards")
    async def list_shards():
        """List all known shards with event counts and Merkle roots."""
        store = _require_store()
        shards = sorted(store.known_shards())
        result = []
        for sk in shards:
            entry: dict = {
                "shard_key":   sk,
                "event_count": store.shard_event_count(sk),
            }
            if ctx.forest_manager:
                tree = ctx.forest_manager.get_tree(sk)
                if tree:
                    entry["merkle_root"]  = tree.batch_root()
                    entry["leaf_count"]   = tree.leaf_count
            result.append(entry)
        return {"shards": result, "total_shards": len(result)}

    return router


# ──────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────

def _serialize_stored_event(se) -> dict:
    """Serialize a StoredEvent to a JSON-safe dict."""
    return {
        "ledger_sequence":    se.ledger_sequence,
        "shard_sequence":     se.shard_sequence,
        "shard_key":          se.shard_key,
        "stored_event_hash":  se.stored_event_hash,
        "prev_ledger_hash":   se.prev_ledger_hash,
        "stored_at_ns":       se.stored_at_ns,
        "event": {
            "ledger_event_id": se.event.ledger_event_id,
            "content_hash":    se.event.content_hash,
            "event_class":     se.event.event_class.value,
            "symbol":          se.event.symbol,
            "timestamp_ns":    se.event.timestamp_ns,
            "ingested_at_ns":  se.event.ingested_at_ns,
            "source_file":     se.event.source_file,
            "source_line":     se.event.source_line,
            "payload":         se.event.raw_payload,
        }
    }