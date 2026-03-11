"""
api/routes_proofs.py

Proof Generation Routes for VFEL API — Phase 8.

Endpoints:
    GET  /proofs/inclusion/{event_id}              Generate inclusion proof
    GET  /proofs/ordering/{event_a_id}/{event_b_id} Generate ordering proof
    GET  /proofs/latency/{event_id}                Generate latency proof
    GET  /proofs/latency/stats                     Latency stats for a batch
    GET  /proofs/consistency/{shard_key}           Generate consistency proof

All proof responses are JSON-serializable and self-contained —
they can be saved and verified offline without access to the ledger.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_router(ctx):
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    router = APIRouter()

    # ── GET /proofs/inclusion/{event_id} ──────────────────────────────

    @router.get("/inclusion/{event_id}")
    async def get_inclusion_proof(event_id: str):
        """
        Generate a Merkle inclusion proof for an event.

        The proof cryptographically demonstrates that event `event_id`
        is included in the ledger at the current forest root.

        Proof chain:
            event_bytes → content_hash → stored_event_hash
            → Merkle leaf → shard root → block header → forest root

        The returned proof is self-contained and verifiable offline.
        POST it to /verify/inclusion to verify without re-generating.
        """
        gen = ctx.inclusion_proof_generator()
        if not gen:
            raise HTTPException(status_code=503, detail="Proof generator not available")

        proof = gen.generate(event_id)
        if not proof:
            raise HTTPException(
                status_code=404,
                detail={
                    "type":   "event-not-found",
                    "title":  "Cannot Generate Inclusion Proof",
                    "detail": f"Event not found or Merkle tree not built: {event_id}",
                }
            )
        return proof.to_dict()

    # ── GET /proofs/ordering/{event_a_id}/{event_b_id} ────────────────

    @router.get("/ordering/{event_a_id}/{event_b_id}")
    async def get_ordering_proof(event_a_id: str, event_b_id: str):
        """
        Generate a proof of ordering between two events.

        Answers: "Was event A processed before event B?"

        Proof type:
            intra_shard  — both events in same shard (uses shard_sequence)
            inter_shard  — events in different shards (uses global_sequence)
            block_level  — uses DAG block total order position

        The proof is self-contained — POST to /verify/ordering to verify.
        """
        gen = ctx.ordering_proof_generator()
        if not gen:
            raise HTTPException(status_code=503, detail="Proof generator not available")

        proof = gen.generate(event_a_id, event_b_id)
        if not proof:
            raise HTTPException(
                status_code=404,
                detail=f"One or both events not found: {event_a_id}, {event_b_id}"
            )
        return proof.to_dict()

    # ── GET /proofs/latency/{event_id} ────────────────────────────────

    @router.get("/latency/{event_id}")
    async def get_latency_proof(event_id: str):
        """
        Generate a latency proof for an event.

        Returns cryptographic evidence of:
        - ingestion_latency_ms: time from event timestamp to VFEL store
        - sealing_latency_ms: time from store to block seal
        - total_latency_ms: time from event timestamp to block seal

        Use POST /verify/latency/sla to verify against an SLA threshold.
        """
        gen = ctx.latency_proof_generator()
        if not gen:
            raise HTTPException(status_code=503, detail="Proof generator not available")

        proof = gen.generate(event_id)
        if not proof:
            raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
        return proof.to_dict()

    # ── GET /proofs/latency/stats ──────────────────────────────────────

    @router.get("/latency/stats")
    async def get_latency_stats(
        limit: int = Query(default=1000, ge=1, le=10000),
        shard: Optional[str] = Query(default=None),
    ):
        """
        Compute latency statistics across a batch of recent events.

        Returns: min, p50, p95, p99, max ingestion latency in milliseconds.
        Useful for SLA monitoring and performance dashboards.
        """
        gen = ctx.latency_proof_generator()
        if not gen or not ctx.event_store:
            raise HTTPException(status_code=503, detail="Services not available")

        event_ids = []
        total = ctx.event_store.total_events()
        start = max(0, total - limit)
        for se in ctx.event_store.replay_global(start, total - 1):
            if shard and se.shard_key != shard:
                continue
            event_ids.append(se.event.ledger_event_id)
            if len(event_ids) >= limit:
                break

        stats = gen.generate_batch_stats(event_ids)
        return {
            "shard":        shard,
            "sample_size":  len(event_ids),
            "stats_ms":     stats,
        }

    # ── GET /proofs/consistency/{shard_key} ───────────────────────────

    @router.get("/consistency/{shard_key}")
    async def get_consistency_proof(
        shard_key: str,
        old_leaf_count: int = Query(..., ge=1, description="Leaf count of old snapshot"),
    ):
        """
        Generate a consistency proof between two snapshots of a shard.

        Proves that the shard at its current state is an append-only
        extension of the state at `old_leaf_count` leaves.

        Requires that a snapshot was taken at `old_leaf_count`.
        Use GET /proofs/consistency/{shard_key}/snapshots to see available checkpoints.
        """
        if not ctx.forest_manager:
            raise HTTPException(status_code=503, detail="ForestManager not available")

        tree = ctx.forest_manager.get_tree(shard_key)
        if not tree:
            raise HTTPException(status_code=404, detail=f"Shard not found: {shard_key}")

        current_count = tree.leaf_count
        if old_leaf_count >= current_count:
            raise HTTPException(
                status_code=400,
                detail=f"old_leaf_count ({old_leaf_count}) must be less than current ({current_count})"
            )

        snap_store = ctx.forest_manager._snapshot_store if hasattr(ctx.forest_manager, '_snapshot_store') else None
        if not snap_store:
            raise HTTPException(status_code=503, detail="Snapshot store not available")

        from proof_system.proofs import ConsistencyProofGenerator
        gen = ConsistencyProofGenerator(snap_store)
        proof = gen.generate(shard_key, old_leaf_count, current_count)

        if not proof:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No snapshot found at leaf_count={old_leaf_count} for shard={shard_key}. "
                    "Snapshots are taken every snapshot_interval leaves during ingestion."
                )
            )
        return proof.to_dict()

    # ── GET /proofs/consistency/{shard_key}/snapshots ─────────────────

    @router.get("/consistency/{shard_key}/snapshots")
    async def list_shard_snapshots(shard_key: str):
        """List available snapshot checkpoints for a shard."""
        snap_store = (
            ctx.forest_manager._snapshot_store
            if ctx.forest_manager and hasattr(ctx.forest_manager, '_snapshot_store')
            else None
        )
        if not snap_store:
            raise HTTPException(status_code=503, detail="Snapshot store not available")

        snaps = snap_store.list_snapshots(shard_key)
        return {
            "shard_key": shard_key,
            "snapshots": [
                {
                    "leaf_count": s.leaf_count,
                    "root_hash":  s.root_hash,
                    "trigger":    s.trigger,
                    "snapshot_id": s.snapshot_id,
                }
                for s in snaps
            ],
        }

    return router