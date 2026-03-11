"""
api/routes_verification.py

Proof Verification Routes for VFEL API — Phase 8.

These endpoints are STATELESS — they verify proofs from their JSON
representation without querying the ledger. An external auditor can
run these against any VFEL node without needing ledger access.

Endpoints:
    POST /verify/inclusion          Verify an inclusion proof JSON
    POST /verify/ordering           Verify an ordering proof JSON
    POST /verify/latency/sla        Verify event meets latency SLA
    POST /verify/block              Verify a block's self-consistency
    GET  /verify/chain/{shard_key}  Verify the hash chain of a shard
    GET  /verify/all                Full ledger integrity scan
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_router(ctx):
    try:
        from fastapi import APIRouter, HTTPException, Body, Query
    except ImportError:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    router = APIRouter()

    # ── POST /verify/inclusion ────────────────────────────────────────

    @router.post("/inclusion")
    async def verify_inclusion(proof_dict: dict = Body(...)):
        """
        Verify an inclusion proof.

        Input: the JSON dict returned by GET /proofs/inclusion/{event_id}
        Output: {valid, checks, failed_check, error}

        This endpoint is STATELESS — it only uses the proof itself.
        No ledger access required. Safe to call from external auditors.

        Checks performed:
        1. leaf_hash = hash_leaf(stored_event_hash)
        2. Merkle path walk produces claimed shard_root
        3. Proof version is recognized
        """
        try:
            from proof_system.inclusion_proof import InclusionProof, InclusionProofVerifier
            proof = InclusionProof.from_dict(proof_dict)
            verifier = InclusionProofVerifier()
            result = verifier.verify(proof)
            return {
                "valid":        result.valid,
                "checks":       result.checks,
                "failed_check": result.failed_check,
                "error":        result.error,
                "summary":      result.summary(),
            }
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"Invalid proof format: missing field {e}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Proof parse error: {e}")

    # ── POST /verify/ordering ─────────────────────────────────────────

    @router.post("/ordering")
    async def verify_ordering(proof_dict: dict = Body(...)):
        """
        Verify an ordering proof.

        Input: the JSON dict returned by GET /proofs/ordering/{a}/{b}
        Output: {valid, reason}

        Verification logic:
        - intra_shard: checks shard_sequences are consistent with claimed order
        - inter_shard: checks global_sequences are consistent with claimed order
        """
        try:
            from proof_system.proofs import OrderingProof, OrderingProofVerifier
            proof = OrderingProof(
                event_a_id=proof_dict["event_a_id"],
                event_b_id=proof_dict["event_b_id"],
                a_before_b=proof_dict["a_before_b"],
                proof_type=proof_dict["ordering_basis"],
                shard_key=proof_dict.get("shard_key"),
                a_shard_seq=proof_dict.get("a_shard_seq"),
                b_shard_seq=proof_dict.get("b_shard_seq"),
                a_global_seq=proof_dict.get("a_global_seq"),
                b_global_seq=proof_dict.get("b_global_seq"),
            )
            verifier = OrderingProofVerifier()
            valid, reason = verifier.verify(proof)
            return {
                "valid":      valid,
                "reason":     reason,
                "a_before_b": proof.a_before_b,
                "proof_type": proof.proof_type,
            }
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"Invalid proof format: missing {e}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Proof parse error: {e}")

    # ── POST /verify/consistency ──────────────────────────────────────

    @router.post("/consistency")
    async def verify_consistency(proof_dict: dict = Body(...)):
        """
        Verify a consistency (append-only extension) proof.

        Proves that the new ledger state is a valid append-only extension
        of the old ledger state — no events were removed or altered.
        """
        try:
            from proof_system.proofs import ConsistencyProof, ConsistencyProofVerifier
            proof = ConsistencyProof.from_dict(proof_dict)
            verifier = ConsistencyProofVerifier()
            valid = verifier.verify(proof)
            return {
                "valid":           valid,
                "shard_key":       proof.shard_key,
                "old_leaf_count":  proof.old_leaf_count,
                "new_leaf_count":  proof.new_leaf_count,
                "old_root":        proof.old_root,
                "new_root":        proof.new_root,
                "summary": (
                    "✓ Ledger is append-only extension" if valid
                    else "✗ Consistency check failed — possible tampering"
                ),
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Proof parse error: {e}")

    # ── POST /verify/latency/sla ──────────────────────────────────────

    @router.post("/latency/sla")
    async def verify_latency_sla(
        proof_dict: dict = Body(...),
        max_ingestion_ms: float = Query(default=100.0, ge=0),
        max_total_ms: float = Query(default=None),
    ):
        """
        Verify that an event's latency meets an SLA threshold.

        Query params:
            max_ingestion_ms  — maximum allowed ingestion latency (default 100ms)
            max_total_ms      — maximum total latency (optional)

        Returns:
            {within_sla, ingestion_latency_ms, total_latency_ms, message}
        """
        try:
            from proof_system.proofs import LatencyProof, LatencyProofVerifier
            proof = LatencyProof(
                ledger_event_id=proof_dict["ledger_event_id"],
                event_timestamp_ns=proof_dict["event_timestamp_ns"],
                ingested_at_ns=proof_dict["ingested_at_ns"],
                stored_at_ns=proof_dict["stored_at_ns"],
                block_hash=proof_dict.get("block_hash"),
                block_sealed_at_ns=proof_dict.get("block_sealed_at_ns"),
            )
            verifier = LatencyProofVerifier()
            within_sla, msg = verifier.verify_sla(
                proof,
                max_ingestion_ms=max_ingestion_ms,
                max_total_ms=max_total_ms,
            )
            return {
                "within_sla":          within_sla,
                "ingestion_latency_ms": proof.ingestion_latency_ms,
                "total_latency_ms":    proof.total_latency_ms,
                "max_ingestion_ms":    max_ingestion_ms,
                "max_total_ms":        max_total_ms,
                "message":             msg,
            }
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"Invalid proof format: missing {e}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Proof parse error: {e}")

    # ── POST /verify/block ────────────────────────────────────────────

    @router.post("/block")
    async def verify_block(block_dict: dict = Body(...)):
        """
        Verify a block's self-consistency.

        Recomputes the block hash from its header fields and
        checks it matches the stored block_hash.
        """
        try:
            from block_dag.block_model import Block
            block = Block.from_dict(block_dict)
            valid, error = block.verify_self()
            return {
                "valid":       valid,
                "block_hash":  block.block_hash,
                "block_height": block.height,
                "error":       error,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Block parse error: {e}")

    # ── GET /verify/chain/{shard_key} ─────────────────────────────────

    @router.get("/chain/{shard_key}")
    async def verify_shard_chain(shard_key: str):
        """
        Verify the cryptographic hash chain of a shard.

        Each StoredEvent in the shard references the hash of the
        previous event via prev_ledger_hash, forming an immutable chain.
        Any tampering breaks the chain.

        This is O(n) in the number of events in the shard.
        """
        if not ctx.event_store:
            raise HTTPException(status_code=503, detail="EventStore not initialized")

        ok, error = ctx.event_store.verify_shard_chain(shard_key)
        count = ctx.event_store.shard_event_count(shard_key)
        return {
            "shard_key":   shard_key,
            "valid":       ok,
            "event_count": count,
            "error":       error,
            "summary": (
                f"✓ Chain valid ({count} events)" if ok
                else f"✗ Chain broken: {error}"
            ),
        }

    # ── GET /verify/all ────────────────────────────────────────────────

    @router.get("/all")
    async def verify_all():
        """
        Full ledger integrity scan.

        Runs all available integrity checks:
        1. Hash chain verification for all shards
        2. Block hash self-consistency for all blocks
        3. DAG total order validity

        This is O(n) in total events + O(V+E) in DAG size.
        May be slow on large ledgers — use with care in production.
        """
        results: dict = {}

        # Shard chain verification
        if ctx.event_store:
            chain_results = ctx.event_store.verify_all_chains()
            chain_ok = all(ok for ok, _ in chain_results.values())
            results["shard_chains"] = {
                "valid":   chain_ok,
                "shards":  len(chain_results),
                "details": {
                    sk: {"valid": ok, "error": err}
                    for sk, (ok, err) in chain_results.items()
                },
            }

        # Block integrity
        if ctx.block_store:
            block_results = ctx.block_store.verify_all()
            blocks_ok = all(ok for ok, _ in block_results.values())
            results["blocks"] = {
                "valid":       blocks_ok,
                "total_blocks": len(block_results),
                "failed": [
                    {"block_hash": bh[:16] + "...", "error": err}
                    for bh, (ok, err) in block_results.items()
                    if not ok
                ],
            }

        # DAG ordering
        if ctx.block_store and ctx.dag_builder:
            from block_dag.ordering_algorithm import DeterministicOrdering
            ordering = DeterministicOrdering()
            all_blocks = ctx.block_store.all_blocks()
            order_result = ordering.compute(all_blocks)
            order_ok, order_err = ordering.verify_order(
                order_result.ordered_hashes, all_blocks
            )
            results["dag_ordering"] = {
                "valid":          order_ok and not order_result.cycle_detected,
                "cycle_detected": order_result.cycle_detected,
                "blocks_ordered": order_result.total_blocks,
                "error":          order_err or order_result.cycle_info,
            }

        all_valid = all(
            v.get("valid", False) for v in results.values()
            if isinstance(v, dict)
        )
        results["overall"] = {
            "valid": all_valid,
            "summary": "✓ All checks passed" if all_valid else "✗ Integrity issues detected",
        }
        return results

    return router