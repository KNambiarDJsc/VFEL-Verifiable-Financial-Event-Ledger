"""
cli/verify_cli.py

VFEL Verification CLI — Phase 9.

Commands:
    vfel-verify inclusion <event_id>             Generate + verify inclusion proof
    vfel-verify ordering <event_a_id> <event_b_id>  Ordering proof
    vfel-verify latency <event_id> [--sla-ms N]  Latency proof + SLA check
    vfel-verify chain [--shard KEY]              Verify hash chain(s)
    vfel-verify all                              Full integrity scan
    vfel-verify proof <proof.json>              Verify a saved proof file

Usage:
    python -m cli.verify_cli inclusion abc123def456
    python -m cli.verify_cli all --verbose
    python -m cli.verify_cli proof saved_proof.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_stack_with_data(jsonl_path: str = None):
    """Build and optionally pre-populate a VFEL stack."""
    from ledger_core.event_store import EventStore
    from ledger_core.sequence_manager import SequenceManager
    from merkle_forest.forest_manager import ForestManager
    from merkle_forest.tree_snapshot import SnapshotStore
    from block_dag.block_store import BlockStore
    from block_dag.dag_builder import DAGBuilder

    seq_mgr   = SequenceManager()
    store     = EventStore(sequence_manager=seq_mgr)
    snap_store = SnapshotStore()
    forest    = ForestManager(snapshot_store=snap_store)
    blk_store = BlockStore()
    builder   = DAGBuilder(store, forest, blk_store, events_per_block=100)
    builder.initialize_genesis()

    if jsonl_path and os.path.exists(jsonl_path):
        from data_ingestion.pipeline import IngestionPipeline
        from merkle_forest.tree_snapshot import TreeSnapshot
        import uuid

        pipeline = IngestionPipeline(file_paths=jsonl_path, num_shards=8)
        for se in pipeline.run():
            ar = store.append(se.event)
            if ar.success:
                forest.append(ar.stored_event.shard_key, ar.stored_event.stored_event_hash)
                builder.feed(ar.stored_event)
        builder.flush()

        # Snapshot all shards
        for sk in store.known_shards():
            tree = forest.get_tree(sk)
            if tree and tree.leaf_count > 0:
                snap = TreeSnapshot(
                    snapshot_id=str(uuid.uuid4()),
                    shard_key=sk, leaf_count=tree.leaf_count,
                    root_hash=tree.batch_root(), frontier=tree.frontier_snapshot(),
                    leaf_hashes=tree.get_all_leaf_hashes(), trigger="cli",
                )
                snap_store.save(snap)

    return store, forest, blk_store, builder, snap_store


def _first_event_ids(store, n: int = 2) -> list[str]:
    ids = []
    for se in store.replay_global(0, min(n * 2 - 1, store.total_events() - 1)):
        ids.append(se.event.ledger_event_id)
        if len(ids) >= n:
            break
    return ids


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

def cmd_inclusion(args):
    """Generate and verify an inclusion proof."""
    store, forest, blk_store, builder, snap_store = _build_stack_with_data(args.data_file)

    event_id = args.event_id
    if event_id == "FIRST":
        ids = _first_event_ids(store, 1)
        if not ids:
            print("✗ No events in ledger.", file=sys.stderr); sys.exit(1)
        event_id = ids[0]
        print(f"  Using first event: {event_id[:24]}...")

    from proof_system.inclusion_proof import InclusionProofGenerator, InclusionProofVerifier
    gen  = InclusionProofGenerator(store, forest, blk_store)
    verif = InclusionProofVerifier()

    t0 = time.perf_counter()
    proof = gen.generate(event_id)
    gen_ms = (time.perf_counter() - t0) * 1000

    if not proof:
        print(f"✗ Cannot generate proof for: {event_id}", file=sys.stderr); sys.exit(1)

    t1 = time.perf_counter()
    result = verif.verify(proof)
    verif_ms = (time.perf_counter() - t1) * 1000

    print(f"\nInclusion Proof")
    print(f"  Event ID     : {proof.ledger_event_id[:32]}...")
    print(f"  Shard        : {proof.shard_key}")
    print(f"  Shard seq    : {proof.shard_sequence}")
    print(f"  Global seq   : {proof.global_sequence}")
    print(f"  Merkle path  : {len(proof.merkle_path)} hashes")
    print(f"  Shard root   : {proof.shard_root[:32]}...")
    print(f"  Forest root  : {proof.forest_root[:32]}...")
    print(f"\n  Generated in : {gen_ms:.2f}ms")
    print(f"  Verified in  : {verif_ms:.2f}ms")
    print(f"\n  {result.summary()}")

    for check, ok in result.checks.items():
        print(f"    {'✓' if ok else '✗'} {check}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(proof.to_dict(), f, indent=2)
        print(f"\n  Proof saved to: {args.output}")

    sys.exit(0 if result.valid else 1)


def cmd_ordering(args):
    """Generate and verify an ordering proof."""
    store, forest, blk_store, builder, _ = _build_stack_with_data(args.data_file)

    event_a = args.event_a
    event_b = args.event_b
    if event_a == "FIRST" or event_b == "SECOND":
        ids = _first_event_ids(store, 25)
        if len(ids) < 2:
            print("✗ Need at least 2 events.", file=sys.stderr); sys.exit(1)
        event_a = ids[0] if event_a == "FIRST" else event_a
        event_b = ids[15] if event_b == "SECOND" else event_b

    from proof_system.proofs import OrderingProofGenerator, OrderingProofVerifier
    gen   = OrderingProofGenerator(store, blk_store, builder)
    verif = OrderingProofVerifier()
    proof = gen.generate(event_a, event_b)

    if not proof:
        print(f"✗ Cannot generate ordering proof", file=sys.stderr); sys.exit(1)

    valid, reason = verif.verify(proof)

    print(f"\nOrdering Proof")
    print(f"  Event A      : {proof.event_a_id[:24]}... (global_seq={proof.a_global_seq})")
    print(f"  Event B      : {proof.event_b_id[:24]}... (global_seq={proof.b_global_seq})")
    print(f"  A before B   : {'✓ YES' if proof.a_before_b else '✗ NO'}")
    print(f"  Proof type   : {proof.proof_type}")
    print(f"\n  {'✓ VALID' if valid else '✗ INVALID'} — {reason}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(proof.to_dict(), f, indent=2)
        print(f"\n  Proof saved to: {args.output}")

    sys.exit(0 if valid else 1)


def cmd_latency(args):
    """Generate a latency proof and optionally check SLA."""
    store, forest, blk_store, builder, _ = _build_stack_with_data(args.data_file)

    event_id = args.event_id
    if event_id == "FIRST":
        ids = _first_event_ids(store, 1)
        if not ids:
            print("✗ No events.", file=sys.stderr); sys.exit(1)
        event_id = ids[0]

    from proof_system.proofs import LatencyProofGenerator, LatencyProofVerifier
    gen   = LatencyProofGenerator(store, blk_store)
    verif = LatencyProofVerifier()
    proof = gen.generate(event_id)

    if not proof:
        print(f"✗ Cannot generate latency proof", file=sys.stderr); sys.exit(1)

    print(f"\nLatency Proof")
    print(f"  Event ID         : {proof.ledger_event_id[:32]}...")
    print(f"  Ingestion latency: {proof.ingestion_latency_ms:.3f}ms")
    if proof.total_latency_ms:
        print(f"  Total latency    : {proof.total_latency_ms:.3f}ms")

    if args.sla_ms:
        within, msg = verif.verify_sla(proof, max_ingestion_ms=args.sla_ms)
        print(f"\n  SLA check (≤{args.sla_ms}ms): {'✓ PASS' if within else '✗ FAIL'}")
        print(f"  {msg}")

        # Batch stats
        print(f"\nBatch latency stats (all events):")
        total = store.total_events()
        ids = [se.event.ledger_event_id
               for se in store.replay_global(0, min(999, total - 1))]
        stats = gen.generate_batch_stats(ids)
        if stats.get("count", 0) > 0:
            for k in ("count", "min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                v = stats[k]
                label = k.replace("_ms", " (ms)").replace("_", " ")
                print(f"  {label:20s}: {v:.3f}" if isinstance(v, float) else f"  {label:20s}: {v}")

    sys.exit(0)


def cmd_chain(args):
    """Verify shard hash chain(s)."""
    store, *_ = _build_stack_with_data(args.data_file)

    if args.shard:
        shards = [args.shard]
    else:
        shards = sorted(store.known_shards())

    all_ok = True
    print(f"\nHash Chain Verification ({len(shards)} shard(s))")
    print(f"{'Shard':20s}  {'Events':>7}  {'Status'}")
    print("-" * 50)

    for sk in shards:
        ok, err = store.verify_shard_chain(sk)
        count = store.shard_event_count(sk)
        status = "✓ valid" if ok else f"✗ BROKEN: {err}"
        print(f"{sk:20s}  {count:>7}  {status}")
        if not ok:
            all_ok = False

    print(f"\n{'✓ All chains valid' if all_ok else '✗ Chain integrity failures detected'}")
    sys.exit(0 if all_ok else 1)


def cmd_verify_all(args):
    """Full ledger integrity scan."""
    store, forest, blk_store, builder, _ = _build_stack_with_data(args.data_file)

    print("\nFull Ledger Integrity Scan")
    print("=" * 50)
    all_ok = True

    # Shard chains
    print("\n[1/3] Shard hash chains...")
    chain_results = store.verify_all_chains()
    chains_ok = all(ok for ok, _ in chain_results.values())
    for sk, (ok, err) in sorted(chain_results.items()):
        if not ok or args.verbose:
            print(f"  {'✓' if ok else '✗'} {sk}: {err or 'OK'}")
    print(f"  → {'✓ PASS' if chains_ok else '✗ FAIL'} ({len(chain_results)} shards)")
    all_ok = all_ok and chains_ok

    # Blocks
    print("\n[2/3] Block hash integrity...")
    block_results = blk_store.verify_all()
    blocks_ok = all(ok for ok, _ in block_results.values())
    failures = [(bh, err) for bh, (ok, err) in block_results.items() if not ok]
    for bh, err in failures:
        print(f"  ✗ {bh[:20]}: {err}")
    print(f"  → {'✓ PASS' if blocks_ok else '✗ FAIL'} ({len(block_results)} blocks)")
    all_ok = all_ok and blocks_ok

    # DAG ordering
    print("\n[3/3] DAG topological order...")
    from block_dag.ordering_algorithm import DeterministicOrdering
    ordering = DeterministicOrdering()
    all_blocks = blk_store.all_blocks()
    order_result = ordering.compute(all_blocks)
    order_ok, order_err = ordering.verify_order(order_result.ordered_hashes, all_blocks)
    dag_ok = order_ok and not order_result.cycle_detected
    print(f"  {'✓' if not order_result.cycle_detected else '✗'} No cycles detected")
    print(f"  {'✓' if order_ok else '✗'} Valid topological order{': '+order_err if order_err else ''}")
    print(f"  → {'✓ PASS' if dag_ok else '✗ FAIL'} ({order_result.total_blocks} blocks)")
    all_ok = all_ok and dag_ok

    print(f"\n{'✓ ALL CHECKS PASSED' if all_ok else '✗ INTEGRITY FAILURES DETECTED'}")
    print(f"  Events : {store.total_events():,}")
    print(f"  Blocks : {len(all_blocks)}")
    print(f"  Shards : {len(chain_results)}")
    sys.exit(0 if all_ok else 1)


def cmd_verify_proof_file(args):
    """Verify a saved proof JSON file."""
    if not os.path.exists(args.file):
        print(f"✗ File not found: {args.file}", file=sys.stderr); sys.exit(1)

    with open(args.file) as f:
        proof_dict = json.load(f)

    proof_type = proof_dict.get("proof_type", "unknown")
    print(f"Verifying proof type: {proof_type}")

    if proof_type == "inclusion":
        from proof_system.inclusion_proof import InclusionProof, InclusionProofVerifier
        proof = InclusionProof.from_dict(proof_dict)
        result = InclusionProofVerifier().verify(proof)
        print(f"\n  {result.summary()}")
        for check, ok in result.checks.items():
            print(f"    {'✓' if ok else '✗'} {check}")
        sys.exit(0 if result.valid else 1)

    elif proof_type == "ordering":
        from proof_system.proofs import OrderingProof, OrderingProofVerifier
        proof = OrderingProof(
            event_a_id=proof_dict["event_a_id"], event_b_id=proof_dict["event_b_id"],
            a_before_b=proof_dict["a_before_b"], proof_type=proof_dict["ordering_basis"],
            a_shard_seq=proof_dict.get("a_shard_seq"), b_shard_seq=proof_dict.get("b_shard_seq"),
            a_global_seq=proof_dict.get("a_global_seq"), b_global_seq=proof_dict.get("b_global_seq"),
        )
        valid, reason = OrderingProofVerifier().verify(proof)
        print(f"\n  {'✓ VALID' if valid else '✗ INVALID'} — {reason}")
        sys.exit(0 if valid else 1)

    else:
        print(f"  Unknown proof type: {proof_type}", file=sys.stderr); sys.exit(1)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vfel-verify",
        description="VFEL Verification CLI — generate and verify cryptographic proofs",
    )
    parser.add_argument("--data-file", default="data/vcp_rta_events.jsonl",
                        help="JSONL data file to load (default: data/vcp_rta_events.jsonl)")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inc = sub.add_parser("inclusion", help="Inclusion proof for an event")
    p_inc.add_argument("event_id", nargs="?", default="FIRST")
    p_inc.add_argument("--output", "-o", help="Save proof to JSON file")
    p_inc.set_defaults(func=cmd_inclusion)

    p_ord = sub.add_parser("ordering", help="Ordering proof between two events")
    p_ord.add_argument("event_a", nargs="?", default="FIRST")
    p_ord.add_argument("event_b", nargs="?", default="SECOND")
    p_ord.add_argument("--output", "-o")
    p_ord.set_defaults(func=cmd_ordering)

    p_lat = sub.add_parser("latency", help="Latency proof for an event")
    p_lat.add_argument("event_id", nargs="?", default="FIRST")
    p_lat.add_argument("--sla-ms", type=float, default=None)
    p_lat.set_defaults(func=cmd_latency)

    p_chain = sub.add_parser("chain", help="Verify shard hash chain(s)")
    p_chain.add_argument("--shard", default=None)
    p_chain.set_defaults(func=cmd_chain)

    p_all = sub.add_parser("all", help="Full integrity scan")
    p_all.set_defaults(func=cmd_verify_all)

    p_file = sub.add_parser("proof", help="Verify a saved proof file")
    p_file.add_argument("file")
    p_file.set_defaults(func=cmd_verify_proof_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()