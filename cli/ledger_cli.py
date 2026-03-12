"""
cli/ledger_cli.py

VFEL Ledger CLI — Phase 9.

Commands:
    vfel-ledger ingest <file.jsonl> [--shards N] [--block-size N]
    vfel-ledger status
    vfel-ledger replay --from SEQ --to SEQ [--shard KEY]
    vfel-ledger blocks [--limit N]
    vfel-ledger forest-root

Usage:
    python -m cli.ledger_cli ingest data/vcp_rta_events.jsonl
    python -m cli.ledger_cli status
    python -m cli.ledger_cli blocks --limit 5

Can also be used as a library:
    from cli.ledger_cli import LedgerCLI
    cli = LedgerCLI.from_data_dir("./ledger_data")
    cli.run_ingest("events.jsonl")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _build_stack(data_dir: str = None):
    """Build the full VFEL stack. Returns (store, forest, block_store, builder)."""
    from ledger_core.event_store import EventStore
    from ledger_core.sequence_manager import SequenceManager
    from merkle_forest.forest_manager import ForestManager
    from merkle_forest.tree_snapshot import SnapshotStore
    from block_dag.block_store import BlockStore
    from block_dag.dag_builder import DAGBuilder

    seq_mgr   = SequenceManager()
    store     = EventStore(sequence_manager=seq_mgr)
    forest    = ForestManager(snapshot_store=SnapshotStore())
    blk_store = BlockStore(store_dir=f"{data_dir}/blocks" if data_dir else None)
    builder   = DAGBuilder(store, forest, blk_store, events_per_block=100)
    builder.initialize_genesis()
    return store, forest, blk_store, builder


# ──────────────────────────────────────────────
# Command implementations
# ──────────────────────────────────────────────

def cmd_ingest(args):
    """Ingest a JSONL event file into the ledger."""
    from data_ingestion.pipeline import IngestionPipeline

    if not os.path.exists(args.file):
        print(f"✗ File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"▶ Ingesting: {args.file}")
    print(f"  Shards: {args.shards} | Block size: {args.block_size}")

    store, forest, blk_store, builder = _build_stack(args.data_dir)
    pipeline = IngestionPipeline(file_paths=args.file, num_shards=args.shards)

    t0 = time.perf_counter()
    events_stored  = 0
    events_skipped = 0
    blocks_produced = []

    for se in pipeline.run():
        ar = store.append(se.event)
        if not ar.success:
            events_skipped += 1
            continue
        events_stored += 1
        forest.append(ar.stored_event.shard_key, ar.stored_event.stored_event_hash)
        result = builder.feed(ar.stored_event)
        if result:
            blocks_produced.append(result)
            if args.verbose:
                print(f"  Block sealed: h={result.block.height} "
                      f"events={result.event_count} "
                      f"trigger={result.trigger}")

    final = builder.flush()
    if final:
        blocks_produced.append(final)

    elapsed = time.perf_counter() - t0
    throughput = events_stored / elapsed if elapsed > 0 else 0

    print(f"\n✓ Ingest complete in {elapsed:.2f}s")
    print(f"  Events stored  : {events_stored:,}")
    print(f"  Events skipped : {events_skipped}")
    print(f"  Blocks produced: {len(blocks_produced)}")
    print(f"  Throughput     : {throughput:,.0f} events/sec")
    print(f"  Forest root    : {forest.forest_root()[:32]}...")
    print(f"  Shards         : {sorted(store.known_shards())}")


def cmd_status(args):
    """Show current ledger status."""
    store, forest, blk_store, builder = _build_stack(args.data_dir)

    print("VFEL Ledger Status")
    print("=" * 50)

    # Events
    total = store.total_events()
    shards = sorted(store.known_shards())
    print(f"\nEvents")
    print(f"  Total          : {total:,}")
    print(f"  Shards ({len(shards):2d})    : {', '.join(shards[:8])}"
          + (" ..." if len(shards) > 8 else ""))

    # Blocks
    dag = blk_store.describe()
    print(f"\nBlocks")
    print(f"  Total          : {dag['total_blocks']}")
    print(f"  Max height     : {dag['max_height']}")
    print(f"  Tips           : {dag['num_tips']}")

    # Forest
    print(f"\nMerkle Forest")
    print(f"  Forest root    : {forest.forest_root()}")

    # Per-shard stats
    if shards and args.verbose:
        print(f"\nPer-Shard Stats")
        for sk in shards:
            count = store.shard_event_count(sk)
            tree  = forest.get_tree(sk)
            root  = tree.batch_root()[:20] + "..." if tree else "N/A"
            print(f"  {sk:20s}: {count:5d} events | root={root}")


def cmd_replay(args):
    """Replay and display events from the ledger."""
    store, forest, blk_store, builder = _build_stack(args.data_dir)

    total = store.total_events()
    if total == 0:
        print("No events in ledger.")
        return

    start = args.from_seq
    end   = args.to_seq if args.to_seq is not None else min(start + 9, total - 1)

    print(f"Replaying events {start} → {end} "
          + (f"(shard={args.shard})" if args.shard else "(all shards)"))
    print("-" * 60)

    count = 0
    for se in store.replay_global(start, end):
        if args.shard and se.shard_key != args.shard:
            continue
        if args.json:
            print(json.dumps(_event_summary(se), indent=2))
        else:
            _print_event_row(se)
        count += 1
        if count >= args.limit:
            print(f"  ... (limit {args.limit} reached)")
            break

    print(f"\n{count} events displayed.")


def cmd_blocks(args):
    """List DAG blocks."""
    store, forest, blk_store, builder = _build_stack(args.data_dir)

    ordered = builder.get_total_order()
    if not ordered:
        print("No blocks in ledger (only genesis).")
        return

    total = len(ordered)
    page = ordered[-(args.limit):]  # Most recent N blocks

    print(f"DAG Blocks (showing last {len(page)} of {total})")
    print(f"{'Height':>7}  {'Hash':>20}  {'Events':>7}  {'Parents':>8}  {'Forest Root':>22}")
    print("-" * 75)

    for bh in page:
        blk = blk_store.get(bh)
        if blk:
            h     = blk.height
            hash_ = bh[:18] + ".."
            evts  = blk.header.event_range.event_count
            pars  = len(blk.header.parent_hashes)
            root  = blk.header.forest_root[:20] + ".."
            print(f"{h:>7}  {hash_:>20}  {evts:>7}  {pars:>8}  {root:>22}")

    print(f"\nForest root: {forest.forest_root()}")


def cmd_forest_root(args):
    """Print the current forest root."""
    store, forest, blk_store, builder = _build_stack(args.data_dir)
    root = forest.forest_root()
    print(root)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _event_summary(se) -> dict:
    return {
        "global_seq":     se.ledger_sequence,
        "shard_seq":      se.shard_sequence,
        "shard_key":      se.shard_key,
        "event_id":       se.event.ledger_event_id,
        "event_class":    se.event.event_class.value,
        "symbol":         se.event.symbol,
        "timestamp_ns":   se.event.timestamp_ns,
        "stored_at_ns":   se.stored_at_ns,
        "stored_hash":    se.stored_event_hash[:20] + "...",
    }

def _print_event_row(se):
    print(
        f"  [{se.ledger_sequence:5d}] "
        f"{se.shard_key:15s} | "
        f"seq={se.shard_sequence:4d} | "
        f"class={se.event.event_class.value:8s} | "
        f"id={se.event.ledger_event_id[:16]}..."
    )


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vfel-ledger",
        description="VFEL Ledger CLI — ingest, inspect, and manage the event ledger",
    )
    parser.add_argument("--data-dir", default=None, help="Ledger data directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a JSONL event file")
    p_ingest.add_argument("file", help="Path to .jsonl file")
    p_ingest.add_argument("--shards",     type=int, default=8)
    p_ingest.add_argument("--block-size", type=int, default=100)
    p_ingest.set_defaults(func=cmd_ingest)

    # status
    p_status = sub.add_parser("status", help="Show ledger status")
    p_status.set_defaults(func=cmd_status)

    # replay
    p_replay = sub.add_parser("replay", help="Replay events")
    p_replay.add_argument("--from-seq", type=int, default=0)
    p_replay.add_argument("--to-seq",   type=int, default=None)
    p_replay.add_argument("--shard",    default=None)
    p_replay.add_argument("--limit",    type=int, default=20)
    p_replay.add_argument("--json",     action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    # blocks
    p_blocks = sub.add_parser("blocks", help="List DAG blocks")
    p_blocks.add_argument("--limit", type=int, default=10)
    p_blocks.set_defaults(func=cmd_blocks)

    # forest-root
    p_root = sub.add_parser("forest-root", help="Print current forest root")
    p_root.set_defaults(func=cmd_forest_root)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()