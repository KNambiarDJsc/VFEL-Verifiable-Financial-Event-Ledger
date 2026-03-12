"""
cli/anchor_cli.py

VFEL Anchor CLI — Phase 9.

Commands:
    vfel-anchor anchor [--networks local,btc,eth]  Anchor current forest root
    vfel-anchor status                             Show all anchor records
    vfel-anchor verify <anchor_id>                 Verify a local timestamp proof
    vfel-anchor poll                               Poll pending anchors for confirmation

Usage:
    python -m cli.anchor_cli anchor
    python -m cli.anchor_cli anchor --networks local,btc
    python -m cli.anchor_cli status
    python -m cli.anchor_cli verify <anchor_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_anchor_manager(networks: list[str]) -> object:
    """Build AnchorManager with requested backends."""
    from anchoring.anchor_manager import AnchorManager
    from anchoring.timestamp_anchor import LocalTimestampAnchor
    from anchoring.bitcoin_anchor import BitcoinAnchor
    from anchoring.ethereum_anchor import EthereumAnchor

    mgr = AnchorManager()

    for net in networks:
        net = net.strip().lower()
        if net in ("local", "timestamp"):
            mgr.register_backend(LocalTimestampAnchor(node_id=f"vfel-cli-{os.getpid()}"))
        elif net in ("btc", "bitcoin"):
            mode = os.environ.get("VFEL_BTC_MODE", "simulation")
            mgr.register_backend(BitcoinAnchor(mode=mode))
        elif net in ("eth", "ethereum"):
            mode = os.environ.get("VFEL_ETH_MODE", "simulation")
            mgr.register_backend(EthereumAnchor(mode=mode))
        else:
            print(f"  Warning: Unknown network '{net}' — skipping", file=sys.stderr)

    return mgr


def _get_forest_root(data_file: str) -> str:
    """Load ledger data and return current forest root."""
    from ledger_core.event_store import EventStore
    from ledger_core.sequence_manager import SequenceManager
    from merkle_forest.forest_manager import ForestManager
    from merkle_forest.tree_snapshot import SnapshotStore
    from block_dag.block_store import BlockStore
    from block_dag.dag_builder import DAGBuilder
    from data_ingestion.pipeline import IngestionPipeline

    seq_mgr   = SequenceManager()
    store     = EventStore(sequence_manager=seq_mgr)
    forest    = ForestManager(snapshot_store=SnapshotStore())
    blk_store = BlockStore()
    builder   = DAGBuilder(store, forest, blk_store, events_per_block=100)
    builder.initialize_genesis()

    if data_file and os.path.exists(data_file):
        pipeline = IngestionPipeline(file_paths=data_file, num_shards=8)
        for se in pipeline.run():
            ar = store.append(se.event)
            if ar.success:
                forest.append(ar.stored_event.shard_key, ar.stored_event.stored_event_hash)
                builder.feed(ar.stored_event)
        builder.flush()

    return forest.forest_root(), store.total_events()


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

def cmd_anchor(args):
    """Anchor the current forest root to specified networks."""
    networks = [n.strip() for n in args.networks.split(",")]
    mgr = _build_anchor_manager(networks)

    forest_root, event_count = _get_forest_root(args.data_file)

    print(f"\nAnchoring forest root")
    print(f"  Forest root  : {forest_root}")
    print(f"  Event count  : {event_count:,}")
    print(f"  Networks     : {', '.join(networks)}")
    print()

    metadata = {
        "event_count":  event_count,
        "cli_pid":      os.getpid(),
        "anchored_by":  "vfel-anchor-cli",
    }

    records = mgr.anchor(forest_root, metadata=metadata)

    for record in records:
        status_sym = {
            "CONFIRMED": "✓",
            "PENDING":   "⏳",
            "FAILED":    "✗",
            "EXPIRED":   "✗",
        }.get(record.status.value, "?")

        print(f"  {status_sym} {record.network.value:15s} → anchor_id={record.anchor_id[:16]}...")
        if record.tx_id:
            print(f"    tx_id = {record.tx_id}")
        if record.error:
            print(f"    error = {record.error}")
        if record.proof_bytes and args.verbose:
            print(f"    proof = {record.proof_bytes[:64]}...")

    confirmed = sum(1 for r in records if r.status.value == "CONFIRMED")
    print(f"\n  {confirmed}/{len(records)} anchors confirmed")

    if args.output:
        output_data = {
            "forest_root": forest_root,
            "anchored_at": time.time_ns(),
            "records": [r.to_dict() for r in records],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Anchor records saved to: {args.output}")


def cmd_status(args):
    """Show status of anchor records from a saved file."""
    if args.records_file and os.path.exists(args.records_file):
        with open(args.records_file) as f:
            data = json.load(f)
        records_data = data.get("records", [])
        forest_root  = data.get("forest_root", "unknown")
    else:
        print("No anchor records file specified. Use --records-file <path>")
        print("Or run: vfel-anchor anchor --output anchors.json")
        return

    print(f"\nAnchor Records")
    print(f"  Forest root: {forest_root[:32]}...")
    print(f"  Records    : {len(records_data)}")
    print()
    print(f"  {'Network':15s}  {'Status':10s}  {'TX ID':35s}  {'Anchor ID'}")
    print("  " + "-" * 85)

    for rd in records_data:
        net    = rd.get("network", "?")
        status = rd.get("status", "?")
        tx     = (rd.get("tx_id") or "—")[:33]
        aid    = rd.get("anchor_id", "?")[:16] + "..."
        sym    = "✓" if status == "CONFIRMED" else ("⏳" if status == "PENDING" else "✗")
        print(f"  {net:15s}  {sym} {status:9s}  {tx:35s}  {aid}")


def cmd_verify_anchor(args):
    """Verify a local timestamp anchor proof."""
    if args.records_file and os.path.exists(args.records_file):
        with open(args.records_file) as f:
            data = json.load(f)
        records = data.get("records", [])
    else:
        print("✗ Provide --records-file <path>", file=sys.stderr); sys.exit(1)

    from anchoring.timestamp_anchor import LocalTimestampAnchor
    anchor = LocalTimestampAnchor()
    verifier = anchor.verify_proof

    found = False
    for rd in records:
        if rd.get("network") != "local_timestamp":
            continue
        proof_bytes = rd.get("proof_bytes")
        if not proof_bytes:
            continue
        found = True
        anchor_id = rd.get("anchor_id", "?")
        valid, msg = verifier(proof_bytes)
        sym = "✓" if valid else "✗"
        print(f"  {sym} anchor_id={anchor_id[:16]}... : {msg}")

    if not found:
        print("No local_timestamp anchors found in records file.")


def cmd_poll(args):
    """Poll pending anchors for confirmation status."""
    if not args.records_file or not os.path.exists(args.records_file):
        print("✗ Provide --records-file <path>", file=sys.stderr); sys.exit(1)

    from anchoring.anchor_manager import AnchorRecord, AnchorManager

    with open(args.records_file) as f:
        data = json.load(f)
    records = [AnchorRecord.from_dict(rd) for rd in data.get("records", [])]

    networks = list({r.network for r in records})
    mgr = _build_anchor_manager([n.value for n in networks])
    mgr._records = {r.anchor_id: r for r in records}

    print(f"\nPolling {len(records)} anchor records...")
    updated = mgr.poll_pending()

    if not updated:
        print("  No status changes.")
    else:
        for record in updated:
            print(f"  {record.network.value}: {record.anchor_id[:16]}... → {record.status.value}")

    # Save updated records
    data["records"] = [r.to_dict() for r in mgr.all_records()]
    with open(args.records_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Updated records saved to: {args.records_file}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vfel-anchor",
        description="VFEL Anchor CLI — anchor and verify forest roots externally",
    )
    parser.add_argument("--data-file", default="data/vcp_rta_events.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_anch = sub.add_parser("anchor", help="Anchor current forest root")
    p_anch.add_argument("--networks", default="local,btc,eth",
                        help="Comma-separated networks: local,btc,eth (default: all)")
    p_anch.add_argument("--output", "-o", help="Save anchor records to JSON file")
    p_anch.set_defaults(func=cmd_anchor)

    p_stat = sub.add_parser("status", help="Show anchor record status")
    p_stat.add_argument("--records-file", "-f")
    p_stat.set_defaults(func=cmd_status)

    p_ver = sub.add_parser("verify", help="Verify a local timestamp proof")
    p_ver.add_argument("--records-file", "-f")
    p_ver.set_defaults(func=cmd_verify_anchor)

    p_poll = sub.add_parser("poll", help="Poll pending anchors for confirmation")
    p_poll.add_argument("--records-file", "-f")
    p_poll.set_defaults(func=cmd_poll)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()