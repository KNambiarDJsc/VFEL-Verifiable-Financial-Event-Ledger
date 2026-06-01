"""
benchmarks/feasibility.py — VFEL feasibility microbenchmark (Table III).

Measures the wall-clock latency (mean ± std, μs) and throughput (ops/s) for
every primitive in the VFEL pipeline on this machine. All operations are
purely in-process (no network, no disk).

Operations benchmarked
──────────────────────
  1. SHA-256 leaf hash          (merkle_math.hash_leaf)
  2. Merkle node hash           (merkle_math.hash_node)
  3. Incremental tree append    (IncrementalMerkleTree.append_raw_hash)
  4. Merkle root seal           (IncrementalMerkleTree.root)
  5. Proof path construction    (merkle_math.compute_proof_path, n=1024)
  6. Proof verification         (merkle_math.verify_proof)
  7. Canonical JSON encode      (crypto.canonical_json.canonical_encode)
  8. Ed25519 key generation     (crypto.signature_engine.KeyPair.generate)
  9. Ed25519 sign               (SignatureEngine.sign_dict)
 10. Ed25519 verify             (SignatureEngine.verify_dict)
 11. Intent commitment (H_I)    (SemanticIntent.intent_hash)
 12. Full intent-action bind    (IntentBinder.bind)
 13. Full intent-action verify  (IntentBinder.verify)
 14. EGE credit observe         (ExecutionCreditController.observe)

Run:
  python -m benchmarks.feasibility
  python -m benchmarks.feasibility --reps 5000 --warmup 500 --tree-size 4096
"""

from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import time
from typing import Callable

# ── VFEL imports ──────────────────────────────────────────────────────────────
from crypto.canonical_json import canonical_encode
from crypto.signature_engine import SignatureEngine, KeyPair
from governance.ege import ExecutionCreditController
from governance.intent_binding import (
    IntentBinder, SemanticIntent, FinancialAction,
)
from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree
from merkle_forest.merkle_math import (
    hash_leaf, hash_node, compute_proof_path, compute_root, verify_proof,
)

# ── Timing helpers ────────────────────────────────────────────────────────────

def _bench(fn: Callable, reps: int, warmup: int) -> tuple[float, float]:
    """
    Returns (mean_us, stdev_us) over `reps` timed repetitions after `warmup`
    throw-away calls. Uses time.perf_counter_ns for sub-microsecond resolution.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1_000.0)  # → μs
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def _fmt(mean_us: float, stdev_us: float) -> str:
    if mean_us >= 1_000:
        return f"{mean_us/1000:8.3f} ms +/- {stdev_us/1000:.3f} ms"
    return f"{mean_us:8.3f} us +/- {stdev_us:.3f} us"


def _throughput(mean_us: float) -> str:
    if mean_us == 0:
        return "∞"
    ops = 1_000_000 / mean_us
    if ops >= 1_000_000:
        return f"{ops/1_000_000:.2f} M ops/s"
    if ops >= 1_000:
        return f"{ops/1_000:.1f} K ops/s"
    return f"{ops:.1f} ops/s"


# ── Benchmark suite ───────────────────────────────────────────────────────────

def run(reps: int = 2000, warmup: int = 200, tree_size: int = 1024) -> None:
    rng = random.Random(42)
    engine = SignatureEngine()
    kp = KeyPair.generate("bench-agent")

    # Pre-build a populated tree for proof benchmarks
    leaves: list[str] = [
        hashlib.sha256(f"leaf-{i}".encode()).hexdigest()
        for i in range(tree_size)
    ]
    tree = IncrementalMerkleTree()
    for lh in leaves:
        tree.append_raw_hash(lh)
    proof_idx = tree_size // 3
    proof = compute_proof_path(leaves, proof_idx)
    root = tree.root()
    leaf_hash_val = leaves[proof_idx]

    # Canonical JSON payload
    sample_dict = {
        "agent_id": "bench-agent",
        "action_type": "ORDER",
        "payload": {"symbol": "AAPL", "side": "BUY", "qty": 100, "px": 190.0},
        "timestamp_ns": 1_000_000_000,
    }

    # Intent / action pair for binding benchmarks
    intent = SemanticIntent(
        agent_id="bench-agent",
        reasoning_trace="spread favourable; size within risk limit; all checks pass",
        policy_tag="liquidity_v1",
        timestamp_ns=1_000_000_000,
    )
    action = FinancialAction(
        agent_id="bench-agent",
        action_type="ORDER",
        payload={"symbol": "AAPL", "side": "BUY", "qty": 100, "px": 190.0},
        intent_ref=intent.intent_hash,
        timestamp_ns=1_000_000_001,
    )
    binder = IntentBinder(engine)
    binding = binder.bind(intent, action, kp)

    # EGE controller (pre-warmed through baseline)
    ege = ExecutionCreditController(baseline_samples=48, lam=4.0, floor_ratio=0.25)
    _t = 1_000_000_000
    for _ in range(60):
        _t += rng.randint(800_000, 1_200_000)
        ege.observe("bench-agent", _t)

    # ── Table ─────────────────────────────────────────────────────────────────
    ROWS: list[tuple[str, Callable]] = [
        ("SHA-256 leaf hash",
         lambda: hash_leaf(b"benchmark-payload-data-32-bytes!")),

        ("Merkle node hash",
         lambda: hash_node(
             "a" * 64,
             "b" * 64,
         )),

        (f"Incremental tree append (n={tree_size})",
         lambda: IncrementalMerkleTree().append_raw_hash("a" * 64)),

        (f"Merkle root seal (n={tree_size})",
         lambda: tree.root()),

        (f"Proof path construct (n={tree_size})",
         lambda: compute_proof_path(leaves, proof_idx)),

        (f"Proof verify (depth={len(proof)})",
         lambda: verify_proof(leaf_hash_val, proof, root)),

        ("Canonical JSON encode",
         lambda: canonical_encode(sample_dict)),

        ("Ed25519 key generation",
         lambda: KeyPair.generate("bench-keygen")),

        ("Ed25519 sign",
         lambda: engine.sign_dict({"H_I": "a" * 64, "H_A": "b" * 64}, kp)),

        ("Ed25519 verify",
         lambda: engine.verify_dict(
             {"H_I": "a" * 64, "H_A": "b" * 64},
             binding.signature_hex, kp.public_key_hex)),

        ("Intent commitment H_I",
         lambda: intent.intent_hash),

        ("Intent-action bind (full)",
         lambda: binder.bind(intent, action, kp)),

        ("Intent-action verify (full)",
         lambda: binder.verify(intent, action, binding, kp.public_key_hex)),

        ("EGE credit observe",
         lambda: ege.observe("bench-agent", _t + rng.randint(800_000, 1_200_000))),
    ]

    col_w = max(len(r[0]) for r in ROWS) + 2
    header = (
        f"\n{'Operation':<{col_w}}  {'Latency (mean +/- stdev)':>24}  "
        f"{'Throughput':>16}  {'Reps':>6}"
    )
    sep = "-" * len(header.strip())

    print()
    print("VFEL Feasibility Microbenchmark — Table III")
    print(f"  Python {__import__('sys').version.split()[0]}  |  "
          f"reps={reps}  warmup={warmup}  tree_size={tree_size}")
    print(sep)
    print(header)
    print(sep)
    for label, fn in ROWS:
        mean_us, stdev_us = _bench(fn, reps, warmup)
        print(f"{label:<{col_w}}  {_fmt(mean_us, stdev_us):>24}  "
              f"{_throughput(mean_us):>16}  {reps:>6,}")
    print(sep)
    print()
    print("Notes:")
    print("  - All operations are purely in-process (no network / disk I/O).")
    print("  - Ed25519 via Python cryptography package (cffi + libsodium backend).")
    print(f"  - Proof depth = ceil(log2({tree_size})) = {len(proof)} hashes.")
    print("  - EGE observe includes entropy computation over a sliding window.")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VFEL feasibility microbenchmark")
    p.add_argument("--reps",      type=int, default=2000, help="timed repetitions")
    p.add_argument("--warmup",    type=int, default=200,  help="warm-up iterations")
    p.add_argument("--tree-size", type=int, default=1024, help="leaf count for tree ops")
    args = p.parse_args()
    run(reps=args.reps, warmup=args.warmup, tree_size=args.tree_size)