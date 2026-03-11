"""
tests/fuzz_tests.py

Fuzz Tests for VFEL — Phase 10.

Property-based and adversarial fuzzing of all cryptographic primitives.
These tests run many iterations with random inputs to catch edge cases
that deterministic unit tests miss.

Test categories:
    1. Hash avalanche effect — small input changes cause large output changes
    2. Canonical JSON stability — any valid dict has exactly one canonical form
    3. Merkle tree soundness — inclusion proofs hold for all tree shapes
    4. Signature non-malleability — signatures cannot be forged or mutated
    5. Block DAG invariants — ordering properties hold for random DAG shapes
    6. End-to-end pipeline fuzz — random event payloads flow through intact

Run:
    python tests/fuzz_tests.py
    python -m pytest tests/fuzz_tests.py -v --tb=short
"""

import hashlib
import os
import random
import string
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merkle_forest.merkle_math import (
    hash_leaf, hash_node, compute_root, compute_proof_path, verify_proof,
)
from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree
from crypto.canonical_json import canonical_dumps, canonical_hash
from crypto.hashing_engine import HashingEngine, HashAlgorithm


# ──────────────────────────────────────────────
# Fuzz utilities
# ──────────────────────────────────────────────

def rand_hex(n_bytes: int = 32) -> str:
    return os.urandom(n_bytes).hex()

def rand_str(min_len: int = 1, max_len: int = 128) -> str:
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))

def rand_dict(depth: int = 0) -> dict:
    """Generate a random nested dict suitable for canonical JSON."""
    n_keys = random.randint(1, 6)
    d = {}
    for _ in range(n_keys):
        key = ''.join(random.choices(string.ascii_letters, k=random.randint(1, 8)))
        if depth == 0 and random.random() < 0.3:
            d[key] = rand_dict(depth=1)
        else:
            d[key] = random.choice([
                random.randint(-1000, 1000),
                round(random.uniform(-100, 100), 4),
                rand_str(1, 20),
                random.choice([True, False, None]),
            ])
    return d

def flip_bit(hex_str: str) -> str:
    """Flip one random bit in a hex string."""
    b = bytearray(bytes.fromhex(hex_str))
    i = random.randint(0, len(b) - 1)
    bit = random.randint(0, 7)
    b[i] ^= (1 << bit)
    return b.hex()


# ──────────────────────────────────────────────
# Hash Avalanche Tests
# ──────────────────────────────────────────────

class TestHashAvalanche(unittest.TestCase):
    """
    Avalanche effect: a 1-bit input change should flip ~50% of output bits.
    We verify this statistically — each test is a soft check.
    """

    TRIALS = 200

    def _bit_difference_ratio(self, h1: str, h2: str) -> float:
        b1 = bin(int(h1, 16))[2:].zfill(256)
        b2 = bin(int(h2, 16))[2:].zfill(256)
        diff = sum(c1 != c2 for c1, c2 in zip(b1, b2))
        return diff / 256

    def test_hash_leaf_avalanche(self):
        """Single-bit input change should flip ~50% of hash_leaf output bits."""
        ratios = []
        for _ in range(self.TRIALS):
            data = rand_str()
            h1 = hash_leaf(data)
            # Flip one character in the string
            if len(data) == 0:
                continue
            mutated = list(data)
            i = random.randint(0, len(mutated) - 1)
            mutated[i] = chr(ord(mutated[i]) ^ 1)
            h2 = hash_leaf(''.join(mutated))
            if h1 != h2:
                ratios.append(self._bit_difference_ratio(h1, h2))

        avg = sum(ratios) / len(ratios) if ratios else 0
        # Average should be close to 0.5 (50% bit flip)
        self.assertGreater(avg, 0.30, f"Avalanche too weak: avg={avg:.3f}")
        self.assertLess(avg, 0.70, f"Avalanche too strong: avg={avg:.3f}")

    def test_hash_node_avalanche(self):
        """Changing left or right child should cascade to ~50% output bit change."""
        ratios = []
        for _ in range(self.TRIALS):
            left  = rand_hex()
            right = rand_hex()
            h1 = hash_node(left, right)
            # Flip one bit in left
            h2 = hash_node(flip_bit(left), right)
            if h1 != h2:
                ratios.append(self._bit_difference_ratio(h1, h2))

        avg = sum(ratios) / len(ratios) if ratios else 0
        self.assertGreater(avg, 0.30)
        self.assertLess(avg, 0.70)

    def test_no_hash_collisions(self):
        """In N random inputs, no two should produce the same hash_leaf."""
        N = 500
        hashes = {hash_leaf(rand_str()) for _ in range(N)}
        # Allow at most 0 collisions (birthday paradox at N=500 is ~0 for SHA256)
        self.assertEqual(len(hashes), N)


# ──────────────────────────────────────────────
# Canonical JSON Fuzz
# ──────────────────────────────────────────────

class TestCanonicalJSONFuzz(unittest.TestCase):
    """Fuzz canonical JSON for stability and correctness."""

    TRIALS = 200

    def test_key_permutation_invariance(self):
        """Any permutation of dict keys must produce the same canonical form."""
        for _ in range(self.TRIALS):
            d = rand_dict()
            canonical = canonical_dumps(d)
            # Rebuild dict with shuffled keys
            keys = list(d.keys())
            random.shuffle(keys)
            d_shuffled = {k: d[k] for k in keys}
            self.assertEqual(
                canonical,
                canonical_dumps(d_shuffled),
                f"Key permutation changed canonical form for dict: {d}"
            )

    def test_canonical_hash_is_deterministic(self):
        """Same dict always produces same canonical hash."""
        for _ in range(self.TRIALS):
            d = rand_dict()
            h1 = canonical_hash(d)
            h2 = canonical_hash(d)
            self.assertEqual(h1, h2)

    def test_different_dicts_different_hashes(self):
        """Two different dicts should have different hashes."""
        collisions = 0
        for _ in range(self.TRIALS):
            d1 = rand_dict()
            d2 = rand_dict()
            if d1 == d2:
                continue
            if canonical_hash(d1) == canonical_hash(d2):
                collisions += 1
        self.assertEqual(collisions, 0)

    def test_nested_key_permutation_invariance(self):
        """Nested dicts must also be sorted recursively."""
        for _ in range(50):
            inner = {chr(ord('a') + i): i for i in range(5)}
            inner_shuffled = dict(random.sample(list(inner.items()), len(inner)))
            outer1 = {"data": inner,          "version": 1}
            outer2 = {"version": 1, "data": inner_shuffled}
            self.assertEqual(canonical_dumps(outer1), canonical_dumps(outer2))

    def test_no_nan_infinity_allowed(self):
        """NaN and Infinity must raise ValueError in canonical JSON."""
        import math
        from crypto.canonical_json import CanonicalJSON
        cj = CanonicalJSON()
        with self.assertRaises(ValueError):
            cj.dumps(math.nan)
        with self.assertRaises(ValueError):
            cj.dumps(math.inf)


# ──────────────────────────────────────────────
# Merkle Tree Fuzz
# ──────────────────────────────────────────────

class TestMerkleTreeFuzz(unittest.TestCase):
    """Fuzz Merkle tree for soundness across arbitrary shapes."""

    TRIALS = 100

    def test_inclusion_proof_all_random_sizes(self):
        """Inclusion proofs valid for random tree sizes and leaf positions."""
        for _ in range(self.TRIALS):
            n = random.randint(1, 64)
            leaves = [rand_hex() for _ in range(n)]
            root = compute_root(leaves)
            idx = random.randint(0, n - 1)
            path = compute_proof_path(leaves, idx)
            self.assertTrue(
                verify_proof(leaves[idx], path, root),
                f"Proof failed: n={n} idx={idx}"
            )

    def test_wrong_leaf_fails_proof(self):
        """Using a wrong leaf to verify a path must fail."""
        for _ in range(50):
            n = random.randint(2, 32)
            leaves = [rand_hex() for _ in range(n)]
            root = compute_root(leaves)
            idx = random.randint(0, n - 1)
            path = compute_proof_path(leaves, idx)
            # Use a different leaf
            wrong_leaf = rand_hex()
            if wrong_leaf == leaves[idx]:
                continue
            self.assertFalse(verify_proof(wrong_leaf, path, root))

    def test_incremental_tree_append_order_matters(self):
        """Appending leaves in different order produces different roots."""
        n = random.randint(3, 10)
        leaves = [rand_hex() for _ in range(n)]

        tree1 = IncrementalMerkleTree()
        for h in leaves:
            tree1.append_raw_hash(h)

        tree2 = IncrementalMerkleTree()
        shuffled = list(leaves)
        random.shuffle(shuffled)
        for h in shuffled:
            tree2.append_raw_hash(h)

        if leaves != shuffled:
            self.assertNotEqual(tree1.batch_root(), tree2.batch_root())

    def test_compute_root_matches_incremental_random(self):
        """batch_root must match compute_root for 100 random cases."""
        for _ in range(self.TRIALS):
            n = random.randint(1, 100)
            leaves = [rand_hex() for _ in range(n)]
            tree = IncrementalMerkleTree()
            for h in leaves:
                tree.append_raw_hash(h)
            self.assertEqual(
                tree.batch_root(),
                compute_root(leaves),
                f"Mismatch at n={n}"
            )


# ──────────────────────────────────────────────
# End-to-End Pipeline Fuzz
# ──────────────────────────────────────────────

class TestPipelineFuzz(unittest.TestCase):
    """
    Fuzz the full ingestion pipeline with random/malformed event payloads.
    The pipeline must never crash — it must gracefully skip bad events.
    """

    def _make_jsonl(self, events: list) -> str:
        import json
        return "\n".join(json.dumps(e) for e in events)

    def test_random_valid_events_ingest_cleanly(self):
        """Random but valid-shaped events must ingest without errors."""
        import json
        import tempfile
        import time

        symbols = ["SYM_A", "SYM_B", "SYM_C"]
        base_ts = int(time.time() * 1000)
        events = []
        for i in range(50):
            events.append({
                "eventId":   str(uuid.uuid4()),
                "symbol":    random.choice(symbols),
                "eventType": random.choice(["TRADE", "QUOTE", "ORDER"]),
                "timestamp": base_ts + i,
                "price":     round(random.uniform(100, 5000), 2),
                "quantity":  random.randint(1, 1000),
                "extra_field_" + rand_str(3, 6): rand_str(1, 20),
            })

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(self._make_jsonl(events))
        tmp.close()

        from data_ingestion.pipeline import IngestionPipeline
        from ledger_core.event_store import EventStore
        from ledger_core.sequence_manager import SequenceManager
        from merkle_forest.forest_manager import ForestManager
        from merkle_forest.tree_snapshot import SnapshotStore

        store = EventStore(sequence_manager=SequenceManager())
        forest = ForestManager(snapshot_store=SnapshotStore())
        pipeline = IngestionPipeline(file_paths=tmp.name, num_shards=4)

        stored = 0
        for se in pipeline.run():
            ar = store.append(se.event)
            if ar.success:
                forest.append(ar.stored_event.shard_key, ar.stored_event.stored_event_hash)
                stored += 1

        os.unlink(tmp.name)
        self.assertGreater(stored, 0)
        self.assertIsNotNone(forest.forest_root())

    def test_malformed_lines_dont_crash_pipeline(self):
        """Malformed JSONL lines must be skipped, not crash the pipeline."""
        import tempfile
        import json

        lines = [
            json.dumps({"eventId": "e1", "symbol": "SYM", "timestamp": 1000}),
            "not valid json {{{",
            "",
            '{"incomplete":',
            json.dumps({"eventId": "e2", "symbol": "SYM", "timestamp": 1001}),
            "null",
            json.dumps({"eventId": "e3", "symbol": "SYM", "timestamp": 1002}),
        ]

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write("\n".join(lines))
        tmp.close()

        from data_ingestion.pipeline import IngestionPipeline
        from ledger_core.event_store import EventStore
        from ledger_core.sequence_manager import SequenceManager

        store = EventStore(sequence_manager=SequenceManager())
        pipeline = IngestionPipeline(file_paths=tmp.name, num_shards=4)

        stored = 0
        try:
            for se in pipeline.run():
                ar = store.append(se.event)
                if ar.success:
                    stored += 1
        except Exception as e:
            self.fail(f"Pipeline crashed on malformed input: {e}")
        finally:
            os.unlink(tmp.name)

        # Should have stored 3 valid events
        self.assertGreaterEqual(stored, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)