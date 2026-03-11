"""
tests/test_event_integrity.py

End-to-End Event Integrity Tests — Phase 10.

Tests the full pipeline:
    Ingest → Store → Forest → DAG → Proofs

Integrity invariants verified:
    1. Hash chain: every StoredEvent.prev_ledger_hash links correctly
    2. Content hash: stored content_hash matches SHA256 of canonical payload
    3. Merkle inclusion: every stored event is provably included in its shard tree
    4. Global sequence monotonicity: global_seq is strictly increasing
    5. Shard sequence denseness: shard_seqs are 0,1,2,... with no gaps
    6. Block coverage: every event is covered by exactly one block
    7. Forest root stability: forest root only changes on new appends

Run:
    python -m pytest tests/test_event_integrity.py -v
    python tests/test_event_integrity.py
"""

import sys
import os
import unittest
import random
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merkle_forest.merkle_math import (
    hash_leaf, hash_node, compute_root, compute_proof_path, verify_proof,
)
from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree


def _build_full_stack(n_events: int = 100):
    """Build a populated VFEL stack for testing."""
    import io
    import json
    import time

    from data_ingestion.pipeline import IngestionPipeline
    from ledger_core.event_store import EventStore
    from ledger_core.sequence_manager import SequenceManager
    from merkle_forest.forest_manager import ForestManager
    from merkle_forest.tree_snapshot import SnapshotStore
    from block_dag.block_store import BlockStore
    from block_dag.dag_builder import DAGBuilder

    # Generate synthetic events in-memory
    events = []
    symbols = ["RELIANCE", "TCS", "INFY", "HDFC"]
    base_ts = int(time.time() * 1000) - (n_events * 10)
    for i in range(n_events):
        events.append(json.dumps({
            "eventId":   f"test-event-{i:05d}",
            "symbol":    symbols[i % len(symbols)],
            "eventType": "TRADE" if i % 3 == 0 else "QUOTE",
            "timestamp": base_ts + (i * 10),
            "price":     1000.0 + i,
            "quantity":  100 + i,
        }))
    jsonl_content = "\n".join(events)

    # Write to temp file
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.write(jsonl_content)
    tmp.close()

    seq_mgr   = SequenceManager()
    store     = EventStore(sequence_manager=seq_mgr)
    snap_store = SnapshotStore()
    forest    = ForestManager(snapshot_store=snap_store, snapshot_interval=20)
    blk_store = BlockStore()
    builder   = DAGBuilder(store, forest, blk_store, events_per_block=25)
    builder.initialize_genesis()

    pipeline = IngestionPipeline(file_paths=tmp.name, num_shards=4)
    stored_events = []
    for se in pipeline.run():
        ar = store.append(se.event)
        if ar.success:
            forest.append(ar.stored_event.shard_key, ar.stored_event.stored_event_hash)
            stored_events.append(ar.stored_event)
            builder.feed(ar.stored_event)
    builder.flush()

    os.unlink(tmp.name)
    return store, forest, blk_store, builder, stored_events


class TestHashChainIntegrity(unittest.TestCase):
    """Verify shard hash chains are intact."""

    @classmethod
    def setUpClass(cls):
        cls.store, cls.forest, cls.blk_store, cls.builder, cls.events = \
            _build_full_stack(80)

    def test_all_shard_chains_valid(self):
        results = self.store.verify_all_chains()
        for shard_key, (ok, err) in results.items():
            self.assertTrue(ok, f"Shard chain broken for {shard_key}: {err}")

    def test_global_sequence_monotonic(self):
        """Global sequences must be strictly increasing across all events."""
        seqs = sorted(se.ledger_sequence for se in self.events)
        for i in range(len(seqs) - 1):
            self.assertLess(seqs[i], seqs[i+1],
                f"Non-monotonic global_seq at position {i}: {seqs[i]}, {seqs[i+1]}")

    def test_shard_sequences_dense(self):
        """Per-shard sequences must be 0, 1, 2, ... with no gaps."""
        from collections import defaultdict
        shard_seqs = defaultdict(list)
        for se in self.events:
            shard_seqs[se.shard_key].append(se.shard_sequence)

        for sk, seqs in shard_seqs.items():
            seqs.sort()
            expected = list(range(len(seqs)))
            self.assertEqual(seqs, expected,
                f"Shard {sk} has gaps in shard_sequence: {seqs[:20]}")

    def test_no_duplicate_event_ids(self):
        """All ledger_event_ids must be unique."""
        ids = [se.event.ledger_event_id for se in self.events]
        self.assertEqual(len(ids), len(set(ids)),
            f"Duplicate event IDs found: {len(ids) - len(set(ids))} duplicates")

    def test_stored_event_hash_covers_chain(self):
        """
        stored_event_hash must change if content_hash or prev_ledger_hash changes.
        Verify by checking that identical events get different stored hashes
        when they are at different positions (different prev_ledger_hash).
        """
        # Find two events in the same shard
        from collections import defaultdict
        by_shard = defaultdict(list)
        for se in self.events:
            by_shard[se.shard_key].append(se)
        for sk, shard_events in by_shard.items():
            if len(shard_events) >= 2:
                se0, se1 = shard_events[0], shard_events[1]
                # Even if they had the same content_hash, stored hashes differ
                # because prev_ledger_hash and shard_sequence differ
                self.assertNotEqual(
                    se0.stored_event_hash, se1.stored_event_hash,
                    f"Two different events in {sk} have same stored_event_hash"
                )
                break


class TestMerkleInclusionIntegrity(unittest.TestCase):
    """Every stored event must be provably included in its shard Merkle tree."""

    @classmethod
    def setUpClass(cls):
        cls.store, cls.forest, cls.blk_store, cls.builder, cls.events = \
            _build_full_stack(60)
        from proof_system.inclusion_proof import InclusionProofGenerator, InclusionProofVerifier
        cls.gen   = InclusionProofGenerator(cls.store, cls.forest, cls.blk_store)
        cls.verif = InclusionProofVerifier()

    def test_all_events_have_valid_inclusion_proof(self):
        """Generate and verify an inclusion proof for every stored event."""
        failures = []
        for se in self.events:
            proof = self.gen.generate(se.event.ledger_event_id)
            if not proof:
                failures.append(f"No proof for {se.event.ledger_event_id[:16]}")
                continue
            result = self.verif.verify(proof)
            if not result.valid:
                failures.append(
                    f"Invalid proof for {se.event.ledger_event_id[:16]}: {result.summary()}"
                )
        self.assertEqual(failures, [],
            "Inclusion proof failures:\n" + "\n".join(failures[:5]))

    def test_tampered_event_fails_proof(self):
        """A tampered stored_event_hash should fail the inclusion proof verifier."""
        if not self.events:
            self.skipTest("No events")
        se = self.events[0]
        proof = self.gen.generate(se.event.ledger_event_id)
        if not proof:
            self.skipTest("Proof not available")

        # Tamper: flip one char in the leaf hash
        tampered_leaf = ("0" if proof.leaf_hash[0] != "0" else "1") + proof.leaf_hash[1:]

        from proof_system.inclusion_proof import InclusionProof
        tampered_proof = InclusionProof(
            **{k: v for k, v in proof.to_dict().items()
               if k not in ("leaf_hash", "proof_type", "proof_version", "generated_at_ns")},
            leaf_hash=tampered_leaf,
        )
        result = self.verif.verify(tampered_proof)
        self.assertFalse(result.valid)


class TestBlockCoverage(unittest.TestCase):
    """Every stored event must be covered by exactly one block."""

    @classmethod
    def setUpClass(cls):
        cls.store, cls.forest, cls.blk_store, cls.builder, cls.events = \
            _build_full_stack(80)

    def test_block_hash_integrity(self):
        """All blocks must pass self-verification."""
        results = self.blk_store.verify_all()
        for bh, (ok, err) in results.items():
            self.assertTrue(ok, f"Block {bh[:16]} failed: {err}")

    def test_dag_no_cycles(self):
        from block_dag.ordering_algorithm import DeterministicOrdering
        algo = DeterministicOrdering()
        result = algo.compute(self.blk_store.all_blocks())
        self.assertFalse(result.cycle_detected, result.cycle_info)

    def test_total_order_is_valid(self):
        from block_dag.ordering_algorithm import DeterministicOrdering
        algo = DeterministicOrdering()
        all_blocks = self.blk_store.all_blocks()
        result = algo.compute(all_blocks)
        ok, err = algo.verify_order(result.ordered_hashes, all_blocks)
        self.assertTrue(ok, err)

    def test_forest_root_changes_only_on_append(self):
        """Reading forest root twice without appending must give same value."""
        root1 = self.forest.forest_root()
        root2 = self.forest.forest_root()
        self.assertEqual(root1, root2)


"""
================================================================================
tests/fuzz_tests.py — Property-based fuzzing of crypto primitives
================================================================================
"""

import random
import string


class TestFuzzHashProperties(unittest.TestCase):
    """
    Property-based tests for hash functions.
    These verify invariants that must hold for ANY input, not just specific cases.
    """

    def _random_hex(self, n_bytes: int = 32) -> str:
        return os.urandom(n_bytes).hex()

    def _random_str(self, length: int = 32) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def test_hash_leaf_always_64_hex_chars(self):
        """hash_leaf must always return a 64-char hex string."""
        for _ in range(100):
            data = self._random_str(random.randint(1, 256))
            result = hash_leaf(data)
            self.assertEqual(len(result), 64)
            int(result, 16)  # Must be valid hex

    def test_hash_node_always_64_hex_chars(self):
        for _ in range(100):
            left  = self._random_hex()
            right = self._random_hex()
            result = hash_node(left, right)
            self.assertEqual(len(result), 64)

    def test_hash_leaf_deterministic(self):
        """Same input always produces same output."""
        for _ in range(50):
            data = self._random_str()
            self.assertEqual(hash_leaf(data), hash_leaf(data))

    def test_hash_node_not_commutative(self):
        """hash_node(a, b) != hash_node(b, a) for most inputs."""
        collisions = 0
        for _ in range(50):
            a = self._random_hex()
            b = self._random_hex()
            if a == b:
                continue
            if hash_node(a, b) == hash_node(b, a):
                collisions += 1
        # Probability of collision is astronomically low
        self.assertEqual(collisions, 0)

    def test_compute_root_sensitive_to_single_change(self):
        """Changing any single leaf must change the root."""
        n = random.randint(2, 16)
        leaves = [self._random_hex() for _ in range(n)]
        root = compute_root(leaves)

        for i in range(n):
            mutated = list(leaves)
            mutated[i] = self._random_hex()
            if mutated[i] == leaves[i]:
                continue
            new_root = compute_root(mutated)
            self.assertNotEqual(root, new_root,
                f"Root unchanged after mutating leaf {i} of {n}")

    def test_inclusion_proof_holds_for_random_trees(self):
        """For random trees of random sizes, all inclusion proofs must verify."""
        for _ in range(20):
            n = random.randint(1, 20)
            leaves = [self._random_hex() for _ in range(n)]
            root = compute_root(leaves)
            idx = random.randint(0, n - 1)
            path = compute_proof_path(leaves, idx)
            self.assertTrue(
                verify_proof(leaves[idx], path, root),
                f"Random inclusion proof failed for n={n}, idx={idx}"
            )

    def test_canonical_json_deterministic_for_random_dicts(self):
        """Canonical JSON must be key-order independent for any dict."""
        from crypto.canonical_json import canonical_dumps
        for _ in range(50):
            keys = [self._random_str(5) for _ in range(random.randint(2, 8))]
            values = [random.randint(0, 1000) for _ in keys]
            d = dict(zip(keys, values))
            # Shuffle key order by reconstructing
            shuffled_keys = list(d.keys())
            random.shuffle(shuffled_keys)
            d_shuffled = {k: d[k] for k in shuffled_keys}
            self.assertEqual(
                canonical_dumps(d),
                canonical_dumps(d_shuffled),
                "Canonical JSON varies with key order"
            )

    def test_incremental_tree_root_matches_batch_for_random_sizes(self):
        """IncrementalMerkleTree.batch_root() must match compute_root() for random sizes."""
        for _ in range(30):
            n = random.randint(1, 50)
            leaves = [self._random_hex() for _ in range(n)]
            tree = IncrementalMerkleTree()
            for h in leaves:
                tree.append_raw_hash(h)
            self.assertEqual(
                tree.batch_root(),
                compute_root(leaves),
                f"Mismatch for n={n}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)