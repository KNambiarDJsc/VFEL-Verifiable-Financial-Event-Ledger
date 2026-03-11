"""
tests/test_merkle_forest.py

Merkle Forest Tests — Phase 10.

Tests:
    - RFC 6962 hash domain separation (leaf vs node prefixes)
    - IncrementalMerkleTree: append, root, frontier correctness
    - Inclusion proof generation and verification
    - batch_root vs incremental root relationship
    - ForestManager: per-shard trees, forest root, sealing
    - TreeSnapshot: save/load round-trip
    - Edge cases: single leaf, power-of-two, non-power-of-two counts

Run:
    python -m pytest tests/test_merkle_forest.py -v
    python tests/test_merkle_forest.py           (no pytest needed)
"""

import hashlib
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merkle_forest.merkle_math import (
    hash_leaf, hash_node, empty_hash,
    compute_root, compute_proof_path, verify_proof,
    tree_height, next_power_of_two,
)
from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree
from merkle_forest.forest_manager import ForestManager
from merkle_forest.tree_snapshot import SnapshotStore, TreeSnapshot


class TestMerkleMath(unittest.TestCase):
    """Test RFC 6962 hash primitives."""

    def test_leaf_domain_separation(self):
        """hash_leaf must use 0x00 prefix per RFC 6962."""
        data = "test_data"
        expected = hashlib.sha256(b"\x00" + data.encode()).hexdigest()
        self.assertEqual(hash_leaf(data), expected)

    def test_node_domain_separation(self):
        """hash_node must use 0x01 prefix per RFC 6962."""
        left  = "a" * 64
        right = "b" * 64
        combined = b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)
        expected = hashlib.sha256(combined).hexdigest()
        self.assertEqual(hash_node(left, right), expected)

    def test_leaf_node_not_equal(self):
        """Same data hashed as leaf vs node must produce different results."""
        data = "a" * 64
        self.assertNotEqual(hash_leaf(data), hash_node(data, data))

    def test_compute_root_single(self):
        """Single leaf: root == hash_leaf(leaf)."""
        leaf = hash_leaf("only_leaf")
        root = compute_root([leaf])
        self.assertEqual(root, leaf)

    def test_compute_root_two(self):
        """Two leaves: root == hash_node(leaf0, leaf1)."""
        l0 = hash_leaf("leaf0")
        l1 = hash_leaf("leaf1")
        root = compute_root([l0, l1])
        self.assertEqual(root, hash_node(l0, l1))

    def test_compute_root_deterministic(self):
        """Same leaves always produce same root."""
        leaves = [hash_leaf(f"leaf{i}") for i in range(7)]
        r1 = compute_root(leaves)
        r2 = compute_root(leaves)
        self.assertEqual(r1, r2)

    def test_compute_root_order_sensitive(self):
        """Reordering leaves changes the root."""
        leaves = [hash_leaf(f"leaf{i}") for i in range(4)]
        r1 = compute_root(leaves)
        r2 = compute_root(list(reversed(leaves)))
        self.assertNotEqual(r1, r2)

    def test_inclusion_proof_all_sizes(self):
        """Inclusion proof valid for all leaf positions, all tree sizes 1-16."""
        for n in range(1, 17):
            leaves = [hash_leaf(f"leaf{i}") for i in range(n)]
            root = compute_root(leaves)
            for idx in range(n):
                path = compute_proof_path(leaves, idx)
                self.assertTrue(
                    verify_proof(leaves[idx], path, root),
                    f"Proof failed for n={n}, idx={idx}"
                )

    def test_inclusion_proof_tampered_leaf_rejected(self):
        """Tampered leaf must fail proof verification."""
        leaves = [hash_leaf(f"leaf{i}") for i in range(8)]
        root   = compute_root(leaves)
        path   = compute_proof_path(leaves, 3)
        tampered = hash_leaf("TAMPERED")
        self.assertFalse(verify_proof(tampered, path, root))

    def test_inclusion_proof_tampered_root_rejected(self):
        """Tampered root must fail proof verification."""
        leaves = [hash_leaf(f"leaf{i}") for i in range(8)]
        root   = compute_root(leaves)
        path   = compute_proof_path(leaves, 3)
        self.assertFalse(verify_proof(leaves[3], path, "0" * 64))

    def test_next_power_of_two(self):
        cases = [(1,1),(2,2),(3,4),(4,4),(5,8),(7,8),(8,8),(9,16)]
        for n, expected in cases:
            self.assertEqual(next_power_of_two(n), expected, f"n={n}")


class TestIncrementalMerkleTree(unittest.TestCase):
    """Test IncrementalMerkleTree O(log n) append correctness."""

    def test_empty_tree(self):
        tree = IncrementalMerkleTree()
        self.assertEqual(tree.leaf_count, 0)

    def test_single_append(self):
        tree = IncrementalMerkleTree()
        tree.append_raw_hash(hash_leaf("leaf0"))
        self.assertEqual(tree.leaf_count, 1)
        self.assertEqual(tree.batch_root(), hash_leaf("leaf0"))

    def test_batch_root_matches_compute_root(self):
        """batch_root() must match compute_root(all_leaves) for all sizes 1-20."""
        for n in range(1, 21):
            tree = IncrementalMerkleTree()
            leaves = []
            for i in range(n):
                h = hash_leaf(f"leaf{i}")
                leaves.append(h)
                tree.append_raw_hash(h)
            expected_root = compute_root(leaves)
            self.assertEqual(
                tree.batch_root(), expected_root,
                f"Mismatch at n={n}"
            )

    def test_inclusion_proof_correctness(self):
        """Every leaf in the tree should have a valid inclusion proof."""
        n = 13
        tree = IncrementalMerkleTree()
        leaves = []
        for i in range(n):
            h = hash_leaf(f"leaf{i}")
            leaves.append(h)
            tree.append_raw_hash(h)

        root = tree.batch_root()
        for idx in range(n):
            path = tree.prove_inclusion(idx)
            self.assertIsNotNone(path, f"No proof for idx={idx}")
            leaf_h = tree.get_leaf_hash(idx)
            self.assertTrue(
                tree.verify_inclusion(leaf_h, idx, path),
                f"Verification failed for idx={idx}"
            )

    def test_append_result(self):
        tree = IncrementalMerkleTree()
        h = hash_leaf("test")
        result = tree.append_raw_hash(h)
        self.assertEqual(result.leaf_index, 0)
        self.assertEqual(result.leaf_hash, h)

    def test_get_all_leaf_hashes(self):
        tree = IncrementalMerkleTree()
        hashes = [hash_leaf(f"l{i}") for i in range(5)]
        for h in hashes:
            tree.append_raw_hash(h)
        self.assertEqual(tree.get_all_leaf_hashes(), hashes)

    def test_frontier_snapshot_restore(self):
        """Restoring from frontier snapshot must give same root."""
        tree = IncrementalMerkleTree()
        for i in range(17):
            tree.append_raw_hash(hash_leaf(f"leaf{i}"))
        snapshot = tree.frontier_snapshot()
        leaf_count = tree.leaf_count

        # Restore and check root
        restored = IncrementalMerkleTree.restore(
            frontier=snapshot,
            leaf_hashes=tree.get_all_leaf_hashes(),
        )
        self.assertEqual(restored.batch_root(), tree.batch_root())
        self.assertEqual(restored.leaf_count, tree.leaf_count)


class TestForestManager(unittest.TestCase):
    """Test ForestManager multi-shard forest operations."""

    def setUp(self):
        self.snap_store = SnapshotStore()
        self.forest = ForestManager(snapshot_store=self.snap_store)

    def test_append_creates_shard_tree(self):
        self.forest.append("RELIANCE", hash_leaf("event1"))
        self.assertIsNotNone(self.forest.get_tree("RELIANCE"))

    def test_multiple_shards(self):
        shards = ["RELIANCE", "TCS", "INFY", "HDFC"]
        for sk in shards:
            for i in range(5):
                self.forest.append(sk, hash_leaf(f"{sk}_{i}"))
        self.assertEqual(set(self.forest.known_shards()), set(shards))

    def test_forest_root_changes_on_append(self):
        r0 = self.forest.forest_root()
        self.forest.append("SHARD1", hash_leaf("event1"))
        r1 = self.forest.forest_root()
        self.assertNotEqual(r0, r1)

    def test_forest_root_deterministic(self):
        """Same events in same order → same forest root."""
        forest2 = ForestManager(snapshot_store=SnapshotStore())
        events = [("SHARD_A", hash_leaf(f"ev{i}")) for i in range(10)]
        events += [("SHARD_B", hash_leaf(f"ev{i}")) for i in range(5)]
        for sk, h in events:
            self.forest.append(sk, h)
            forest2.append(sk, h)
        self.assertEqual(self.forest.forest_root(), forest2.forest_root())

    def test_seal_forest_root(self):
        for i in range(5):
            self.forest.append("TEST_SHARD", hash_leaf(f"ev{i}"))
        sealed = self.forest.seal_forest_root(trigger="test")
        self.assertIsNotNone(sealed.forest_root)
        self.assertIn("TEST_SHARD", sealed.shard_roots)
        # Sealed shard root must be a valid 64-char hex string
        shard_root = sealed.shard_roots["TEST_SHARD"]
        self.assertEqual(len(shard_root), 64)
        int(shard_root, 16)  # Must be valid hex


class TestTreeSnapshot(unittest.TestCase):
    """Test TreeSnapshot save/load round-trip."""

    def test_save_load_round_trip(self):
        import uuid
        snap_store = SnapshotStore()
        leaves = [hash_leaf(f"l{i}") for i in range(8)]

        snap = TreeSnapshot(
            snapshot_id=str(uuid.uuid4()),
            shard_key="TEST",
            leaf_count=8,
            root_hash=compute_root(leaves),
            frontier=[None] * 48,
            leaf_hashes=leaves,
            trigger="test",
        )
        snap_store.save(snap)
        loaded = snap_store.get_at_leaf_count("TEST", 8)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.root_hash, snap.root_hash)
        self.assertEqual(loaded.leaf_hashes, leaves)

    def test_get_nearest_before(self):
        import uuid
        snap_store = SnapshotStore()
        for n in [5, 10, 20]:
            leaves = [hash_leaf(f"l{i}") for i in range(n)]
            snap_store.save(TreeSnapshot(
                snapshot_id=str(uuid.uuid4()), shard_key="SHARD",
                leaf_count=n, root_hash=compute_root(leaves),
                frontier=[None]*48, leaf_hashes=leaves, trigger="test",
            ))
        snap = snap_store.get_nearest_before("SHARD", 15)
        self.assertIsNotNone(snap)
        self.assertLessEqual(snap.leaf_count, 15)


# ──────────────────────────────────────────────
# Run directly (no pytest)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)