"""Soundness: the committed/anchored root and the proof-verification root must
be identical for ALL leaf counts, and inclusion proofs must verify against the
committed root. Guards against regression of bug B1 (frontier vs batch root)."""
import hashlib, unittest
from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree
from merkle_forest.merkle_math import compute_root, compute_proof_path, verify_proof


class TestRootConsistency(unittest.TestCase):
    def test_frontier_root_equals_mth(self):
        for n in range(1, 200):
            t = IncrementalMerkleTree(); leaves = []
            for i in range(n):
                h = hashlib.sha256(f"{n}-{i}".encode()).hexdigest()
                t.append_raw_hash(h); leaves.append(t._leaf_hashes[-1])
            self.assertEqual(t.root(), compute_root(leaves), f"root mismatch at n={n}")

    def test_proofs_verify_against_committed_root(self):
        for n in (1, 2, 3, 5, 8, 13, 100, 129):
            t = IncrementalMerkleTree(); leaves = []
            for i in range(n):
                h = hashlib.sha256(f"{n}-{i}".encode()).hexdigest()
                t.append_raw_hash(h); leaves.append(t._leaf_hashes[-1])
            root = t.root()
            for idx in range(n):
                p = compute_proof_path(leaves, idx)
                self.assertTrue(verify_proof(leaves[idx], p, root))
                self.assertFalse(verify_proof("0"*64, p, root))


if __name__ == "__main__":
    unittest.main()