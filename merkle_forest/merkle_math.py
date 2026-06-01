"""
merkle_forest/merkle_math.py

Merkle Tree Mathematical Primitives for VFEL.

All pure functions — no state, no side effects.
These are the cryptographic building blocks used by every other module
in the merkle_forest package.

Design:
- Binary Merkle tree (arity=2) — standard, well-studied, proof-friendly
- Leaf hash: H(0x00 || data)    ← domain separation prevents second-preimage
- Node hash: H(0x01 || left || right)   ← RFC 6962 style
- Empty tree: deterministic empty root per height (no null values in the tree)
- All hashes: hex strings (64 chars for SHA256)

Why domain separation?
  Without 0x00/0x01 prefixes, an attacker can construct a valid proof for an
  internal node by treating it as a leaf. This is the second-preimage attack
  on Merkle trees. RFC 6962 (Certificate Transparency) uses this exact scheme.

Phase 5 note: hash_leaf() and hash_node() will be upgraded to BLAKE3 when
the crypto layer is implemented. The interface is stable.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# ──────────────────────────────────────────────
# Domain separation prefixes (RFC 6962 style)
# ──────────────────────────────────────────────
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"

# Sentinel: empty node hash (H(0x02 || "EMPTY"))
# Used to pad odd-length levels to even width
_EMPTY_HASH = hashlib.sha256(b"\x02EMPTY").hexdigest()


# ──────────────────────────────────────────────
# Core Hash Functions
# ──────────────────────────────────────────────

def hash_leaf(data: bytes | str) -> str:
    """
    Hash a leaf value with domain separation.
    H(0x00 || data)
    Input can be bytes or a hex string (auto-decoded).
    """
    if isinstance(data, str):
        # Treat hex strings as raw bytes; fallback to UTF-8 for non-hex
        try:
            data = bytes.fromhex(data)
        except ValueError:
            data = data.encode("utf-8")

    return hashlib.sha256(_LEAF_PREFIX + data).hexdigest()


def hash_node(left: str, right: str) -> str:
    """
    Hash two child hashes into a parent node.
    H(0x01 || left_bytes || right_bytes)
    Inputs are hex strings.
    """
    left_bytes = bytes.fromhex(left)
    right_bytes = bytes.fromhex(right)
    return hashlib.sha256(_NODE_PREFIX + left_bytes + right_bytes).hexdigest()


def empty_hash() -> str:
    """Canonical empty node hash for tree padding."""
    return _EMPTY_HASH


# ──────────────────────────────────────────────
# Tree Structure Math
# ──────────────────────────────────────────────

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n. Returns 1 for n=0."""
    if n <= 0:
        return 1
    if n == 1:
        return 1
    return 1 << (n - 1).bit_length()


def tree_height(leaf_count: int) -> int:
    """
    Height of a binary Merkle tree holding leaf_count leaves.
    Height 0 = root only (1 leaf). Height n = 2^n leaves.
    """
    if leaf_count <= 1:
        return 0
    return (leaf_count - 1).bit_length()


def sibling_index(index: int) -> int:
    """
    Given a node index in a level, return its sibling's index.
    Even index → right sibling (index + 1)
    Odd index  → left sibling (index - 1)
    """
    return index ^ 1  # XOR with 1 flips the last bit


def parent_index(index: int) -> int:
    """Given a node index in a level, return its parent's index in the level above."""
    return index >> 1  # Integer divide by 2


def is_left_child(index: int) -> bool:
    """True if this index is a left child (even index in its level)."""
    return (index & 1) == 0


# ──────────────────────────────────────────────
# Full Tree Computation
# ──────────────────────────────────────────────

def _largest_power_of_two_below(n: int) -> int:
    """Largest power of two strictly less than n (n >= 2). RFC 6962 split point k."""
    return 1 << ((n - 1).bit_length() - 1)


def compute_root(leaves: list[str]) -> str:
    """
    Compute the RFC 6962 Merkle Tree Hash (MTH) over a list of leaf hashes.

    MTH({})      = empty_hash()
    MTH({d0})    = d0
    MTH(D[n])    = hash_node(MTH(D[0:k]), MTH(D[k:n])),  k = largest 2^i < n

    This is an UNBALANCED, left-perfect tree — NOT a power-of-two-padded tree.
    Using this definition makes compute_root() bit-identical to the incremental
    frontier root() for every leaf count, which is required for proof soundness:
    an inclusion proof must verify against the same root that gets sealed and
    anchored. (Previously this padded to next_power_of_two with empty_hash(),
    which diverged from root() at every non-power-of-two count.)

    O(n) time — not the hot path. Use IncrementalMerkleTree for streaming append.
    """
    n = len(leaves)
    if n == 0:
        return empty_hash()
    if n == 1:
        return leaves[0]
    k = _largest_power_of_two_below(n)
    return hash_node(compute_root(leaves[:k]), compute_root(leaves[k:]))


def compute_proof_path(leaves: list[str], leaf_index: int) -> list[tuple[str, str]]:
    """
    RFC 6962 Merkle audit path for the leaf at leaf_index.

    Returns a list of (direction, sibling_hash) from leaf to root, where
    direction = "R" if the sibling is on the right (our node is a left child),
                "L" if the sibling is on the left  (our node is a right child).

    Consistent with the MTH definition in compute_root(): the path verifies
    against the SAME root that is sealed and anchored. O(n) to build, O(log n)
    in size and to verify.
    """
    n = len(leaves)
    if n == 0 or leaf_index < 0 or leaf_index >= n:
        return []
    if n == 1:
        return []
    k = _largest_power_of_two_below(n)
    if leaf_index < k:
        # our leaf is in the left subtree; sibling is the right subtree root
        return compute_proof_path(leaves[:k], leaf_index) + [("R", compute_root(leaves[k:]))]
    else:
        # our leaf is in the right subtree; sibling is the left subtree root
        return compute_proof_path(leaves[k:], leaf_index - k) + [("L", compute_root(leaves[:k]))]


def verify_proof(leaf_hash: str, proof: list[tuple[str, str]], expected_root: str) -> bool:
    """
    Verify a Merkle inclusion proof.

    leaf_hash: hash of the leaf being proved
    proof: list of (direction, sibling_hash) from compute_proof_path()
    expected_root: the known-good root hash

    Returns True if the proof is valid.
    """
    current = leaf_hash
    for direction, sibling in proof:
        if direction == "R":
            # current is left child
            current = hash_node(current, sibling)
        else:
            # current is right child
            current = hash_node(sibling, current)
    return current == expected_root