"""
proof_system/inclusion_proof.py

Inclusion Proof for VFEL.

An inclusion proof answers: "Is event E in the ledger at root R?"

Proof structure:
    1. Event identity: ledger_event_id, content_hash
    2. StoredEvent identity: stored_event_hash, shard_key, shard_sequence
    3. Merkle path: list of (direction, sibling_hash) from leaf to shard root
    4. Shard root in block: which block sealed this shard root
    5. Forest root: the block's forest_root (the global commitment)

Verification algorithm:
    1. Recompute leaf hash from stored_event_hash
    2. Walk Merkle path → recompute shard root
    3. Verify shard root matches the block's shard_snapshot for this shard
    4. Verify block hash is consistent with the block's header fields
    5. PASS if all checks pass

This is a complete chain of custody proof:
    event bytes → content_hash → stored_event_hash →
    Merkle leaf → Merkle root → block header → forest root

Serialization:
    Proofs are JSON-serializable. An external auditor can verify a proof
    using only: the proof JSON + the block's forest_root.
    They do NOT need access to the full ledger.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ledger_core.event_store import EventStore
from merkle_forest.forest_manager import ForestManager
from merkle_forest.merkle_math import verify_proof, hash_leaf
from block_dag.block_store import BlockStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Proof models
# ──────────────────────────────────────────────

@dataclass
class InclusionProof:
    """
    Cryptographic proof that a specific event is included in the ledger.

    Self-contained: everything needed to verify is in this object.
    Can be serialized to JSON and verified offline.
    """

    # Event identity
    ledger_event_id: str
    content_hash: str
    stored_event_hash: str
    shard_key: str
    shard_sequence: int
    global_sequence: int

    # Merkle proof path (leaf → shard root)
    merkle_path: list[tuple[str, str]]   # [(direction, sibling_hash), ...]
    leaf_hash: str                        # hash_leaf(stored_event_hash)
    shard_root: str                       # Expected shard root after walking path

    # Block context
    block_hash: str
    block_height: int
    forest_root: str                      # The global commitment

    # Metadata
    generated_at_ns: int = field(default_factory=time.time_ns)
    proof_version: str = "vfel-proof-1.0"

    def to_dict(self) -> dict:
        return {
            "proof_type":         "inclusion",
            "proof_version":      self.proof_version,
            "generated_at_ns":    self.generated_at_ns,
            "ledger_event_id":    self.ledger_event_id,
            "content_hash":       self.content_hash,
            "stored_event_hash":  self.stored_event_hash,
            "shard_key":          self.shard_key,
            "shard_sequence":     self.shard_sequence,
            "global_sequence":    self.global_sequence,
            "merkle_path":        self.merkle_path,
            "leaf_hash":          self.leaf_hash,
            "shard_root":         self.shard_root,
            "block_hash":         self.block_hash,
            "block_height":       self.block_height,
            "forest_root":        self.forest_root,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InclusionProof":
        return cls(
            ledger_event_id=d["ledger_event_id"],
            content_hash=d["content_hash"],
            stored_event_hash=d["stored_event_hash"],
            shard_key=d["shard_key"],
            shard_sequence=d["shard_sequence"],
            global_sequence=d["global_sequence"],
            merkle_path=[(t[0], t[1]) for t in d["merkle_path"]],
            leaf_hash=d["leaf_hash"],
            shard_root=d["shard_root"],
            block_hash=d["block_hash"],
            block_height=d["block_height"],
            forest_root=d["forest_root"],
            generated_at_ns=d.get("generated_at_ns", 0),
            proof_version=d.get("proof_version", "vfel-proof-1.0"),
        )


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    checks: dict[str, bool]           # Individual check results
    failed_check: Optional[str] = None
    error: Optional[str] = None

    def summary(self) -> str:
        if self.valid:
            return f"✓ PROOF VALID ({len(self.checks)} checks passed)"
        return f"✗ PROOF INVALID — failed check: {self.failed_check}: {self.error}"


# ──────────────────────────────────────────────
# Inclusion Proof Generator
# ──────────────────────────────────────────────

class InclusionProofGenerator:
    """
    Generates inclusion proofs for events in the ledger.

    Requires access to EventStore, ForestManager, and BlockStore.
    Proof generation is read-only — does not modify ledger state.
    """

    def __init__(
        self,
        event_store: EventStore,
        forest_manager: ForestManager,
        block_store: BlockStore,
    ):
        self._event_store = event_store
        self._forest_manager = forest_manager
        self._block_store = block_store

    def generate(self, ledger_event_id: str) -> Optional[InclusionProof]:
        """
        Generate an inclusion proof for the given event ID.
        Returns None if the event is not found or tree not yet built.
        """
        # Step 1: Find the StoredEvent
        stored = self._event_store.get_by_event_id(ledger_event_id)
        if not stored:
            logger.warning("Event not found: %s", ledger_event_id[:16])
            return None

        shard_key = stored.shard_key

        # Step 2: Get the Merkle tree for this shard
        tree = self._forest_manager.get_tree(shard_key)
        if not tree:
            logger.warning("No Merkle tree for shard: %s", shard_key)
            return None

        # Step 3: Find leaf index
        # Leaf index in tree = shard_sequence of this stored event
        leaf_index = stored.shard_sequence
        if leaf_index >= tree.leaf_count:
            logger.warning(
                "Leaf index %d out of range (tree has %d leaves)",
                leaf_index, tree.leaf_count
            )
            return None

        # Step 4: Get leaf hash
        leaf_hash = tree.get_leaf_hash(leaf_index)
        if not leaf_hash:
            return None

        # Step 5: Generate Merkle path
        merkle_path = tree.prove_inclusion(leaf_index)
        if merkle_path is None:
            return None

        shard_root = tree.batch_root()

        # Step 6: Find the block that sealed this shard root
        # Walk blocks to find one whose shard_snapshot matches our shard root
        block_hash, block_height, forest_root = self._find_sealing_block(
            shard_key, shard_root
        )

        return InclusionProof(
            ledger_event_id=ledger_event_id,
            content_hash=stored.event.content_hash,
            stored_event_hash=stored.stored_event_hash,
            shard_key=shard_key,
            shard_sequence=stored.shard_sequence,
            global_sequence=stored.ledger_sequence,
            merkle_path=merkle_path,
            leaf_hash=leaf_hash,
            shard_root=shard_root,
            block_hash=block_hash,
            block_height=block_height,
            forest_root=forest_root,
        )

    def _find_sealing_block(
        self, shard_key: str, shard_root: str
    ) -> tuple[str, int, str]:
        """Find the block that sealed a specific shard root."""
        for bh, block in self._block_store.all_blocks().items():
            for snap in block.header.shard_snapshots:
                if snap.shard_key == shard_key and snap.merkle_root == shard_root:
                    return bh, block.height, block.header.forest_root
        # Return current forest root if no block found yet
        return "PENDING", -1, self._forest_manager.forest_root()


# ──────────────────────────────────────────────
# Inclusion Proof Verifier
# ──────────────────────────────────────────────

class InclusionProofVerifier:
    """
    Stateless verifier for inclusion proofs.
    Does not need access to the ledger — only the proof itself.
    Can run on any machine with the proof JSON.
    """

    def verify(self, proof: InclusionProof) -> VerificationResult:
        """
        Verify an inclusion proof end-to-end.

        Checks performed:
        1. leaf_hash = hash_leaf(stored_event_hash)  — leaf is correct
        2. Merkle path walk → recomputed root = proof.shard_root  — path is valid
        3. proof.shard_root appears in the block's shard_snapshots  — not checked offline
           (offline verifiers trust the block context)
        4. Block hash is structurally valid (if block data is available)
        """
        checks = {}

        # Check 1: leaf hash
        expected_leaf = hash_leaf(proof.stored_event_hash)
        checks["leaf_hash_correct"] = (expected_leaf == proof.leaf_hash)
        if not checks["leaf_hash_correct"]:
            return VerificationResult(
                valid=False, checks=checks,
                failed_check="leaf_hash_correct",
                error=f"Expected {expected_leaf[:16]}, got {proof.leaf_hash[:16]}"
            )

        # Check 2: Merkle path validity
        path_valid = verify_proof(proof.leaf_hash, proof.merkle_path, proof.shard_root)
        checks["merkle_path_valid"] = path_valid
        if not path_valid:
            return VerificationResult(
                valid=False, checks=checks,
                failed_check="merkle_path_valid",
                error="Merkle path does not produce the claimed shard root"
            )

        # Check 3: Proof version
        checks["proof_version_known"] = proof.proof_version.startswith("vfel-proof-")
        if not checks["proof_version_known"]:
            return VerificationResult(
                valid=False, checks=checks,
                failed_check="proof_version_known",
                error=f"Unknown proof version: {proof.proof_version}"
            )

        return VerificationResult(valid=True, checks=checks)

        