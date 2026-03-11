"""
anchoring/anchor_manager.py

Anchor Manager for VFEL — Phase 7.

The anchor manager is the single coordinator for all external anchoring.
It takes the current forest root from the Merkle forest, submits it to
one or more anchoring backends, and records the results in LedgerState.

Why anchor externally?
    The VFEL forest root is a cryptographic commitment to ALL events in the ledger.
    Anchoring it to an external system (Bitcoin, Ethereum, a trusted timestamp server)
    means that even if the VFEL node is compromised or manipulated, an attacker
    cannot retroactively change the ledger without breaking the external anchor.

    This is the key property: external anchors are immutable public records.
    Anyone can independently verify "the VFEL ledger had root R at time T"
    by looking at the Bitcoin/Ethereum blockchain or OpenTimestamps proof.

Anchor lifecycle:
    PENDING   → submitted to backend, waiting for confirmation
    CONFIRMED → confirmed on-chain / in the timestamp server
    FAILED    → submission failed (will retry)
    EXPIRED   → anchor attempt abandoned after max retries

Anchor policy:
    - Anchor every N blocks (default: every block)
    - Anchor on explicit flush/shutdown
    - Multiple backends can anchor the same root (redundancy)
    - Each backend runs independently (one failure doesn't block others)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class AnchorStatus(str, Enum):
    PENDING   = "PENDING"
    CONFIRMED = "CONFIRMED"
    FAILED    = "FAILED"
    EXPIRED   = "EXPIRED"


class AnchorNetwork(str, Enum):
    BITCOIN          = "bitcoin"
    ETHEREUM         = "ethereum"
    OPENTIMESTAMPS   = "opentimestamps"
    LOCAL_TIMESTAMP  = "local_timestamp"   # Fallback: signed local timestamp


# ──────────────────────────────────────────────
# Anchor record
# ──────────────────────────────────────────────

@dataclass
class AnchorRecord:
    """
    A single anchor submission and its result.

    One AnchorRecord per (forest_root, network) pair.
    Multiple records may exist for the same forest_root if using multiple networks.
    """
    anchor_id: str                    # UUID
    forest_root: str                  # The VFEL forest root being anchored
    network: AnchorNetwork
    status: AnchorStatus = AnchorStatus.PENDING

    # Submission details
    submitted_at_ns: int = field(default_factory=time.time_ns)
    confirmed_at_ns: Optional[int] = None

    # Network-specific transaction reference
    tx_id: Optional[str] = None           # Bitcoin txid / Ethereum tx hash
    block_number: Optional[int] = None    # On-chain block number
    block_hash: Optional[str] = None      # On-chain block hash
    proof_bytes: Optional[str] = None     # OTS proof (base64) or similar

    # Error tracking
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: dict = field(default_factory=dict)

    def mark_confirmed(
        self,
        tx_id: str,
        block_number: Optional[int] = None,
        block_hash: Optional[str] = None,
        proof_bytes: Optional[str] = None,
    ) -> None:
        self.status = AnchorStatus.CONFIRMED
        self.tx_id = tx_id
        self.block_number = block_number
        self.block_hash = block_hash
        self.proof_bytes = proof_bytes
        self.confirmed_at_ns = time.time_ns()

    def mark_failed(self, error: str) -> None:
        self.retry_count += 1
        self.error = error
        if self.retry_count >= self.max_retries:
            self.status = AnchorStatus.EXPIRED
            logger.error(
                "Anchor %s expired after %d retries: %s",
                self.anchor_id[:8], self.max_retries, error
            )
        else:
            self.status = AnchorStatus.FAILED
            logger.warning(
                "Anchor %s failed (attempt %d/%d): %s",
                self.anchor_id[:8], self.retry_count, self.max_retries, error
            )

    def to_dict(self) -> dict:
        return {
            "anchor_id":      self.anchor_id,
            "forest_root":    self.forest_root,
            "network":        self.network.value,
            "status":         self.status.value,
            "submitted_at_ns": self.submitted_at_ns,
            "confirmed_at_ns": self.confirmed_at_ns,
            "tx_id":          self.tx_id,
            "block_number":   self.block_number,
            "block_hash":     self.block_hash,
            "proof_bytes":    self.proof_bytes,
            "error":          self.error,
            "retry_count":    self.retry_count,
            "metadata":       self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorRecord":
        r = cls(
            anchor_id=d["anchor_id"],
            forest_root=d["forest_root"],
            network=AnchorNetwork(d["network"]),
        )
        r.status = AnchorStatus(d["status"])
        r.submitted_at_ns = d.get("submitted_at_ns", 0)
        r.confirmed_at_ns = d.get("confirmed_at_ns")
        r.tx_id = d.get("tx_id")
        r.block_number = d.get("block_number")
        r.block_hash = d.get("block_hash")
        r.proof_bytes = d.get("proof_bytes")
        r.error = d.get("error")
        r.retry_count = d.get("retry_count", 0)
        r.metadata = d.get("metadata", {})
        return r


# ──────────────────────────────────────────────
# Base anchor backend interface
# ──────────────────────────────────────────────

class AnchorBackend:
    """
    Abstract base class for all anchoring backends.
    Subclasses implement submit() and check_status().
    """

    @property
    def network(self) -> AnchorNetwork:
        raise NotImplementedError

    def submit(self, forest_root: str, metadata: dict) -> AnchorRecord:
        """
        Submit a forest root for anchoring.
        Returns an AnchorRecord (may be PENDING or CONFIRMED immediately).
        """
        raise NotImplementedError

    def check_status(self, record: AnchorRecord) -> AnchorRecord:
        """
        Check and update the status of a pending anchor.
        Returns the updated record.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Returns True if this backend is reachable/configured."""
        return True


# ──────────────────────────────────────────────
# Anchor Manager
# ──────────────────────────────────────────────

class AnchorManager:
    """
    Orchestrates anchoring across multiple backends.

    Usage:
        manager = AnchorManager()
        manager.register_backend(LocalTimestampAnchor())
        manager.register_backend(OpenTimestampsAnchor())

        # After sealing a block:
        records = manager.anchor(forest_root, metadata={"block_hash": bh})

        # Later, check pending anchors:
        manager.poll_pending()
    """

    def __init__(self, store_dir: Optional[str | Path] = None):
        self._backends: dict[AnchorNetwork, AnchorBackend] = {}
        self._records: dict[str, AnchorRecord] = {}         # anchor_id → record
        self._root_index: dict[str, list[str]] = {}         # forest_root → [anchor_ids]
        self._store_dir = Path(store_dir) if store_dir else None
        if self._store_dir:
            self._store_dir.mkdir(parents=True, exist_ok=True)

    def register_backend(self, backend: AnchorBackend) -> None:
        self._backends[backend.network] = backend
        logger.info("Anchor backend registered: %s", backend.network.value)

    def anchor(
        self,
        forest_root: str,
        metadata: Optional[dict] = None,
        networks: Optional[list[AnchorNetwork]] = None,
    ) -> list[AnchorRecord]:
        """
        Anchor a forest root to all registered backends (or specified networks).
        Returns list of AnchorRecords, one per backend.
        """
        metadata = metadata or {}
        targets = networks or list(self._backends.keys())
        results = []

        for network in targets:
            backend = self._backends.get(network)
            if not backend:
                logger.warning("No backend registered for network: %s", network)
                continue
            if not backend.is_available():
                logger.warning("Backend unavailable: %s", network.value)
                continue

            try:
                record = backend.submit(forest_root, metadata)
                self._store_record(record)
                results.append(record)
                logger.info(
                    "Anchored forest_root=%s... via %s → anchor_id=%s status=%s",
                    forest_root[:16], network.value,
                    record.anchor_id[:8], record.status.value
                )
            except Exception as exc:
                logger.error("Anchor submission failed for %s: %s", network.value, exc)
                # Create a failed record so we have audit trail
                failed = AnchorRecord(
                    anchor_id=str(uuid.uuid4()),
                    forest_root=forest_root,
                    network=network,
                    metadata=metadata,
                )
                failed.mark_failed(str(exc))
                self._store_record(failed)
                results.append(failed)

        return results

    def poll_pending(self) -> list[AnchorRecord]:
        """Check status of all PENDING and FAILED (retryable) anchors."""
        updated = []
        for record in list(self._records.values()):
            if record.status in (AnchorStatus.PENDING, AnchorStatus.FAILED):
                if record.retry_count >= record.max_retries:
                    continue
                backend = self._backends.get(record.network)
                if backend:
                    try:
                        updated_record = backend.check_status(record)
                        self._store_record(updated_record)
                        updated.append(updated_record)
                    except Exception as exc:
                        record.mark_failed(str(exc))
        return updated

    def get_records_for_root(self, forest_root: str) -> list[AnchorRecord]:
        ids = self._root_index.get(forest_root, [])
        return [self._records[aid] for aid in ids if aid in self._records]

    def get_confirmed_for_root(self, forest_root: str) -> list[AnchorRecord]:
        return [r for r in self.get_records_for_root(forest_root)
                if r.status == AnchorStatus.CONFIRMED]

    def all_records(self) -> list[AnchorRecord]:
        return list(self._records.values())

    def _store_record(self, record: AnchorRecord) -> None:
        self._records[record.anchor_id] = record
        root = record.forest_root
        if root not in self._root_index:
            self._root_index[root] = []
        if record.anchor_id not in self._root_index[root]:
            self._root_index[root].append(record.anchor_id)
        if self._store_dir:
            self._persist(record)

    def _persist(self, record: AnchorRecord) -> None:
        path = self._store_dir / f"anchor_{record.anchor_id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
            tmp.rename(path)
        except Exception as exc:
            logger.error("Failed to persist anchor record: %s", exc)

    def load_from_disk(self) -> int:
        if not self._store_dir:
            return 0
        loaded = 0
        for path in self._store_dir.glob("anchor_*.json"):
            try:
                with open(path) as f:
                    record = AnchorRecord.from_dict(json.load(f))
                self._store_record(record)
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load anchor record %s: %s", path.name, exc)
        logger.info("Loaded %d anchor records from disk", loaded)
        return loaded

    def describe(self) -> dict:
        by_status = {}
        by_network = {}
        for r in self._records.values():
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
            by_network[r.network.value] = by_network.get(r.network.value, 0) + 1
        return {
            "total_records":     len(self._records),
            "registered_backends": [n.value for n in self._backends],
            "by_status":         by_status,
            "by_network":        by_network,
        }