"""
ledger_core/ledger_state.py

Ledger State Manager for VFEL.

Responsibilities:
- Track the global ledger epoch and configuration
- Checkpoint sequence state to disk (JSON) for crash recovery
- Track per-shard Merkle root hashes (updated by Phase 3)
- Track anchoring state (updated by Phase 7)
- Provide a single consistent view of "what is the current state of the ledger"

Design:
- Lightweight — this is metadata, not event data
- Crash-safe: write to temp file, then atomic rename
- Human-readable JSON: ops teams can inspect/debug without tooling
- Epoch concept: changing shard count or crypto config bumps the epoch.
  Different epochs cannot share sequence spaces.

File layout (when persisted):
    ledger_state/
        state.json          ← atomic write target
        state.json.tmp      ← temp file (rename → state.json)
        checkpoints/        ← historical checkpoints
            cp_000001.json
            cp_000002.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ledger_core.sequence_manager import SequenceManager, SequenceSnapshot

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Ledger Configuration (immutable per epoch)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class LedgerConfig:
    """
    Immutable configuration for a ledger epoch.
    Changing any of these = new epoch = new ledger instance.
    """
    num_shards: int = 16
    hash_algorithm: str = "sha256"       # Phase 5 upgrades to blake3
    merkle_tree_arity: int = 2           # Binary Merkle tree (Phase 3)
    version: str = "vfel-0.2.0"

    def to_dict(self) -> dict:
        return {
            "num_shards":        self.num_shards,
            "hash_algorithm":    self.hash_algorithm,
            "merkle_tree_arity": self.merkle_tree_arity,
            "version":           self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerConfig":
        return cls(
            num_shards=d.get("num_shards", 16),
            hash_algorithm=d.get("hash_algorithm", "sha256"),
            merkle_tree_arity=d.get("merkle_tree_arity", 2),
            version=d.get("version", "vfel-0.2.0"),
        )


# ──────────────────────────────────────────────
# Ledger State (mutable, persisted)
# ──────────────────────────────────────────────

@dataclass
class LedgerState:
    """
    Current state of the VFEL ledger.

    Updated atomically on:
    - Checkpoint (periodic or on flush)
    - Merkle root update (Phase 3)
    - Anchor event (Phase 7)
    - Epoch change (config update)
    """

    # Identity
    epoch: int = 0
    created_at_ns: int = field(default_factory=time.time_ns)
    last_updated_ns: int = field(default_factory=time.time_ns)

    # Config
    config: LedgerConfig = field(default_factory=LedgerConfig)

    # Sequence state (snapshot from SequenceManager)
    sequence_snapshot: SequenceSnapshot = field(default_factory=SequenceSnapshot.empty)

    # Per-shard Merkle roots (updated by Phase 3)
    # shard_key → hex root hash
    shard_merkle_roots: dict[str, str] = field(default_factory=dict)

    # Forest root (global Merkle root across all shard trees — Phase 3)
    forest_root: Optional[str] = None

    # Last anchor (Phase 7)
    last_anchor_hash: Optional[str] = None
    last_anchor_at_ns: Optional[int] = None
    last_anchor_network: Optional[str] = None

    # Stats
    total_events_stored: int = 0
    total_checkpoints: int = 0

    def update_sequence(self, seq_manager: SequenceManager) -> None:
        """Sync sequence state from a live SequenceManager."""
        self.sequence_snapshot = seq_manager.snapshot()
        self.total_events_stored = seq_manager.total_assigned()
        self.last_updated_ns = time.time_ns()

    def update_shard_merkle_root(self, shard_key: str, root_hash: str) -> None:
        """Called by Phase 3 after each Merkle tree update."""
        self.shard_merkle_roots[shard_key] = root_hash
        self.last_updated_ns = time.time_ns()

    def update_forest_root(self, forest_root: str) -> None:
        """Called by Phase 3 after forest root recomputation."""
        self.forest_root = forest_root
        self.last_updated_ns = time.time_ns()

    def record_anchor(self, anchor_hash: str, network: str) -> None:
        """Called by Phase 7 after external anchoring."""
        self.last_anchor_hash = anchor_hash
        self.last_anchor_at_ns = time.time_ns()
        self.last_anchor_network = network
        self.last_updated_ns = time.time_ns()

    def to_dict(self) -> dict:
        return {
            "epoch":               self.epoch,
            "created_at_ns":       self.created_at_ns,
            "last_updated_ns":     self.last_updated_ns,
            "config":              self.config.to_dict(),
            "sequence_snapshot":   self.sequence_snapshot.to_dict(),
            "shard_merkle_roots":  self.shard_merkle_roots,
            "forest_root":         self.forest_root,
            "last_anchor_hash":    self.last_anchor_hash,
            "last_anchor_at_ns":   self.last_anchor_at_ns,
            "last_anchor_network": self.last_anchor_network,
            "total_events_stored": self.total_events_stored,
            "total_checkpoints":   self.total_checkpoints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerState":
        state = cls()
        state.epoch = d.get("epoch", 0)
        state.created_at_ns = d.get("created_at_ns", time.time_ns())
        state.last_updated_ns = d.get("last_updated_ns", time.time_ns())
        state.config = LedgerConfig.from_dict(d.get("config", {}))
        state.sequence_snapshot = SequenceSnapshot.from_dict(d.get("sequence_snapshot", {}))
        state.shard_merkle_roots = d.get("shard_merkle_roots", {})
        state.forest_root = d.get("forest_root")
        state.last_anchor_hash = d.get("last_anchor_hash")
        state.last_anchor_at_ns = d.get("last_anchor_at_ns")
        state.last_anchor_network = d.get("last_anchor_network")
        state.total_events_stored = d.get("total_events_stored", 0)
        state.total_checkpoints = d.get("total_checkpoints", 0)
        return state

    @classmethod
    def new(cls, config: Optional[LedgerConfig] = None) -> "LedgerState":
        """Create a fresh ledger state for a new epoch."""
        state = cls()
        state.config = config or LedgerConfig()
        state.epoch = 0
        logger.info("New LedgerState created: epoch=0, config=%s", state.config.to_dict())
        return state


# ──────────────────────────────────────────────
# State Manager (persistence layer)
# ──────────────────────────────────────────────

class StateManager:
    """
    Manages persistence of LedgerState to disk.

    Atomic writes: temp file → rename. Safe against power loss / SIGKILL.
    Checkpoints: keeps last N checkpoint files for point-in-time recovery.
    """

    MAX_CHECKPOINTS = 50

    def __init__(self, state_dir: str | Path = "ledger_state"):
        self.state_dir = Path(state_dir)
        self.checkpoint_dir = self.state_dir / "checkpoints"
        self.state_file = self.state_dir / "state.json"
        self.tmp_file = self.state_dir / "state.json.tmp"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_or_new(self, config: Optional[LedgerConfig] = None) -> LedgerState:
        """Load existing state from disk, or create fresh if none exists."""
        if self.state_file.exists():
            try:
                state = self._load(self.state_file)
                logger.info(
                    "LedgerState loaded: epoch=%d, %d events",
                    state.epoch, state.total_events_stored
                )
                return state
            except Exception as exc:
                logger.error("Failed to load state file: %s — starting fresh", exc)

        return LedgerState.new(config=config)

    def save(self, state: LedgerState) -> None:
        """Atomically save state to disk."""
        state.last_updated_ns = time.time_ns()
        self._write_atomic(self.state_file, state.to_dict())
        logger.debug("LedgerState saved: epoch=%d", state.epoch)

    def checkpoint(self, state: LedgerState, seq_manager: SequenceManager) -> str:
        """
        Save a named checkpoint of current state.
        Returns the checkpoint file path.
        """
        state.update_sequence(seq_manager)
        state.total_checkpoints += 1

        cp_name = f"cp_{state.total_checkpoints:06d}.json"
        cp_path = self.checkpoint_dir / cp_name
        self._write_atomic(cp_path, state.to_dict())

        # Also update the live state file
        self.save(state)

        # Prune old checkpoints
        self._prune_checkpoints()

        logger.info("Checkpoint saved: %s (events=%d)", cp_name, state.total_events_stored)
        return str(cp_path)

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint files, sorted newest first."""
        checkpoints = sorted(self.checkpoint_dir.glob("cp_*.json"), reverse=True)
        return [str(p) for p in checkpoints]

    def load_checkpoint(self, checkpoint_path: str) -> LedgerState:
        """Load a specific checkpoint by path."""
        return self._load(Path(checkpoint_path))

    def _load(self, path: Path) -> LedgerState:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LedgerState.from_dict(data)

    def _write_atomic(self, target: Path, data: dict) -> None:
        """Write JSON to temp file, then atomically rename to target."""
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        shutil.move(str(tmp), str(target))

    def _prune_checkpoints(self) -> None:
        checkpoints = sorted(self.checkpoint_dir.glob("cp_*.json"))
        while len(checkpoints) > self.MAX_CHECKPOINTS:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug("Pruned old checkpoint: %s", oldest.name)