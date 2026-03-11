from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_HEIGHT = 48


@dataclass
class TreeSnapshot:
    shard_key: str
    leaf_count: int
    root_hash: str
    frontier: list[Optional[str]]
    leaf_hashes: list[str]
    snapshot_id: str
    captured_at_ns: int = field(default_factory=time.time_ns)
    trigger: str = "manual"

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "shard_key": self.shard_key,
            "leaf_count": self.leaf_count,
            "root_hash": self.root_hash,
            "frontier": self.frontier,
            "leaf_hashes": self.leaf_hashes,
            "captured_at_ns": self.captured_at_ns,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeSnapshot":
        return cls(
            snapshot_id=d["snapshot_id"],
            shard_key=d["shard_key"],
            leaf_count=d["leaf_count"],
            root_hash=d["root_hash"],
            frontier=d["frontier"],
            leaf_hashes=d["leaf_hashes"],
            captured_at_ns=d.get("captured_at_ns", 0),
            trigger=d.get("trigger", "manual"),
        )

    def size_bytes_estimate(self) -> int:
        return self.leaf_count * 32 + MAX_HEIGHT * 32


class SnapshotStore:

    def __init__(self, state_dir: Optional[str | Path] = None):
        self._index: dict[str, list[tuple[int, str]]] = {}
        self._snapshots: dict[str, TreeSnapshot] = {}

        self._state_dir = Path(state_dir) if state_dir else None
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: TreeSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot

        if snapshot.shard_key not in self._index:
            self._index[snapshot.shard_key] = []

        self._index[snapshot.shard_key].append((snapshot.leaf_count, snapshot.snapshot_id))
        self._index[snapshot.shard_key].sort()

        if self._state_dir:
            self._persist(snapshot)

        logger.debug(
            "Snapshot saved: shard=%s leaves=%d root=%s...",
            snapshot.shard_key, snapshot.leaf_count, snapshot.root_hash[:16]
        )

    def get_by_id(self, snapshot_id: str) -> Optional[TreeSnapshot]:
        return self._snapshots.get(snapshot_id)

    def get_at_leaf_count(
        self, shard_key: str, leaf_count: int
    ) -> Optional[TreeSnapshot]:
        entries = self._index.get(shard_key, [])
        for count, snap_id in entries:
            if count == leaf_count:
                return self._snapshots.get(snap_id)
        return None

    def get_latest(self, shard_key: str) -> Optional[TreeSnapshot]:
        entries = self._index.get(shard_key, [])
        if not entries:
            return None
        _, snap_id = entries[-1]
        return self._snapshots.get(snap_id)

    def get_nearest_before(self, shard_key: str, leaf_count: int) -> Optional[TreeSnapshot]:
        entries = self._index.get(shard_key, [])
        best = None
        for count, snap_id in entries:
            if count <= leaf_count:
                best = self._snapshots.get(snap_id)
            else:
                break
        return best

    def list_for_shard(self, shard_key: str) -> list[dict]:
        entries = self._index.get(shard_key, [])
        result = []
        for count, snap_id in entries:
            snap = self._snapshots.get(snap_id)
            if snap:
                result.append({
                    "snapshot_id": snap.snapshot_id,
                    "leaf_count": snap.leaf_count,
                    "root_hash": snap.root_hash,
                    "trigger": snap.trigger,
                    "captured_at_ns": snap.captured_at_ns,
                })
        return result

    def known_shards(self) -> list[str]:
        return list(self._index.keys())

    def total_snapshots(self) -> int:
        return len(self._snapshots)

    def _persist(self, snapshot: TreeSnapshot) -> None:
        safe_shard = snapshot.shard_key.replace("/", "_").replace(":", "_")
        filename = f"snap_{safe_shard}_{snapshot.leaf_count:012d}_{snapshot.snapshot_id[:8]}.json"
        path = self._state_dir / filename
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(snapshot.to_dict(), f)
            tmp.rename(path)
        except Exception as exc:
            logger.error("Failed to persist snapshot %s: %s", snapshot.snapshot_id, exc)

    def load_from_disk(self) -> int:
        if not self._state_dir:
            return 0

        loaded = 0
        for path in sorted(self._state_dir.glob("snap_*.json")):
            try:
                with open(path, "r") as f:
                    snap = TreeSnapshot.from_dict(json.load(f))
                self.save(snap)
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load snapshot %s: %s", path.name, exc)

        logger.info("Loaded %d snapshots from disk", loaded)
        return loaded

    def describe(self) -> dict:
        return {
            "total_snapshots": self.total_snapshots(),
            "shards_tracked": len(self._index),
            "per_shard": {k: len(v) for k, v in self._index.items()},
        }