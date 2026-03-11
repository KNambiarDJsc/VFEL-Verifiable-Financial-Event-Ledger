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
class FrontierState:
    shard_key: str
    leaf_count: int
    root_hash: str
    frontier: list[Optional[str]]
    captured_at_ns: int = field(default_factory=time.time_ns)
    last_leaf_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "shard_key": self.shard_key,
            "leaf_count": self.leaf_count,
            "root_hash": self.root_hash,
            "frontier": self.frontier,
            "captured_at_ns": self.captured_at_ns,
            "last_leaf_hash": self.last_leaf_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrontierState":
        return cls(
            shard_key=d["shard_key"],
            leaf_count=d["leaf_count"],
            root_hash=d["root_hash"],
            frontier=d["frontier"],
            captured_at_ns=d.get("captured_at_ns", 0),
            last_leaf_hash=d.get("last_leaf_hash"),
        )

    def is_valid(self) -> bool:
        if len(self.frontier) != MAX_HEIGHT:
            return False
        if self.leaf_count < 0:
            return False
        if not self.root_hash or len(self.root_hash) != 64:
            return False
        return True


class FrontierRegistry:

    def __init__(self, state_dir: Optional[str | Path] = None):
        self._frontiers: dict[str, FrontierState] = {}
        self._state_dir = Path(state_dir) if state_dir else None

        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        shard_key: str,
        leaf_count: int,
        root_hash: str,
        frontier: list[Optional[str]],
        last_leaf_hash: Optional[str] = None,
    ) -> FrontierState:
        state = FrontierState(
            shard_key=shard_key,
            leaf_count=leaf_count,
            root_hash=root_hash,
            frontier=list(frontier),
            last_leaf_hash=last_leaf_hash,
        )
        self._frontiers[shard_key] = state
        return state

    def get(self, shard_key: str) -> Optional[FrontierState]:
        return self._frontiers.get(shard_key)

    def all_shard_keys(self) -> list[str]:
        return list(self._frontiers.keys())

    def all_roots(self) -> dict[str, str]:
        return {k: v.root_hash for k, v in self._frontiers.items()}

    def flush_to_disk(self) -> int:
        if not self._state_dir:
            return 0

        written = 0
        for shard_key, state in self._frontiers.items():
            path = self._frontier_path(shard_key)
            try:
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(state.to_dict(), f)
                tmp.rename(path)
                written += 1
            except Exception as exc:
                logger.error("Failed to flush frontier for %s: %s", shard_key, exc)

        logger.debug("Flushed %d frontier states to disk", written)
        return written

    def load_from_disk(self) -> int:
        if not self._state_dir:
            return 0

        loaded = 0
        for path in self._state_dir.glob("frontier_*.json"):
            try:
                with open(path, "r") as f:
                    state = FrontierState.from_dict(json.load(f))
                if state.is_valid():
                    self._frontiers[state.shard_key] = state
                    loaded += 1
                else:
                    logger.warning("Invalid frontier state in %s — skipping", path.name)
            except Exception as exc:
                logger.error("Failed to load frontier from %s: %s", path.name, exc)

        logger.info("Loaded %d frontier states from disk", loaded)
        return loaded

    def _frontier_path(self, shard_key: str) -> Path:
        safe_key = shard_key.replace("/", "_").replace(":", "_")
        return self._state_dir / f"frontier_{safe_key}.json"

    def describe(self) -> dict:
        return {
            "num_shards": len(self._frontiers),
            "state_dir": str(self._state_dir) if self._state_dir else None,
            "shard_summary": {
                k: {
                    "leaf_count": v.leaf_count,
                    "root": v.root_hash[:16] + "...",
                }
                for k, v in self._frontiers.items()
            },
        }