from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from merkle_forest.incremental_merkle_tree import IncrementalMerkleTree
from merkle_forest.frontier_state import FrontierRegistry
from merkle_forest.tree_snapshot import TreeSnapshot, SnapshotStore
from merkle_forest.merkle_math import compute_root, hash_leaf, empty_hash

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_INTERVAL = 10_000


@dataclass(frozen=True)
class ForestAppendResult:
    shard_key: str
    leaf_hash: str
    leaf_index: int
    shard_root: str
    forest_root: str
    snapshot_taken: bool = False


@dataclass(frozen=True)
class SealedForestRoot:
    forest_root: str
    shard_roots: dict[str, str]
    shard_leaf_counts: dict[str, int]
    sealed_at_ns: int
    seal_id: str


class ForestManager:

    def __init__(
        self,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        frontier_registry: Optional[FrontierRegistry] = None,
        snapshot_store: Optional[SnapshotStore] = None,
    ):
        self._snapshot_interval = snapshot_interval
        self._trees: dict[str, IncrementalMerkleTree] = {}
        self._frontier_registry = frontier_registry or FrontierRegistry()
        self._snapshot_store = snapshot_store or SnapshotStore()

        self._forest_root_cache: Optional[str] = None

        self._total_appended: int = 0
        self._total_snapshots: int = 0

    def append(self, shard_key: str, event_hash: str) -> ForestAppendResult:
        tree = self._get_or_create_tree(shard_key)
        result = tree.append(event_hash)

        self._forest_root_cache = None
        self._total_appended += 1

        self._frontier_registry.update(
            shard_key=shard_key,
            leaf_count=tree.leaf_count,
            root_hash=tree.root(),
            frontier=tree.frontier_snapshot(),
            last_leaf_hash=result.leaf_hash,
        )

        snapshot_taken = False
        if tree.leaf_count % self._snapshot_interval == 0:
            self._take_snapshot(shard_key, tree, trigger="periodic")
            snapshot_taken = True

        new_forest_root = self.forest_root()

        return ForestAppendResult(
            shard_key=shard_key,
            leaf_hash=result.leaf_hash,
            leaf_index=result.leaf_index,
            shard_root=tree.root(),
            forest_root=new_forest_root,
            snapshot_taken=snapshot_taken,
        )

    def append_batch(
        self, events: list[tuple[str, str]]
    ) -> list[ForestAppendResult]:

        results = []

        for shard_key, event_hash in events:
            tree = self._get_or_create_tree(shard_key)
            tree_result = tree.append(event_hash)

            self._frontier_registry.update(
                shard_key=shard_key,
                leaf_count=tree.leaf_count,
                root_hash=tree.root(),
                frontier=tree.frontier_snapshot(),
                last_leaf_hash=tree_result.leaf_hash,
            )

            if tree.leaf_count % self._snapshot_interval == 0:
                self._take_snapshot(shard_key, tree, trigger="periodic")

            results.append(ForestAppendResult(
                shard_key=shard_key,
                leaf_hash=tree_result.leaf_hash,
                leaf_index=tree_result.leaf_index,
                shard_root=tree.root(),
                forest_root="",
                snapshot_taken=(tree.leaf_count % self._snapshot_interval == 0),
            ))

        self._forest_root_cache = None
        fr = self.forest_root()

        results = [
            ForestAppendResult(
                shard_key=r.shard_key,
                leaf_hash=r.leaf_hash,
                leaf_index=r.leaf_index,
                shard_root=r.shard_root,
                forest_root=fr,
                snapshot_taken=r.snapshot_taken,
            )
            for r in results
        ]

        self._total_appended += len(events)
        return results

    def forest_root(self) -> str:

        if self._forest_root_cache is not None:
            return self._forest_root_cache

        if not self._trees:
            self._forest_root_cache = empty_hash()
            return self._forest_root_cache

        shard_entries = sorted(
            (shard_key, tree.root())
            for shard_key, tree in self._trees.items()
        )

        forest_leaves = [
            hash_leaf(f"{shard_key}:{shard_root}")
            for shard_key, shard_root in shard_entries
        ]

        self._forest_root_cache = compute_root(forest_leaves)
        return self._forest_root_cache

    def shard_roots(self) -> dict[str, str]:
        return {k: v.root() for k, v in self._trees.items()}

    def shard_root(self, shard_key: str) -> Optional[str]:
        tree = self._trees.get(shard_key)
        return tree.root() if tree else None

    def seal_forest_root(self, trigger: str = "pre_block") -> SealedForestRoot:

        shard_roots = {}
        shard_leaf_counts = {}

        for shard_key, tree in self._trees.items():
            self._take_snapshot(shard_key, tree, trigger=trigger)
            shard_roots[shard_key] = tree.root()
            shard_leaf_counts[shard_key] = tree.leaf_count

        sealed = SealedForestRoot(
            forest_root=self.forest_root(),
            shard_roots=shard_roots,
            shard_leaf_counts=shard_leaf_counts,
            sealed_at_ns=time.time_ns(),
            seal_id=str(uuid.uuid4()),
        )

        logger.info(
            "Forest sealed: %d shards, root=%s..., seal_id=%s",
            len(shard_roots), sealed.forest_root[:16], sealed.seal_id
        )

        return sealed

    def get_tree(self, shard_key: str) -> Optional[IncrementalMerkleTree]:
        return self._trees.get(shard_key)

    def known_shards(self) -> list[str]:
        return list(self._trees.keys())

    def total_leaves(self) -> int:
        return sum(t.leaf_count for t in self._trees.values())

    def describe(self) -> dict:
        return {
            "num_shards": len(self._trees),
            "total_leaves": self.total_leaves(),
            "total_appended": self._total_appended,
            "total_snapshots": self._total_snapshots,
            "forest_root": self.forest_root(),
            "shard_summary": {
                k: t.describe()
                for k, t in sorted(self._trees.items())
            },
        }

    def _get_or_create_tree(self, shard_key: str) -> IncrementalMerkleTree:
        if shard_key not in self._trees:
            self._trees[shard_key] = IncrementalMerkleTree()
            logger.debug("New Merkle tree created for shard: %s", shard_key)
        return self._trees[shard_key]

    def _take_snapshot(
        self,
        shard_key: str,
        tree: IncrementalMerkleTree,
        trigger: str = "manual",
    ) -> TreeSnapshot:

        snapshot = TreeSnapshot(
            snapshot_id=str(uuid.uuid4()),
            shard_key=shard_key,
            leaf_count=tree.leaf_count,
            root_hash=tree.root(),
            frontier=tree.frontier_snapshot(),
            leaf_hashes=tree.get_all_leaf_hashes(),
            trigger=trigger,
        )

        self._snapshot_store.save(snapshot)

        self._total_snapshots += 1

        return snapshot