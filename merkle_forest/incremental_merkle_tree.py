from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from merkle_forest.merkle_math import (
    hash_leaf,
    hash_node,
    empty_hash,
    compute_root,
    compute_proof_path,
    verify_proof,
    tree_height,
)

logger = logging.getLogger(__name__)

MAX_HEIGHT = 48


@dataclass(frozen=True)
class AppendResult:
    leaf_hash: str
    leaf_index: int
    new_root: str


class IncrementalMerkleTree:

    def __init__(self):
        self._frontier: list[Optional[str]] = [None] * MAX_HEIGHT
        self._leaf_hashes: list[str] = []
        self._leaf_count: int = 0
        self._root_cache: Optional[str] = None

    def append_raw_hash(self, leaf_hash: str) -> AppendResult:
        index = self._leaf_count
        self._leaf_hashes.append(leaf_hash)
        self._leaf_count += 1
        self._root_cache = None

        current = leaf_hash
        height = 0

        while height < MAX_HEIGHT:
            if self._frontier[height] is None:
                self._frontier[height] = current
                break
            else:
                left = self._frontier[height]
                current = hash_node(left, current)
                self._frontier[height] = None
                height += 1

        new_root = self.root()

        return AppendResult(
            leaf_hash=leaf_hash,
            leaf_index=index,
            new_root=new_root,
        )

    def append(self, data: bytes | str) -> AppendResult:
        leaf_hash = hash_leaf(data)
        return self.append_raw_hash(leaf_hash)

    def root(self) -> str:
        if self._root_cache is not None:
            return self._root_cache

        if self._leaf_count == 0:
            self._root_cache = empty_hash()
            return self._root_cache

        accumulated: Optional[str] = None

        for height in range(MAX_HEIGHT):
            node = self._frontier[height]

            if node is not None:
                if accumulated is None:
                    accumulated = node
                else:
                    accumulated = hash_node(node, accumulated)

        self._root_cache = accumulated or empty_hash()
        return self._root_cache

    def prove_inclusion(self, leaf_index: int) -> Optional[list[tuple[str, str]]]:
        if leaf_index < 0 or leaf_index >= self._leaf_count:
            return None

        return compute_proof_path(self._leaf_hashes, leaf_index)

    def batch_root(self) -> str:
        if not self._leaf_hashes:
            return empty_hash()

        return compute_root(self._leaf_hashes)

    def verify_inclusion(
        self,
        leaf_hash: str,
        leaf_index: int,
        proof: list[tuple[str, str]],
    ) -> bool:
        return verify_proof(leaf_hash, proof, self.batch_root())

    @property
    def leaf_count(self) -> int:
        return self._leaf_count

    def get_leaf_hash(self, index: int) -> Optional[str]:
        if 0 <= index < self._leaf_count:
            return self._leaf_hashes[index]

        return None

    def get_all_leaf_hashes(self) -> list[str]:
        return list(self._leaf_hashes)

    def frontier_snapshot(self) -> list[Optional[str]]:
        return list(self._frontier)

    def height(self) -> int:
        return tree_height(self._leaf_count)

    @classmethod
    def restore(
        cls,
        leaf_hashes: list[str],
        frontier: Optional[list[Optional[str]]] = None,
    ) -> "IncrementalMerkleTree":

        tree = cls()

        if frontier is not None and len(frontier) == MAX_HEIGHT:
            tree._frontier = list(frontier)
            tree._leaf_hashes = list(leaf_hashes)
            tree._leaf_count = len(leaf_hashes)

            logger.debug("Tree restored from snapshot: %d leaves", tree._leaf_count)

        else:
            logger.debug("Tree rebuilding from %d leaf hashes...", len(leaf_hashes))

            for lh in leaf_hashes:
                tree.append_raw_hash(lh)

        return tree

    def describe(self) -> dict:
        return {
            "leaf_count": self._leaf_count,
            "height": self.height(),
            "root": self.root(),
            "frontier_slots": sum(1 for f in self._frontier if f is not None),
        }