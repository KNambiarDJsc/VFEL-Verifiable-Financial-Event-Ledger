from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from block_dag.block_model import Block

logger = logging.getLogger(__name__)


@dataclass
class OrderingResult:
    ordered_hashes: list[str]
    total_blocks: int
    cycle_detected: bool = False
    cycle_info: Optional[str] = None

    @property
    def success(self) -> bool:
        return not self.cycle_detected and len(self.ordered_hashes) == self.total_blocks


class DeterministicOrdering:

    def compute(self, blocks: dict[str, Block]) -> OrderingResult:
        if not blocks:
            return OrderingResult(ordered_hashes=[], total_blocks=0)

        in_degree: dict[str, int] = {bh: 0 for bh in blocks}
        children: dict[str, list[str]] = defaultdict(list)

        for block_hash, block in blocks.items():
            for parent_hash in block.header.parent_hashes:
                if parent_hash in blocks:
                    in_degree[block_hash] += 1
                    children[parent_hash].append(block_hash)

        ready: list[str] = [bh for bh, deg in in_degree.items() if deg == 0]
        ordered: list[str] = []

        while ready:
            ready.sort(key=lambda bh: self._sort_key(blocks[bh]))
            chosen = ready.pop(0)
            ordered.append(chosen)

            for child_hash in children[chosen]:
                in_degree[child_hash] -= 1
                if in_degree[child_hash] == 0:
                    ready.append(child_hash)

        if len(ordered) != len(blocks):
            unprocessed = [bh for bh in blocks if bh not in set(ordered)]
            cycle_info = f"Cycle detected involving {len(unprocessed)} blocks: {[h[:12] for h in unprocessed[:5]]}"
            logger.error(cycle_info)
            return OrderingResult(
                ordered_hashes=ordered,
                total_blocks=len(blocks),
                cycle_detected=True,
                cycle_info=cycle_info,
            )

        logger.debug("Ordering complete: %d blocks", len(ordered))
        return OrderingResult(ordered_hashes=ordered, total_blocks=len(blocks))

    def _sort_key(self, block: Block) -> tuple:
        min_ts = block.header.event_range.global_seq_start
        return (
            block.header.block_height,
            min_ts,
            block.header.block_hash,
        )

    def verify_order(
        self,
        ordered_hashes: list[str],
        blocks: dict[str, Block],
    ) -> tuple[bool, Optional[str]]:

        position: dict[str, int] = {bh: i for i, bh in enumerate(ordered_hashes)}

        for block_hash in ordered_hashes:
            block = blocks.get(block_hash)
            if not block:
                return False, f"Unknown block in ordering: {block_hash[:16]}"

            for parent_hash in block.header.parent_hashes:
                if parent_hash not in position:
                    continue
                if position[parent_hash] >= position[block_hash]:
                    return False, (
                        f"Invalid order: block {block_hash[:12]} at pos {position[block_hash]} "
                        f"but parent {parent_hash[:12]} at pos {position[parent_hash]}"
                    )

        return True, None


class IncrementalOrdering:

    def __init__(self):
        self._ordered: list[str] = []
        self._position: dict[str, int] = {}
        self._algo = DeterministicOrdering()

    def rebuild(self, blocks: dict[str, Block]) -> OrderingResult:
        result = self._algo.compute(blocks)
        if result.success:
            self._ordered = list(result.ordered_hashes)
            self._position = {bh: i for i, bh in enumerate(self._ordered)}
        return result

    def get_ordered(self) -> list[str]:
        return list(self._ordered)

    def get_position(self, block_hash: str) -> Optional[int]:
        return self._position.get(block_hash)

    def total_ordered(self) -> int:
        return len(self._ordered)