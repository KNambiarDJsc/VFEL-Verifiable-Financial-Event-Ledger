from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from block_dag.block_model import Block, BlockStatus

logger = logging.getLogger(__name__)


class BlockStore:

    def __init__(self, store_dir: Optional[str | Path] = None):
        self._blocks: dict[str, Block] = {}
        self._id_index: dict[str, str] = {}
        self._height_index: dict[int, list[str]] = {}
        self._tips: set[str] = set()
        self._children: dict[str, set[str]] = {}

        self._store_dir = Path(store_dir) if store_dir else None
        if self._store_dir:
            (self._store_dir / "blocks").mkdir(parents=True, exist_ok=True)

    def put(self, block: Block) -> bool:
        if not block.is_sealed:
            raise ValueError(f"Cannot store unsealed block: {block.block_id}")

        if block.block_hash in self._blocks:
            logger.debug("Block already stored: %s", block.block_hash[:16])
            return False

        self._blocks[block.block_hash] = block
        self._id_index[block.block_id] = block.block_hash

        h = block.height
        if h not in self._height_index:
            self._height_index[h] = []
        self._height_index[h].append(block.block_hash)

        self._tips.add(block.block_hash)
        self._children[block.block_hash] = set()

        for parent_hash in block.header.parent_hashes:
            self._tips.discard(parent_hash)
            if parent_hash not in self._children:
                self._children[parent_hash] = set()
            self._children[parent_hash].add(block.block_hash)

        if self._store_dir:
            self._persist_block(block)

        logger.debug(
            "Block stored: h=%d hash=%s parents=%d",
            block.height, block.block_hash[:16], len(block.header.parent_hashes)
        )
        return True

    def update_status(self, block_hash: str, new_status: BlockStatus) -> bool:
        block = self._blocks.get(block_hash)
        if not block:
            return False
        block.header.status = new_status
        if self._store_dir:
            self._persist_block(block)
        return True

    def get(self, block_hash: str) -> Optional[Block]:
        return self._blocks.get(block_hash)

    def get_by_id(self, block_id: str) -> Optional[Block]:
        bh = self._id_index.get(block_id)
        return self._blocks.get(bh) if bh else None

    def get_at_height(self, height: int) -> list[Block]:
        hashes = self._height_index.get(height, [])
        return [self._blocks[bh] for bh in hashes]

    def get_children(self, block_hash: str) -> list[Block]:
        child_hashes = self._children.get(block_hash, set())
        return [self._blocks[ch] for ch in child_hashes if ch in self._blocks]

    def get_parents(self, block_hash: str) -> list[Block]:
        block = self._blocks.get(block_hash)
        if not block:
            return []
        return [self._blocks[ph] for ph in block.header.parent_hashes if ph in self._blocks]

    def contains(self, block_hash: str) -> bool:
        return block_hash in self._blocks

    def tips(self) -> list[Block]:
        return [self._blocks[bh] for bh in sorted(self._tips) if bh in self._blocks]

    def tip_hashes(self) -> list[str]:
        return sorted(self._tips)

    def max_height(self) -> int:
        if not self._height_index:
            return -1
        return max(self._height_index.keys())

    def all_blocks(self) -> dict[str, Block]:
        return dict(self._blocks)

    def blocks_since_height(self, min_height: int) -> dict[str, Block]:
        result = {}
        for h, hashes in self._height_index.items():
            if h >= min_height:
                for bh in hashes:
                    result[bh] = self._blocks[bh]
        return result

    def verify_block(self, block_hash: str) -> tuple[bool, Optional[str]]:
        block = self._blocks.get(block_hash)
        if not block:
            return False, f"Block not found: {block_hash[:16]}"
        return block.verify_self()

    def verify_all(self) -> dict[str, tuple[bool, Optional[str]]]:
        return {bh: block.verify_self() for bh, block in self._blocks.items()}

    def _persist_block(self, block: Block) -> None:
        blocks_dir = self._store_dir / "blocks"
        path = blocks_dir / f"{block.block_hash}.json"
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(block.to_dict(), f, indent=2)
            tmp.rename(path)
        except Exception as exc:
            logger.error("Failed to persist block %s: %s", block.block_hash[:16], exc)

    def load_from_disk(self) -> int:
        if not self._store_dir:
            return 0

        blocks_dir = self._store_dir / "blocks"
        loaded = 0
        for path in sorted(blocks_dir.glob("*.json")):
            try:
                with open(path, "r") as f:
                    block = Block.from_dict(json.load(f))
                self.put(block)
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load block from %s: %s", path.name, exc)

        logger.info("Loaded %d blocks from disk", loaded)
        return loaded

    def describe(self) -> dict:
        tip_heights = [self._blocks[bh].height for bh in self._tips if bh in self._blocks]
        return {
            "total_blocks": len(self._blocks),
            "max_height": self.max_height(),
            "num_tips": len(self._tips),
            "tip_heights": sorted(tip_heights),
            "tip_hashes": [h[:16] + "..." for h in sorted(self._tips)],
        }