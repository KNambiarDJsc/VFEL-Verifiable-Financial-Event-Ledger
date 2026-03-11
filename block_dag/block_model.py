from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockStatus(str, Enum):
    PENDING = "PENDING"
    SEALED = "SEALED"
    VERIFIED = "VERIFIED"
    ANCHORED = "ANCHORED"


@dataclass(frozen=True)
class EventRange:
    global_seq_start: int
    global_seq_end: int
    event_count: int

    @property
    def is_empty(self) -> bool:
        return self.event_count == 0

    def to_dict(self) -> dict:
        return {
            "global_seq_start": self.global_seq_start,
            "global_seq_end": self.global_seq_end,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventRange":
        return cls(
            global_seq_start=d["global_seq_start"],
            global_seq_end=d["global_seq_end"],
            event_count=d["event_count"],
        )

    @classmethod
    def empty(cls) -> "EventRange":
        return cls(global_seq_start=0, global_seq_end=0, event_count=0)


@dataclass(frozen=True)
class ShardSnapshot:
    shard_key: str
    shard_seq_end: int
    merkle_root: str
    leaf_count: int

    def to_dict(self) -> dict:
        return {
            "shard_key": self.shard_key,
            "shard_seq_end": self.shard_seq_end,
            "merkle_root": self.merkle_root,
            "leaf_count": self.leaf_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardSnapshot":
        return cls(
            shard_key=d["shard_key"],
            shard_seq_end=d["shard_seq_end"],
            merkle_root=d["merkle_root"],
            leaf_count=d["leaf_count"],
        )


@dataclass
class BlockHeader:
    block_id: str
    block_height: int
    parent_hashes: list[str]
    forest_root: str
    shard_snapshots: list[ShardSnapshot]
    event_range: EventRange
    sealed_at_ns: int = field(default_factory=time.time_ns)
    block_hash: str = ""
    status: BlockStatus = BlockStatus.PENDING

    def seal(self) -> str:
        if self.status == BlockStatus.SEALED:
            return self.block_hash
        self.block_hash = self._compute_hash()
        self.status = BlockStatus.SEALED
        return self.block_hash

    def _compute_hash(self) -> str:
        canonical = {
            "block_id": self.block_id,
            "block_height": self.block_height,
            "parent_hashes": sorted(self.parent_hashes),
            "forest_root": self.forest_root,
            "event_range": self.event_range.to_dict(),
            "shard_snapshots": sorted(
                [s.to_dict() for s in self.shard_snapshots],
                key=lambda x: x["shard_key"],
            ),
            "sealed_at_s": self.sealed_at_ns // 1_000_000_000,
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "block_height": self.block_height,
            "parent_hashes": self.parent_hashes,
            "forest_root": self.forest_root,
            "shard_snapshots": [s.to_dict() for s in self.shard_snapshots],
            "event_range": self.event_range.to_dict(),
            "sealed_at_ns": self.sealed_at_ns,
            "block_hash": self.block_hash,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockHeader":
        h = cls(
            block_id=d["block_id"],
            block_height=d["block_height"],
            parent_hashes=d["parent_hashes"],
            forest_root=d["forest_root"],
            shard_snapshots=[ShardSnapshot.from_dict(s) for s in d["shard_snapshots"]],
            event_range=EventRange.from_dict(d["event_range"]),
            sealed_at_ns=d["sealed_at_ns"],
        )
        h.block_hash = d.get("block_hash", "")
        h.status = BlockStatus(d.get("status", "PENDING"))
        return h


@dataclass
class Block:
    header: BlockHeader
    event_hashes: list[str]

    @property
    def block_hash(self) -> str:
        return self.header.block_hash

    @property
    def block_id(self) -> str:
        return self.header.block_id

    @property
    def height(self) -> int:
        return self.header.block_height

    @property
    def is_genesis(self) -> bool:
        return len(self.header.parent_hashes) == 0

    @property
    def is_sealed(self) -> bool:
        return self.header.status == BlockStatus.SEALED

    def seal(self) -> str:
        return self.header.seal()

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "event_hashes": self.event_hashes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        return cls(
            header=BlockHeader.from_dict(d["header"]),
            event_hashes=d.get("event_hashes", []),
        )

    def verify_self(self) -> tuple[bool, Optional[str]]:
        if not self.is_sealed:
            return False, "Block is not sealed"
        recomputed = self.header._compute_hash()
        if recomputed != self.block_hash:
            return False, f"Hash mismatch: stored={self.block_hash[:16]}, computed={recomputed[:16]}"
        return True, None


def make_genesis_block(forest_root: str, block_id: str = "GENESIS") -> Block:
    header = BlockHeader(
        block_id=block_id,
        block_height=0,
        parent_hashes=[],
        forest_root=forest_root,
        shard_snapshots=[],
        event_range=EventRange.empty(),
    )
    block = Block(header=header, event_hashes=[])
    block.seal()
    return block