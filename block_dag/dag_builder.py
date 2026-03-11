from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from ledger_core.event_store import EventStore, StoredEvent
from merkle_forest.forest_manager import ForestManager, SealedForestRoot
from block_dag.block_model import (
    Block, BlockHeader, BlockStatus,
    EventRange, ShardSnapshot, make_genesis_block,
)
from block_dag.block_store import BlockStore
from block_dag.ordering_algorithm import DeterministicOrdering, IncrementalOrdering

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PER_BLOCK = 100
DEFAULT_MAX_BLOCK_INTERVAL_S = 5.0


@dataclass
class BlockProductionResult:
    block: Block
    parent_hashes: list[str]
    event_count: int
    forest_root: str
    trigger: str
    build_time_ms: float


class DAGBuilder:

    def __init__(
        self,
        event_store: EventStore,
        forest_manager: ForestManager,
        block_store: BlockStore,
        events_per_block: int = DEFAULT_EVENTS_PER_BLOCK,
        max_block_interval_s: float = DEFAULT_MAX_BLOCK_INTERVAL_S,
        on_block_sealed: Optional[Callable[[Block], None]] = None,
    ):
        self._event_store = event_store
        self._forest_manager = forest_manager
        self._block_store = block_store
        self._events_per_block = events_per_block
        self._max_block_interval_s = max_block_interval_s
        self._on_block_sealed = on_block_sealed

        self._pending_events: list[StoredEvent] = []
        self._last_seal_time: float = time.monotonic()

        self._ordering = IncrementalOrdering()

        self._blocks_produced: int = 0
        self._total_events_blocked: int = 0

    def initialize_genesis(self) -> Block:
        existing = self._block_store.get_by_id("GENESIS")
        if existing:
            logger.debug("Genesis block already exists: %s", existing.block_hash[:16])
            return existing

        forest_root = self._forest_manager.forest_root()
        genesis = make_genesis_block(forest_root=forest_root)
        self._block_store.put(genesis)
        self._ordering.rebuild(self._block_store.all_blocks())

        logger.info("Genesis block created: %s", genesis.block_hash[:16])
        return genesis

    def feed(self, stored_event: StoredEvent) -> Optional[BlockProductionResult]:
        self._pending_events.append(stored_event)

        if self._should_seal("events"):
            return self._seal_block(trigger="events")
        if self._should_seal("time"):
            return self._seal_block(trigger="time")
        return None

    def feed_batch(self, stored_events) -> list[BlockProductionResult]:
        results = []
        for event in stored_events:
            result = self.feed(event)
            if result:
                results.append(result)
        return results

    def flush(self) -> Optional[BlockProductionResult]:
        if not self._pending_events:
            return None
        return self._seal_block(trigger="manual")

    def _seal_block(self, trigger: str) -> BlockProductionResult:
        t0 = time.perf_counter()

        pending = list(self._pending_events)
        self._pending_events.clear()
        self._last_seal_time = time.monotonic()

        global_seqs = [e.ledger_sequence for e in pending]
        seq_start = min(global_seqs)
        seq_end = max(global_seqs) + 1
        event_count = len(pending)

        sealed_forest: SealedForestRoot = self._forest_manager.seal_forest_root(
            trigger=f"block_{trigger}"
        )

        shard_snapshots = [
            ShardSnapshot(
                shard_key=sk,
                shard_seq_end=sealed_forest.shard_leaf_counts.get(sk, 0) - 1,
                merkle_root=root,
                leaf_count=sealed_forest.shard_leaf_counts.get(sk, 0),
            )
            for sk, root in sorted(sealed_forest.shard_roots.items())
        ]

        parent_hashes = self._block_store.tip_hashes()
        if not parent_hashes:
            logger.warning("No tips found — block will be parentless (post-genesis)")

        parent_heights = [
            self._block_store.get(ph).height
            for ph in parent_hashes
            if self._block_store.contains(ph)
        ]
        block_height = (max(parent_heights) + 1) if parent_heights else 0

        header = BlockHeader(
            block_id=str(uuid.uuid4()),
            block_height=block_height,
            parent_hashes=parent_hashes,
            forest_root=sealed_forest.forest_root,
            shard_snapshots=shard_snapshots,
            event_range=EventRange(
                global_seq_start=seq_start,
                global_seq_end=seq_end,
                event_count=event_count,
            ),
        )

        event_hashes = [e.stored_event_hash for e in pending]

        block = Block(header=header, event_hashes=event_hashes)
        block.seal()

        self._block_store.put(block)
        self._ordering.rebuild(self._block_store.all_blocks())

        self._blocks_produced += 1
        self._total_events_blocked += event_count

        build_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Block sealed: h=%d hash=%s... events=%d parents=%d trigger=%s (%.1fms)",
            block_height,
            block.block_hash[:16],
            event_count,
            len(parent_hashes),
            trigger,
            build_time_ms,
        )

        result = BlockProductionResult(
            block=block,
            parent_hashes=parent_hashes,
            event_count=event_count,
            forest_root=sealed_forest.forest_root,
            trigger=trigger,
            build_time_ms=build_time_ms,
        )

        if self._on_block_sealed:
            self._on_block_sealed(block)

        return result

    def _should_seal(self, trigger: str) -> bool:
        if trigger == "events":
            return len(self._pending_events) >= self._events_per_block
        if trigger == "time":
            elapsed = time.monotonic() - self._last_seal_time
            return (
                elapsed >= self._max_block_interval_s
                and len(self._pending_events) > 0
            )
        return False

    def get_total_order(self) -> list[str]:
        return self._ordering.get_ordered()

    def get_block_position(self, block_hash: str) -> Optional[int]:
        return self._ordering.get_position(block_hash)

    def pending_count(self) -> int:
        return len(self._pending_events)

    def describe(self) -> dict:
        return {
            "blocks_produced": self._blocks_produced,
            "total_events_blocked": self._total_events_blocked,
            "pending_events": len(self._pending_events),
            "events_per_block": self._events_per_block,
            "dag_store": self._block_store.describe(),
            "total_order_length": self._ordering.total_ordered(),
        }