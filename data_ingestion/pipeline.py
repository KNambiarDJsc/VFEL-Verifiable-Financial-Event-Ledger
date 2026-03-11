from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from data_ingestion.jsonl_loader import JSONLLoader, MultiFileLoader
from data_ingestion.event_parser import EventParser
from data_ingestion.event_normalizer import EventNormalizer
from data_ingestion.shard_router import ShardRouter, ShardedEvent, ShardAffinityMap

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    source_files: list[str]
    num_shards: int
    duration_seconds: float

    total_lines: int = 0
    parse_errors: int = 0
    normalize_errors: int = 0
    events_routed: int = 0

    shard_distribution: dict = field(default_factory=dict)
    top_symbols: list = field(default_factory=list)

    @property
    def throughput_eps(self) -> float:
        """Events per second."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.events_routed / self.duration_seconds

    def summary(self) -> str:
        return (
            f"VFEL Ingestion Report\n"
            f"  Sources     : {', '.join(self.source_files)}\n"
            f"  Shards      : {self.num_shards}\n"
            f"  Total lines : {self.total_lines}\n"
            f"  Routed      : {self.events_routed}\n"
            f"  Parse errors: {self.parse_errors}\n"
            f"  Duration    : {self.duration_seconds:.3f}s\n"
            f"  Throughput  : {self.throughput_eps:.0f} events/sec\n"
            f"  Top symbols : {self.top_symbols[:5]}\n"
        )


class IngestionPipeline:

    def __init__(
        self,
        file_paths: str | Path | list[str | Path],
        num_shards: int = 16,
    ):
        # Normalize to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        self._file_paths = [Path(p) for p in file_paths]
        self._num_shards = num_shards

        # Pipeline components
        self._loader = MultiFileLoader(self._file_paths)
        self._parser = EventParser()
        self._normalizer = EventNormalizer()
        self._router = ShardRouter(num_shards=num_shards)
        self._affinity_map = ShardAffinityMap(num_shards=num_shards)

        # Runtime stats
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._events_routed: int = 0

    def run(self) -> Generator[ShardedEvent, None, None]:
        self._start_time = time.perf_counter()
        logger.info(
            "Starting ingestion pipeline: %d files, %d shards",
            len(self._file_paths), self._num_shards,
        )

        # Compose the pipeline as a lazy generator chain
        raw_stream = self._loader.stream()
        parsed_stream = self._parser.parse_stream(raw_stream)
        normalized_stream = self._normalizer.normalize_stream(parsed_stream)
        routed_stream = self._router.route_stream(normalized_stream)

        for sharded_event in routed_stream:
            self._affinity_map.record(sharded_event)
            self._events_routed += 1

            if self._events_routed % 10_000 == 0:
                logger.info("Ingested %d events...", self._events_routed)

            yield sharded_event

        self._end_time = time.perf_counter()
        logger.info(
            "Pipeline complete: %d events in %.3fs",
            self._events_routed,
            self._end_time - self._start_time,
        )

    def get_report(self) -> PipelineReport:
        """Build a summary report after run() completes."""
        duration = (self._end_time or time.perf_counter()) - (self._start_time or 0)

        # Aggregate loader stats across all files
        total_lines = sum(
            loader.stats.total_lines for loader in self._loader.loaders
        )
        parse_errors = sum(
            loader.stats.parse_errors for loader in self._loader.loaders
        )

        return PipelineReport(
            source_files=[str(p) for p in self._file_paths],
            num_shards=self._num_shards,
            duration_seconds=duration,
            total_lines=total_lines,
            parse_errors=parse_errors,
            events_routed=self._events_routed,
            shard_distribution=self._affinity_map.get_shard_distribution(),
            top_symbols=self._affinity_map.top_keys(n=10),
        )