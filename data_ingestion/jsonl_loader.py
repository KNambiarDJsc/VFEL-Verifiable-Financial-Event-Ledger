"""
data_ingestion/jsonl_loader.py

Streaming JSONL loader for VFEL — Verifiable Financial Event Ledger.

Design principles:
- Generator-based: never loads full file into memory (critical for million-event datasets)
- Fault-tolerant: malformed lines are captured, logged, and skipped — not fatal
- Stateful: tracks line number, byte offset, and parse errors for observability
- Interface: yields raw dicts with source metadata attached (origin file, line, offset)

This is the entry point for all historical JSONL replay and live file tailing.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawEvent:
    """
    A raw parsed JSON line from the JSONL stream.
    Carries source provenance — essential for debugging replays and audit trails.
    """
    data: dict
    source_file: str
    line_number: int
    byte_offset: int


@dataclass
class LoaderStats:
    """
    Tracks ingestion health metrics during a streaming session.
    Exposed for observability — feed into Prometheus/Datadog in prod.
    """
    total_lines: int = 0
    parsed_ok: int = 0
    parse_errors: int = 0
    empty_lines: int = 0
    error_samples: list = field(default_factory=list)  # Keep first N errors for debugging

    MAX_ERROR_SAMPLES = 10

    def record_error(self, line_no: int, raw: str, exc: Exception):
        self.parse_errors += 1
        if len(self.error_samples) < self.MAX_ERROR_SAMPLES:
            self.error_samples.append({
                "line": line_no,
                "raw_snippet": raw[:120],  # Truncate — don't bloat memory
                "error": str(exc),
            })


class JSONLLoader:
    """
    Streaming JSONL reader. Generator-based, O(1) memory per line.

    Usage:
        loader = JSONLLoader("vcp_rta_events.jsonl")
        for raw_event in loader.stream():
            process(raw_event)

    Can stream multiple files sequentially via stream_files([...]).
    """

    def __init__(self, file_path: str | Path, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.stats = LoaderStats()

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")

    def stream(self) -> Generator[RawEvent, None, None]:
        """
        Stream events one line at a time.
        Yields RawEvent for each valid JSON line.
        Skips and logs malformed lines — never raises mid-stream.
        """
        byte_offset = 0
        self.stats = LoaderStats()  # Reset stats on each stream call

        with open(self.file_path, "r", encoding=self.encoding) as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                self.stats.total_lines += 1
                line_bytes = len(raw_line.encode(self.encoding))

                stripped = raw_line.strip()

                if not stripped:
                    self.stats.empty_lines += 1
                    byte_offset += line_bytes
                    continue

                try:
                    data = json.loads(stripped)

                    if not isinstance(data, dict):
                        # Reject non-object JSON — VFEL only handles event objects
                        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

                    self.stats.parsed_ok += 1
                    yield RawEvent(
                        data=data,
                        source_file=str(self.file_path),
                        line_number=line_number,
                        byte_offset=byte_offset,
                    )

                except (json.JSONDecodeError, ValueError) as exc:
                    self.stats.record_error(line_number, raw_line, exc)
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_number, self.file_path.name, exc
                    )

                finally:
                    byte_offset += line_bytes

        logger.info(
            "Stream complete: %s — %d ok / %d errors / %d total",
            self.file_path.name,
            self.stats.parsed_ok,
            self.stats.parse_errors,
            self.stats.total_lines,
        )

    def get_stats(self) -> LoaderStats:
        return self.stats


class MultiFileLoader:
    """
    Sequences multiple JSONL files into a single event stream.
    Useful for replaying sharded historical data dumps or multi-day datasets.
    """

    def __init__(self, file_paths: list[str | Path], encoding: str = "utf-8"):
        self.loaders = [JSONLLoader(p, encoding) for p in file_paths]

    def stream(self) -> Generator[RawEvent, None, None]:
        """Yields events from all files in order."""
        for loader in self.loaders:
            yield from loader.stream()

    def aggregate_stats(self) -> dict:
        return {
            str(loader.file_path): {
                "total": loader.stats.total_lines,
                "ok": loader.stats.parsed_ok,
                "errors": loader.stats.parse_errors,
            }
            for loader in self.loaders
        }