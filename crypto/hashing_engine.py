"""
crypto/hashing_engine.py

Hashing Engine for VFEL — unified SHA256 + BLAKE3 interface.

Design:
- Single HashingEngine class abstracts over both algorithms
- Default algorithm is configurable per-instance
- All methods return hex strings (64 chars for SHA256, configurable for BLAKE3)
- BLAKE3 is ~3-6x faster than SHA256 for large inputs on modern hardware
- SHA256 remains available for VCP cross-verification compatibility

BLAKE3 dependency:
    pip install blake3
    If not installed, BLAKE3 falls back to SHA256 with a warning.
    This ensures the system works out-of-the-box without the dependency.

Algorithm selection guidance:
    - SHA256: VCP event hash cross-verification, external anchor compatibility
    - BLAKE3: internal Merkle tree nodes, event content hashing (Phase 5+)
    - Phase 3 Merkle trees currently use SHA256; upgrade path is swap HashingEngine default.
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Union

logger = logging.getLogger(__name__)

# Try to import BLAKE3 — optional dependency
try:
    import blake3 as _blake3_module
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
    logger.warning(
        "blake3 not installed — BLAKE3 calls will fall back to SHA256. "
        "Install with: pip install blake3"
    )


class HashAlgorithm(str, Enum):
    SHA256 = "sha256"
    BLAKE3 = "blake3"
    SHA3_256 = "sha3_256"


class HashingEngine:
    """
    Unified hashing interface for VFEL.

    Usage:
        engine = HashingEngine(default_algorithm=HashAlgorithm.BLAKE3)
        h = engine.hash_bytes(b"some data")
        h = engine.hash_str("some string")
        h = engine.hash_dict({"key": "value"})   # canonical JSON hash

    All methods return lowercase hex strings.
    """

    def __init__(self, default_algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        self.default_algorithm = default_algorithm
        if default_algorithm == HashAlgorithm.BLAKE3 and not BLAKE3_AVAILABLE:
            logger.warning("BLAKE3 requested but not available — defaulting to SHA256")
            self.default_algorithm = HashAlgorithm.SHA256

    def hash_bytes(
        self,
        data: bytes,
        algorithm: HashAlgorithm = None,
    ) -> str:
        """Hash raw bytes. Returns hex string."""
        algo = algorithm or self.default_algorithm
        return self._dispatch(data, algo)

    def hash_str(
        self,
        data: str,
        encoding: str = "utf-8",
        algorithm: HashAlgorithm = None,
    ) -> str:
        """Hash a string (UTF-8 encoded). Returns hex string."""
        return self.hash_bytes(data.encode(encoding), algorithm)

    def hash_dict(
        self,
        data: dict,
        algorithm: HashAlgorithm = None,
    ) -> str:
        """
        Hash a dict via canonical JSON serialization.
        Keys sorted, no extra whitespace, deterministic across runs.
        """
        import json
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return self.hash_str(canonical, algorithm=algorithm)

    def hash_concat(
        self,
        *parts: Union[bytes, str],
        algorithm: HashAlgorithm = None,
    ) -> str:
        """Hash the concatenation of multiple parts."""
        combined = b""
        for part in parts:
            if isinstance(part, str):
                combined += part.encode("utf-8")
            else:
                combined += part
        return self.hash_bytes(combined, algorithm)

    def hash_file(
        self,
        path: str,
        algorithm: HashAlgorithm = None,
        chunk_size: int = 65536,
    ) -> str:
        """Hash a file incrementally (streaming, O(1) memory)."""
        algo = algorithm or self.default_algorithm

        if algo == HashAlgorithm.BLAKE3 and BLAKE3_AVAILABLE:
            h = _blake3_module.blake3()
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
            return h.hexdigest()
        else:
            h = self._sha_hasher(algo)
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
            return h.hexdigest()

    def verify(self, data: bytes, expected_hash: str, algorithm: HashAlgorithm = None) -> bool:
        """Constant-time hash verification."""
        import hmac
        computed = self.hash_bytes(data, algorithm)
        # hmac.compare_digest prevents timing attacks
        return hmac.compare_digest(computed, expected_hash.lower())

    def algorithm_name(self) -> str:
        return self.default_algorithm.value

    # ── Internal dispatch ──────────────────────────────────────────────

    def _dispatch(self, data: bytes, algo: HashAlgorithm) -> str:
        if algo == HashAlgorithm.BLAKE3:
            if BLAKE3_AVAILABLE:
                return _blake3_module.blake3(data).hexdigest()
            else:
                logger.debug("BLAKE3 unavailable — falling back to SHA256")
                return hashlib.sha256(data).hexdigest()

        return self._sha_hasher(algo, data).hexdigest()

    def _sha_hasher(self, algo: HashAlgorithm, data: bytes = None):
        if algo == HashAlgorithm.SHA3_256:
            h = hashlib.sha3_256()
        else:
            h = hashlib.sha256()
        if data:
            h.update(data)
        return h


# ──────────────────────────────────────────────
# Module-level convenience instances
# ──────────────────────────────────────────────

# Default engine (SHA256 — no external deps required)
sha256_engine = HashingEngine(HashAlgorithm.SHA256)

# BLAKE3 engine (falls back to SHA256 if not installed)
blake3_engine = HashingEngine(HashAlgorithm.BLAKE3)

def quick_hash(data: Union[bytes, str, dict]) -> str:
    """
    Quick SHA256 hash for any input type.
    Convenience function — no engine instantiation needed.
    """
    if isinstance(data, dict):
        return sha256_engine.hash_dict(data)
    if isinstance(data, str):
        return sha256_engine.hash_str(data)
    return sha256_engine.hash_bytes(data)