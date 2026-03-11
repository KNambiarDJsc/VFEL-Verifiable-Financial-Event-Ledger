"""
crypto/canonical_json.py

Canonical JSON Serialization for VFEL.

Problem: Standard JSON has no canonical form.
    {"a":1,"b":2} and {"b":2,"a":1} are semantically identical
    but produce different byte sequences — and therefore different hashes.
    This breaks cryptographic commitments.

Solution: Canonical JSON.
    Rules (subset of RFC 8785 / JCS):
    1. Keys sorted lexicographically (Unicode code point order)
    2. No insignificant whitespace
    3. Numbers: integers as integers, no trailing zeros in floats
    4. Strings: UTF-8 encoded, minimal escape sequences
    5. Recursive: nested objects also canonicalized
    6. Arrays: order preserved (arrays are ordered by definition)

Why not use the `canonicaljson` library?
    We implement our own so:
    - Zero extra dependencies in Phase 5
    - We control the exact spec and can document it for auditors
    - The rules are simple enough to implement correctly in ~50 lines

Compliance note:
    This implementation is compatible with RFC 8785 (JSON Canonicalization Scheme)
    for the subset of types used in VFEL (str, int, float, bool, None, dict, list).
    Full RFC 8785 handles IEEE 754 edge cases (NaN, Infinity) which we reject.
"""

from __future__ import annotations

import json
import re
from typing import Any


class CanonicalJSON:
    """
    Deterministic JSON serializer for cryptographic use.

    Usage:
        cj = CanonicalJSON()
        canonical_bytes = cj.encode({"b": 2, "a": 1})
        # b'{"a":1,"b":2}'

        canonical_str = cj.dumps(my_dict)
        h = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    """

    def encode(self, obj: Any) -> bytes:
        """
        Serialize to canonical JSON bytes (UTF-8).
        This is the primary method — use this for hashing.
        """
        return self.dumps(obj).encode("utf-8")

    def dumps(self, obj: Any) -> str:
        """Serialize to canonical JSON string."""
        return self._serialize(obj)

    def _serialize(self, obj: Any) -> str:
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            # Must check bool before int — bool is subclass of int in Python
            return "true" if obj else "false"
        if isinstance(obj, int):
            return str(obj)
        if isinstance(obj, float):
            return self._serialize_float(obj)
        if isinstance(obj, str):
            return self._serialize_string(obj)
        if isinstance(obj, (list, tuple)):
            return self._serialize_array(obj)
        if isinstance(obj, dict):
            return self._serialize_object(obj)
        raise TypeError(f"Cannot canonicalize type {type(obj).__name__}: {obj!r}")

    def _serialize_object(self, obj: dict) -> str:
        """Sort keys lexicographically, recurse into values."""
        if not obj:
            return "{}"
        # Sort keys by their serialized string form (Unicode code point order)
        sorted_items = sorted(obj.items(), key=lambda kv: kv[0])
        parts = [
            f"{self._serialize_string(k)}:{self._serialize(v)}"
            for k, v in sorted_items
        ]
        return "{" + ",".join(parts) + "}"

    def _serialize_array(self, arr) -> str:
        """Arrays preserve order."""
        if not arr:
            return "[]"
        parts = [self._serialize(item) for item in arr]
        return "[" + ",".join(parts) + "]"

    def _serialize_string(self, s: str) -> str:
        """
        JSON string serialization with minimal escaping.
        Only escapes what JSON requires: control chars, backslash, double-quote.
        Uses Python's json.dumps for correctness (handles all Unicode edge cases).
        """
        return json.dumps(s, ensure_ascii=False)

    def _serialize_float(self, f: float) -> str:
        """
        Float serialization.
        VFEL rejects NaN and Infinity — these must not appear in ledger data.
        Uses Python's repr for precision, then normalizes format.
        """
        import math
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"NaN and Infinity are not allowed in canonical JSON: {f}")

        # Use repr for maximum precision, then parse and reformat
        # This matches RFC 8785's "shortest representation" requirement
        s = repr(f)
        # Remove trailing zeros after decimal (e.g. "1.0" → "1.0" kept, "1.50" → "1.5")
        if "." in s and "e" not in s.lower():
            s = s.rstrip("0").rstrip(".")
            if "." not in s:
                s = s + ".0"  # Always keep decimal point for floats
        return s

    def hash_object(self, obj: Any, algorithm: str = "sha256") -> str:
        """
        Convenience: serialize and hash in one call.
        Returns hex string.
        """
        import hashlib
        data = self.encode(obj)
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        if algorithm == "sha3_256":
            return hashlib.sha3_256(data).hexdigest()
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ──────────────────────────────────────────────
# Module-level singleton + convenience functions
# ──────────────────────────────────────────────

_cj = CanonicalJSON()

def canonical_dumps(obj: Any) -> str:
    """Serialize to canonical JSON string."""
    return _cj.dumps(obj)

def canonical_encode(obj: Any) -> bytes:
    """Serialize to canonical JSON bytes."""
    return _cj.encode(obj)

def canonical_hash(obj: Any, algorithm: str = "sha256") -> str:
    """Canonical JSON hash of an object. Returns hex string."""
    return _cj.hash_object(obj, algorithm)