"""
crypto/key_registry.py

Key Registry for VFEL.

Manages public keys for all nodes, signers, and external verifiers.
Think of this as a lightweight PKI for the ledger system.

Responsibilities:
- Store public keys indexed by key_id
- Track key validity windows (created_at, expires_at)
- Handle key rotation (old key → new key with overlap period)
- Provide public key lookup for signature verification
- Persist to JSON for sharing with external verifiers

NOT a CA (Certificate Authority):
    VFEL's key registry is intentionally simple — it stores trusted public
    keys that have been registered out-of-band. There's no certificate chain,
    no X.509, no revocation list. For production deployments, this would
    integrate with an HSM or a distributed key management service.

Key lifecycle:
    ACTIVE   — valid for signing and verification
    EXPIRING — within rotation window (accept but warn)
    EXPIRED  — reject for new signatures, accept for historical verification
    REVOKED  — reject always (key compromise)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class KeyStatus(str, Enum):
    ACTIVE   = "ACTIVE"
    EXPIRING = "EXPIRING"
    EXPIRED  = "EXPIRED"
    REVOKED  = "REVOKED"


@dataclass
class PublicKeyRecord:
    """
    A public key record in the registry.
    Only stores the PUBLIC key — private keys never enter the registry.
    """
    key_id: str
    public_key_hex: str           # 32-byte Ed25519 public key as hex
    algorithm: str = "Ed25519"
    owner: str = ""               # Human-readable owner (e.g. "node-1", "auditor")
    created_at_ns: int = field(default_factory=time.time_ns)
    expires_at_ns: Optional[int] = None   # None = never expires
    status: KeyStatus = KeyStatus.ACTIVE
    metadata: dict = field(default_factory=dict)

    def is_valid_for_signing(self) -> bool:
        return self.status == KeyStatus.ACTIVE

    def is_valid_for_verification(self) -> bool:
        """Expired keys still valid for verifying historical signatures."""
        return self.status not in (KeyStatus.REVOKED,)

    def current_status(self) -> KeyStatus:
        """Compute effective status (checks expiry against wall clock)."""
        if self.status == KeyStatus.REVOKED:
            return KeyStatus.REVOKED

        if self.expires_at_ns is not None:
            now = time.time_ns()
            warning_window_ns = 7 * 24 * 3600 * 1_000_000_000  # 7 days
            if now > self.expires_at_ns:
                return KeyStatus.EXPIRED
            if now > self.expires_at_ns - warning_window_ns:
                return KeyStatus.EXPIRING

        return KeyStatus.ACTIVE

    def to_dict(self) -> dict:
        return {
            "key_id":          self.key_id,
            "public_key_hex":  self.public_key_hex,
            "algorithm":       self.algorithm,
            "owner":           self.owner,
            "created_at_ns":   self.created_at_ns,
            "expires_at_ns":   self.expires_at_ns,
            "status":          self.status.value,
            "metadata":        self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PublicKeyRecord":
        r = cls(
            key_id=d["key_id"],
            public_key_hex=d["public_key_hex"],
            algorithm=d.get("algorithm", "Ed25519"),
            owner=d.get("owner", ""),
            created_at_ns=d.get("created_at_ns", 0),
            expires_at_ns=d.get("expires_at_ns"),
            metadata=d.get("metadata", {}),
        )
        r.status = KeyStatus(d.get("status", "ACTIVE"))
        return r


class KeyRegistry:
    """
    In-memory + optional disk-persisted public key registry.

    Thread-safe for concurrent reads (key lookups during verification).
    Write operations (register, revoke, rotate) should be serialized by caller.

    Usage:
        registry = KeyRegistry()
        registry.register(key_pair.to_dict(), owner="node-1")
        record = registry.lookup("node-1-key")
        ok = record and record.is_valid_for_verification()
    """

    def __init__(self, store_path: Optional[str | Path] = None):
        self._keys: dict[str, PublicKeyRecord] = {}  # key_id → record
        self._owner_index: dict[str, list[str]] = {}  # owner → [key_ids]
        self._store_path = Path(store_path) if store_path else None

    def register(
        self,
        public_key_hex: str,
        key_id: str,
        owner: str = "",
        algorithm: str = "Ed25519",
        expires_at_ns: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> PublicKeyRecord:
        """Register a new public key. Returns the created record."""
        if key_id in self._keys:
            raise ValueError(f"Key ID already registered: {key_id}")

        record = PublicKeyRecord(
            key_id=key_id,
            public_key_hex=public_key_hex,
            algorithm=algorithm,
            owner=owner,
            expires_at_ns=expires_at_ns,
            metadata=metadata or {},
        )
        self._keys[key_id] = record

        if owner not in self._owner_index:
            self._owner_index[owner] = []
        self._owner_index[owner].append(key_id)

        logger.info("Key registered: %s (owner=%s)", key_id, owner)
        self._persist()
        return record

    def register_from_dict(self, d: dict, owner: str = "") -> PublicKeyRecord:
        """Register from a key dict (output of KeyPair.to_dict())."""
        return self.register(
            public_key_hex=d["public_key_hex"],
            key_id=d["key_id"],
            owner=owner or d.get("owner", ""),
            algorithm=d.get("algorithm", "Ed25519"),
        )

    def lookup(self, key_id: str) -> Optional[PublicKeyRecord]:
        """Look up a key by key_id. Returns None if not found."""
        return self._keys.get(key_id)

    def lookup_active_for_owner(self, owner: str) -> list[PublicKeyRecord]:
        """Get all active keys for an owner (multiple keys during rotation)."""
        key_ids = self._owner_index.get(owner, [])
        return [
            self._keys[kid]
            for kid in key_ids
            if kid in self._keys and self._keys[kid].is_valid_for_signing()
        ]

    def revoke(self, key_id: str, reason: str = "") -> bool:
        """Revoke a key. Revoked keys fail all future signature verifications."""
        record = self._keys.get(key_id)
        if not record:
            return False
        record.status = KeyStatus.REVOKED
        record.metadata["revocation_reason"] = reason
        record.metadata["revoked_at_ns"] = time.time_ns()
        logger.warning("Key revoked: %s (reason=%s)", key_id, reason)
        self._persist()
        return True

    def rotate(
        self,
        old_key_id: str,
        new_public_key_hex: str,
        new_key_id: str,
        owner: str = "",
    ) -> PublicKeyRecord:
        """
        Key rotation: register new key, expire old key.
        Old key remains valid for historical verification (not revoked, just expired).
        """
        old_record = self._keys.get(old_key_id)
        if old_record:
            old_record.status = KeyStatus.EXPIRED
            old_record.expires_at_ns = time.time_ns()

        new_record = self.register(
            public_key_hex=new_public_key_hex,
            key_id=new_key_id,
            owner=owner or (old_record.owner if old_record else ""),
            metadata={"rotated_from": old_key_id},
        )
        logger.info("Key rotated: %s → %s", old_key_id, new_key_id)
        return new_record

    def all_keys(self) -> list[PublicKeyRecord]:
        return list(self._keys.values())

    def _persist(self) -> None:
        if not self._store_path:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._store_path.with_suffix(".tmp")
        data = {"keys": [r.to_dict() for r in self._keys.values()]}
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.rename(self._store_path)
        except Exception as exc:
            logger.error("Failed to persist key registry: %s", exc)

    def load_from_disk(self) -> int:
        if not self._store_path or not self._store_path.exists():
            return 0
        try:
            with open(self._store_path, "r") as f:
                data = json.load(f)
            for kd in data.get("keys", []):
                rec = PublicKeyRecord.from_dict(kd)
                self._keys[rec.key_id] = rec
                if rec.owner not in self._owner_index:
                    self._owner_index[rec.owner] = []
                self._owner_index[rec.owner].append(rec.key_id)
            logger.info("Loaded %d keys from registry", len(self._keys))
            return len(self._keys)
        except Exception as exc:
            logger.error("Failed to load key registry: %s", exc)
            return 0

    def describe(self) -> dict:
        status_counts = {}
        for r in self._keys.values():
            s = r.current_status().value
            status_counts[s] = status_counts.get(s, 0) + 1
        return {
            "total_keys":    len(self._keys),
            "status_counts": status_counts,
            "owners":        list(self._owner_index.keys()),
        }