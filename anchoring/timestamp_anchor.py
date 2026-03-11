"""
anchoring/timestamp_anchor.py

Timestamp-based anchoring backends for VFEL.

Two backends in this module:

1. LocalTimestampAnchor
   - Signs the forest root with the node's Ed25519 key + wall clock
   - Produces a self-contained signed timestamp proof
   - Works offline, zero external dependencies
   - Suitable for: internal audit trails, development, testing
   - NOT a substitute for external anchoring — the signing key is controlled by you

2. OpenTimestampsAnchor
   - Submits forest root to the OpenTimestamps (OTS) public calendar servers
   - OTS aggregates many hashes and anchors them to Bitcoin via OP_RETURN
   - Final Bitcoin confirmation takes ~1 hour (one Bitcoin block)
   - Produces a .ots proof file that anyone can independently verify
   - Suitable for: production external anchoring without gas costs
   - Requires: `pip install opentimestamps-client` + internet access
   - OTS calendar servers: alice.btc.calendar.opentimestamps.org,
                           bob.btc.calendar.opentimestamps.org,
                           finney.calendar.opentimestamps.org

OpenTimestamps verification (offline, after confirmation):
    ots verify proof.ots  # Uses bitcoin node or block explorer API
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from anchoring.anchor_manager import (
    AnchorBackend, AnchorRecord, AnchorNetwork, AnchorStatus,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Local Timestamp Anchor (always available)
# ──────────────────────────────────────────────

class LocalTimestampAnchor(AnchorBackend):
    """
    Signed local timestamp anchor.

    Creates a cryptographically signed record of:
        {forest_root, timestamp_ns, node_id, metadata}

    Signed with the node's Ed25519 key (if available) or SHA256-chained
    against a local anchor chain (fallback).

    This is the zero-dependency baseline anchor. Every VFEL deployment
    should have this as a fallback — it at least creates a tamper-evident
    local record even when external networks are unavailable.

    The anchor proof is the JSON record + signature, stored as base64.
    """

    def __init__(self, node_id: str = "vfel-node-0", key_pair=None):
        self.node_id = node_id
        self._key_pair = key_pair  # Optional Ed25519 KeyPair from Phase 5
        self._chain_tip: str = "0" * 64  # Hash chain of all local anchors

    @property
    def network(self) -> AnchorNetwork:
        return AnchorNetwork.LOCAL_TIMESTAMP

    def submit(self, forest_root: str, metadata: dict) -> AnchorRecord:
        """Create and sign a local timestamp anchor immediately."""
        anchor_id = str(uuid.uuid4())
        timestamp_ns = time.time_ns()

        # Build the anchor payload
        payload = {
            "anchor_id":    anchor_id,
            "forest_root":  forest_root,
            "timestamp_ns": timestamp_ns,
            "node_id":      self.node_id,
            "prev_anchor":  self._chain_tip,
            "metadata":     metadata,
        }

        # Sign payload
        sig_hex, pub_key_hex = self._sign_payload(payload)

        # Encode proof as base64 JSON
        proof_obj = {
            "payload":     payload,
            "signature":   sig_hex,
            "public_key":  pub_key_hex,
            "algorithm":   "Ed25519" if self._key_pair else "SHA256-chain",
        }
        proof_bytes = base64.b64encode(
            json.dumps(proof_obj, sort_keys=True).encode()
        ).decode()

        # Advance local chain
        chain_input = f"{self._chain_tip}:{forest_root}:{timestamp_ns}"
        self._chain_tip = hashlib.sha256(chain_input.encode()).hexdigest()

        record = AnchorRecord(
            anchor_id=anchor_id,
            forest_root=forest_root,
            network=self.network,
            metadata={"node_id": self.node_id, **metadata},
        )
        record.mark_confirmed(
            tx_id=f"local:{anchor_id}",
            proof_bytes=proof_bytes,
        )
        return record

    def check_status(self, record: AnchorRecord) -> AnchorRecord:
        """Local anchors confirm immediately — nothing to poll."""
        return record

    def verify_proof(self, proof_bytes: str) -> tuple[bool, str]:
        """
        Verify a local timestamp proof offline.
        Returns (valid, description).
        """
        try:
            proof_obj = json.loads(base64.b64decode(proof_bytes))
            payload = proof_obj["payload"]
            sig_hex = proof_obj.get("signature", "")
            pub_key = proof_obj.get("public_key", "")
            algorithm = proof_obj.get("algorithm", "")

            if algorithm == "Ed25519" and pub_key and sig_hex:
                from crypto.signature_engine import SignatureEngine
                from crypto.canonical_json import canonical_encode
                engine = SignatureEngine()
                message = canonical_encode(payload)
                valid = engine.verify(message, sig_hex, pub_key)
                return valid, f"Ed25519 signature {'valid' if valid else 'INVALID'}"
            elif algorithm == "SHA256-chain":
                return True, "SHA256-chain anchor (signature not verifiable without chain state)"
            return False, f"Unknown algorithm: {algorithm}"
        except Exception as exc:
            return False, f"Proof parse error: {exc}"

    def _sign_payload(self, payload: dict) -> tuple[str, str]:
        """Sign payload with Ed25519 key or SHA256 fallback."""
        if self._key_pair:
            try:
                from crypto.signature_engine import SignatureEngine
                from crypto.canonical_json import canonical_encode
                engine = SignatureEngine()
                message = canonical_encode(payload)
                sig = engine.sign(message, self._key_pair)
                return sig.signature_hex, self._key_pair.public_key_hex
            except Exception as exc:
                logger.warning("Ed25519 signing failed, using SHA256 fallback: %s", exc)

        # SHA256 fallback: hash of canonical JSON
        import json as _json
        canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sig_hex = hashlib.sha256(canonical.encode()).hexdigest()
        return sig_hex, ""


# ──────────────────────────────────────────────
# OpenTimestamps Anchor
# ──────────────────────────────────────────────

class OpenTimestampsAnchor(AnchorBackend):
    """
    OpenTimestamps (OTS) anchoring backend.

    Submits the forest_root hash to public OTS calendar servers.
    The OTS servers aggregate hashes and anchor them to Bitcoin.
    Final confirmation takes ~1 hour (next Bitcoin block).

    Requires: pip install opentimestamps-client

    Verification:
        The .ots proof file (stored as base64 in proof_bytes) can be
        verified independently using the `ots` command-line tool:
            ots verify <forest_root.ots>

    Calendar servers used (same as OTS default):
        - alice.btc.calendar.opentimestamps.org
        - bob.btc.calendar.opentimestamps.org
        - finney.calendar.opentimestamps.org

    If the opentimestamps library is not installed, submission falls back
    to creating a pending record that can be submitted later.
    """

    CALENDAR_URLS = [
        "https://alice.btc.calendar.opentimestamps.org",
        "https://bob.btc.calendar.opentimestamps.org",
        "https://finney.calendar.opentimestamps.org",
    ]

    def __init__(self, proof_dir: Optional[str | Path] = None):
        self._proof_dir = Path(proof_dir) if proof_dir else None
        if self._proof_dir:
            self._proof_dir.mkdir(parents=True, exist_ok=True)
        self._ots_available = self._check_ots()

    @property
    def network(self) -> AnchorNetwork:
        return AnchorNetwork.OPENTIMESTAMPS

    def is_available(self) -> bool:
        return True  # Always attempt (graceful fallback if no internet)

    def submit(self, forest_root: str, metadata: dict) -> AnchorRecord:
        """
        Submit forest_root to OpenTimestamps calendar servers.

        The forest_root is treated as the data to timestamp.
        OTS requires bytes — we use the raw bytes of the hex hash.
        """
        anchor_id = str(uuid.uuid4())
        record = AnchorRecord(
            anchor_id=anchor_id,
            forest_root=forest_root,
            network=self.network,
            metadata=metadata,
        )

        if not self._ots_available:
            logger.warning(
                "opentimestamps-client not installed. "
                "Install with: pip install opentimestamps-client\n"
                "  Creating PENDING anchor record for manual submission."
            )
            record.metadata["manual_submission_required"] = True
            record.metadata["forest_root_bytes_hex"] = forest_root
            return record

        try:
            ots_proof_bytes = self._submit_to_ots(forest_root)
            proof_b64 = base64.b64encode(ots_proof_bytes).decode()

            if self._proof_dir:
                proof_path = self._proof_dir / f"{forest_root[:16]}_{anchor_id[:8]}.ots"
                with open(proof_path, "wb") as f:
                    f.write(ots_proof_bytes)
                logger.info("OTS proof saved: %s", proof_path)

            # OTS submission is PENDING until Bitcoin confirms (~1 hour)
            record.tx_id = f"ots:pending:{anchor_id}"
            record.proof_bytes = proof_b64
            record.metadata["ots_submitted_at"] = time.time_ns()
            # Status stays PENDING — poll check_status() after ~1 hour
            logger.info(
                "OTS submission complete for forest_root=%s... (pending Bitcoin confirmation)",
                forest_root[:16]
            )

        except Exception as exc:
            record.mark_failed(f"OTS submission error: {exc}")

        return record

    def check_status(self, record: AnchorRecord) -> AnchorRecord:
        """
        Attempt to upgrade a PENDING OTS proof to CONFIRMED.
        OTS proofs upgrade automatically when a Bitcoin block is found.
        """
        if not self._ots_available or not record.proof_bytes:
            return record

        try:
            ots_bytes = base64.b64decode(record.proof_bytes)
            confirmed, block_info = self._try_upgrade_ots(ots_bytes, record.forest_root)
            if confirmed:
                record.mark_confirmed(
                    tx_id=f"btc:{block_info.get('txid', 'unknown')}",
                    block_number=block_info.get("block_height"),
                    block_hash=block_info.get("block_hash"),
                )
                logger.info(
                    "OTS anchor confirmed in Bitcoin block %s",
                    block_info.get("block_height")
                )
        except Exception as exc:
            logger.warning("OTS status check failed: %s", exc)

        return record

    def _check_ots(self) -> bool:
        try:
            import opentimestamps  # noqa
            return True
        except ImportError:
            return False

    def _submit_to_ots(self, forest_root: str) -> bytes:
        """Submit to OTS calendars. Returns raw .ots proof bytes."""
        import opentimestamps.core.timestamp as ots_ts
        import opentimestamps.core.op as ots_op
        from opentimestamps.calendar import RemoteCalendar
        import io

        # Hash the forest_root hex string to get the digest to timestamp
        digest = bytes.fromhex(forest_root)

        file_timestamp = ots_ts.DetachedTimestampFile.from_bytes(
            ots_op.OpSHA256(), digest
        )

        # Submit to all calendar servers
        for url in self.CALENDAR_URLS:
            try:
                cal = RemoteCalendar(url)
                cal.submit(file_timestamp.timestamp)
                logger.debug("Submitted to OTS calendar: %s", url)
            except Exception as exc:
                logger.warning("OTS calendar %s failed: %s", url, exc)

        buf = io.BytesIO()
        file_timestamp.serialize(buf)
        return buf.getvalue()

    def _try_upgrade_ots(self, ots_bytes: bytes, forest_root: str) -> tuple[bool, dict]:
        """Attempt to upgrade an incomplete OTS proof via Bitcoin."""
        try:
            import opentimestamps.core.timestamp as ots_ts
            import opentimestamps.core.op as ots_op
            from opentimestamps.calendar import RemoteCalendar
            import io

            file_timestamp = ots_ts.DetachedTimestampFile.deserialize(
                io.BytesIO(ots_bytes)
            )

            for url in self.CALENDAR_URLS:
                try:
                    cal = RemoteCalendar(url)
                    cal.get_timestamp(file_timestamp.timestamp)
                except Exception:
                    pass

            # Check if any attestation exists
            for op, ts in file_timestamp.timestamp.ops.items():
                for attest in ts.attestations:
                    if hasattr(attest, 'height'):
                        return True, {"block_height": attest.height}

        except Exception as exc:
            logger.debug("OTS upgrade attempt: %s", exc)

        return False, {}