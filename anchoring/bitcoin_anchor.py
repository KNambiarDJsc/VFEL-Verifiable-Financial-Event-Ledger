"""
anchoring/bitcoin_anchor.py

Bitcoin Anchoring Backend for VFEL.

Embeds the VFEL forest root into the Bitcoin blockchain via OP_RETURN.

OP_RETURN:
    Bitcoin script opcode that marks a transaction output as provably unspendable.
    Used to embed arbitrary 80-byte data into the blockchain permanently.
    We embed: b"VFEL" (4 bytes prefix) + forest_root bytes (32 bytes) = 36 bytes total.

    Example OP_RETURN output script:
        OP_RETURN 56464c4c<32-byte-forest-root-hex>
        (VFEL = 0x56 0x46 0x45 0x4c)

Why Bitcoin?
    - Longest proof-of-work chain = highest tamper cost
    - Immutable: once buried under 6+ blocks, effectively irreversible
    - Public: anyone can verify with a Bitcoin node or block explorer API
    - Permanent: Bitcoin blockchain is replicated by 10,000+ nodes globally

Connection modes:
    1. Bitcoin Core RPC  — requires running a full node (most secure)
    2. Electrum protocol — lightweight, connects to Electrum servers
    3. Block explorer API — Blockstream.info / mempool.space REST API (easiest)
    4. Simulation mode  — no network, generates mock tx for testing

Production note:
    This module uses mode 3 (block explorer API) as default since it requires
    no local Bitcoin node. For production deployments with highest security,
    use mode 1 (Bitcoin Core RPC) with your own node.

    Broadcasting a real transaction requires:
    - A Bitcoin address with funds (for transaction fees ~$1-5)
    - Private key access (handled by your wallet, not stored in VFEL)
    - The `bitcoinlib` or `bit` library: pip install bit
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from anchoring.anchor_manager import (
    AnchorBackend, AnchorRecord, AnchorNetwork, AnchorStatus,
)

logger = logging.getLogger(__name__)

# VFEL magic prefix for OP_RETURN identification
VFEL_PREFIX = b"VFEL"
VFEL_PREFIX_HEX = VFEL_PREFIX.hex()  # "56464c4c"

# Mainnet block explorer APIs
BLOCKSTREAM_API = "https://blockstream.info/api"
MEMPOOL_API     = "https://mempool.space/api"


class BitcoinAnchor(AnchorBackend):
    """
    Bitcoin OP_RETURN anchoring backend.

    Modes:
        simulation  — no real transaction, generates deterministic mock tx
        explorer    — uses block explorer API to broadcast (requires funded wallet)
        rpc         — uses Bitcoin Core JSON-RPC (requires local node)

    Default: simulation mode (safe for development/testing).
    Set mode="explorer" + provide wif_key for production use.

    The simulation mode generates a deterministic transaction ID that uniquely
    identifies the forest_root — useful for testing proof pipelines without
    spending real Bitcoin.
    """

    def __init__(
        self,
        mode: str = "simulation",
        wif_key: Optional[str] = None,         # WIF-encoded private key for signing
        rpc_url: Optional[str] = None,          # Bitcoin Core RPC endpoint
        rpc_user: Optional[str] = None,
        rpc_password: Optional[str] = None,
        network: str = "mainnet",               # "mainnet" or "testnet"
    ):
        self._mode = mode
        self._wif_key = wif_key
        self._rpc_url = rpc_url
        self._rpc_user = rpc_user
        self._rpc_password = rpc_password
        self._btc_network = network

        if mode not in ("simulation", "explorer", "rpc"):
            raise ValueError(f"Unknown Bitcoin anchor mode: {mode}")

        logger.info("BitcoinAnchor initialized: mode=%s network=%s", mode, network)

    @property
    def network(self) -> AnchorNetwork:
        return AnchorNetwork.BITCOIN

    def is_available(self) -> bool:
        if self._mode == "simulation":
            return True
        if self._mode == "explorer":
            return self._wif_key is not None
        if self._mode == "rpc":
            return self._rpc_url is not None
        return False

    def submit(self, forest_root: str, metadata: dict) -> AnchorRecord:
        """Embed forest_root in Bitcoin blockchain via OP_RETURN."""
        anchor_id = str(uuid.uuid4())
        record = AnchorRecord(
            anchor_id=anchor_id,
            forest_root=forest_root,
            network=self.network,
            metadata={**metadata, "mode": self._mode, "btc_network": self._btc_network},
        )

        if self._mode == "simulation":
            return self._simulate(record, forest_root)
        elif self._mode == "explorer":
            return self._submit_via_explorer(record, forest_root)
        elif self._mode == "rpc":
            return self._submit_via_rpc(record, forest_root)

        record.mark_failed("Unknown mode")
        return record

    def check_status(self, record: AnchorRecord) -> AnchorRecord:
        """Query block explorer to check if tx is confirmed."""
        if record.status == AnchorStatus.CONFIRMED:
            return record
        if not record.tx_id or record.tx_id.startswith("sim:"):
            return record  # Simulation tx — always "confirmed"

        if self._mode in ("explorer", "rpc"):
            try:
                tx_info = self._get_tx_info(record.tx_id)
                if tx_info and tx_info.get("confirmed"):
                    record.mark_confirmed(
                        tx_id=record.tx_id,
                        block_number=tx_info.get("block_height"),
                        block_hash=tx_info.get("block_hash"),
                    )
            except Exception as exc:
                logger.warning("Bitcoin tx status check failed: %s", exc)

        return record

    def build_op_return_data(self, forest_root: str) -> bytes:
        """
        Build the OP_RETURN data payload.
        Format: VFEL (4 bytes) + forest_root_bytes (32 bytes) = 36 bytes
        Well within Bitcoin's 80-byte OP_RETURN limit.
        """
        forest_root_bytes = bytes.fromhex(forest_root)
        return VFEL_PREFIX + forest_root_bytes

    def decode_op_return(self, op_return_hex: str) -> Optional[str]:
        """
        Decode an OP_RETURN script and extract the forest root if it's a VFEL anchor.
        Returns the forest_root hex string, or None if not a VFEL anchor.
        """
        try:
            data = bytes.fromhex(op_return_hex)
            if data[:4] == VFEL_PREFIX and len(data) >= 36:
                return data[4:36].hex()
        except Exception:
            pass
        return None

    # ── Simulation mode ───────────────────────────────────────────────

    def _simulate(self, record: AnchorRecord, forest_root: str) -> AnchorRecord:
        """
        Generate a deterministic mock Bitcoin transaction.
        The simulated txid is SHA256(VFEL_PREFIX + forest_root + anchor_id).
        Confirms immediately. Used for testing and development.
        """
        op_return_data = self.build_op_return_data(forest_root)
        sim_input = op_return_data + record.anchor_id.encode()
        sim_txid = hashlib.sha256(sim_input).hexdigest()

        record.mark_confirmed(
            tx_id=f"sim:{sim_txid}",
            block_number=None,
            block_hash=None,
        )
        record.metadata["op_return_hex"] = op_return_data.hex()
        record.metadata["simulation"] = True
        logger.info(
            "Bitcoin anchor simulated: forest_root=%s... txid=sim:%s...",
            forest_root[:16], sim_txid[:16]
        )
        return record

    # ── Explorer mode ─────────────────────────────────────────────────

    def _submit_via_explorer(self, record: AnchorRecord, forest_root: str) -> AnchorRecord:
        """
        Broadcast a real Bitcoin transaction via block explorer API.
        Requires `bit` library: pip install bit
        """
        try:
            import bit  # type: ignore

            key = bit.Key(self._wif_key) if self._btc_network == "mainnet" \
                else bit.PrivateKeyTestnet(self._wif_key)

            op_return_data = self.build_op_return_data(forest_root)

            # Build transaction: no outputs (just OP_RETURN)
            tx_hex = key.create_transaction(
                [],
                message=op_return_data.decode("latin-1"),
                custom_pushdata=True,
            )

            # Broadcast
            txid = key.send([], message=op_return_data.decode("latin-1"), custom_pushdata=True)

            record.tx_id = txid
            record.metadata["op_return_hex"] = op_return_data.hex()
            logger.info("Bitcoin tx broadcast: txid=%s", txid)

        except ImportError:
            record.mark_failed("bit library not installed. Run: pip install bit")
        except Exception as exc:
            record.mark_failed(f"Bitcoin broadcast error: {exc}")

        return record

    # ── RPC mode ──────────────────────────────────────────────────────

    def _submit_via_rpc(self, record: AnchorRecord, forest_root: str) -> AnchorRecord:
        """
        Submit via Bitcoin Core JSON-RPC.
        Requires a running Bitcoin node with wallet enabled.
        """
        try:
            import requests
            op_return_data = self.build_op_return_data(forest_root)
            op_return_hex = op_return_data.hex()

            # createrawtransaction → fundrawtransaction → signrawtransaction → sendrawtransaction
            rpc = lambda method, params: requests.post(
                self._rpc_url,
                json={"jsonrpc": "1.0", "method": method, "params": params},
                auth=(self._rpc_user, self._rpc_password),
                timeout=30,
            ).json()

            # Create raw TX with OP_RETURN output
            raw = rpc("createrawtransaction", [
                [],
                [{"data": op_return_hex}]
            ])

            # Fund it
            funded = rpc("fundrawtransaction", [raw["result"]])
            # Sign it
            signed = rpc("signrawtransactionwithwallet", [funded["result"]["hex"]])
            # Send it
            txid_resp = rpc("sendrawtransaction", [signed["result"]["hex"]])

            record.tx_id = txid_resp["result"]
            record.metadata["op_return_hex"] = op_return_hex
            logger.info("Bitcoin RPC tx broadcast: txid=%s", record.tx_id)

        except Exception as exc:
            record.mark_failed(f"Bitcoin RPC error: {exc}")

        return record

    def _get_tx_info(self, txid: str) -> Optional[dict]:
        """Query Blockstream API for transaction status."""
        try:
            import urllib.request
            url = f"{BLOCKSTREAM_API}/tx/{txid}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
                status = data.get("status", {})
                return {
                    "confirmed":    status.get("confirmed", False),
                    "block_height": status.get("block_height"),
                    "block_hash":   status.get("block_hash"),
                }
        except Exception as exc:
            logger.debug("Blockstream API query failed: %s", exc)
            return None