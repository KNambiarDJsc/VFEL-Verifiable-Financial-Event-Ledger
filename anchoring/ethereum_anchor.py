"""
anchoring/ethereum_anchor.py

Ethereum Anchoring Backend for VFEL.

Emits a ForestRootAnchored event on an Ethereum smart contract.
The event is permanently recorded on the Ethereum blockchain and
indexable via standard event logs.

Solidity contract interface (deploy once, use forever):
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;

    contract VFELAnchor {
        event ForestRootAnchored(
            bytes32 indexed forestRoot,
            uint256 indexed blockSeq,
            address indexed anchorer,
            uint256 timestamp
        );

        function anchor(bytes32 forestRoot, uint256 blockSeq) external {
            emit ForestRootAnchored(forestRoot, blockSeq, msg.sender, block.timestamp);
        }
    }

    Deployed on:
        Mainnet:  0x... (deploy your own)
        Goerli:   0x... (testnet)
        Sepolia:  0x... (testnet — preferred for testing)

Why Ethereum over Bitcoin for some use cases?
    - Sub-second finality with PoS (vs ~1 hour Bitcoin)
    - Smart contract provides structured event logs (indexed, queryable)
    - ERC-165 compatible — can build permissioned anchoring with access control
    - Gas costs: ~21,000 gas per anchor event ≈ $0.05-2 at typical gas prices

Connection modes:
    simulation  — no network, deterministic mock tx hash
    web3        — uses web3.py + Infura/Alchemy RPC endpoint
    raw_rpc     — direct JSON-RPC (no web3.py dependency)

Requires for web3 mode: pip install web3
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

# ABI for the VFELAnchor contract's anchor() function and ForestRootAnchored event
VFEL_ANCHOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "forestRoot", "type": "bytes32"},
            {"internalType": "uint256", "name": "blockSeq",   "type": "uint256"},
        ],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "forestRoot", "type": "bytes32"},
            {"indexed": True, "internalType": "uint256", "name": "blockSeq",   "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "anchorer",   "type": "address"},
            {"indexed": False,"internalType": "uint256", "name": "timestamp",  "type": "uint256"},
        ],
        "name": "ForestRootAnchored",
        "type": "event",
    },
]


class EthereumAnchor(AnchorBackend):
    """
    Ethereum smart contract anchoring backend.

    Usage (simulation mode — no ETH required):
        anchor = EthereumAnchor(mode="simulation")
        record = anchor.submit(forest_root, metadata={})

    Usage (web3 mode — requires Infura/Alchemy key + funded wallet):
        anchor = EthereumAnchor(
            mode="web3",
            rpc_url="https://mainnet.infura.io/v3/YOUR_KEY",
            private_key="0x...",
            contract_address="0x...",
        )
        record = anchor.submit(forest_root, metadata={"block_seq": 42})
    """

    def __init__(
        self,
        mode: str = "simulation",
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,   # Hex-encoded Ethereum private key
        contract_address: Optional[str] = None,
        chain_id: int = 1,                   # 1=mainnet, 11155111=Sepolia
    ):
        self._mode = mode
        self._rpc_url = rpc_url
        self._private_key = private_key
        self._contract_address = contract_address
        self._chain_id = chain_id

        if mode not in ("simulation", "web3", "raw_rpc"):
            raise ValueError(f"Unknown Ethereum anchor mode: {mode}")

        logger.info(
            "EthereumAnchor initialized: mode=%s chain_id=%d",
            mode, chain_id
        )

    @property
    def network(self) -> AnchorNetwork:
        return AnchorNetwork.ETHEREUM

    def is_available(self) -> bool:
        if self._mode == "simulation":
            return True
        return (
            self._rpc_url is not None
            and self._private_key is not None
            and self._contract_address is not None
        )

    def submit(self, forest_root: str, metadata: dict) -> AnchorRecord:
        anchor_id = str(uuid.uuid4())
        block_seq = metadata.get("block_seq", 0)

        record = AnchorRecord(
            anchor_id=anchor_id,
            forest_root=forest_root,
            network=self.network,
            metadata={
                **metadata,
                "mode":          self._mode,
                "chain_id":      self._chain_id,
                "contract":      self._contract_address,
                "block_seq":     block_seq,
            },
        )

        if self._mode == "simulation":
            return self._simulate(record, forest_root, block_seq)
        elif self._mode == "web3":
            return self._submit_via_web3(record, forest_root, block_seq)
        elif self._mode == "raw_rpc":
            return self._submit_via_raw_rpc(record, forest_root, block_seq)

        record.mark_failed("Unknown mode")
        return record

    def check_status(self, record: AnchorRecord) -> AnchorRecord:
        """Query transaction receipt to check confirmation."""
        if record.status == AnchorStatus.CONFIRMED:
            return record
        if not record.tx_id or record.tx_id.startswith("sim:"):
            return record

        try:
            receipt = self._get_receipt(record.tx_id)
            if receipt and receipt.get("status") == 1:
                record.mark_confirmed(
                    tx_id=record.tx_id,
                    block_number=receipt.get("blockNumber"),
                    block_hash=receipt.get("blockHash"),
                )
        except Exception as exc:
            logger.warning("Ethereum tx receipt check failed: %s", exc)

        return record

    def build_calldata(self, forest_root: str, block_seq: int) -> str:
        """
        Build the ABI-encoded calldata for anchor(bytes32,uint256).
        Function selector = keccak256("anchor(bytes32,uint256)")[:4]
        """
        selector = hashlib.sha3_256(b"anchor(bytes32,uint256)").digest()[:4]
        root_padded = bytes.fromhex(forest_root).ljust(32, b"\x00")  # bytes32
        seq_padded  = block_seq.to_bytes(32, "big")                  # uint256
        return "0x" + (selector + root_padded + seq_padded).hex()

    # ── Simulation ────────────────────────────────────────────────────

    def _simulate(self, record: AnchorRecord, forest_root: str, block_seq: int) -> AnchorRecord:
        """
        Generate deterministic mock Ethereum transaction hash.
        Simulates an event emission without real network access.
        """
        calldata = self.build_calldata(forest_root, block_seq)
        sim_input = f"{forest_root}:{block_seq}:{record.anchor_id}:{self._chain_id}"
        sim_txhash = "0x" + hashlib.sha256(sim_input.encode()).hexdigest()

        # Simulate event log
        event_log = {
            "event":       "ForestRootAnchored",
            "forestRoot":  "0x" + forest_root,
            "blockSeq":    block_seq,
            "anchorer":    "0x" + "0" * 40,
            "timestamp":   int(time.time()),
            "txHash":      sim_txhash,
            "simulation":  True,
        }

        record.mark_confirmed(
            tx_id=f"sim:{sim_txhash}",
            block_number=None,
        )
        record.metadata["calldata"]  = calldata
        record.metadata["event_log"] = event_log
        record.metadata["simulation"] = True

        logger.info(
            "Ethereum anchor simulated: forest_root=%s... txhash=%s...",
            forest_root[:16], sim_txhash[:20]
        )
        return record

    # ── Web3.py mode ─────────────────────────────────────────────────

    def _submit_via_web3(self, record: AnchorRecord, forest_root: str, block_seq: int) -> AnchorRecord:
        """Submit via web3.py library."""
        try:
            from web3 import Web3  # type: ignore
            w3 = Web3(Web3.HTTPProvider(self._rpc_url))

            if not w3.is_connected():
                record.mark_failed(f"Cannot connect to Ethereum RPC: {self._rpc_url}")
                return record

            account = w3.eth.account.from_key(self._private_key)
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(self._contract_address),
                abi=VFEL_ANCHOR_ABI,
            )

            forest_root_bytes32 = bytes.fromhex(forest_root)

            tx = contract.functions.anchor(
                forest_root_bytes32,
                block_seq,
            ).build_transaction({
                "from":     account.address,
                "nonce":    w3.eth.get_transaction_count(account.address),
                "gas":      80_000,
                "gasPrice": w3.eth.gas_price,
                "chainId":  self._chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            txhash_hex = tx_hash.hex()

            record.tx_id = txhash_hex
            record.metadata["from_address"] = account.address
            logger.info("Ethereum tx broadcast: txhash=%s", txhash_hex[:20])

        except ImportError:
            record.mark_failed("web3 library not installed. Run: pip install web3")
        except Exception as exc:
            record.mark_failed(f"Ethereum web3 error: {exc}")

        return record

    # ── Raw RPC mode ──────────────────────────────────────────────────

    def _submit_via_raw_rpc(self, record: AnchorRecord, forest_root: str, block_seq: int) -> AnchorRecord:
        """
        Submit via raw Ethereum JSON-RPC (no web3.py).
        Requires manual tx signing via eth_account or similar.
        """
        try:
            import urllib.request
            calldata = self.build_calldata(forest_root, block_seq)

            # eth_call to estimate gas
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_estimateGas",
                "params": [{
                    "to":   self._contract_address,
                    "data": calldata,
                }],
            }
            req = urllib.request.Request(
                self._rpc_url,
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                gas_estimate = int(result.get("result", "0x15f90"), 16)

            record.metadata["gas_estimate"] = gas_estimate
            record.metadata["calldata"] = calldata
            record.metadata["note"] = "Raw RPC: sign and send tx externally using calldata"
            logger.info(
                "Ethereum raw RPC: gas_estimate=%d, calldata=%s...",
                gas_estimate, calldata[:32]
            )

        except Exception as exc:
            record.mark_failed(f"Ethereum raw RPC error: {exc}")

        return record

    def _get_receipt(self, tx_hash: str) -> Optional[dict]:
        """Get transaction receipt via raw JSON-RPC."""
        try:
            import urllib.request
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_getTransactionReceipt",
                "params": [tx_hash],
            }
            req = urllib.request.Request(
                self._rpc_url,
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read()).get("result")
                if result:
                    return {
                        "status":      int(result.get("status", "0x0"), 16),
                        "blockNumber": int(result.get("blockNumber", "0x0"), 16),
                        "blockHash":   result.get("blockHash"),
                    }
        except Exception as exc:
            logger.debug("Receipt query failed: %s", exc)
        return None