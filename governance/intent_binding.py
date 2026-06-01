"""
governance/intent_binding.py — Intent-Action binding (paper Eq. 10).

The fundamental unit of verifiable agency is the Intent-Action Pair (I, A):

    I = SemanticIntent  : the agent's reasoning trace / policy rationale
    A = FinancialAction : the resulting order / cancel / replace payload

An action is valid only if it explicitly references a pre-committed intent.
The binding identity is:

    H_I = SHA256(canonical(intent))                      # intent commitment
    H_A = SHA256(canonical(action_payload))              # action commitment
    sigma = Ed25519_sign_agent( canonical({H_I, H_A}) )  # non-repudiation
    B = SHA256( H_I_bytes || H_A_bytes || sigma_bytes )  # Eq. 10

Because B commits to the reasoning trace BEFORE the action is accepted, the
operator cannot later alter the trace to fit an observed market outcome
without changing B (which is sealed into the Merkle Forest and anchored).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from crypto.canonical_json import canonical_encode
from crypto.signature_engine import SignatureEngine, KeyPair


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _commit(obj: dict) -> str:
    """Deterministic commitment to a dict via canonical JSON."""
    return _sha256_hex(canonical_encode(obj))


@dataclass(frozen=True)
class SemanticIntent:
    """I — the agent's committed decision context."""
    agent_id: str
    reasoning_trace: str          # tau: raw rationale from the inference engine
    policy_tag: str               # e.g. "mean_reversion_v3", "liquidity_provision"
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def intent_hash(self) -> str:
        """H_I — commitment to the decision context."""
        return _commit({
            "agent_id": self.agent_id,
            "reasoning_trace": self.reasoning_trace,
            "policy_tag": self.policy_tag,
            "timestamp_ns": self.timestamp_ns,
        })

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "reasoning_trace": self.reasoning_trace,
            "policy_tag": self.policy_tag,
            "timestamp_ns": self.timestamp_ns,
            "intent_hash": self.intent_hash,
        }


@dataclass(frozen=True)
class FinancialAction:
    """A — the market interaction. Must reference a pre-committed intent_hash."""
    agent_id: str
    action_type: str              # ORDER | CANCEL | REPLACE
    payload: dict                 # symbol, side, qty, price, ...
    intent_ref: str               # the H_I this action claims to act on
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def action_hash(self) -> str:
        """H_A — commitment to the order payload."""
        return _commit({
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "payload": self.payload,
            "timestamp_ns": self.timestamp_ns,
        })

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "payload": self.payload,
            "intent_ref": self.intent_ref,
            "timestamp_ns": self.timestamp_ns,
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True)
class IntentActionBinding:
    """The signed binding B (Eq. 10), self-describing and offline-verifiable."""
    agent_id: str
    intent_hash: str
    action_hash: str
    signature_hex: str
    key_id: str
    binding_id: str               # B
    binding_version: str = "vfel-binding-1.0"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "intent_hash": self.intent_hash,
            "action_hash": self.action_hash,
            "signature_hex": self.signature_hex,
            "key_id": self.key_id,
            "binding_id": self.binding_id,
            "binding_version": self.binding_version,
        }


class IntentBinder:
    """Builds and verifies Intent-Action bindings."""

    def __init__(self, engine: Optional[SignatureEngine] = None):
        self._engine = engine or SignatureEngine()

    def bind(self, intent: SemanticIntent, action: FinancialAction,
             key_pair: KeyPair) -> IntentActionBinding:
        """Produce a signed binding B for a matching (I, A) pair."""
        if action.intent_ref != intent.intent_hash:
            raise ValueError(
                "action.intent_ref does not reference this intent "
                f"({action.intent_ref[:12]}.. != {intent.intent_hash[:12]}..)"
            )
        h_i = intent.intent_hash
        h_a = action.action_hash
        sig = self._engine.sign_dict({"H_I": h_i, "H_A": h_a}, key_pair)
        binding_id = _sha256_hex(
            bytes.fromhex(h_i) + bytes.fromhex(h_a) + bytes.fromhex(sig.signature_hex)
        )
        return IntentActionBinding(
            agent_id=intent.agent_id,
            intent_hash=h_i,
            action_hash=h_a,
            signature_hex=sig.signature_hex,
            key_id=key_pair.key_id,
            binding_id=binding_id,
        )

    def verify(self, intent: SemanticIntent, action: FinancialAction,
               binding: IntentActionBinding, public_key_hex: str) -> bool:
        """
        Offline verification of Verify(I, A) = True. Checks, in order:
          1. action references this intent (intent_ref == H_I)
          2. recomputed H_I, H_A match the binding
          3. signature is valid under the agent's public key
          4. binding_id (B) recomputes correctly (Eq. 10)
        """
        if action.intent_ref != intent.intent_hash:
            return False
        if binding.intent_hash != intent.intent_hash:
            return False
        if binding.action_hash != action.action_hash:
            return False
        sig_ok = self._engine.verify_dict(
            {"H_I": binding.intent_hash, "H_A": binding.action_hash},
            binding.signature_hex, public_key_hex,
        )
        if not sig_ok:
            return False
        recomputed_b = _sha256_hex(
            bytes.fromhex(binding.intent_hash)
            + bytes.fromhex(binding.action_hash)
            + bytes.fromhex(binding.signature_hex)
        )
        return recomputed_b == binding.binding_id