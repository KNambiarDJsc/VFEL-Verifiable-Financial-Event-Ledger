"""
governance/governance_shard.py — enforces the Governance Fault invariant
(paper Invariant 1) and closes the intent -> action -> EGE governance loop.

Invariant 1 (Governance Fault): for every action A pending ledger inclusion,
there must exist a pre-committed intent I such that Verify(I, A) = True. An
action with no matching, signature-valid intent puts the system into a
GOVERNANCE_FAULT state — a cryptographic circuit breaker that refuses to seal
the action until the Intent-Action invariant is restored.

Every decision (ACCEPT / GOVERNANCE_FAULT / THROTTLED) is recorded as a
deterministic, hashable governance record. When a ForestManager is supplied,
the record hash is appended to a dedicated "GOV" shard so the full governance
history is tamper-evident and offline-verifiable alongside trade events.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from crypto.canonical_json import canonical_encode
from governance.intent_binding import (
    IntentBinder, SemanticIntent, FinancialAction, IntentActionBinding,
)
from governance.ege import (
    ExecutionCreditController, MultiSignalCreditController, ThrottleRecord)

GOV_SHARD = "GOV"


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    GOVERNANCE_FAULT = "GOVERNANCE_FAULT"
    THROTTLED = "THROTTLED"


@dataclass(frozen=True)
class GovernanceRecord:
    verdict: Verdict
    agent_id: str
    intent_hash: Optional[str]
    action_hash: Optional[str]
    binding_id: Optional[str]
    reason: str
    throttle: Optional[dict] = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "agent_id": self.agent_id,
            "intent_hash": self.intent_hash,
            "action_hash": self.action_hash,
            "binding_id": self.binding_id,
            "reason": self.reason,
            "throttle": self.throttle,
            "timestamp_ns": self.timestamp_ns,
        }

    @property
    def record_hash(self) -> str:
        return hashlib.sha256(canonical_encode(self.to_dict())).hexdigest()


class GovernanceShard:
    """
    Holds pre-committed intents and adjudicates submitted actions.

    Args:
        forest:   optional ForestManager; if given, every record hash is
                  appended to the GOV shard (tamper-evident governance log).
        registry: maps agent_id -> public_key_hex for signature verification.
        ege:      optional ExecutionCreditController for entropy gating.
    """

    def __init__(self, forest=None, registry: Optional[dict] = None,
                 ege=None, signal_fn=None):
        self._binder = IntentBinder()
        self._intents: dict[str, SemanticIntent] = {}   # intent_hash -> I
        self._forest = forest
        self._registry = registry or {}                  # agent_id -> pubkey hex
        self._ege = ege
        self._signal_fn = signal_fn
        self.records: list[GovernanceRecord] = []

    # ---- key management -------------------------------------------------
    def register_agent(self, agent_id: str, public_key_hex: str) -> None:
        self._registry[agent_id] = public_key_hex

    # ---- intent pre-commitment -----------------------------------------
    def precommit(self, intent: SemanticIntent) -> str:
        """Pre-commit an intent. Returns its intent_hash (H_I)."""
        h_i = intent.intent_hash
        self._intents[h_i] = intent
        if self._forest is not None:
            self._forest.append(GOV_SHARD, hashlib.sha256(
                canonical_encode({"precommit": h_i})).hexdigest())
        return h_i

    # ---- action adjudication -------------------------------------------
    def submit_action(self, action: FinancialAction,
                      binding: IntentActionBinding) -> GovernanceRecord:
        """
        Adjudicate a submitted action against Invariant 1, then EGE.
        Returns a GovernanceRecord; appends its hash to the GOV shard.
        """
        intent = self._intents.get(action.intent_ref)
        pubkey = self._registry.get(action.agent_id)

        # Invariant 1: matching, signature-valid pre-committed intent must exist.
        if intent is None:
            return self._emit(GovernanceRecord(
                Verdict.GOVERNANCE_FAULT, action.agent_id, None,
                action.action_hash, binding.binding_id,
                "no pre-committed intent for action.intent_ref"))
        if pubkey is None:
            return self._emit(GovernanceRecord(
                Verdict.GOVERNANCE_FAULT, action.agent_id, intent.intent_hash,
                action.action_hash, binding.binding_id,
                "unknown agent identity (no registered public key)"))
        if not self._binder.verify(intent, action, binding, pubkey):
            return self._emit(GovernanceRecord(
                Verdict.GOVERNANCE_FAULT, action.agent_id, intent.intent_hash,
                action.action_hash, binding.binding_id,
                "binding verification failed (Verify(I,A) = False)"))

        # Entropy-Gated Execution (optional active defence).
        throttle: Optional[ThrottleRecord] = None
        if self._ege is not None:
            if isinstance(self._ege, MultiSignalCreditController):
                signal = (self._signal_fn(action) if self._signal_fn
                          else action.action_type)
                throttle = self._ege.observe(
                    action.agent_id, action.timestamp_ns, signal, binding.binding_id)
            else:
                throttle = self._ege.observe(
                    action.agent_id, action.timestamp_ns, binding.binding_id)
            if throttle.throttled:
                return self._emit(GovernanceRecord(
                    Verdict.THROTTLED, action.agent_id, intent.intent_hash,
                    action.action_hash, binding.binding_id,
                    "execution credit below floor (entropy drift)",
                    throttle.to_dict()))

        return self._emit(GovernanceRecord(
            Verdict.ACCEPT, action.agent_id, intent.intent_hash,
            action.action_hash, binding.binding_id, "intent-action verified",
            throttle.to_dict() if throttle else None))

    # ---- internals ------------------------------------------------------
    def _emit(self, record: GovernanceRecord) -> GovernanceRecord:
        self.records.append(record)
        if self._forest is not None:
            self._forest.append(GOV_SHARD, record.record_hash)
        return record

    def binder(self) -> IntentBinder:
        return self._binder

    def summary(self) -> dict:
        out = {v.value: 0 for v in Verdict}
        for r in self.records:
            out[r.verdict.value] += 1
        return out