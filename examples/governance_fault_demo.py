"""
examples/governance_fault_demo.py — one worked, reproducible scenario that
replaces the unrun 16-agent benchmark in the paper's evaluation section.

Demonstrates, end to end:
  1. ACCEPT           — a benign agent: intent pre-committed, action bound and verified
  2. GOVERNANCE_FAULT — a malicious agent: action with no pre-committed intent
  3. THROTTLED        — a drifting agent: timing entropy collapse decays credit
  4. OFFLINE VERIFY   — a third party re-verifies a binding with only the public
                        key, with NO access to the governance shard

Run:  python -m examples.governance_fault_demo
"""

from crypto.signature_engine import SignatureEngine, KeyPair
from governance.intent_binding import (
    IntentBinder, SemanticIntent, FinancialAction)
from governance.ege import ExecutionCreditController
from governance.governance_shard import GovernanceShard
from governance.agent_harness import run_harness


def main():
    print("=" * 64)
    print("VFEL governance demo — Intent-Action binding + Governance Fault + EGE")
    print("=" * 64)

    ege = ExecutionCreditController(baseline_samples=48, lam=4.0, floor_ratio=0.25)
    shard = GovernanceShard(ege=ege)
    engine = SignatureEngine()

    # --- 1. ACCEPT: benign agent --------------------------------------
    kp = KeyPair.generate("alice")
    shard.register_agent("alice", kp.public_key_hex)
    intent = SemanticIntent("alice", "spread favourable; size within risk limit",
                            "liquidity_v1", 1_000_000_000)
    shard.precommit(intent)
    action = FinancialAction("alice", "ORDER",
                             {"symbol": "AAPL", "side": "BUY", "qty": 100, "px": 190.0},
                             intent.intent_hash, 1_000_500_000)
    binding = shard.binder().bind(intent, action, kp)
    rec = shard.submit_action(action, binding)
    print(f"\n[1] benign agent      -> {rec.verdict.value}: {rec.reason}")
    print(f"    binding_id B = {binding.binding_id[:24]}...")

    # --- 2. GOVERNANCE_FAULT: malicious agent -------------------------
    mk = KeyPair.generate("mallory")
    shard.register_agent("mallory", mk.public_key_hex)
    forged_intent = SemanticIntent("mallory", "post-hoc justification", "x", 2_000_000_000)
    # action references an intent that was NEVER pre-committed
    rogue_action = FinancialAction("mallory", "ORDER",
                                   {"symbol": "TSLA", "side": "SELL", "qty": 9999},
                                   ("f" * 64), 2_000_000_001)
    forged_binding = shard.binder().bind(
        forged_intent,
        FinancialAction("mallory", "ORDER", {"symbol": "TSLA", "side": "SELL", "qty": 9999},
                        forged_intent.intent_hash, 2_000_000_001),
        mk)
    rec = shard.submit_action(rogue_action, forged_binding)
    print(f"\n[2] malicious agent   -> {rec.verdict.value}: {rec.reason}")

    # --- 3. THROTTLED: drifting agent (entropy collapse) --------------
    res = run_harness(seed=7, steps=80)
    drift = res.per_agent["agent-drifting"]
    print(f"\n[3] drifting agent    -> THROTTLED {drift['THROTTLED']}x after "
          f"timing-entropy collapse (ACCEPT {drift['ACCEPT']}x before)")

    # --- 4. OFFLINE VERIFICATION --------------------------------------
    third_party = IntentBinder(SignatureEngine())   # fresh verifier, no shard access
    ok = third_party.verify(intent, action, binding, kp.public_key_hex)
    tampered = SemanticIntent("alice", "I actually intended something else",
                              "liquidity_v1", 1_000_000_000)
    ok_tampered = third_party.verify(tampered, action, binding, kp.public_key_hex)
    print(f"\n[4] offline verify (no operator trust):")
    print(f"    genuine binding verifies      : {ok}")
    print(f"    tampered reasoning trace fails : {not ok_tampered}")

    print("\nHarness summary (seed=7):", res.summary)
    print("=" * 64)


if __name__ == "__main__":
    main()