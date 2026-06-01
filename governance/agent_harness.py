"""
governance/agent_harness.py — small, deterministic, seeded agent harness.

This REPLACES the unrun "16 autonomous agents on NASDAQ replay" claim with a
reproducible mechanism demonstration. It is intentionally NOT a market
simulation and makes NO accuracy/precision claims. It deterministically drives
a handful of scripted agents through the governance loop so that the paper's
qualitative claims — faults are caught at submission, entropy collapse decays
execution credit — are reproducible from a fixed seed.

Agent profiles:
  benign    : jittered inter-arrival times -> stable timing entropy
  drifting  : benign warmup, then hyper-deterministic low-jitter burst
              (entropy collapse) -> EGE credit decays below floor -> THROTTLED
  malicious : submits an action that references an intent never pre-committed
              -> GOVERNANCE_FAULT at submission time
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from crypto.signature_engine import SignatureEngine, KeyPair
from governance.intent_binding import SemanticIntent, FinancialAction
from governance.ege import ExecutionCreditController
from governance.governance_shard import GovernanceShard, Verdict


@dataclass
class HarnessResult:
    summary: dict
    per_agent: dict
    total_events: int


def _mk_intent(agent_id: str, i: int, t_ns: int) -> SemanticIntent:
    return SemanticIntent(
        agent_id=agent_id,
        reasoning_trace=f"{agent_id} rationale step {i}: spread favourable, size within limit",
        policy_tag="liquidity_provision_v1",
        timestamp_ns=t_ns,
    )


def _mk_action(agent_id: str, i: int, t_ns: int, intent_ref: str) -> FinancialAction:
    side = "BUY" if i % 2 == 0 else "SELL"
    return FinancialAction(
        agent_id=agent_id,
        action_type="ORDER",
        payload={"symbol": "AAPL", "side": side, "qty": 100, "px": 190.0 + (i % 5)},
        intent_ref=intent_ref,
        timestamp_ns=t_ns,
    )


def run_harness(seed: int = 7, steps: int = 80,
                forest=None) -> HarnessResult:
    rng = random.Random(seed)
    engine = SignatureEngine()
    ege = ExecutionCreditController(baseline_samples=48, window=64, lam=4.0,
                                    floor_ratio=0.25)
    shard = GovernanceShard(forest=forest, ege=ege)

    agents = {
        "agent-benign":    KeyPair.generate("agent-benign"),
        "agent-drifting":  KeyPair.generate("agent-drifting"),
        "agent-malicious": KeyPair.generate("agent-malicious"),
    }
    for aid, kp in agents.items():
        shard.register_agent(aid, kp.public_key_hex)

    binder = shard.binder()
    clocks = {aid: 1_000_000_000 for aid in agents}
    per_agent = {aid: {v.value: 0 for v in Verdict} for aid in agents}

    for i in range(steps):
        for aid, kp in agents.items():
            # advance per-agent clock by a profile-specific inter-arrival gap
            if aid == "agent-benign":
                gap = rng.randint(800_000, 1_200_000)            # jittered
            elif aid == "agent-drifting":
                if i < 50:
                    gap = rng.randint(800_000, 1_200_000)        # warmup jitter
                else:
                    gap = 1_000_000                              # collapse: constant
            else:  # malicious
                gap = rng.randint(800_000, 1_200_000)
            clocks[aid] += gap
            t = clocks[aid]

            intent = _mk_intent(aid, i, t)

            if aid == "agent-malicious":
                # never pre-commit; reference a fabricated intent_hash
                bogus_ref = ("f" * 64)
                action = _mk_action(aid, i, t, bogus_ref)
                # sign over a self-serving (forged) binding to mimic post-hoc justification
                forged_intent = SemanticIntent(aid, "forged", "x", t)
                try:
                    binding = binder.bind(forged_intent,
                                          _mk_action(aid, i, t, forged_intent.intent_hash),
                                          kp)
                except ValueError:
                    continue
                rec = shard.submit_action(action, binding)
            else:
                shard.precommit(intent)
                action = _mk_action(aid, i, t, intent.intent_hash)
                binding = binder.bind(intent, action, kp)
                rec = shard.submit_action(action, binding)

            per_agent[aid][rec.verdict.value] += 1

    return HarnessResult(
        summary=shard.summary(),
        per_agent=per_agent,
        total_events=len(shard.records),
    )


if __name__ == "__main__":
    res = run_harness()
    print("Governance summary:", res.summary)
    for aid, counts in res.per_agent.items():
        print(f"  {aid:18s} {counts}")
    print("total governance records:", res.total_events)