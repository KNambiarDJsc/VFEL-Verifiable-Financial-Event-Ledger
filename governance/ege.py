"""
governance/ege.py — Entropy-Gated Execution (paper Section VI, Eq. 11, 12).

EGE is non-invasive: it never inspects model weights or private strategy logic.
It observes only the TIMING structure of ledgered events per agent and throttles
execution authority when an agent's operational signature diverges from baseline.

  Temporal-jitter fingerprint:  inter-arrival times dt_i = t_i - t_{i-1}
  Shannon entropy (Eq. 11):     H(X_a) = - sum p(dt) * log2 p(dt)
  Execution credit (Eq. 12):    C_a = V_max * exp(-lambda * |H_obs - H_base|)

A benign strategy has stable timing entropy within a regime. Unsafe loops
produce hyper-deterministic low-jitter bursts (entropy collapse) or erratic
spikes — both move H_obs away from H_base and decay C_a toward zero. If C_a
falls below a floor, the action is diverted to the Governance Shard for review.
All throttle decisions are emitted as tamper-evident records.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass(frozen=True)
class ThrottleRecord:
    """Tamper-evident record of an EGE decision (a, H_obs, H_base, C_a, B)."""
    agent_id: str
    h_observed: float
    h_baseline: float
    execution_credit: float
    ceiling: float
    throttled: bool
    binding_id: Optional[str] = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "h_observed": round(self.h_observed, 6),
            "h_baseline": round(self.h_baseline, 6),
            "execution_credit": round(self.execution_credit, 6),
            "ceiling": self.ceiling,
            "throttled": self.throttled,
            "binding_id": self.binding_id,
            "timestamp_ns": self.timestamp_ns,
        }


class TemporalJitterFingerprint:
    """
    Sliding-window estimator of the Shannon entropy of inter-arrival times.
    Entropy is computed over a fixed-width histogram of the most recent N gaps.
    """

    def __init__(self, window: int = 64, bins: int = 16):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.bins = bins
        self._times: Deque[int] = deque(maxlen=window + 1)

    def observe(self, timestamp_ns: int) -> None:
        self._times.append(timestamp_ns)

    def _gaps(self) -> list[int]:
        ts = list(self._times)
        return [ts[i] - ts[i - 1] for i in range(1, len(ts))]

    def entropy(self) -> float:
        """H(X_a) in bits. Returns 0.0 until at least 2 gaps are observed."""
        gaps = self._gaps()
        if len(gaps) < 2:
            return 0.0
        lo, hi = min(gaps), max(gaps)
        if hi == lo:
            return 0.0  # perfectly deterministic timing -> zero entropy
        width = (hi - lo) / self.bins
        counts = [0] * self.bins
        for g in gaps:
            idx = min(int((g - lo) / width), self.bins - 1)
            counts[idx] += 1
        total = sum(counts)
        h = 0.0
        for c in counts:
            if c:
                p = c / total
                h -= p * math.log2(p)
        return h


class ExecutionCreditController:
    """
    Maintains per-agent jitter fingerprints and computes execution credit.

    Baselines are learned during a warmup phase (the first `baseline_samples`
    observations per agent), then frozen. After that, every observation yields
    a credit value and (when below floor) a throttle decision.
    """

    def __init__(self, v_max: float = 1_000_000.0, lam: float = 4.0,
                 floor_ratio: float = 0.25, window: int = 64, bins: int = 16,
                 baseline_samples: int = 48):
        self.v_max = v_max
        self.lam = lam
        self.floor = floor_ratio * v_max
        self.window = window
        self.bins = bins
        self.baseline_samples = baseline_samples
        self._fp: dict[str, TemporalJitterFingerprint] = {}
        self._baseline: dict[str, float] = {}
        self._seen: dict[str, int] = {}

    def _agent(self, agent_id: str) -> TemporalJitterFingerprint:
        if agent_id not in self._fp:
            self._fp[agent_id] = TemporalJitterFingerprint(self.window, self.bins)
            self._seen[agent_id] = 0
        return self._fp[agent_id]

    def observe(self, agent_id: str, timestamp_ns: int,
                binding_id: Optional[str] = None) -> ThrottleRecord:
        fp = self._agent(agent_id)
        fp.observe(timestamp_ns)
        self._seen[agent_id] += 1
        h_obs = fp.entropy()

        # Warmup: learn baseline, full credit, never throttle.
        if self._seen[agent_id] <= self.baseline_samples:
            self._baseline[agent_id] = h_obs
            return ThrottleRecord(
                agent_id=agent_id, h_observed=h_obs, h_baseline=h_obs,
                execution_credit=self.v_max, ceiling=self.v_max,
                throttled=False, binding_id=binding_id,
            )

        h_base = self._baseline.get(agent_id, h_obs)
        credit = self.v_max * math.exp(-self.lam * abs(h_obs - h_base))
        throttled = credit < self.floor
        return ThrottleRecord(
            agent_id=agent_id, h_observed=h_obs, h_baseline=h_base,
            execution_credit=credit, ceiling=self.v_max,
            throttled=throttled, binding_id=binding_id,
        )

    def baseline(self, agent_id: str) -> Optional[float]:
        return self._baseline.get(agent_id)


# ──────────────────────────────────────────────────────────────────────
# Multi-signal extension
#
# Finding from real LangGraph agent traces (experiments/, Dataset 1):
# inter-arrival-time entropy alone is unreliable as a drift signal because
# real CPU jitter from reasoning steps masks tool-loop collapse — benign
# agents get false-throttled and policy-drift agents are missed.
#
# A reliable signal must augment timing entropy with POLICY entropy: the
# distribution over discrete decisions the agent makes (e.g. tool calls).
# An agent stuck in a tight loop produces near-zero policy entropy even
# when its timing remains noisy.
# ──────────────────────────────────────────────────────────────────────

class PolicyDistributionFingerprint:
    """Sliding-window Shannon entropy over discrete agent decisions (tool names)."""

    def __init__(self, window: int = 32):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._choices: Deque[str] = deque(maxlen=window)

    def observe(self, choice: str) -> None:
        self._choices.append(choice)

    def entropy(self) -> float:
        if len(self._choices) < 2:
            return 0.0
        counts: dict[str, int] = {}
        for c in self._choices:
            counts[c] = counts.get(c, 0) + 1
        total = len(self._choices)
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * math.log2(p)
        return h


class MultiSignalCreditController:
    """
    Combined timing + policy entropy controller. Credit decays on EITHER
    signal diverging from its agent-specific baseline:

        C_a = V_max * exp( - (lambda_t * |dH_t| + lambda_p * |dH_p|) )

    where dH_t is timing-entropy deviation and dH_p is policy-entropy deviation.
    Default weights make policy entropy the stronger signal because it is
    immune to CPU jitter.
    """

    def __init__(self, v_max: float = 1_000_000.0,
                 lambda_t: float = 1.0, lambda_p: float = 6.0,
                 floor_ratio: float = 0.25,
                 time_window: int = 48, time_bins: int = 12,
                 policy_window: int = 24,
                 baseline_samples: int = 32):
        self.v_max = v_max
        self.lambda_t = lambda_t
        self.lambda_p = lambda_p
        self.floor = floor_ratio * v_max
        self.time_window = time_window
        self.time_bins = time_bins
        self.policy_window = policy_window
        self.baseline_samples = baseline_samples
        self._tfp: dict[str, TemporalJitterFingerprint] = {}
        self._pfp: dict[str, PolicyDistributionFingerprint] = {}
        self._baseline_t: dict[str, float] = {}
        self._baseline_p: dict[str, float] = {}
        self._seen: dict[str, int] = {}

    def _agent(self, agent_id: str):
        if agent_id not in self._tfp:
            self._tfp[agent_id] = TemporalJitterFingerprint(self.time_window, self.time_bins)
            self._pfp[agent_id] = PolicyDistributionFingerprint(self.policy_window)
            self._seen[agent_id] = 0
        return self._tfp[agent_id], self._pfp[agent_id]

    def observe(self, agent_id: str, timestamp_ns: int, choice: str,
                binding_id: Optional[str] = None) -> ThrottleRecord:
        tfp, pfp = self._agent(agent_id)
        tfp.observe(timestamp_ns)
        pfp.observe(choice)
        self._seen[agent_id] += 1
        h_t = tfp.entropy()
        h_p = pfp.entropy()

        if self._seen[agent_id] <= self.baseline_samples:
            self._baseline_t[agent_id] = h_t
            self._baseline_p[agent_id] = h_p
            return ThrottleRecord(
                agent_id=agent_id, h_observed=h_p, h_baseline=h_p,
                execution_credit=self.v_max, ceiling=self.v_max,
                throttled=False, binding_id=binding_id,
            )

        bt = self._baseline_t.get(agent_id, h_t)
        bp = self._baseline_p.get(agent_id, h_p)
        deviation = self.lambda_t * abs(h_t - bt) + self.lambda_p * abs(h_p - bp)
        credit = self.v_max * math.exp(-deviation)
        throttled = credit < self.floor
        return ThrottleRecord(
            agent_id=agent_id, h_observed=h_p, h_baseline=bp,
            execution_credit=credit, ceiling=self.v_max,
            throttled=throttled, binding_id=binding_id,
        )