"""Entropy-Gated Execution: stable timing keeps credit high; collapse decays it."""
import unittest
from governance.ege import ExecutionCreditController, TemporalJitterFingerprint


class TestEGE(unittest.TestCase):
    def test_constant_timing_zero_entropy(self):
        fp = TemporalJitterFingerprint(window=32, bins=8)
        t = 0
        for _ in range(40):
            t += 1_000_000; fp.observe(t)
        self.assertEqual(fp.entropy(), 0.0)

    def test_stable_agent_keeps_credit(self):
        import random
        rng = random.Random(1)
        c = ExecutionCreditController(baseline_samples=48, lam=4.0, floor_ratio=0.25)
        t = 0; last = None
        for _ in range(80):
            t += rng.randint(800_000, 1_200_000)
            last = c.observe("a", t)
        self.assertFalse(last.throttled)
        self.assertGreater(last.execution_credit, c.floor)

    def test_entropy_collapse_throttles(self):
        import random
        rng = random.Random(2)
        c = ExecutionCreditController(baseline_samples=48, lam=4.0, floor_ratio=0.25)
        t = 0
        for _ in range(48):                       # warmup with jitter
            t += rng.randint(800_000, 1_200_000); c.observe("b", t)
        last = None
        for _ in range(40):                       # collapse to constant gap
            t += 1_000_000; last = c.observe("b", t)
        self.assertTrue(last.throttled)
        self.assertLess(last.execution_credit, c.floor)


if __name__ == "__main__":
    unittest.main()


from governance.ege import PolicyDistributionFingerprint, MultiSignalCreditController


class TestMultiSignal(unittest.TestCase):
    def test_policy_entropy_constant_choice_zero(self):
        pf = PolicyDistributionFingerprint(window=16)
        for _ in range(20):
            pf.observe("hash")
        self.assertEqual(pf.entropy(), 0.0)

    def test_policy_entropy_uniform_high(self):
        pf = PolicyDistributionFingerprint(window=16)
        for c in (["a", "b", "c", "d"] * 4):
            pf.observe(c)
        self.assertGreater(pf.entropy(), 1.5)  # 4 uniform symbols -> 2.0 bits

    def test_multi_signal_catches_policy_drift_under_jitter(self):
        """Real-data finding: timing entropy alone misses tool-loop drift under
        CPU jitter, but multi-signal (timing + policy) catches it reliably."""
        import random
        rng = random.Random(3)
        c = MultiSignalCreditController(baseline_samples=32, policy_window=16,
                                        lambda_t=1.0, lambda_p=6.0)
        t = 0
        # warmup: jittered timing + varied tools
        tools = ["hash", "sort", "json", "wc"]
        for i in range(40):
            t += rng.randint(800_000, 1_200_000)
            c.observe("a", t, rng.choice(tools))
        # collapse: tool fixed to "hash" but timing still jittered (CPU noise)
        last = None
        for _ in range(30):
            t += rng.randint(800_000, 1_200_000)        # keep timing noisy
            last = c.observe("a", t, "hash")            # but policy collapses
        self.assertTrue(last.throttled)