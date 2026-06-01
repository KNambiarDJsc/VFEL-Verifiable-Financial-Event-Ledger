"""Governance Fault invariant + closed-loop adjudication."""
import unittest
from crypto.signature_engine import KeyPair
from governance.intent_binding import SemanticIntent, FinancialAction
from governance.governance_shard import GovernanceShard, Verdict
from governance.agent_harness import run_harness


class TestGovernanceShard(unittest.TestCase):
    def setUp(self):
        self.kp = KeyPair.generate("agent-1")
        self.shard = GovernanceShard()
        self.shard.register_agent("agent-1", self.kp.public_key_hex)

    def test_matched_action_accepted(self):
        intent = SemanticIntent("agent-1", "trace", "p", 1)
        self.shard.precommit(intent)
        action = FinancialAction("agent-1", "ORDER", {"q": 1}, intent.intent_hash, 2)
        binding = self.shard.binder().bind(intent, action, self.kp)
        rec = self.shard.submit_action(action, binding)
        self.assertEqual(rec.verdict, Verdict.ACCEPT)

    def test_unmatched_action_faults(self):
        intent = SemanticIntent("agent-1", "trace", "p", 1)
        # NOT pre-committed
        action = FinancialAction("agent-1", "ORDER", {"q": 1}, intent.intent_hash, 2)
        binding = self.shard.binder().bind(intent, action, self.kp)
        rec = self.shard.submit_action(action, binding)
        self.assertEqual(rec.verdict, Verdict.GOVERNANCE_FAULT)

    def test_record_hash_is_deterministic(self):
        intent = SemanticIntent("agent-1", "trace", "p", 1)
        self.shard.precommit(intent)
        action = FinancialAction("agent-1", "ORDER", {"q": 1}, intent.intent_hash, 2)
        binding = self.shard.binder().bind(intent, action, self.kp)
        rec = self.shard.submit_action(action, binding)
        self.assertEqual(len(rec.record_hash), 64)

    def test_harness_catches_all_three_profiles(self):
        res = run_harness(seed=7, steps=80)
        self.assertGreater(res.per_agent["agent-malicious"]["GOVERNANCE_FAULT"], 0)
        self.assertGreater(res.per_agent["agent-drifting"]["THROTTLED"], 0)
        self.assertGreater(res.per_agent["agent-benign"]["ACCEPT"], 0)


if __name__ == "__main__":
    unittest.main()