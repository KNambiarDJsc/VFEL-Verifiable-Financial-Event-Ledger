"""Intent-Action binding (Eq. 10) soundness."""
import unittest
from crypto.signature_engine import SignatureEngine, KeyPair
from governance.intent_binding import (
    IntentBinder, SemanticIntent, FinancialAction)


class TestIntentBinding(unittest.TestCase):
    def setUp(self):
        self.kp = KeyPair.generate("agent-1")
        self.binder = IntentBinder(SignatureEngine())
        self.intent = SemanticIntent("agent-1", "buy dip, size ok", "p_v1", 1000)
        self.action = FinancialAction(
            "agent-1", "ORDER", {"symbol": "AAPL", "qty": 100},
            self.intent.intent_hash, 1001)

    def test_valid_binding_verifies(self):
        b = self.binder.bind(self.intent, self.action, self.kp)
        self.assertTrue(self.binder.verify(self.intent, self.action, b,
                                           self.kp.public_key_hex))

    def test_tampered_trace_fails(self):
        b = self.binder.bind(self.intent, self.action, self.kp)
        forged = SemanticIntent("agent-1", "DIFFERENT trace", "p_v1", 1000)
        self.assertFalse(self.binder.verify(forged, self.action, b,
                                            self.kp.public_key_hex))

    def test_mismatched_intent_ref_refused_at_bind(self):
        bad_action = FinancialAction("agent-1", "ORDER", {"x": 1}, "deadbeef", 1)
        with self.assertRaises(ValueError):
            self.binder.bind(self.intent, bad_action, self.kp)

    def test_wrong_key_fails(self):
        b = self.binder.bind(self.intent, self.action, self.kp)
        other = KeyPair.generate("agent-2")
        self.assertFalse(self.binder.verify(self.intent, self.action, b,
                                            other.public_key_hex))


if __name__ == "__main__":
    unittest.main()