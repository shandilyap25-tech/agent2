from __future__ import annotations

import unittest

from autostream_agent.service import AutoStreamAgent


class AutoStreamAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AutoStreamAgent()

    def test_pricing_question_uses_local_knowledge(self) -> None:
        result = self.agent.chat("pricing-session", "Hi, tell me about your pricing.")

        self.assertEqual(result.intent, "product_or_pricing_inquiry")
        self.assertIn("$29/month", result.reply)
        self.assertIn("$79/month", result.reply)

    def test_policy_question_uses_rag_context(self) -> None:
        result = self.agent.chat(
            "policy-session",
            "What is your refund policy and do you have support?",
        )

        self.assertIn("refund", result.reply.lower())
        self.assertIn("24/7 support", result.reply)

    def test_high_intent_flow_waits_for_all_required_fields(self) -> None:
        session_id = "lead-session"

        first = self.agent.chat(session_id, "Hi, tell me about your pricing.")
        self.assertFalse(first.lead_captured)

        second = self.agent.chat(
            session_id,
            "That sounds good, I want to try the Pro plan for my YouTube channel.",
        )
        self.assertEqual(second.intent, "high_intent_lead")
        self.assertEqual(second.lead_info.get("platform"), "YouTube")
        self.assertEqual(second.missing_fields, ["name", "email"])
        self.assertFalse(second.lead_captured)

        third = self.agent.chat(session_id, "My name is Alice Creator")
        self.assertEqual(third.missing_fields, ["email"])
        self.assertFalse(third.lead_captured)

        fourth = self.agent.chat(session_id, "alice@example.com")
        self.assertTrue(fourth.lead_captured)
        self.assertIn("Alice Creator", fourth.reply)
        self.assertIn("alice@example.com", fourth.reply)
        self.assertEqual(fourth.lead_info.get("platform"), "YouTube")


if __name__ == "__main__":
    unittest.main()

