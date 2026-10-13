import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-1"}})
        self.prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Taylor Example",
            "email": "taylor@example.com",
        }
        self.from_rep = {"name": "Alex Rep", "email": "alex@example.com"}

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch(
        "gtm_agent.gtm_agent.data_service.get_prospect_record",
        return_value={"disqualified": True},
    )
    def test_disqualified_prospect_is_blocked_without_sending(self, get_record, uuid4):
        result = send_prospect_email.func(
            self.prospect, "Hello", "Message", self.runtime, self.from_rep
        )

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=None)
    def test_missing_source_record_is_blocked_without_sending(self, get_record, uuid4):
        result = send_prospect_email.func(
            self.prospect, "Hello", "Message", self.runtime, self.from_rep
        )

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect record not found; email not sent."},
        )
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    @patch(
        "gtm_agent.gtm_agent.data_service.get_prospect_record",
        return_value={"disqualified": False},
    )
    def test_non_disqualified_prospect_is_sent(self, get_record):
        result = send_prospect_email.func(
            self.prospect, "Hello", "Message", self.runtime, self.from_rep
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "taylor@example.com")
        get_record.assert_called_once_with("LEAD-50001")

    @patch(
        "gtm_agent.gtm_agent.data_service.get_prospect_record",
        return_value={"disqualified": False},
    )
    def test_missing_email_retains_existing_failure(self, get_record):
        prospect = {"prospect_id": "LEAD-50001", "name": "Taylor Example"}
        result = send_prospect_email.func(
            prospect, "Hello", "Message", self.runtime, self.from_rep
        )

        self.assertEqual(
            result,
            {"status": "failed", "error": "Prospect record has no email address."},
        )
        get_record.assert_called_once_with("LEAD-50001")

    def test_prompt_reports_blocked_send_reason_without_mislabeling_missing_record(self):
        self.assertIn("Clearly tell the rep that no email was sent", SYSTEM_PROMPT)
        self.assertIn("report the actual reason returned by the tool", SYSTEM_PROMPT)
        self.assertIn("do not describe a missing prospect record as disqualified", SYSTEM_PROMPT)
