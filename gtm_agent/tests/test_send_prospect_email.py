import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_NAME = "_gtm_agent_engine_workshop"
PACKAGE_PATH = Path(__file__).resolve().parents[1]
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.gtm_agent", PACKAGE_PATH / "gtm_agent.py"
)
gtm_agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gtm_agent
with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    spec.loader.exec_module(gtm_agent)


class SendProspectEmailTests(unittest.TestCase):
    def send_email(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect,
            "Hello",
            "Following up",
            SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )

    def test_disqualified_source_record_blocks_send(self):
        prospect = gtm_agent.data_service.get_prospect_record("LEAD-50001")
        result = self.send_email({"prospect_id": "LEAD-50001", **prospect})

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["error"], "Prospect is disqualified; email not sent.")

    def test_missing_source_record_blocks_send(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=None):
            result = self.send_email(
                {"prospect_id": "LEAD-MISSING", "email": "lead@example.com"}
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["error"], "Prospect record not found; email not sent.")

    def test_non_disqualified_prospect_still_sends(self):
        prospect = gtm_agent.data_service.get_prospect_record("LEAD-12853")
        result = self.send_email({"prospect_id": "LEAD-12853", **prospect})

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])

    def test_missing_email_keeps_existing_failure(self):
        with patch.object(
            gtm_agent.data_service,
            "get_prospect_record",
            return_value={"disqualified": False},
        ):
            result = self.send_email({"prospect_id": "LEAD-NO-EMAIL"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "Prospect record has no email address.")

    def test_prompt_requires_clear_explanation_for_blocked_send(self):
        self.assertIn("tell the rep clearly that the email was not sent", gtm_agent.SYSTEM_PROMPT)
        self.assertIn("whether the prospect is disqualified", gtm_agent.SYSTEM_PROMPT)
        self.assertIn("source record could not be found", gtm_agent.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
