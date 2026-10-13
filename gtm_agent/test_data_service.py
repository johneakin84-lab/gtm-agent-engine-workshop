import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service, gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.original_record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.original_profile = data_service._PROFILES.get(self.prospect_id)

    def tearDown(self):
        data_service.PROSPECTS[self.prospect_id] = self.original_record
        if self.original_profile is None:
            data_service._PROFILES.pop(self.prospect_id, None)
        else:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_and_rebuilds_profile(self):
        data_service.save_profile_to_db(self.prospect_id, {"tech_stack": ["stale"]})

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        self.assertIn("Terraform", profile["tech_stack"])

    def test_missing_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("LEAD-MISSING", "Terraform")

        self.assertFalse(result["updated"])
        self.assertFalse(result["found"])

    def test_repeat_update_does_not_claim_a_change(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertFalse(result["updated"])
        self.assertTrue(result["found"])
        self.assertEqual(
            data_service.fetch_tech_stack(self.prospect_id).count("Terraform"), 1
        )

    def test_score_uses_the_updated_source_tech_stack(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        profile = {"prospect_id": self.prospect_id, "tech_stack": ["stale"]}
        offering = {
            "required_tech_stack": ["Terraform"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        result = Mock()
        result.model_dump.return_value = {"score": 100}

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = result
            gtm_agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertIn("Terraform", scored_profile["tech_stack"])
