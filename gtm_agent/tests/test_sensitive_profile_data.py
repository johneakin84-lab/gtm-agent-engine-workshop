import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


def test_get_prospect_returns_only_allowed_fields():
    result = gtm_agent.get_prospect.func("LEAD-12853")

    assert result["found"] is True
    assert "billing_qualification" not in result["prospect"]
    assert set(result["prospect"]) <= {
        "prospect_id", "name", "email", "disqualified", "annual_revenue",
        "enrichment_source",
    }


def test_build_prospect_profile_excludes_billing_data_from_new_profile(monkeypatch):
    monkeypatch.setattr(data_service, "_PROFILES", {})

    result = gtm_agent.build_prospect_profile.func("LEAD-12853")

    assert result["found"] is True
    assert "billing_qualification" not in result["prospect_profile"]
    assert "billing_qualification" not in data_service._PROFILES["LEAD-12853"]


def test_build_prospect_profile_removes_billing_data_from_cached_profile(monkeypatch):
    monkeypatch.setattr(data_service, "_PROFILES", {})
    data_service._PROFILES["LEAD-12853"] = {
        "prospect_id": "LEAD-12853",
        "billing_qualification": {"tax_id": "sensitive"},
    }

    cached = data_service.get_profile_from_db("LEAD-12853")["prospect_profile"]
    result = gtm_agent.build_prospect_profile.func("LEAD-12853")

    assert "billing_qualification" not in cached
    assert result["found"] is True
    assert "billing_qualification" not in result["prospect_profile"]
    assert "billing_qualification" not in data_service._PROFILES["LEAD-12853"]


def test_score_prospect_sends_only_allowed_profile_fields(monkeypatch):
    profile = {
        "prospect_id": "LEAD-12853",
        "name": "Omar Okafor",
        "annual_revenue": 1000000,
        "tech_stack": ["Salesforce"],
        "account_details": {"industry": "technology"},
        "engagement_history": [],
        "billing_qualification": {"tax_id": "sensitive"},
    }
    offering = {
        "required_tech_stack": ["Salesforce"],
        "min_annual_revenue": 1,
        "description": "Test offering",
    }
    result = SimpleNamespace(model_dump=lambda: {"score": 90})
    invoke = Mock(return_value=result)
    monkeypatch.setattr(gtm_agent, "_scoring_llm", SimpleNamespace(invoke=invoke))

    gtm_agent.score_prospect.func(profile, offering)

    user_message = invoke.call_args.args[0][1]["content"]
    serialized_profile = user_message.split("\n\nProspect profile:\n", 1)[1]
    scoring_profile = json.loads(serialized_profile)

    assert "billing_qualification" not in scoring_profile
    assert set(scoring_profile) == {
        "prospect_id", "name", "annual_revenue", "tech_stack",
        "account_details", "engagement_history",
    }
