"""Tests for models/models.py — Pydantic model validation."""

import pytest
from pydantic import ValidationError

from models.models import WorkflowInput, ProposedAction, ApprovalRequest, ApprovalDecision


class TestWorkflowInput:
    def test_defaults(self):
        wi = WorkflowInput(user_request="Do something")
        assert wi.user_request == "Do something"
        assert wi.approval_timeout_seconds == 300

    def test_custom_timeout(self):
        wi = WorkflowInput(user_request="Delete data", approval_timeout_seconds=60)
        assert wi.approval_timeout_seconds == 60

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            WorkflowInput()


class TestProposedAction:
    def test_risky_action(self):
        action = ProposedAction(
            action_type="delete_data",
            description="Delete all test data",
            reasoning="User requested deletion",
            risky_action=True,
        )
        assert action.risky_action is True
        assert action.action_type == "delete_data"

    def test_safe_action(self):
        action = ProposedAction(
            action_type="list_files",
            description="List files in directory",
            reasoning="User wants to see files",
            risky_action=False,
        )
        assert action.risky_action is False


class TestApprovalRequest:
    def test_construction(self):
        action = ProposedAction(
            action_type="test",
            description="desc",
            reasoning="reason",
            risky_action=True,
        )
        req = ApprovalRequest(
            request_id="req-123",
            proposed_action=action,
            context="User said something",
            requested_at="2025-01-01T00:00:00",
        )
        assert req.request_id == "req-123"
        assert req.proposed_action.action_type == "test"


class TestApprovalDecision:
    def test_approved_with_notes(self):
        decision = ApprovalDecision(
            request_id="req-123",
            approved=True,
            reviewer_notes="Looks good",
            decided_at="2025-01-01T00:01:00",
        )
        assert decision.approved is True
        assert decision.reviewer_notes == "Looks good"

    def test_rejected_without_notes(self):
        decision = ApprovalDecision(
            request_id="req-123",
            approved=False,
            decided_at="2025-01-01T00:01:00",
        )
        assert decision.approved is False
        assert decision.reviewer_notes is None

    def test_serialization_roundtrip(self):
        decision = ApprovalDecision(
            request_id="req-456",
            approved=True,
            reviewer_notes="OK",
            decided_at="2025-01-01T00:00:00",
        )
        json_str = decision.model_dump_json()
        restored = ApprovalDecision.model_validate_json(json_str)
        assert restored.request_id == decision.request_id
        assert restored.approved == decision.approved
