from temporalio import workflow
from datetime import timedelta
from typing import Optional
import asyncio

with workflow.unsafe.imports_passed_through():
    from models.approval import ProposedAction, ApprovalRequest, ApprovalDecision
    from activities import openai_responses, execute_action, notify_approval_needed


@workflow.defn
class HumanInTheLoopWorkflow:
    def __init__(self):
        self.current_decision: Optional[ApprovalDecision] = None
        self.pending_request_id: Optional[str] = None

    @workflow.signal
    async def approval_decision(self, decision: ApprovalDecision):
        """Signal handler for receiving approval decisions from humans."""
        # Verify this decision is for the current pending request
        if decision.request_id == self.pending_request_id:
            self.current_decision = decision
            workflow.logger.info(
                f"Received approval decision: {'approved' if decision.approved else 'rejected'}",
                extra={"decision": decision.model_dump()}
            )
        else:
            workflow.logger.warning(
                f"Received decision for wrong request ID: {decision.request_id}, expected: {self.pending_request_id}"
            )

    @workflow.run
    async def run(self, user_request: str, approval_timeout_seconds: int = 300) -> str:
        """Execute an AI agent workflow with human-in-the-loop approval.
        
        Args:
            user_request: The user's request for the AI agent
            approval_timeout_seconds: How long to wait for approval (default: 5 minutes)
            
        Returns:
            Result of the workflow execution
        """
        workflow.logger.info(f"Starting human-in-the-loop workflow for request: {user_request}")

        # Step 1: AI analyzes the request and proposes an action
        proposed_action = await self._analyze_and_propose_action(user_request)
        
        workflow.logger.info(
            f"AI proposed action: {proposed_action.action_type}",
            extra={"proposed_action": proposed_action.model_dump()}
        )

        # Step 2: Request human approval
        approval_result = await self._request_approval(
            proposed_action, 
            user_request,
            timeout_seconds=approval_timeout_seconds
        )

        # Step 3: Handle the approval result
        if approval_result == "approved":
            workflow.logger.info("Action approved, proceeding with execution")
            result = await workflow.execute_activity(
                execute_action.execute_action,
                proposed_action,
                start_to_close_timeout=timedelta(seconds=60),
            )
            return f"Action completed successfully: {result}"
        
        elif approval_result == "rejected":
            workflow.logger.info("Action rejected by human reviewer")
            reviewer_notes = self.current_decision.reviewer_notes or 'None provided'
            return f"Action rejected. Reviewer notes: {reviewer_notes}"
        
        else:  # timeout
            workflow.logger.warning("Approval request timed out")
            timeout_msg = f"Action cancelled: approval request timed out after {approval_timeout_seconds} seconds"
            return timeout_msg

    async def _analyze_and_propose_action(self, user_request: str) -> ProposedAction:
        """Use LLM to analyze request and propose an action."""
        system_instructions = """
You are an AI assistant that analyzes user requests and proposes actions.
For each request, you should:
1. Determine what action needs to be taken
2. Provide a clear description of the action
3. Explain your reasoning for why this action addresses the request

Be thorough and clear in your analysis.

Respond with a JSON string in this structure:

{
  "action_type": "A short name for the action (e.g., \\"delete_test_data\\")",
  "description": "A clear description of what the action will do",
  "reasoning": "Your explanation for why this action addresses the request"
}
"""

        result = await workflow.execute_activity(
            openai_responses.create,
            openai_responses.OpenAIResponsesRequest(
                model="gpt-4o-mini",
                instructions=system_instructions,
                input=user_request,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Parse the JSON output into our ProposedAction model
        return ProposedAction.model_validate_json(result)

    async def _request_approval(
        self, 
        proposed_action: ProposedAction, 
        context: str,
        timeout_seconds: int
    ) -> str:
        """Request human approval and wait for response.
        
        Returns:
            "approved", "rejected", or "timeout"
        """
        # Generate unique request ID using workflow's deterministic UUID
        self.current_decision = None
        self.pending_request_id = str(workflow.uuid4())
        
        # Create approval request
        approval_request = ApprovalRequest(
            request_id=self.pending_request_id,
            proposed_action=proposed_action,
            context=context,
            requested_at=workflow.now().isoformat(),
        )

        # Send notification to external systems
        await workflow.execute_activity(
            notify_approval_needed.notify_approval_needed,
            approval_request,
            start_to_close_timeout=timedelta(seconds=10),
        )

        # Wait for approval decision with timeout
        try:
            await workflow.wait_condition(
                lambda: self.current_decision is not None,
                timeout=timedelta(seconds=timeout_seconds),
            )
            
            # Decision received
            if self.current_decision.approved:
                return "approved"
            else:
                return "rejected"
                
        except asyncio.TimeoutError:
            # Timeout waiting for approval
            return "timeout"
