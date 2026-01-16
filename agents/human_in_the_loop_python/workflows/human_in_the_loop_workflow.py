from temporalio import workflow
from datetime import timedelta
from typing import Optional
import asyncio

with workflow.unsafe.imports_passed_through():
    from models.models import WorkflowInput, ProposedAction, ApprovalRequest, ApprovalDecision
    from models.agent_instructions import SYSTEM_INSTRUCTIONS
    from activities import openai_responses, execute_action, notify_approval_needed


@workflow.defn
class HumanInTheLoopWorkflow:
    def __init__(self):
        self.current_decision: Optional[ApprovalDecision] = None
        self.pending_request_id: Optional[str] = None

    @workflow.run
    async def run(self, input: WorkflowInput) -> str:
        """Execute an AI agent workflow with human-in-the-loop approval.
        
        Args:
            input: Workflow input containing user_request and approval_timeout_seconds
            
        Returns:
            Result of the workflow execution
        """
        workflow.logger.info(f"Starting human-in-the-loop workflow for request: {input.user_request}")

        # Step 1: AI analyzes the request and proposes an action
        proposed_action = await self._analyze_and_propose_action(input.user_request)
        
        risk_status = "RISKY" if proposed_action.risky_action else "SAFE"
        workflow.logger.info(
            f"AI proposed action: {proposed_action.action_type} (Risk level: {risk_status})",
            extra={"proposed_action": proposed_action.model_dump()}
        )

        # Step 2: Request human approval only if action is risky
        if proposed_action.risky_action:
            workflow.logger.info("Action is risky, requesting human approval")
            approval_result = await self._request_approval(
                proposed_action, 
                input.user_request,
                timeout_seconds=input.approval_timeout_seconds
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
                timeout_msg = f"Action cancelled: approval request timed out after {input.approval_timeout_seconds} seconds"
                return timeout_msg
        else:
            # Auto-approve non-risky actions
            workflow.logger.info("Action is safe, auto-approving and proceeding with execution")
            result = await workflow.execute_activity(
                execute_action.execute_action,
                proposed_action,
                start_to_close_timeout=timedelta(seconds=60),
            )
            print(f"\n{'='*60}")
            print(f"Action completed successfully (auto-approved)")
            print(f"{'='*60}\n")

            return f"Action completed successfully (auto-approved): {result}"

    async def _analyze_and_propose_action(self, user_request: str) -> ProposedAction:
        """Use LLM to analyze request and propose an action."""
        result = await workflow.execute_activity(
            openai_responses.create,
            openai_responses.OpenAIResponsesRequest(
                model="gpt-4o-mini",
                instructions=SYSTEM_INSTRUCTIONS,
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