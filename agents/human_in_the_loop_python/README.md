# Human-in-the-Loop AI Agent

This example demonstrates how to build an AI agent that requires human approval before executing certain actions using Temporal Signals.

## Overview

The workflow:
1. Uses an LLM to analyze a user request and propose an action
2. Pauses and waits for human approval via Temporal Signal
3. Executes the action if approved, or cancels if rejected/timed out

Key features:
- **Durable waiting**: Can wait hours or days for approval without consuming resources
- **Signal-based approval**: External systems send approval decisions via Temporal Signals
- **Timeout handling**: Automatically handles cases where approval is not received
- **Complete audit trail**: All decisions are logged for compliance

## Prerequisites

- Python 3.11+
- Temporal server running locally
- OpenAI API key

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Start Temporal Dev Server:
```bash
temporal server start-dev
```

## Running

### Start the Worker

In one terminal:
```bash
uv run python -m worker
```

### Start a Workflow

In another terminal:
```bash
uv run python -m start_workflow "Delete all test data from the production database"
```

The workflow will start, analyze the request, and pause for approval. Watch the worker output for instructions.

### Send Approval Decision

The worker output will show the workflow ID and request ID. Use the `send_approval` script to approve or reject:

**To approve:**
```bash
uv run python -m send_approval <workflow-id> <request-id> approve "Looks good"
```

**To reject:**
```bash
uv run python -m send_approval <workflow-id> <request-id> reject "Too risky"
```

### Testing Timeout

To test timeout behavior, simply don't send any approval signal. After 5 minutes (default), the workflow will automatically complete with a timeout result.

## Architecture

- **Models** (`models/approval.py`): Data structures for approval requests and decisions
- **Activities**:
  - `openai_responses.py`: Generic LLM invocation activity
  - `execute_action.py`: Executes approved actions
  - `notify_approval_needed.py`: Notifies external systems of approval requests
- **Workflow** (`workflows/human_in_the_loop_workflow.py`): Orchestrates the approval process
- **Scripts**:
  - `worker.py`: Runs the Temporal worker
  - `start_workflow.py`: Starts workflow execution
  - `send_approval.py`: Helper script to send approval signals

## Key Patterns

### Signal Handler
The workflow uses a signal handler to receive approval decisions asynchronously:
```python
@workflow.signal
async def approval_decision(self, decision: ApprovalDecision):
    if decision.request_id == self.pending_request_id:
        self.approval_decision = decision
```

### Waiting with Timeout
The workflow waits for approval with a configurable timeout:
```python
await workflow.wait_condition(
    lambda: self.approval_decision is not None,
    timeout=timedelta(seconds=timeout_seconds),
)
```

### Request Validation
Each approval request has a unique ID to prevent confusion from stale or duplicate approvals.

## Extensions

This pattern can be extended to support:
- Multiple approvers with voting
- Escalation workflows
- Conditional approval based on action risk
- Integration with Slack, email, or custom UIs
- Query handlers to check approval status
