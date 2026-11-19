"""
Agent Execution Workflow

Orchestrates durable agent execution using the Claude Agent SDK.

Follows Temporal best practices:
- Deterministic workflow code (no direct I/O)
- Long timeouts for agent execution
- String-based activity names to avoid importing activity modules
- Models errors as completions (not exceptions)
"""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from models import AgentInput, AgentOutput


@workflow.defn
class AgentExecutionWorkflow:
    """
    Durable agent execution workflow.

    Steps:
    1. Execute agent via Claude Agent SDK (long-running, heartbeated)
    2. Log the result (lightweight, quick)

    The workflow is durable — it survives process restarts and will
    automatically retry failed activities with exponential backoff.
    """

    @workflow.run
    async def run(self, input_data: AgentInput) -> AgentOutput:
        workflow.logger.info(
            f"Starting AgentExecutionWorkflow: prompt_length={len(input_data.prompt)}"
        )

        # Step 1: Execute agent
        #
        # Timeout rationale:
        # - start_to_close_timeout (30 min): agents can run complex multi-step
        #   tasks with many tool calls; 30 minutes is a reasonable upper bound.
        # - heartbeat_timeout (10 min): the activity sends heartbeats every 60s
        #   via a background task.  10 minutes gives plenty of buffer.  If the
        #   staleness guard stops heartbeating (after 15 min idle), Temporal
        #   kills the activity ~25 min after the last event.
        #
        # Retry rationale:
        # - 3 attempts with exponential backoff for transient failures.
        # - ValueError and PermissionError are non-retryable because they
        #   indicate bad input, not a transient issue.
        execution_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            non_retryable_error_types=["ValueError", "PermissionError"],
        )

        output: AgentOutput = await workflow.execute_activity(
            "execute_agent_activity",
            input_data,
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=10),
            retry_policy=execution_retry_policy,
            result_type=AgentOutput,
        )

        workflow.logger.info(
            f"Agent execution completed: status={output.status}, "
            f"response_length={len(output.response)}"
        )

        # Step 2: Log result
        #
        # In a production system, this step would persist the result to a
        # database (messages table, run records, etc.).  Here we keep it
        # simple and just log.
        log_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=10),
            backoff_coefficient=2.0,
            maximum_attempts=5,
        )

        await workflow.execute_activity(
            "log_result_activity",
            output,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=log_retry_policy,
        )

        workflow.logger.info(
            f"AgentExecutionWorkflow completed: status={output.status}"
        )

        return output
