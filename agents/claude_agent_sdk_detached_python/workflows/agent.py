from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from models import RunRequest, RunResult


@workflow.defn
class ClaudeRun:
    @workflow.run
    async def run(self, request: RunRequest) -> RunResult:
        return await workflow.execute_activity(
            "run_claude",
            request,
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
            result_type=RunResult,
        )
