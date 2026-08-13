from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.classify import classify, ClassifyRequest
    from models.signals import ContentSignals
    from models.verdict import Verdict


@workflow.defn
class ClassifyContentWorkflow:
    @workflow.run
    async def run(self, signals: ContentSignals) -> Verdict:
        return await workflow.execute_activity(
            classify,
            ClassifyRequest(signals=signals),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
