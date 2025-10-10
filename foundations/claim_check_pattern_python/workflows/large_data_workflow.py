from temporalio import workflow
from activities.large_data_processing import (
    LargeDataset,
    SummaryResult,
    transform_large_dataset,
    generate_summary,
)
from datetime import timedelta


@workflow.defn
class LargeDataProcessingWorkflow:
    """Workflow that demonstrates the Claim Check pattern with large datasets.
    
    This workflow demonstrates the claim check pattern by:
    1. Taking a large dataset as input (large workflow input)
    2. Transforming it into another large dataset (large activity input/output)
    3. Generating a summary from the transformed data (large activity input, small output)
    
    This shows how the claim check pattern handles large payloads at multiple stages.
    """

    @workflow.run
    async def run(self, dataset: LargeDataset) -> SummaryResult:
        """Process large dataset using claim check pattern with two-stage processing.
        
        Args:
            dataset: Large dataset to process (passed from client)
            
        Returns:
            SummaryResult with aggregated statistics
        """
        # Step 1: Transform the large dataset (large input -> large output)
        transformed_dataset = await workflow.execute_activity(
            transform_large_dataset,
            dataset,
            start_to_close_timeout=timedelta(minutes=10),
            summary="Transform large dataset"
        )
        
        # Step 2: Generate summary from transformed data (large input -> small output)
        summary_result = await workflow.execute_activity(
            generate_summary,
            transformed_dataset,
            start_to_close_timeout=timedelta(minutes=5),
            summary="Generate summary from transformed data"
        )
        
        return summary_result
