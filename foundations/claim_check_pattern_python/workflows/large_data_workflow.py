from temporalio import workflow
from activities.large_data_processing import (
    LargeDataset,
    ProcessingResult,
    process_large_dataset,
)
from datetime import timedelta


@workflow.defn
class LargeDataProcessingWorkflow:
    """Workflow that demonstrates the Claim Check pattern with large datasets.
    
    This workflow processes a large dataset passed from the client, showing how
    the claim check pattern enables handling of large payloads transparently.
    """

    @workflow.run
    async def run(self, dataset: LargeDataset) -> ProcessingResult:
        """Process large dataset using claim check pattern.
        
        Args:
            dataset: Large dataset to process (passed from client)
            
        Returns:
            ProcessingResult with processing statistics
        """
        # Process the large dataset
        result = await workflow.execute_activity(
            process_large_dataset,
            dataset,
            start_to_close_timeout=timedelta(minutes=10),
            summary="Process large dataset"
        )
        
        return result
