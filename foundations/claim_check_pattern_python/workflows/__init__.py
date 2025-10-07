from temporalio import workflow
from activities.large_data_processing import (
    LargeDataset,
    ProcessingResult,
    generate_large_dataset,
    process_large_dataset,
)
from datetime import timedelta


@workflow.defn
class LargeDataProcessingWorkflow:
    """Workflow that demonstrates the Claim Check pattern with large datasets.
    
    This workflow generates a large dataset and processes it, showing how
    the claim check pattern enables handling of large payloads transparently.
    """

    @workflow.run
    async def run(self, dataset_size: int = 1000) -> ProcessingResult:
        """Process large dataset using claim check pattern.
        
        Args:
            dataset_size: Number of items to generate in the dataset
            
        Returns:
            ProcessingResult with processing statistics
        """
        # Generate a large dataset
        dataset = await workflow.execute_activity(
            generate_large_dataset,
            dataset_size,
            start_to_close_timeout=timedelta(minutes=5),
            summary="Generate large dataset"
        )
        
        # Process the large dataset
        result = await workflow.execute_activity(
            process_large_dataset,
            dataset,
            start_to_close_timeout=timedelta(minutes=10),
            summary="Process large dataset"
        )
        
        return result
