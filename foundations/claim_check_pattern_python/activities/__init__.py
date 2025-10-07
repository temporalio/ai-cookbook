from temporalio import activity
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class LargeDataset:
    """Represents a large dataset that would benefit from claim check pattern."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Result of processing the large dataset."""
    processed_count: int
    summary: Dict[str, Any]
    errors: List[str]


@activity.defn
async def process_large_dataset(dataset: LargeDataset) -> ProcessingResult:
    """Process a large dataset - this would normally cause payload size issues.
    
    This activity demonstrates how the claim check pattern allows processing
    of large datasets without hitting Temporal's payload size limits.
    
    Args:
        dataset: Large dataset to process
        
    Returns:
        ProcessingResult with processing statistics and any errors
    """
    processed_count = 0
    errors = []
    summary = {"total_items": len(dataset.data)}
    
    for item in dataset.data:
        try:
            # Simulate processing each item
            if "value" in item:
                item["processed_value"] = item["value"] * 2
                processed_count += 1
            elif "text" in item:
                # Simulate text processing
                item["word_count"] = len(item["text"].split())
                processed_count += 1
        except Exception as e:
            errors.append(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
    
    summary["processed_items"] = processed_count
    summary["error_count"] = len(errors)
    
    return ProcessingResult(
        processed_count=processed_count,
        summary=summary,
        errors=errors
    )


@activity.defn
async def generate_large_dataset(size: int = 1000) -> LargeDataset:
    """Generate a large dataset for demonstration purposes.
    
    Args:
        size: Number of items to generate in the dataset
        
    Returns:
        LargeDataset with generated data
    """
    data = []
    for i in range(size):
        item = {
            "id": f"item_{i}",
            "value": i * 10,
            "text": f"This is sample text for item {i}. " * 10,  # Make it larger
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "category": f"category_{i % 10}",
                "tags": [f"tag_{j}" for j in range(i % 5)]
            }
        }
        data.append(item)
    
    metadata = {
        "generated_at": "2024-01-01T00:00:00Z",
        "total_items": size,
        "description": "Sample large dataset for claim check pattern demonstration"
    }
    
    return LargeDataset(data=data, metadata=metadata)
