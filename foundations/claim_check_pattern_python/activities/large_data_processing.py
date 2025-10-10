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


@dataclass
class TransformedDataset:
    """Large dataset after transformation - still large but processed."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    transformation_stats: Dict[str, Any]


@dataclass
class SummaryResult:
    """Final summary result - small payload."""
    total_items: int
    processed_items: int
    transformation_stats: Dict[str, Any]
    summary_stats: Dict[str, Any]
    errors: List[str]


@activity.defn
async def transform_large_dataset(dataset: LargeDataset) -> TransformedDataset:
    """Transform a large dataset - produces another large dataset.
    
    This activity demonstrates how the claim check pattern allows processing
    of large datasets without hitting Temporal's payload size limits.
    The transformation produces another large dataset that gets passed to
    the next activity.
    
    Args:
        dataset: Large dataset to transform
        
    Returns:
        TransformedDataset with enhanced data and transformation statistics
    """
    processed_count = 0
    errors = []
    transformation_stats = {
        "total_items": len(dataset.data),
        "transformations_applied": []
    }
    
    # Create a copy of the data to transform
    transformed_data = []
    
    for item in dataset.data:
        try:
            # Create a new item with transformations
            transformed_item = item.copy()
            
            # Apply various transformations
            if "value" in item:
                transformed_item["processed_value"] = item["value"] * 2
                transformed_item["value_category"] = "high" if item["value"] > 1000 else "low"
                processed_count += 1
                
            if "text" in item:
                # Simulate text processing
                words = item["text"].split()
                transformed_item["word_count"] = len(words)
                transformed_item["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
                transformed_item["text_sentiment"] = "positive" if "good" in item["text"].lower() else "neutral"
                processed_count += 1
                
            # Add additional computed fields
            transformed_item["computed_score"] = (
                transformed_item.get("processed_value", 0) * 0.7 + 
                transformed_item.get("word_count", 0) * 0.3
            )
            
            transformed_data.append(transformed_item)
            
        except Exception as e:
            errors.append(f"Error transforming item {item.get('id', 'unknown')}: {str(e)}")
            # Still add the original item even if transformation failed
            transformed_data.append(item)
    
    transformation_stats["processed_items"] = processed_count
    transformation_stats["error_count"] = len(errors)
    transformation_stats["transformations_applied"] = [
        "value_doubling", "category_assignment", "text_analysis", 
        "sentiment_analysis", "score_computation"
    ]
    
    # Update metadata
    updated_metadata = dataset.metadata.copy()
    updated_metadata["transformed_at"] = "2024-01-01T00:00:00Z"
    updated_metadata["transformation_version"] = "1.0"
    
    return TransformedDataset(
        data=transformed_data,
        metadata=updated_metadata,
        transformation_stats=transformation_stats
    )


@activity.defn
async def generate_summary(transformed_dataset: TransformedDataset) -> SummaryResult:
    """Generate a summary from the transformed dataset.
    
    This activity takes the large transformed dataset and produces a small
    summary result, demonstrating the claim check pattern with large activity
    input and small output.
    
    Args:
        transformed_dataset: Large transformed dataset
        
    Returns:
        SummaryResult with aggregated statistics
    """
    data = transformed_dataset.data
    total_items = len(data)
    
    # Calculate summary statistics
    summary_stats = {
        "total_items": total_items,
        "value_stats": {
            "min_value": min(item.get("value", 0) for item in data),
            "max_value": max(item.get("value", 0) for item in data),
            "avg_value": sum(item.get("value", 0) for item in data) / total_items if total_items > 0 else 0,
            "high_value_count": sum(1 for item in data if item.get("value_category") == "high")
        },
        "text_stats": {
            "total_words": sum(item.get("word_count", 0) for item in data),
            "avg_word_count": sum(item.get("word_count", 0) for item in data) / total_items if total_items > 0 else 0,
            "avg_word_length": sum(item.get("avg_word_length", 0) for item in data) / total_items if total_items > 0 else 0,
            "positive_sentiment_count": sum(1 for item in data if item.get("text_sentiment") == "positive")
        },
        "score_stats": {
            "min_score": min(item.get("computed_score", 0) for item in data),
            "max_score": max(item.get("computed_score", 0) for item in data),
            "avg_score": sum(item.get("computed_score", 0) for item in data) / total_items if total_items > 0 else 0
        }
    }
    
    return SummaryResult(
        total_items=total_items,
        processed_items=transformed_dataset.transformation_stats.get("processed_items", 0),
        transformation_stats=transformed_dataset.transformation_stats,
        summary_stats=summary_stats,
        errors=transformed_dataset.transformation_stats.get("error_count", 0)
    )

