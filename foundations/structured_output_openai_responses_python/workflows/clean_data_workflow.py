from pydantic import BaseModel, Field, field_validator, EmailStr
from pydantic_core import PydanticCustomError
import re
from temporalio import workflow
from activities import invoke_model
from activities.invoke_model import InvokeModelRequest
from typing import List, Optional
from datetime import timedelta
from temporalio.common import RetryPolicy


class Business(BaseModel):
    name: Optional[str] = Field(
        None,
        description="The business name",
        json_schema_extra={"example": "Acme Corporation"},
    )
    email: Optional[EmailStr] = Field(
        None,
        description="Primary business email address",
        json_schema_extra={"example": "info@acmecorp.com"},
    )
    phone: Optional[str] = Field(
        None,
        description="Primary business phone number in E.164 format",
        json_schema_extra={"example": "+12025550173"},
    )
    address: Optional[str] = Field(
        None,
        description="Business mailing address",
        json_schema_extra={
            "example": "123 Business Park Dr, Suite 100, New York, NY 10001"
        },
    )
    website: Optional[str] = Field(
        None,
        description="Business website URL",
        json_schema_extra={"example": "https://www.acmecorp.com"},
    )
    industry: Optional[str] = Field(
        None,
        description="Business industry or sector",
        json_schema_extra={"example": "Technology"},
    )

    @field_validator("phone", mode="before")
    def validate_phone(cls, v):
        # Allow None values
        if v is None:
            return None

        if isinstance(v, str):
            v = v.strip()
            # Allow empty strings to be converted to None for optional fields
            if not v:
                return None

            # E.164 format: + followed by 1-9, then 9-15 more digits
            e164_pattern = r"^\+[1-9]\d{9,15}$"

            if not re.match(e164_pattern, v):
                raise PydanticCustomError(
                    "phone_format",
                    "Phone number must be in E.164 format (e.g., +12025550173)",
                    {"invalid_phone": v},
                )

        return v

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        # Allow None values
        if v is None:
            return None

        if isinstance(v, str):
            v = v.strip()
            # Convert empty strings to None (this is acceptable)
            if not v:
                return None

        return v


class BusinessList(BaseModel):
    businesses: List[Business]


@workflow.defn
class CleanDataWorkflow:
    @workflow.run
    async def run(self, data: str) -> BusinessList:
        results = await workflow.execute_activity(
            invoke_model.invoke_model,
            InvokeModelRequest(
                model="gpt-4o",
                instructions="""Extract and clean business data with these specific rules:

1. BUSINESS NAME: Extract the main business name, normalize capitalization (Title Case for proper nouns)
2. EMAIL:
   - Extract only ONE primary email address
   - If multiple emails, choose the one marked as "primary" or the first valid one
   - Validate format (must have @ and valid domain with .)
   - Set to null if invalid (e.g., "bob@email", "NONE PROVIDED")
3. PHONE:
   - Convert to E.164 format (+1 prefix for US numbers, add if not provided)
   - Convert letters to numbers where appropriate (e.g., "1-800-FLOWERS" → "+18003569377")
   - Set to null if cannot be converted to valid E.164 format
   - Examples: "(555) 123-4567" → "+15551234567", "555 234 5678 ext 349i" → null (invalid), "5551234567" → "+15551234567"
4. ADDRESS:
   - Provide complete, standardized address
   - Set to null if vague/incomplete (e.g., "north end of main st", "unknown", "[PRIVATE]")
5. WEBSITE:
   - Standardize to https:// format
   - Remove "www." prefix, add https:// if missing
   - Set to null if broken/invalid (e.g., "broken-link.com/404", "down for maintenance")
6. INDUSTRY:
   - Use clear, professional industry categories
   - Normalize similar terms (e.g., "fix cars and trucks" → "Automotive Repair")

Return null for any field that cannot be reliably extracted or validated.""",
                input=data,
                response_format=BusinessList,
            ),
            start_to_close_timeout=timedelta(seconds=300),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
            ),
            summary="Clean data",
        )
        return results.response
