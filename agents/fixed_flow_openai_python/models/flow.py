from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["APPROVED", "REVIEW_REQUIRED", "REJECTED"]
HumanDecision = Literal["APPROVED", "REJECTED"]


class MortgageApplication(BaseModel):
    case_id: str
    credit_score: int = Field(ge=300, le=850)
    monthly_income: float = Field(gt=0)
    monthly_debt: float = Field(ge=0)
    loan_amount: float = Field(gt=0)
    property_value: float = Field(gt=0)


class FixedFlowInput(BaseModel):
    application: MortgageApplication
    model: str = "gpt-4o-mini"


class UnderwritingMetrics(BaseModel):
    debt_to_income: float
    loan_to_value: float


class UnderwritingAnalysis(BaseModel):
    summary: str
    risks: list[str]
    missing_evidence: list[str]
    recommended_decision: Decision


class CriticReview(BaseModel):
    accepted: bool
    issues: list[str]
    revision_instructions: list[str]


class DecisionRecommendation(BaseModel):
    decision: Decision
    rationale: str
    conditions: list[str]


class AgentTask(BaseModel):
    application: MortgageApplication
    metrics: UnderwritingMetrics
    previous_analysis: UnderwritingAnalysis | None = None
    critique: CriticReview | None = None
    model: str = "gpt-4o-mini"


class HumanReviewInput(BaseModel):
    reviewer: str
    decision: HumanDecision
    notes: str


class HumanReviewResult(HumanReviewInput):
    timestamp: str


class HumanReviewPacket(BaseModel):
    application: MortgageApplication
    metrics: UnderwritingMetrics
    analysis: UnderwritingAnalysis
    critique: CriticReview
    recommendation: DecisionRecommendation
    policy_violations: list[str]
    review_reasons: list[str]


class FixedFlowResult(BaseModel):
    case_id: str
    final_decision: HumanDecision
    metrics: UnderwritingMetrics
    analysis: UnderwritingAnalysis
    critique: CriticReview
    recommendation: DecisionRecommendation
    policy_violations: list[str]
    revision_count: int
    human_review_required: bool
    human_review: HumanReviewResult | None = None
