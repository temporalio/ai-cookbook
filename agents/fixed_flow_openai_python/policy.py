from models.flow import (
    CriticReview,
    DecisionRecommendation,
    MortgageApplication,
    UnderwritingMetrics,
)


def compute_metrics(application: MortgageApplication) -> UnderwritingMetrics:
    return UnderwritingMetrics(
        debt_to_income=round(application.monthly_debt / application.monthly_income, 4),
        loan_to_value=round(application.loan_amount / application.property_value, 4),
    )


def hard_stop_violations(
    application: MortgageApplication, metrics: UnderwritingMetrics
) -> list[str]:
    violations = []
    if application.credit_score < 620:
        violations.append("Credit score is below the example minimum of 620.")
    if metrics.debt_to_income > 0.50:
        violations.append("Debt-to-income is above the example maximum of 50%.")
    return violations


def review_reasons(
    recommendation: DecisionRecommendation,
    critique: CriticReview,
    violations: list[str],
) -> list[str]:
    reasons = []
    if violations:
        reasons.append("Binding policy checks require human review.")
    if not critique.accepted:
        reasons.append("The critic still found issues after the revision limit.")
    if recommendation.decision == "REVIEW_REQUIRED":
        reasons.append("The decision agent requested human review.")
    return reasons
