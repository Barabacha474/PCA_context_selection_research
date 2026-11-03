"""Evaluation family re-exports."""
from pcadiv.evaluation.budget_curve import budget_curve
from pcadiv.evaluation.metrics import (
    BaseMetric,
    Coverage,
    MeanPairwiseDistance,
    VendiScore,
    get_metric,
    list_metrics,
    register_metric,
)
from pcadiv.evaluation.stats import (
    aggregate_seeds,
    interpolate_at_budgets,
    paired_compare,
    pivot_summary,
)

__all__ = [
    "BaseMetric",
    "Coverage",
    "MeanPairwiseDistance",
    "VendiScore",
    "aggregate_seeds",
    "budget_curve",
    "get_metric",
    "interpolate_at_budgets",
    "list_metrics",
    "paired_compare",
    "pivot_summary",
    "register_metric",
]