"""Small, testable invariants used by the empirical design.

The full estimation pipeline remains in ``01_main_pipeline.py``. These helpers
make the paper's core definitions explicit and independently testable.
"""

from collections.abc import Iterable


def financial_subsidy_rate(private_benchmark: float, realized_soe_return: float) -> float:
    """Return the private benchmark minus realized SOE EBI/A."""

    return private_benchmark - realized_soe_return


def financial_subsidy_amount(subsidy_rate: float, book_assets: float) -> float:
    """Convert a subsidy rate into the signed annual opportunity-cost amount."""

    if book_assets < 0:
        raise ValueError("book_assets must be non-negative")
    return subsidy_rate * book_assets


def validate_out_of_time_window(outcome_years: Iterable[int], forecast_year: int) -> bool:
    """Check that every training outcome predates the forecast year."""

    years = tuple(outcome_years)
    if not years:
        raise ValueError("outcome_years must not be empty")
    return max(years) < forecast_year


def classify_common_support(distance: float, p90: float, p95: float) -> str:
    """Classify the distinct-firm 5-NN distance used in the final paper."""

    if distance < 0 or p90 < 0 or p95 < 0:
        raise ValueError("distances and thresholds must be non-negative")
    if p90 > p95:
        raise ValueError("p90 must not exceed p95")
    if distance <= p90:
        return "Good"
    if distance <= p95:
        return "Boundary"
    return "Extrapolation"
