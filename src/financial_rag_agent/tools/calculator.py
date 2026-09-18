def growth_rate(current: float, previous: float) -> float:
    """(current - previous) / previous. Growth from a zero base is
    undefined, not "infinite" or "0" — raise rather than return a
    misleading number."""
    if previous == 0:
        raise ValueError("Cannot compute growth rate from a zero base value")
    return (current - previous) / previous


def margin(numerator: float, denominator: float) -> float:
    """numerator / denominator (e.g. operating income / revenue)."""
    if denominator == 0:
        raise ValueError("Cannot compute a margin/ratio with a zero denominator")
    return numerator / denominator


def yoy_change(current: float, previous: float) -> dict:
    """Year-over-year change, both as an absolute delta and a percent."""
    return {
        "absolute": current - previous,
        "percent": growth_rate(current, previous) * 100,
    }


def cagr(begin_value: float, end_value: float, periods: float) -> float:
    """Compound annual growth rate over `periods` years. Returns a
    fraction (0.15 == 15%/year), not a percentage."""
    if begin_value <= 0:
        raise ValueError("CAGR requires a positive begin_value")
    if periods <= 0:
        raise ValueError("CAGR requires a positive number of periods")
    return (end_value / begin_value) ** (1 / periods) - 1


# operation name -> (function, required field names, in call order).
# Adding a calculation means adding one entry here (plus the Literal in
# tools/schemas.py) — the router dispatches off this, no if/elif chain.
OPERATIONS: dict[str, tuple[object, tuple[str, ...]]] = {
    "growth_rate": (growth_rate, ("current", "previous")),
    "margin": (margin, ("numerator", "denominator")),
    "yoy_change": (yoy_change, ("current", "previous")),
    "cagr": (cagr, ("begin_value", "end_value", "periods")),
}
