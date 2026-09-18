import pytest

from financial_rag_agent.tools.calculator import cagr, growth_rate, margin, yoy_change


def test_growth_rate_basic():
    assert growth_rate(130497, 130497 / 1.1) == pytest.approx(0.1)


def test_growth_rate_matches_real_nvidia_revenue_figures():
    # Real FY2025 -> FY2026 revenue: 130,497 -> 215,938 ($ millions). The
    # filing itself states this as "65% Change" in its MD&A table.
    result = growth_rate(215938, 130497)
    assert result * 100 == pytest.approx(65.5, abs=0.5)


def test_growth_rate_zero_base_raises():
    with pytest.raises(ValueError, match="zero base"):
        growth_rate(100, 0)


def test_margin_basic():
    assert margin(139297, 215938) == pytest.approx(0.6449, abs=0.001)


def test_margin_zero_denominator_raises():
    with pytest.raises(ValueError, match="zero denominator"):
        margin(100, 0)


def test_yoy_change_structure():
    result = yoy_change(215938, 130497)
    assert result["absolute"] == pytest.approx(85441)
    assert result["percent"] == pytest.approx(65.5, abs=0.5)


def test_cagr_basic():
    # doubling over 3 years
    result = cagr(100, 200, 3)
    assert result == pytest.approx(0.2599, abs=0.001)


def test_cagr_negative_begin_value_raises():
    with pytest.raises(ValueError, match="positive begin_value"):
        cagr(-100, 200, 3)


def test_cagr_zero_periods_raises():
    with pytest.raises(ValueError, match="positive number of periods"):
        cagr(100, 200, 0)
