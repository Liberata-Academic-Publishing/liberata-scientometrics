"""
Tests for get_volatility, get_sharpe_ratio, and get_arc.
"""
import time

import numpy as np
import pytest

from liberata_metrics.metrics.portfolio_metrics import (
    get_arc,
    get_sharpe_ratio,
    get_volatility,
)


class TestGetVolatilityCorrectness:
    """Exact volatility values on hand-computable capital histories."""

    def test_symmetric_history(
        self, cap_t0, cap_t1, cap_t2, small_params
    ):
        """
        History [t0, t1, t2]: returns = [+0.5, -0.5], mean = 0.
        sse = 0.25 + 0.25 = 0.5
        vol = sqrt(0.5 / 2) = sqrt(0.25) = 0.5
        """
        result = get_volatility([cap_t0, cap_t1, cap_t2], small_params["contributor_map"])[0]
        assert result == 0.5

    def test_steady_growth_history(
        self, cap_t0, cap_t1, cap_new, small_params
    ):
        """
        History [t0, t1, new]: returns = [+0.5, +1.0], mean = 0.75.
        sse = (-0.25)^2 + (0.25)^2 = 0.125
        vol = sqrt(0.125 / 2) = sqrt(0.0625) = 0.25
        """
        result = get_volatility([cap_t0, cap_t1, cap_new], small_params["contributor_map"])[0]
        assert result == 0.25

    def test_two_entry_history_zero_vol(
        self, cap_t0, cap_t1, small_params
    ):
        """Single-period history: one return -> sse = 0 -> vol = 0."""
        result = get_volatility([cap_t0, cap_t1], small_params["contributor_map"])[0]
        assert result == 0

    def test_constant_returns_zero_vol(self, cap_t0, small_params):
        """All returns identical -> no spread -> vol = 0."""
        result = get_volatility([cap_t0, cap_t0, cap_t0], small_params["contributor_map"])[0]
        assert result == 0

    def test_insufficient_history_raises(self, cap_t0, small_params):
        """Fewer than 2 entries must raise ValueError."""
        with pytest.raises(ValueError):
            get_volatility([cap_t0], small_params["contributor_map"])

    def test_empty_history_raises(self, small_params):
        """Empty history must raise ValueError."""
        with pytest.raises(ValueError):
            get_volatility([], small_params["contributor_map"])

    def test_non_sparse_element_raises(self, cap_t0, small_params):
        """Non-sparse element in history must raise TypeError."""
        with pytest.raises(TypeError):
            get_volatility([cap_t0, np.array([[1, 0, 0, 0]])], small_params["contributor_map"])


class TestGetSharpeRatioCorrectness:
    """Exact Sharpe ratio values on hand-computable capital histories."""

    def test_zero_mean_return(
        self, cap_t0, cap_t1, cap_t2, small_params
    ):
        """
        History [t0, t1, t2]: expected = 0.0, vol = 0.5.
        Sharpe = 0.0 / 0.5 = 0.0
        """
        result = get_sharpe_ratio(
            [cap_t0, cap_t1, cap_t2], small_params["contributor_map"]
        )
        assert result == 0

    def test_positive_sharpe(
        self, cap_t0, cap_t1, cap_new, small_params
    ):
        """
        History [t0, t1, new]: expected = 0.75, vol = 0.25.
        Sharpe = 0.75 / 0.25 = 3.0
        """
        result = get_sharpe_ratio([cap_t0, cap_t1, cap_new], small_params["contributor_map"])
        assert result == 3

    def test_zero_volatility_returns_zero(self, cap_t0, small_params):
        """vol = 0 (constant returns) -> Sharpe = 0 by the guard."""
        result = get_sharpe_ratio([cap_t0, cap_t0, cap_t0], small_params["contributor_map"])
        assert result == 0

    def test_two_entry_history_zero_vol(
        self, cap_t0, cap_t1, small_params
    ):
        """Single-period history -> vol = 0 -> Sharpe = 0."""
        result = get_sharpe_ratio([cap_t0, cap_t1], small_params["contributor_map"])
        assert result == 0

    def test_insufficient_history_raises(self, cap_t0, small_params):
        """Fewer than 2 entries must raise ValueError."""
        with pytest.raises(ValueError):
            get_sharpe_ratio([cap_t0], small_params["contributor_map"])

    def test_empty_history_raises(self, small_params):
        """Empty history must raise ValueError."""
        with pytest.raises(ValueError):
            get_sharpe_ratio([], small_params["contributor_map"])

    def test_non_sparse_element_raises(self, cap_t0, small_params):
        """Non-sparse element in history must raise TypeError."""
        with pytest.raises(TypeError):
            get_sharpe_ratio([cap_t0, np.array([[1, 0, 0, 0]])],small_params["contributor_map"])


class TestGetArcCorrectness:
    """Exact ARC values on hand-computable capital histories."""

    def test_growth_arc(self, cap_t0, cap_t1, small_params):
        """
        History [t0, t1]:
            recent_return = 0.5, current_capital = 45.0
            ARC = 0.5 / 45 = 1/90
        """
        result = get_arc([cap_t0, cap_t1], small_params["contributor_map"])
        assert result == 0.5/45

    def test_decline_arc(self, cap_t1, cap_t2, small_params):
        """
        History [t1, t2]:
            recent_return = -0.5, current_capital = 22.5
            ARC = -0.5 / 22.5  (negative)
        """
        result = get_arc([cap_t1, cap_t2], small_params["contributor_map"])
        assert result == -0.5/22.5

    def test_uses_only_last_two_entries(
        self, cap_t0, cap_t2, cap_t1, small_params
    ):
        """
        History [t0, t2, t1]: ARC uses only the final pair (t2, t1).
            recent_return = (45-22.5)/22.5 = 1.0
            current_capital = 45.0
            ARC = 1.0 / 45.0
        """
        result = get_arc([cap_t0, cap_t2, cap_t1], small_params["contributor_map"])
        assert result == 1/45

    def test_zero_current_capital_returns_zero(
        self, cap_t0, cap_zero, small_params
    ):
        """ARC = 0 when current capital is zero (divide-by-zero guard)."""
        result = get_arc([cap_t0, cap_zero], small_params["contributor_map"])
        assert result == 0.0

    def test_identical_matrices_zero_arc(self, cap_t0, small_params):
        """Zero recent return -> ARC = 0."""
        result = get_arc([cap_t0, cap_t0], small_params["contributor_map"])
        assert result == 0

    def test_insufficient_history_raises(self, cap_t0, small_params):
        """Fewer than 2 entries must raise ValueError."""
        with pytest.raises(ValueError):
            get_arc([cap_t0], small_params["contributor_map"])

    def test_empty_history_raises(self, small_params):
        """Empty history must raise ValueError."""
        with pytest.raises(ValueError):
            get_arc([], small_params["contributor_map"])

    def test_non_sparse_element_raises(self, cap_t0, small_params):
        """Non-sparse element in history must raise TypeError."""
        with pytest.raises(TypeError):
            get_arc(
                [cap_t0, np.array([[1, 0, 0, 0]])],
                small_params["contributor_map"],
            )

TIMING_BUDGET_S = 5.0  # seconds


@pytest.mark.large
class TestGetVolatilityLarge:
    """Invariant tests for get_volatility on 800x1200 matrices."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """Must return a finite float on a real large matrix."""
        cap = large_data.capital
        result = get_volatility([cap, cap.multiply(1.5), cap.multiply(0.8)],large_contributor_subset)[0]
        assert np.isfinite(result)

    def test_result_is_non_negative(
        self, large_data, large_contributor_subset
    ):
        """Volatility is a standard deviation: always >= 0."""
        cap = large_data.capital
        result = get_volatility([cap, cap.multiply(1.5), cap.multiply(0.8)],large_contributor_subset)[0]
        assert result >= 0.0

    def test_completes_within_time_budget(
        self, large_data, large_contributor_subset
    ):
        """Must complete within the timing budget."""
        cap = large_data.capital
        t0 = time.perf_counter()
        get_volatility(
            [cap, cap.multiply(1.2), cap.multiply(0.9)],
            large_contributor_subset,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < TIMING_BUDGET_S, (
            f"get_volatility took {elapsed:.2f}s "
            f"(budget: {TIMING_BUDGET_S}s)"
        )

    def test_uniform_growth_zero_vol(
        self, large_data, large_contributor_subset
    ):
        """
        When every period return equals k-1, all returns are identical
        -> no spread -> volatility = 0 regardless of matrix content.
        """
        cap = large_data.capital
        k = 1.3
        history = [cap, cap.multiply(k), cap.multiply(k ** 2)]
        result = get_volatility(history, large_contributor_subset)[0]
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_two_entry_history_zero_vol(
        self, large_data, large_contributor_subset
    ):
        """
        Two-entry history: one return, deviation from itself is 0
        -> vol = 0, regardless of matrix size.
        """
        cap = large_data.capital
        result = get_volatility([cap, cap.multiply(1.5)], large_contributor_subset)[0]
        assert result == pytest.approx(0.0, abs=1e-9)


@pytest.mark.large
class TestGetSharpeRatioLarge:
    """Invariant tests for get_sharpe_ratio on 800x1200 matrices."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """Must return a finite float on a real large matrix."""
        cap = large_data.capital
        result = get_sharpe_ratio(
            [cap, cap.multiply(2.0), cap.multiply(2.5)],
            large_contributor_subset,
        )
        assert np.isfinite(result)

    def test_completes_within_time_budget(
        self, large_data, large_contributor_subset
    ):
        """Must complete within the timing budget."""
        cap = large_data.capital
        t0 = time.perf_counter()
        get_sharpe_ratio(
            [cap, cap.multiply(2.0), cap.multiply(2.5)],
            large_contributor_subset,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < TIMING_BUDGET_S, (
            f"get_sharpe_ratio took {elapsed:.2f}s "
            f"(budget: {TIMING_BUDGET_S}s)"
        )

    def test_uniform_growth_zero_sharpe(
        self, large_data, large_contributor_subset
    ):
        """
        Uniform growth -> vol = 0 -> Sharpe = 0 (by zero-vol guard).
        Holds for any matrix content.
        """
        cap = large_data.capital
        k = 1.3
        history = [cap, cap.multiply(k), cap.multiply(k ** 2)]
        result = get_sharpe_ratio(history, large_contributor_subset)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_exact_sharpe_from_scaling(
        self, large_data, large_contributor_subset
    ):
        """
        History [C, 2C, 2.5C]:
            r1 = 1.0,  r2 = 0.25,  mean = 0.625
            sse = 2 * 0.375^2 = 0.28125
            vol = sqrt(0.28125 / 2) = 0.375
            Sharpe = 0.625 / 0.375 = 5/3

        This identity holds for ANY matrix with positive total capital
        because returns depend only on the scale factors (1->2->2.5).
        """
        cap = large_data.capital
        history = [cap, cap.multiply(2.0), cap.multiply(2.5)]
        result = get_sharpe_ratio(history, large_contributor_subset)
        assert result == pytest.approx(5.0 / 3.0, rel=1e-5)


@pytest.mark.large
class TestGetArcLarge:
    """Invariant tests for get_arc on 800x1200 matrices."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """Must return a finite float on a real large matrix."""
        cap = large_data.capital
        result = get_arc(
            [cap, cap.multiply(1.5)], large_contributor_subset
        )
        assert np.isfinite(result)

    def test_completes_within_time_budget(
        self, large_data, large_contributor_subset
    ):
        """Must complete within the timing budget."""
        cap = large_data.capital
        t0 = time.perf_counter()
        get_arc([cap, cap.multiply(1.5)], large_contributor_subset)
        elapsed = time.perf_counter() - t0
        assert elapsed < TIMING_BUDGET_S, (
            f"get_arc took {elapsed:.2f}s "
            f"(budget: {TIMING_BUDGET_S}s)"
        )

    def test_growth_gives_positive_arc(
        self, large_data, large_contributor_subset
    ):
        """Positive recent return and positive capital -> ARC > 0."""
        cap = large_data.capital
        result = get_arc(
            [cap, cap.multiply(1.5)], large_contributor_subset
        )
        assert result > 0.0

    def test_shrinkage_gives_negative_arc(
        self, large_data, large_contributor_subset
    ):
        """Negative recent return and positive capital -> ARC < 0."""
        cap = large_data.capital
        result = get_arc(
            [cap, cap.multiply(0.5)], large_contributor_subset
        )
        assert result < 0.0

    def test_zero_return_gives_zero_arc(
        self, large_data, large_contributor_subset
    ):
        """Zero recent return -> ARC = 0, regardless of capital level."""
        cap = large_data.capital
        result = get_arc([cap, cap], large_contributor_subset)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_uses_only_last_two_entries(
        self, large_data, large_contributor_subset
    ):
        """
        Only the final two matrices determine ARC.
        Prepending extra history must not change the result.
        """
        cap = large_data.capital
        arc_two = get_arc(
            [cap, cap.multiply(1.5)], large_contributor_subset
        )
        arc_longer = get_arc(
            [cap.multiply(0.1), cap, cap.multiply(1.5)],
            large_contributor_subset,
        )
        assert arc_two == pytest.approx(arc_longer, rel=1e-9)
