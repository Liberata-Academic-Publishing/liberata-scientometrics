#Tests for get_academic_capital_growth_rate.
import numpy as np
import pytest
from scipy import sparse

from liberata_metrics.metrics.system_health_metrics import get_academic_capital_growth_rate
from liberata_metrics.metrics.portfolio_metrics import academic_capital


class TestGrowthRateCorrectness:

    def test_positive_growth(self, cap_t0, cap_t1, small_params):
        result = get_academic_capital_growth_rate(
            [cap_t0, cap_t1], small_params["contributor_map"]
        )
        assert result == 0.5

    def test_negative_growth(self, cap_t1, cap_t2, small_params):
        result = get_academic_capital_growth_rate(
            [cap_t1, cap_t2], small_params["contributor_map"]
        )
        assert result == -0.5

    def test_flat_history_returns_zero(self, cap_t0, small_params):
        result = get_academic_capital_growth_rate(
            [cap_t0, cap_t0, cap_t0], small_params["contributor_map"]
        )
        assert result == 0

    def test_zero_start_returns_zero(self, cap_zero, cap_t0, small_params):
        """Starting capital of zero, needs to return 0.0!!"""
        result = get_academic_capital_growth_rate(
            [cap_zero, cap_t0], small_params["contributor_map"]
        )
        assert result == 0.0


    def test_cagr_formula_multi_step(self, cap_t0, small_params):
        #Multi step multiply
        history = [cap_t0.multiply(1.0 + 0.1 * i) for i in range(5)]
        T = len(history) - 1
        cap_start = academic_capital(history[0], small_params["contributor_map"])
        cap_end = academic_capital(history[-1], small_params["contributor_map"])
        expected = (cap_end / cap_start) ** (1.0 / T) - 1.0
        result = get_academic_capital_growth_rate(
            history, small_params["contributor_map"]
        )
        assert result == expected

    def test_too_few_entries_raises(self, cap_t0, small_params):
        """Fewer than two entries must raise ValueError."""
        with pytest.raises(ValueError):
            get_academic_capital_growth_rate(
                [cap_t0], small_params["contributor_map"]
            )

    def test_empty_history_raises(self, small_params):
        """Empty history must raise ValueError."""
        with pytest.raises(ValueError):
            get_academic_capital_growth_rate([], small_params["contributor_map"])

    def test_non_sparse_in_history_raises(self, cap_t0, small_params):
        """A dense array inside the history list must raise TypeError."""
        dense = cap_t0.toarray()
        with pytest.raises(TypeError):
            get_academic_capital_growth_rate(
                [cap_t0, dense], small_params["contributor_map"]
            )


#Large matrices

@pytest.mark.large
class TestGrowthRateLarge:
    """Invariant tests on an 800x1200 real-structure sparse matrix."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """get_academic_capital_growth_rate must produce a finite float."""
        cap = large_data.capital
        history = [cap, cap.multiply(1.2), cap.multiply(1.44)]
        result = get_academic_capital_growth_rate(history, large_contributor_subset)
        assert np.isfinite(result)

    def test_flat_history_returns_zero(self, large_data, large_contributor_subset):
        """Identical matrices -> growth rate must be 0.0."""
        cap = large_data.capital
        result = get_academic_capital_growth_rate(
            [cap, cap, cap], large_contributor_subset
        )
        assert result == 0

    def test_shrinking_history_negative(self, large_data, large_contributor_subset):
        """Capital reduction must produce a strictly negative growth rate."""
        cap = large_data.capital
        result = get_academic_capital_growth_rate(
            [cap, cap.multiply(0.8)], large_contributor_subset
        )
        assert result < 0.0

    def test_monotonicity(self, large_data, large_contributor_subset):
        """Faster-growing history must produce a higher growth rate."""
        cap = large_data.capital
        r_slow = get_academic_capital_growth_rate(
            [cap, cap.multiply(1.1)], large_contributor_subset
        )
        r_fast = get_academic_capital_growth_rate(
            [cap, cap.multiply(2.0)], large_contributor_subset
        )
        assert r_fast > r_slow

    @pytest.mark.parametrize("scale", [0.5, 1.5, 2.0])
    def test_uniform_scale_cagr_invariant(
        self, large_data, large_contributor_subset, scale
    ):
        cap = large_data.capital
        result = get_academic_capital_growth_rate(
            [cap, cap.multiply(scale)], large_contributor_subset
        )
        assert result == scale-1