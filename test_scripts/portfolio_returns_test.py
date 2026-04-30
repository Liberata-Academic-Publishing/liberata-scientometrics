"""
Tests for get_returns and get_expected_returns.

Run only correctness tests:
    pytest test_scripts/portfolio_returns_test.py -m "not large"

Run everything (requires large data on disk):
    pytest test_scripts/portfolio_returns_test.py
"""
import time

import numpy as np
import pytest
from scipy import sparse


from liberata_metrics.metrics.portfolio_metrics import (
    academic_capital,
    get_expected_returns,
    get_returns,
)


# -----------------------------------------------------------------------
# SMALL MATRIX - Correctness
# All values below are computed by hand and have a single correct answer.
# -----------------------------------------------------------------------


class TestGetReturnsCorrectness:
    """Exact return values on 2x4 hand-verifiable matrices."""

    def test_positive_return(self, cap_t0, cap_t1, small_params):
        """total 30->45 over dt=1  ->  return = (45-30)/1 = 15."""
        result = get_returns(
            cap_t0, cap_t1, small_params["contributor_map"], 1.0
        )
        assert result == pytest.approx(15.0, rel=1e-6)

    def test_negative_return(self, cap_t1, cap_t2, small_params):
        """total 45->22.5 over dt=1  ->  return = (22.5-45)/1 = -22.5."""
        result = get_returns(
            cap_t1, cap_t2, small_params["contributor_map"], 1.0
        )
        assert result == pytest.approx(-22.5, rel=1e-6)

    def test_nonpositive_time_interval_raises(
        self, cap_t0, cap_t1, small_params
    ):
        """Non-positive time interval must raise ValueError."""
        with pytest.raises(ValueError):
            get_returns(
                cap_t0, cap_t1, small_params["contributor_map"], 0.0
            )

    def test_unchanged_capital_returns_zero(self, cap_t0, small_params):
        """Identical matrices -> return must be exactly 0.0."""
        result = get_returns(
            cap_t0, cap_t0, small_params["contributor_map"], 1.0
        )
        assert result == 0.0

    @pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 3.0, 10.0])
    def test_uniform_scale_exact(self, cap_t0, small_params, scale):
        """
        Scaling every entry by k -> total_end = k * total_start (=30)
        -> return = (k-1) * 30 / dt, an algebraic identity that must hold exactly.
        """
        cap_scaled = cap_t0.multiply(scale)
        result = get_returns(
            cap_t0, cap_scaled, small_params["contributor_map"], 1.0
        )
        assert result == pytest.approx((scale - 1.0) * 30.0, rel=1e-6)

    def test_single_contributor_single_paper(self, small_params):
        """One contributor, one paper: return fully determined by two scalars."""
        p = small_params
        start = sparse.csr_matrix(
            ([10.0], ([0], [2])), shape=(p["M"], p["total_cols"])
        )
        end = sparse.csr_matrix(
            ([20.0], ([0], [2])), shape=(p["M"], p["total_cols"])
        )
        result = get_returns(start, end, {"Alice": 0}, 1.0)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_type_error_dense_start(self, cap_t1, small_params):
        """Non-sparse start matrix must raise TypeError."""
        with pytest.raises(TypeError):
            get_returns(
                np.array([[1, 0, 0, 0]]),
                cap_t1,
                small_params["contributor_map"],
                1.0,
            )

    def test_type_error_dense_end(self, cap_t0, small_params):
        """Non-sparse end matrix must raise TypeError."""
        with pytest.raises(TypeError):
            get_returns(
                cap_t0,
                np.array([[1, 0, 0, 0]]),
                small_params["contributor_map"],
                1.0,
            )


class TestGetExpectedReturnsCorrectness:
    """Exact expected-return values on hand-computable histories."""

    def test_three_entry_history_arithmetic_mean(
        self, cap_t0, cap_t1, cap_t2, small_params
    ):
        """History [30->45, 45->22.5] over dt=1: mean = (15 + -22.5) / 2 = -3.75."""
        result = get_expected_returns(
            [cap_t0, cap_t1, cap_t2], [0.0, 1.0, 2.0], small_params["contributor_map"]
        )
        assert result == pytest.approx(-3.75, rel=1e-6)

    def test_two_entry_history_equals_single_return(
        self, cap_t0, cap_t1, small_params
    ):
        """Minimum valid history (2 entries) -> expected = single return = 15."""
        result = get_expected_returns(
            [cap_t0, cap_t1], [0.0, 1.0], small_params["contributor_map"]
        )
        assert result == pytest.approx(15.0, rel=1e-6)

    def test_steady_growth(self, cap_t0, cap_t1, small_params):
        """History [30->45, 45->90] over dt=1 each: mean = (15 + 45) / 2 = 30."""
        # cap_t1 total=45 (+15), cap_new total=90 (+45 from cap_t1)
        cap_new = sparse.csr_matrix(
            ([30.0, 60.0], ([0, 1], [2, 3])), shape=(2, 4)
        )
        result = get_expected_returns(
            [cap_t0, cap_t1, cap_new], [0.0, 1.0, 2.0], small_params["contributor_map"]
        )
        assert result == pytest.approx(30.0, rel=1e-6)

    def test_single_entry_history_raises(self, cap_t0, small_params):
        """Fewer than 2 entries must raise ValueError."""
        with pytest.raises(ValueError):
            get_expected_returns(
                [cap_t0], [0.0], small_params["contributor_map"]
            )

    def test_empty_history_raises(self, small_params):
        """Empty history must raise ValueError."""
        with pytest.raises(ValueError):
            get_expected_returns([], [], small_params["contributor_map"])

    def test_non_sparse_element_in_history_raises(
        self, cap_t0, small_params
    ):
        """A dense array inside the history list must raise TypeError."""
        with pytest.raises(TypeError):
            get_expected_returns(
                [cap_t0, np.array([[1, 0, 0, 0]])],
                [0.0, 1.0],
                small_params["contributor_map"],
            )


# -----------------------------------------------------------------------
# LARGE MATRIX - Scalability & mathematical invariants
# -----------------------------------------------------------------------

# Wall-clock budgets - generous to avoid CI flakiness
TIMING_BUDGET_RETURNS_S = 2.0   # seconds per get_returns call
TIMING_BUDGET_EXPECTED_S = 5.0  # seconds for a 3-step get_expected_returns


@pytest.mark.large
class TestGetReturnsLarge:
    """Invariant tests on an 800x1200 real-structure sparse matrix."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """get_returns must produce a finite float on a real large matrix."""
        result = get_returns(
            large_data.capital,
            large_data.capital.multiply(1.5),
            large_contributor_subset,
            1.0,
        )
        assert np.isfinite(result)

    def test_completes_within_time_budget(
        self, large_data, large_contributor_subset
    ):
        """Must finish in time (catches O(n^2) regressions)."""
        cap = large_data.capital
        t0 = time.perf_counter()
        get_returns(cap, cap.multiply(1.5), large_contributor_subset, 1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < TIMING_BUDGET_RETURNS_S, (
            f"get_returns took {elapsed:.2f}s "
            f"(budget: {TIMING_BUDGET_RETURNS_S}s)"
        )

    @pytest.mark.parametrize("scale", [0.5, 1.5, 2.0])
    def test_uniform_scale_invariant(
        self, large_data, large_contributor_subset, scale
    ):
        """
        Multiplying the entire capital matrix by k over dt=1 yields
        return = (k-1) * total_cap.
        Holds for any non-negative matrix and any contributor subset.
        """
        cap = large_data.capital
        total_cap = academic_capital(cap, large_contributor_subset)
        result = get_returns(
            cap, cap.multiply(scale), large_contributor_subset, 1.0
        )
        assert result == pytest.approx((scale - 1.0) * total_cap, rel=1e-5)

    def test_unchanged_matrix_gives_zero(
        self, large_data, large_contributor_subset
    ):
        """Identical start and end matrices -> return = 0.0."""
        cap = large_data.capital
        result = get_returns(cap, cap, large_contributor_subset, 1.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_monotonicity(self, large_data, large_contributor_subset):
        """
        Holding the start matrix fixed, a larger end capital must produce
        a strictly higher return (monotonicity of the return formula).
        """
        cap = large_data.capital
        r_low = get_returns(
            cap, cap.multiply(1.1), large_contributor_subset, 1.0
        )
        r_high = get_returns(
            cap, cap.multiply(2.0), large_contributor_subset, 1.0
        )
        assert r_high > r_low

    def test_shrinkage_yields_negative_return(
        self, large_data, large_contributor_subset
    ):
        """Capital reduction must produce a strictly negative return."""
        cap = large_data.capital
        result = get_returns(
            cap, cap.multiply(0.5), large_contributor_subset, 1.0
        )
        assert result < 0.0

    def test_single_contributor_subset_is_finite(self, large_data):
        """
        A one-contributor portfolio is the smallest valid subset.
        Result must be finite and equal to (k-1) * contributor_cap.
        """
        first_key = next(iter(large_data.contributor_index_map))
        subset = {first_key: large_data.contributor_index_map[first_key]}
        cap = large_data.capital
        scale = 1.2
        total_cap = academic_capital(cap, subset)
        result = get_returns(cap, cap.multiply(scale), subset, 1.0)
        assert np.isfinite(result)
        assert result == pytest.approx((scale - 1.0) * total_cap, rel=1e-5)


@pytest.mark.large
class TestGetExpectedReturnsLarge:
    """Scalability and invariant tests for get_expected_returns."""

    def test_result_is_finite(self, large_data, large_contributor_subset):
        """get_expected_returns must produce a finite float."""
        cap = large_data.capital
        result = get_expected_returns(
            [cap, cap.multiply(1.2), cap.multiply(1.5)],
            [0.0, 1.0, 2.0],
            large_contributor_subset,
        )
        assert np.isfinite(result)

    def test_completes_within_time_budget(
        self, large_data, large_contributor_subset
    ):
        """3-step history must complete within the timing budget."""
        cap = large_data.capital
        t0 = time.perf_counter()
        get_expected_returns(
            [cap, cap.multiply(1.2), cap.multiply(1.44)],
            [0.0, 1.0, 2.0],
            large_contributor_subset,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < TIMING_BUDGET_EXPECTED_S, (
            f"get_expected_returns took {elapsed:.2f}s "
            f"(budget: {TIMING_BUDGET_EXPECTED_S}s)"
        )

    def test_constant_growth_rate_invariant(
        self, large_data, large_contributor_subset
    ):
        """
        Geometric history with per-step growth factor k, dt=1:
            history = [C, k*C, k^2*C]
        r1 = (k-1)*ac, r2 = k*(k-1)*ac -> expected = (k-1)*ac*(1+k)/2.
        Holds for any k > 0 and any non-negative sparse matrix.
        """
        cap = large_data.capital
        k = 1.3
        ac = academic_capital(cap, large_contributor_subset)
        history = [cap, cap.multiply(k), cap.multiply(k ** 2)]
        result = get_expected_returns(history, [0.0, 1.0, 2.0], large_contributor_subset)
        expected = (k - 1.0) * ac * (1.0 + k) / 2.0
        assert result == pytest.approx(expected, rel=1e-5)

    def test_up_then_down_arithmetic(
        self, large_data, large_contributor_subset
    ):
        """
        History [C, 1.5*C, C] with equal dt=1:
            r1 = +0.5*ac,  r2 = -0.5*ac
            expected = 0 (symmetric absolute changes cancel).
        """
        cap = large_data.capital
        result = get_expected_returns(
            [cap, cap.multiply(1.5), cap], [0.0, 1.0, 2.0], large_contributor_subset
        )
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_monotone_growth_history_positive(
        self, large_data, large_contributor_subset
    ):
        """A strictly increasing 5-step history must yield positive mean."""
        cap = large_data.capital
        history = [cap.multiply(1.0 + 0.1 * i) for i in range(5)]
        result = get_expected_returns(history, [float(i) for i in range(5)], large_contributor_subset)
        assert result > 0.0
