"""
Tests for get_funding_efficiency and get_time_efficiency.

Two-tier strategy
-----------------
Small (correctness): 2x4 synthetic matrices, hand-computed exact values.
Large (invariants):  800x1200 pre-generated matrix; scale/monotonicity
                     properties that hold for any valid capital matrix.

Run all tests:
    pytest test_scripts/funding_time_efficiency_test.py -v

Skip large tests (no disk data needed):
    pytest test_scripts/funding_time_efficiency_test.py -v -m "not large"
"""
import math
import sys
from pathlib import Path

import pytest
from scipy import sparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.portfolio_metrics import (
    get_funding_efficiency,
    get_time_efficiency,
)


# ===========================================================================
# get_funding_efficiency -- small correctness tests
#
# Fixtures from conftest:
#   cap_t0  : Alice=10, Bob=20  -> total capital = 30
#   small_params["contributor_map"] = {"Alice": 0, "Bob": 1}
# ===========================================================================

class TestGetFundingEfficiencyCorrectness:

    def test_basic_ratio(self, cap_t0, small_params):
        # 30 capital / 10 funding = 3.0
        result = get_funding_efficiency(
            cap_t0, small_params["contributor_map"], funding=10.0
        )
        assert result == pytest.approx(3.0, rel=1e-9)

    def test_funding_larger_than_capital(self, cap_t0, small_params):
        # 30 capital / 60 funding = 0.5
        result = get_funding_efficiency(
            cap_t0, small_params["contributor_map"], funding=60.0
        )
        assert result == pytest.approx(0.5, rel=1e-9)

    def test_funding_equals_capital(self, cap_t0, small_params):
        # 30 capital / 30 funding = 1.0
        result = get_funding_efficiency(
            cap_t0, small_params["contributor_map"], funding=30.0
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_zero_capital_positive_funding(
        self, cap_zero, small_params
    ):
        # 0 capital / 5 funding = 0.0
        result = get_funding_efficiency(
            cap_zero, small_params["contributor_map"], funding=5.0
        )
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_zero_funding_returns_nan(self, cap_t0, small_params):
        # funding == 0 is undefined; function must return nan
        result = get_funding_efficiency(
            cap_t0, small_params["contributor_map"], funding=0.0
        )
        assert math.isnan(result)

    def test_negative_funding_raises(self, cap_t0, small_params):
        with pytest.raises(ValueError):
            get_funding_efficiency(
                cap_t0, small_params["contributor_map"], funding=-1.0
            )

    def test_non_sparse_raises_type_error(self, small_params):
        dense = np.ones((2, 4))
        with pytest.raises(TypeError):
            get_funding_efficiency(
                dense, small_params["contributor_map"], funding=10.0
            )

    def test_doubling_funding_halves_result(
        self, cap_t0, small_params
    ):
        # Funding efficiency is linear in 1/funding
        cmap = small_params["contributor_map"]
        r1 = get_funding_efficiency(cap_t0, cmap, funding=10.0)
        r2 = get_funding_efficiency(cap_t0, cmap, funding=20.0)
        assert r1 == pytest.approx(2 * r2, rel=1e-9)

    def test_doubling_capital_doubles_result(
        self, cap_t0, cap_new, small_params
    ):
        # cap_new = 2 * cap_t1 = 3 * cap_t0 -- use cap_t1 (45) vs cap_t0 (30)
        # cap_t0 total=30, cap_t1 total=45 -> not exactly double.
        # Use cap_t0 (30) and a 2x version of it.
        cmap = small_params["contributor_map"]
        p = small_params
        cap_double = sparse.csr_matrix(
            ([20.0, 40.0], ([0, 1], [2, 3])),
            shape=(p["M"], p["total_cols"]),
        )  # total = 60 = 2 * 30
        r1 = get_funding_efficiency(cap_t0, cmap, funding=10.0)
        r2 = get_funding_efficiency(cap_double, cmap, funding=10.0)
        assert r2 == pytest.approx(2 * r1, rel=1e-9)


# ===========================================================================
# get_funding_efficiency -- large invariant tests
# ===========================================================================

@pytest.mark.large
class TestGetFundingEfficiencyLarge:

    def test_positive_capital_positive_funding_gives_positive(
        self, large_data, large_contributor_subset
    ):
        result = get_funding_efficiency(
            large_data.capital,
            large_contributor_subset,
            funding=1_000_000.0,
        )
        assert result > 0.0

    def test_result_is_finite(
        self, large_data, large_contributor_subset
    ):
        result = get_funding_efficiency(
            large_data.capital,
            large_contributor_subset,
            funding=1.0,
        )
        assert math.isfinite(result)

    def test_doubling_funding_halves_result(
        self, large_data, large_contributor_subset
    ):
        cmap = large_contributor_subset
        r1 = get_funding_efficiency(
            large_data.capital, cmap, funding=500.0
        )
        r2 = get_funding_efficiency(
            large_data.capital, cmap, funding=1000.0
        )
        assert r1 == pytest.approx(2 * r2, rel=1e-9)

    def test_scaling_capital_scales_result(
        self, large_data, large_contributor_subset
    ):
        # Multiplying capital by k multiplies efficiency by k
        k = 2.5
        cmap = large_contributor_subset
        r1 = get_funding_efficiency(
            large_data.capital, cmap, funding=100.0
        )
        r2 = get_funding_efficiency(
            large_data.capital.multiply(k), cmap, funding=100.0
        )
        assert r2 == pytest.approx(k * r1, rel=1e-6)

    def test_zero_funding_returns_nan(
        self, large_data, large_contributor_subset
    ):
        result = get_funding_efficiency(
            large_data.capital, large_contributor_subset, funding=0.0
        )
        assert math.isnan(result)


# ===========================================================================
# get_time_efficiency -- small correctness tests
#
# Fixtures from conftest:
#   cap_t0  : total = 30
#   cap_t1  : total = 45  (+15 from T0)
#   cap_t2  : total = 22.5 (-22.5 from T1, -7.5 from T0)
# ===========================================================================

class TestGetTimeEfficiencyCorrectness:

    def test_two_periods_positive_growth(
        self, cap_t0, cap_t1, small_params
    ):
        # (45 - 30) / 1 period = 15.0
        result = get_time_efficiency(
            [cap_t0, cap_t1], small_params["contributor_map"]
        )
        assert result == pytest.approx(15.0, rel=1e-9)

    def test_three_periods_net_decline(
        self, cap_t0, cap_t1, cap_t2, small_params
    ):
        # (22.5 - 30) / 2 periods = -3.75
        result = get_time_efficiency(
            [cap_t0, cap_t1, cap_t2], small_params["contributor_map"]
        )
        assert result == pytest.approx(-3.75, rel=1e-9)

    def test_identical_matrices_zero_efficiency(
        self, cap_t0, small_params
    ):
        # No change over time -> efficiency = 0.0
        result = get_time_efficiency(
            [cap_t0, cap_t0], small_params["contributor_map"]
        )
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_single_entry_raises_value_error(
        self, cap_t0, small_params
    ):
        with pytest.raises(ValueError):
            get_time_efficiency(
                [cap_t0], small_params["contributor_map"]
            )

    def test_empty_history_raises_value_error(self, small_params):
        with pytest.raises(ValueError):
            get_time_efficiency([], small_params["contributor_map"])

    def test_non_sparse_raises_type_error(self, cap_t0, small_params):
        dense = np.ones((2, 4))
        with pytest.raises(TypeError):
            get_time_efficiency(
                [cap_t0, dense], small_params["contributor_map"]
            )

    def test_longer_history_uses_endpoints_only(
        self, cap_t0, cap_t1, cap_t2, cap_new, small_params
    ):
        # time_efficiency = (cap_end - cap_start) / T
        # Intermediate values do not matter; only T0 and T_last count.
        # [cap_t0(30), cap_t1(45), cap_t2(22.5), cap_new(90)]
        # -> (90 - 30) / 3 = 20.0
        cmap = small_params["contributor_map"]
        result = get_time_efficiency(
            [cap_t0, cap_t1, cap_t2, cap_new], cmap
        )
        assert result == pytest.approx(20.0, rel=1e-9)


# ===========================================================================
# get_time_efficiency -- large invariant tests
# ===========================================================================

@pytest.mark.large
class TestGetTimeEfficiencyLarge:

    def test_result_is_finite(
        self, large_data, large_contributor_subset
    ):
        cap = large_data.capital
        result = get_time_efficiency(
            [cap, cap.multiply(1.5)], large_contributor_subset
        )
        assert math.isfinite(result)

    def test_stationary_capital_zero_efficiency(
        self, large_data, large_contributor_subset
    ):
        # Same matrix at every time step -> efficiency = 0.0
        cap = large_data.capital
        result = get_time_efficiency(
            [cap, cap, cap], large_contributor_subset
        )
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_uniform_growth_scales_with_capital(
        self, large_data, large_contributor_subset
    ):
        # history [C, 2C]: efficiency = (2*total - total) / 1 = total
        # history [C, 3C]: efficiency = (3*total - total) / 1 = 2*total
        cap = large_data.capital
        cmap = large_contributor_subset
        r1 = get_time_efficiency([cap, cap.multiply(2.0)], cmap)
        r2 = get_time_efficiency([cap, cap.multiply(3.0)], cmap)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)

    def test_more_periods_reduces_rate(
        self, large_data, large_contributor_subset
    ):
        # Same total change over 1 vs 3 periods -> rate is 3x smaller
        cap = large_data.capital
        cmap = large_contributor_subset
        cap_end = cap.multiply(2.0)
        r1 = get_time_efficiency([cap, cap_end], cmap)
        mid = cap.multiply(1.5)
        r3 = get_time_efficiency([cap, mid, mid, cap_end], cmap)
        assert r1 == pytest.approx(3 * r3, rel=1e-6)
