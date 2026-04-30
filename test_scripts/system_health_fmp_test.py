import numpy as np
import pytest
from scipy import sparse

from liberata_metrics.metrics.system_health_metrics import (
    total_fair_market_price,
    get_reviewer_shrinkage_rate,
    get_replicator_shrinkage_rate,
    get_reviewer_fmp_volatility,
    get_replicator_fmp_volatility,
)

class TestTotalFairMarketPrice:

    def test_reviewer_fmp(self, cap_role, role_params):
        result = total_fair_market_price(
            cap_role, role_params["contributor_map"], is_reviewer=True
        )
        assert result == pytest.approx(5.0)

    def test_replicator_fmp_zero(self, cap_role, role_params):
        result = total_fair_market_price(
            cap_role, role_params["contributor_map"], is_reviewer=False
        )
        assert result == pytest.approx(0.0)

    def test_reviewer_fmp_t1(self, cap_role_t1, role_params):
        result = total_fair_market_price(
            cap_role_t1, role_params["contributor_map"], is_reviewer=True
        )
        assert result == pytest.approx(7.5)

    def test_empty_matrix_returns_zero(self, role_params):
        p = role_params
        empty = sparse.csr_matrix((p["M"], p["total_cols"]))
        assert total_fair_market_price(empty, p["contributor_map"], is_reviewer=True) == 0.0
        assert total_fair_market_price(empty, p["contributor_map"], is_reviewer=False) == 0.0

    def test_replicator_fmp_nonzero(self, role_params):
        """Matrix with explicit replicator column values."""
        p = role_params
        M, total_cols = p["M"], p["total_cols"]
        num_c = p["num_contributors"]
        # replicator col = M + 2 * num_c
        replicator_col = M + 2 * num_c
        data = [4.0, 6.0]
        rows = [0, 1]
        cols = [replicator_col, replicator_col]
        cap = sparse.csr_matrix((data, (rows, cols)), shape=(M, total_cols))
        result = total_fair_market_price(cap, p["contributor_map"], is_reviewer=False)
        assert result == pytest.approx(10.0)



class TestReviewerShrinkageRate:

    def test_increasing_fmp_gives_negative_shrinkage(self, cap_role, cap_role_t1, role_params):
        """If reviewer FMP grows, shrinkage rate is negative."""
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role_t1], role_params["contributor_map"]
        )
        assert result < 0.0

    def test_decreasing_fmp_gives_positive_shrinkage(self, cap_role, cap_role_t1, role_params):
        """If reviewer FMP falls, shrinkage rate is positive."""
        result = get_reviewer_shrinkage_rate(
            [cap_role_t1, cap_role], role_params["contributor_map"]
        )
        assert result > 0.0

    def test_flat_history_zero_shrinkage(self, cap_role, role_params):
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == pytest.approx(0.0)

    def test_exact_value_two_steps(self, cap_role, cap_role_t1, role_params):
        # T0 reviewer FMP = 5.0, T1 = 7.5 -> shrinkage = -(7.5 - 5.0) / 1 = -2.5
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role_t1], role_params["contributor_map"]
        )
        assert result == pytest.approx(-2.5)

    def test_multi_step_divides_by_T(self, cap_role, cap_role_t1, role_params):
        # T=2, change = 7.5 - 5.0 = 2.5 -> shrinkage = -2.5 / 2
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role, cap_role_t1], role_params["contributor_map"]
        )
        assert result == pytest.approx(-2.5 / 2)

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_reviewer_shrinkage_rate([cap_role], role_params["contributor_map"])

    def test_empty_history_raises(self, role_params):
        with pytest.raises(ValueError):
            get_reviewer_shrinkage_rate([], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        dense = cap_role.toarray()
        with pytest.raises(TypeError):
            get_reviewer_shrinkage_rate([cap_role, dense], role_params["contributor_map"])


# ── get_replicator_shrinkage_rate ──────────────────────────────────────────────

class TestReplicatorShrinkageRate:

    def test_flat_history_zero_shrinkage(self, cap_role, role_params):
        result = get_replicator_shrinkage_rate(
            [cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == pytest.approx(0.0)

    def test_exact_value(self, role_params):
        """Build a matrix with known replicator values and check exact output."""
        p = role_params
        M, total_cols, num_c = p["M"], p["total_cols"], p["num_contributors"]
        replicator_col = M + 2 * num_c

        def _make(val):
            return sparse.csr_matrix(
                ([val, val], ([0, 1], [replicator_col, replicator_col])),
                shape=(M, total_cols),
            )

        cap_a = _make(5.0)   # total replicator FMP = 10.0
        cap_b = _make(8.0)   # total replicator FMP = 16.0
        result = get_replicator_shrinkage_rate([cap_a, cap_b], p["contributor_map"])
        # shrinkage = -(16 - 10) / 1 = -6.0
        assert result == pytest.approx(-6.0)

    def test_decreasing_fmp_positive_shrinkage(self, role_params):
        p = role_params
        M, total_cols, num_c = p["M"], p["total_cols"], p["num_contributors"]
        replicator_col = M + 2 * num_c

        def _make(val):
            return sparse.csr_matrix(
                ([val, val], ([0, 1], [replicator_col, replicator_col])),
                shape=(M, total_cols),
            )

        result = get_replicator_shrinkage_rate(
            [_make(8.0), _make(5.0)], p["contributor_map"]
        )
        assert result > 0.0

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_replicator_shrinkage_rate([cap_role], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        dense = cap_role.toarray()
        with pytest.raises(TypeError):
            get_replicator_shrinkage_rate([cap_role, dense], role_params["contributor_map"])


# ── get_reviewer_fmp_volatility ────────────────────────────────────────────────

class TestReviewerFmpVolatility:

    def test_constant_prices_zero_volatility(self, cap_role, role_params):
        result = get_reviewer_fmp_volatility(
            [cap_role, cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == pytest.approx(0.0)

    def test_result_is_non_negative(self, cap_role, cap_role_t1, role_params):
        result = get_reviewer_fmp_volatility(
            [cap_role, cap_role_t1, cap_role], role_params["contributor_map"]
        )
        assert result >= 0.0

    def test_exact_value(self, cap_role, cap_role_t1, role_params):
        # prices = [5.0, 7.5], n=2, std(ddof=0) = 1.25, vol = 1.25 * sqrt(2)
        prices = np.array([5.0, 7.5])
        expected = np.std(prices, ddof=0) * np.sqrt(2)
        result = get_reviewer_fmp_volatility(
            [cap_role, cap_role_t1], role_params["contributor_map"]
        )
        assert result == pytest.approx(expected)

    def test_higher_spread_higher_volatility(self, cap_role, cap_role_t1, role_params):
        vol_small = get_reviewer_fmp_volatility(
            [cap_role, cap_role_t1], role_params["contributor_map"]
        )
        cap_role_t2 = cap_role_t1.multiply(2.0)
        vol_large = get_reviewer_fmp_volatility(
            [cap_role, cap_role_t2], role_params["contributor_map"]
        )
        assert vol_large > vol_small

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_reviewer_fmp_volatility([cap_role], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        dense = cap_role.toarray()
        with pytest.raises(TypeError):
            get_reviewer_fmp_volatility([cap_role, dense], role_params["contributor_map"])


# ── get_replicator_fmp_volatility ──────────────────────────────────────────────

class TestReplicatorFmpVolatility:

    def test_constant_prices_zero_volatility(self, cap_role, role_params):
        result = get_replicator_fmp_volatility(
            [cap_role, cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == pytest.approx(0.0)

    def test_exact_value(self, role_params):
        p = role_params
        M, total_cols, num_c = p["M"], p["total_cols"], p["num_contributors"]
        replicator_col = M + 2 * num_c

        def _make(val):
            return sparse.csr_matrix(
                ([val, val], ([0, 1], [replicator_col, replicator_col])),
                shape=(M, total_cols),
            )

        caps = [_make(v) for v in [4.0, 6.0, 8.0]]
        # total FMP per step: 8, 12, 16 ; std(ddof=0) * sqrt(3)
        prices = np.array([8.0, 12.0, 16.0])
        expected = np.std(prices, ddof=0) * np.sqrt(3)
        result = get_replicator_fmp_volatility(caps, p["contributor_map"])
        assert result == pytest.approx(expected)

    def test_result_is_non_negative(self, role_params):
        p = role_params
        M, total_cols, num_c = p["M"], p["total_cols"], p["num_contributors"]
        replicator_col = M + 2 * num_c

        def _make(val):
            return sparse.csr_matrix(
                ([val], ([0], [replicator_col])), shape=(M, total_cols)
            )

        result = get_replicator_fmp_volatility(
            [_make(3.0), _make(7.0)], p["contributor_map"]
        )
        assert result >= 0.0

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_replicator_fmp_volatility([cap_role], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        dense = cap_role.toarray()
        with pytest.raises(TypeError):
            get_replicator_fmp_volatility([cap_role, dense], role_params["contributor_map"])


# ── Large matrix tests ─────────────────────────────────────────────────────────

@pytest.mark.large
class TestSystemHealthFmpLarge:

    def test_reviewer_shrinkage_is_finite(self, large_data):
        cap = large_data.capital.tocsr()
        history = [cap, cap.multiply(0.9)]
        result = get_reviewer_shrinkage_rate(history, large_data.contributor_index_map)
        assert np.isfinite(result)

    def test_replicator_shrinkage_is_finite(self, large_data):
        cap = large_data.capital.tocsr()
        history = [cap, cap.multiply(1.1)]
        result = get_replicator_shrinkage_rate(history, large_data.contributor_index_map)
        assert np.isfinite(result)

    def test_reviewer_volatility_non_negative(self, large_data):
        cap = large_data.capital.tocsr()
        history = [cap, cap.multiply(1.1), cap.multiply(0.9)]
        result = get_reviewer_fmp_volatility(history, large_data.contributor_index_map)
        assert result >= 0.0

    def test_replicator_volatility_non_negative(self, large_data):
        cap = large_data.capital.tocsr()
        history = [cap, cap.multiply(1.1), cap.multiply(0.9)]
        result = get_replicator_fmp_volatility(history, large_data.contributor_index_map)
        assert result >= 0.0

    def test_flat_reviewer_shrinkage_zero(self, large_data):
        cap = large_data.capital.tocsr()
        result = get_reviewer_shrinkage_rate(
            [cap, cap, cap], large_data.contributor_index_map
        )
        assert result == pytest.approx(0.0)

    def test_flat_replicator_shrinkage_zero(self, large_data):
        cap = large_data.capital.tocsr()
        result = get_replicator_shrinkage_rate(
            [cap, cap, cap], large_data.contributor_index_map
        )
        assert result == pytest.approx(0.0)
