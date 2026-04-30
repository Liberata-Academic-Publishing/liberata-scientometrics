import numpy as np
import pytest
from scipy import sparse

from liberata_metrics.metrics.portfolio_metrics import (
    allocation_weights,
    portfolio_hhi,
    portfolio_gini,
    portfolio_normalized_entropy,
)

class TestAllocationWeights:
    def test_correct_per_manuscript(self, cap_t0, small_params):
        w = allocation_weights(cap_t0, small_params["contributor_map"])
        assert w[0] == 1/3
        assert w[1] == 2/3

    def test_zero_matrix_returns_empty(self, cap_zero, small_params):
        assert allocation_weights(cap_zero, small_params["contributor_map"]) == {}

    def test_empty_contributor_map_returns_empty(self, cap_t0):
        assert allocation_weights(cap_t0, {}) == {}

    def test_role_blocked_weights_sum_to_one(self, cap_role, role_params):
        w = allocation_weights(cap_role, role_params["contributor_map"])
        assert sum(w.values()) == 1

    def test_role_blocked_correct_values(self, cap_role, role_params):
        w = allocation_weights(cap_role, role_params["contributor_map"])
        assert w[0] == 13/20
        assert w[1] == 7/20

class TestPortfolioHHI:
    def test_known_value(self, cap_t0, small_params):
        hhi = portfolio_hhi(cap_t0, small_params["contributor_map"])
        assert hhi == 5/9

    def test_zero_matrix_returns_zero(self, cap_zero, small_params):
        assert portfolio_hhi(cap_zero, small_params["contributor_map"]) == 0.0

    def test_range(self, cap_t0, small_params):
        hhi = portfolio_hhi(cap_t0, small_params["contributor_map"])
        assert 0.0 < hhi <= 1.0

    def test_scales_invariant(self, cap_t0, cap_t1, small_params):
        hhi0 = portfolio_hhi(cap_t0, small_params["contributor_map"])
        hhi1 = portfolio_hhi(cap_t1, small_params["contributor_map"])
        assert abs(hhi0 - hhi1) < 1e-9

    def test_role_blocked(self, cap_role, role_params):
        hhi = portfolio_hhi(cap_role, role_params["contributor_map"])
        expected = (13/20)**2 + (7/20)**2
        assert hhi == expected


class TestPortfolioGini:
    def test_non_negative(self, cap_t0, small_params):
        assert portfolio_gini(cap_t0, small_params["contributor_map"]) >= 0.0

    def test_zero_matrix_returns_zero(self, cap_zero, small_params):
        assert portfolio_gini(cap_zero, small_params["contributor_map"]) == 0.0

    def test_non_sparse_raises(self, small_params):
        with pytest.raises(TypeError):
            portfolio_gini(np.eye(3), small_params["contributor_map"])

    def test_scales_invariant(self, cap_t0, cap_t1, small_params):
        g0 = portfolio_gini(cap_t0, small_params["contributor_map"])
        g1 = portfolio_gini(cap_t1, small_params["contributor_map"])
        assert abs(g0 - g1) < 1e-9

    def test_more_unequal_higher_gini(
        self, cap_t0, cap_new, small_params
    ):
        # cap_t0: [1/3, 2/3] — unequal
        # cap_new: [1/3, 2/3] same ratio, so use a more equal matrix for comparison
        # Build a perfectly equal matrix manually
        p = small_params
        cap_equal = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])),
            shape=(p["M"], p["total_cols"]),
        )
        g_equal = portfolio_gini(cap_equal, p["contributor_map"])
        g_unequal = portfolio_gini(cap_t0, p["contributor_map"])
        assert g_equal < g_unequal

    def test_role_blocked_non_negative(self, cap_role, role_params):
        assert portfolio_gini(cap_role, role_params["contributor_map"]) >= 0.0



class TestPortfolioNormalizedEntropy:
    def test_range(self, cap_t0, small_params):
        h = portfolio_normalized_entropy(cap_t0, small_params["contributor_map"])
        assert 0.0 <= h <= 1.0

    def test_zero_matrix_returns_zero(self, cap_zero, small_params):
        assert portfolio_normalized_entropy(
            cap_zero, small_params["contributor_map"]
        ) == 0.0

    def test_non_sparse_raises(self, small_params):
        with pytest.raises(TypeError):
            portfolio_normalized_entropy(np.eye(3), small_params["contributor_map"])

    def test_scales_invariant(self, cap_t0, cap_t1, small_params):
        h0 = portfolio_normalized_entropy(cap_t0, small_params["contributor_map"])
        h1 = portfolio_normalized_entropy(cap_t1, small_params["contributor_map"])
        assert abs(h0 - h1) < 1e-9

    def test_uniform_is_one(self, small_params):
        p = small_params
        cap_equal = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])),
            shape=(p["M"], p["total_cols"]),
        )
        h = portfolio_normalized_entropy(cap_equal, p["contributor_map"])
        assert h == 1

    def test_more_uniform_higher_entropy(self, cap_t0, small_params):
        p = small_params
        cap_equal = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])),
            shape=(p["M"], p["total_cols"]),
        )
        h_equal = portfolio_normalized_entropy(cap_equal, p["contributor_map"])
        h_unequal = portfolio_normalized_entropy(cap_t0, p["contributor_map"])
        assert h_equal > h_unequal

    def test_role_blocked(self, cap_role, role_params):
        h = portfolio_normalized_entropy(cap_role, role_params["contributor_map"])
        assert 0.0 <= h <= 1.0
