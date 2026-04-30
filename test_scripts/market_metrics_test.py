import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics import (
    compute_fair_marketprice,
    compute_relative_performance,
    compute_risk_adjusted_excess_return,
    compute_risk_adjusted_relative_performance,
    compute_risk_premiums,
    compute_sensitivity,
    compute_utility_function,
)
# from liberata_metrics.metrics.distribution_metrics import (
#     hhi_discrepancy,
#     share_splits_inequality,
# )
# from liberata_metrics.metrics.market_metrics import (
#     compute_hhi_discrepancy,
#     compute_share_splits_inequality,
# )


def _role_capital(scale: float = 1.0) -> sparse.csr_matrix:
    """Build a 1-manuscript, 1-contributor-per-role matrix."""
    data = [10.0 * scale, 2.0 * scale, 4.0 * scale]
    rows = [0, 0, 0]
    cols = [1, 2, 3]
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, 4))


def _uniform_shares_matrix(num_rows: int, num_cols: int) -> sparse.csr_matrix:
    values = np.full((num_rows, num_cols), 1.0 / num_cols, dtype=float)
    return sparse.csr_matrix(values)


@pytest.fixture(scope="module")
def small_market_params() -> dict:
    return {
        "M": 1,
        "total_cols": 4,
        "num_contributors": 1,
        "contributor_map": {"C0": 0},
        "author_mask": np.s_[:, 1:2],
        "reviewer_mask": np.s_[:, 2:3],
        "replicator_mask": np.s_[:, 3:4],
    }


@pytest.fixture(scope="module")
def market_capital_t0() -> sparse.csr_matrix:
    return _role_capital(1.0)


@pytest.fixture(scope="module")
def market_capital_t1() -> sparse.csr_matrix:
    return _role_capital(1.5)


@pytest.fixture(scope="module")
def market_capital_t2() -> sparse.csr_matrix:
    return _role_capital(0.75)


@pytest.fixture(scope="module")
def market_memberships() -> sparse.csr_matrix:
    return sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))


@pytest.fixture(scope="module")
def market_memberships_two_tags() -> sparse.csr_matrix:
    """One manuscript assigned to two tags to create a non-symmetric domain baseline."""
    return sparse.csr_matrix(([1.0, 1.0], ([0, 1], [0, 0])), shape=(2, 1))


@pytest.fixture(scope="module")
def market_shares() -> sparse.csr_matrix:
    return _role_capital(1.0)


def _num_contributors(capital: sparse.spmatrix) -> int:
    manuscripts = int(capital.shape[0])
    total_contributor_cols = int(capital.shape[1]) - manuscripts
    return total_contributor_cols // 3 if total_contributor_cols % 3 == 0 else total_contributor_cols


def _as_1d(value: object) -> np.ndarray:
    return np.asarray(value).ravel()


class TestComputeFairMarketpriceCorrectness:
    def test_small_exact_values(self, market_capital_t0, market_memberships, small_market_params):
        reviewer_fmp, replicator_fmp = compute_fair_marketprice(
            capital=market_capital_t0,
            mask_reviewers=small_market_params["reviewer_mask"],
            mask_replicators=small_market_params["replicator_mask"],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
            num_contributors=small_market_params["num_contributors"],
        )

        np.testing.assert_allclose(_as_1d(reviewer_fmp), np.array([2.0]))
        np.testing.assert_allclose(_as_1d(replicator_fmp), np.array([4.0]))


class TestComputeRiskPremiumsCorrectness:
    def test_small_exact_zero_premiums(self, market_capital_t0, market_memberships, small_market_params):
        reviewer_fmp, replicator_fmp = compute_fair_marketprice(
            capital=market_capital_t0,
            mask_reviewers=small_market_params["reviewer_mask"],
            mask_replicators=small_market_params["replicator_mask"],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
            num_contributors=small_market_params["num_contributors"],
        )

        reviewer_risk_premiums, replicator_risk_premiums = compute_risk_premiums(
            capital=market_capital_t0,
            mask_authors=small_market_params["author_mask"],
            mask_reviewers=small_market_params["reviewer_mask"],
            mask_replicators=small_market_params["replicator_mask"],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
            reviewer_fmp=reviewer_fmp,
            replicator_fmp=replicator_fmp,
        )

        np.testing.assert_allclose(_as_1d(reviewer_risk_premiums), np.array([0.0]))
        np.testing.assert_allclose(_as_1d(replicator_risk_premiums), np.array([0.0]))


class TestComputeUtilityFunctionCorrectness:
    def test_small_exact_utility(self, small_market_params, market_capital_t0, market_capital_t1, market_capital_t2):
        utility = compute_utility_function(
            [market_capital_t0, market_capital_t1, market_capital_t2],
            [0.0, 1.0, 2.0],
            small_market_params["contributor_map"],
            risk_willingness=2.0,
        )

        assert utility == pytest.approx(-4.061552812808831, rel=1e-9)


class TestComputeRelativePerformanceCorrectness:
    def test_small_exact_relative_performance(self, market_shares, market_capital_t0, market_capital_t1, market_memberships, small_market_params):
        result = compute_relative_performance(
            shares=market_shares,
            capital_history=[market_capital_t0, market_capital_t1],
            time_history=[0.0, 1.0],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
        )

        assert result == pytest.approx(1.0, rel=1e-9)


class TestComputeSensitivityCorrectness:
    def test_linear_relationship(self):
        manuscript = np.array([2.0, 4.0, 6.0, 8.0])
        domain = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_sensitivity(manuscript, domain) == pytest.approx(2.0, rel=1e-9)

    def test_zero_variance_returns_one(self):
        manuscript = np.array([5.0, 5.0, 5.0])
        domain = np.array([3.0, 3.0, 3.0])
        assert compute_sensitivity(manuscript, domain) == pytest.approx(1.0, rel=1e-9)


class TestComputeRiskAdjustedExcessReturnCorrectness:
    def test_small_exact_alpha(self, small_market_params, market_capital_t0, market_capital_t1, market_memberships):
        result = compute_risk_adjusted_excess_return(
            capital_history=[market_capital_t0, market_capital_t1],
            time_history=[0.0, 1.0],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
            manuscript_index=0,
        )

        assert result == pytest.approx(0.0, abs=1e-12)

    def test_small_nonsymmetric_alpha_is_negative(
        self,
        small_market_params,
        market_capital_t0,
        market_capital_t1,
        market_memberships_two_tags,
    ):
        result = compute_risk_adjusted_excess_return(
            capital_history=[market_capital_t0, market_capital_t1],
            time_history=[0.0, 1.0],
            manuscript_memberships=market_memberships_two_tags,
            contributor_index_map_subset=small_market_params["contributor_map"],
            manuscript_index=0,
        )

        # With two tags on one manuscript, domain return is doubled -> alpha should be negative and non-zero.
        assert result == pytest.approx(-8.0, abs=1e-12)


class TestComputeRiskAdjustedRelativePerformanceCorrectness:
    def test_small_exact_total(self, small_market_params, market_capital_t0, market_capital_t1, market_memberships):
        result = compute_risk_adjusted_relative_performance(
            capital_history=[market_capital_t0, market_capital_t1],
            time_history=[0.0, 1.0],
            manuscript_memberships=market_memberships,
            contributor_index_map_subset=small_market_params["contributor_map"],
        )

        assert result == pytest.approx(0.0, abs=1e-12)

    def test_small_nonsymmetric_total_is_negative(
        self,
        small_market_params,
        market_capital_t0,
        market_capital_t1,
        market_memberships_two_tags,
    ):
        result = compute_risk_adjusted_relative_performance(
            capital_history=[market_capital_t0, market_capital_t1],
            time_history=[0.0, 1.0],
            manuscript_memberships=market_memberships_two_tags,
            contributor_index_map_subset=small_market_params["contributor_map"],
        )

        # From alpha=-8 and domain return=16, expected portfolio total is -0.5 for one manuscript.
        assert result == pytest.approx(-0.5, abs=1e-12)


# class TestShareSplitsAndHHICorrectness:
#     def test_share_splits_inequality_small(self):
#         shares = sparse.csr_matrix(
#             [
#                 [0.75, 0.25],
#                 [0.50, 0.50],
#             ]
#         )
#         expected = (0.75**2 + 0.25**2 + 0.50**2 + 0.50**2) / 2.0

#         assert compute_share_splits_inequality(shares) == pytest.approx(expected, rel=1e-9)
#         assert compute_share_splits_inequality(shares) == pytest.approx(share_splits_inequality(shares), rel=1e-9)

#     def test_hhi_discrepancy_small(self):
#         shares = sparse.csr_matrix(
#             [
#                 [0.75, 0.25],
#                 [0.50, 0.50],
#             ]
#         )
#         expected = abs(((0.75**2 + 0.25**2 + 0.50**2 + 0.50**2) / 2.0) - (0.75**2 + 0.25**2))

#         assert compute_hhi_discrepancy(shares, slice(0, 1)) == pytest.approx(expected, rel=1e-9)
#         assert compute_hhi_discrepancy(shares, slice(0, 1)) == pytest.approx(hhi_discrepancy(shares, slice(0, 1)), rel=1e-9)


@pytest.mark.large
class TestComputeFairMarketpriceLarge:
    def test_outputs_are_finite_and_aligned(self, large_data):
        capital = large_data.capital
        num_contributors = _num_contributors(capital)

        reviewer_fmp, replicator_fmp = compute_fair_marketprice(
            capital=capital,
            mask_reviewers=large_data.reviewers_slice,
            mask_replicators=large_data.replicators_slice,
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_data.contributor_index_map,
            num_contributors=num_contributors,
        )

        reviewer_values = _as_1d(reviewer_fmp)
        replicator_values = _as_1d(replicator_fmp)
        tag_count = int(large_data.primary_tag_memberships.shape[0])

        assert reviewer_values.size == tag_count
        assert replicator_values.size == tag_count
        assert np.isfinite(reviewer_values).all()
        assert np.isfinite(replicator_values).all()


@pytest.mark.large
class TestComputeRiskPremiumsLarge:
    def test_outputs_are_finite_and_aligned(self, large_data, large_contributor_subset):
        capital = large_data.capital
        num_contributors = _num_contributors(capital)

        reviewer_fmp, replicator_fmp = compute_fair_marketprice(
            capital=capital,
            mask_reviewers=large_data.reviewers_slice,
            mask_replicators=large_data.replicators_slice,
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_data.contributor_index_map,
            num_contributors=num_contributors,
        )

        reviewer_risk_premiums, replicator_risk_premiums = compute_risk_premiums(
            capital=capital,
            mask_authors=large_data.authors_slice,
            mask_reviewers=large_data.reviewers_slice,
            mask_replicators=large_data.replicators_slice,
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_contributor_subset,
            reviewer_fmp=reviewer_fmp,
            replicator_fmp=replicator_fmp,
        )

        reviewer_values = _as_1d(reviewer_risk_premiums)
        replicator_values = _as_1d(replicator_risk_premiums)
        tag_count = int(large_data.primary_tag_memberships.shape[0])

        assert reviewer_values.size == tag_count
        assert replicator_values.size == tag_count
        assert np.isfinite(reviewer_values).all()
        assert np.isfinite(replicator_values).all()


@pytest.mark.large
class TestComputeUtilityFunctionLarge:
    def test_risk_willingness_monotonicity(self, large_data, large_contributor_subset):
        capital = large_data.capital
        history = [capital, capital.multiply(1.2), capital.multiply(0.8)]

        utility_low_risk = compute_utility_function(
            history,
            [0.0, 1.0, 2.0],
            large_contributor_subset,
            risk_willingness=0.0,
        )
        utility_high_risk = compute_utility_function(
            history,
            [0.0, 1.0, 2.0],
            large_contributor_subset,
            risk_willingness=2.0,
        )

        assert np.isfinite(utility_low_risk)
        assert np.isfinite(utility_high_risk)
        assert utility_high_risk <= utility_low_risk


@pytest.mark.large
class TestComputeRelativePerformanceLarge:
    def test_uniform_scaling_returns_one(self, large_data, large_contributor_subset):
        capital = large_data.capital
        result = compute_relative_performance(
            shares=capital,
            capital_history=[capital, capital.multiply(1.5)],
            time_history=[0.0, 1.0],
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_contributor_subset,
        )

        assert np.isfinite(result)
        assert result == pytest.approx(1.0, rel=1e-6)


@pytest.mark.large
class TestComputeSensitivityLarge:
    def test_large_linear_arrays(self):
        domain = np.linspace(1.0, 1000.0, 5000)
        manuscript = 3.0 * domain + 7.0

        assert compute_sensitivity(manuscript, domain) == pytest.approx(3.0, rel=1e-9)


@pytest.mark.large
class TestRiskAdjustedMetricsLarge:
    def test_risk_adjusted_excess_return_uniform_scaling(self, large_data, large_contributor_subset):
        capital = large_data.capital
        result = compute_risk_adjusted_excess_return(
            capital_history=[capital, capital.multiply(1.5)],
            time_history=[0.0, 1.0],
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_contributor_subset,
            manuscript_index=0,
        )

        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_risk_adjusted_relative_performance_uniform_scaling(self, large_data, large_contributor_subset):
        capital = large_data.capital
        result = compute_risk_adjusted_relative_performance(
            capital_history=[capital, capital.multiply(1.5)],
            time_history=[0.0, 1.0],
            manuscript_memberships=large_data.primary_tag_memberships,
            contributor_index_map_subset=large_contributor_subset,
        )

        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-12)


# @pytest.mark.large
# class TestShareSplitsAndHHILarge:
#     def test_share_splits_inequality_large_uniform_matrix(self):
#         shares = _uniform_shares_matrix(1000, 40)
#         assert compute_share_splits_inequality(shares) == pytest.approx(1.0 / 40.0, rel=1e-9)
#         assert compute_share_splits_inequality(shares) == pytest.approx(share_splits_inequality(shares), rel=1e-9)

#     def test_hhi_discrepancy_large_uniform_matrix(self):
#         shares = _uniform_shares_matrix(1000, 40)
#         assert compute_hhi_discrepancy(shares, slice(0, 500)) == pytest.approx(0.0, abs=1e-12)
#         assert compute_hhi_discrepancy(shares, slice(0, 500)) == pytest.approx(hhi_discrepancy(shares, slice(0, 500)), rel=1e-9)