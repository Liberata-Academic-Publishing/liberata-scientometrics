"""
Tests for get_proportional_split and get_diversification_ratio.

Two-tier strategy
-----------------
Small (correctness): 2x5 synthetic matrix with exact hand-computed values.
Large (invariants):  800x1200 pre-generated matrix; mathematical properties
                     that hold for any valid capital matrix.

Run all tests:
    pytest test_scripts/proportional_split_diversification_test.py -v

Run only small tests (no disk data needed):
    pytest test_scripts/proportional_split_diversification_test.py -v -m "not large"

Run only large tests:
    pytest test_scripts/proportional_split_diversification_test.py -v -m large
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
    get_diversification_ratio,
    get_proportional_split,
)


# ===========================================================================
# get_proportional_split -- small correctness tests
# ===========================================================================

class TestGetProportionalSplitCorrectness:
    """
    Hand-verified values for the 3-role fixture.

    Cap matrix shape (2, 5):  author col=2, reviewer col=3, replicator col=4
      Author:     m0=10, m1=5  -> total 15
      Reviewer:   m0=3,  m1=2  -> total  5
      Replicator: 0            -> total  0
      Grand total = 20

    Expected splits:  author=0.75, reviewer=0.25, replicator=0.0
    """

    def test_author_split(self, cap_role, role_params):
        # 15 / 20 = 0.75
        p = role_params
        result = get_proportional_split(
            cap_role,
            mask_by_role=p["author_mask"],
            contributor_index_map_subset=p["contributor_map"],
        )
        assert result == pytest.approx(0.75, rel=1e-9)

    def test_reviewer_split(self, cap_role, role_params):
        # 5 / 20 = 0.25
        p = role_params
        result = get_proportional_split(
            cap_role,
            mask_by_role=p["reviewer_mask"],
            contributor_index_map_subset=p["contributor_map"],
        )
        assert result == pytest.approx(0.25, rel=1e-9)

    def test_replicator_split_is_zero(self, cap_role, role_params):
        # 0 / 20 = 0.0
        p = role_params
        result = get_proportional_split(
            cap_role,
            mask_by_role=p["replicator_mask"],
            contributor_index_map_subset=p["contributor_map"],
        )
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_three_splits_sum_to_one(self, cap_role, role_params):
        # author + reviewer + replicator = 1.0 exactly
        p = role_params
        author = get_proportional_split(
            cap_role, p["author_mask"], p["contributor_map"]
        )
        reviewer = get_proportional_split(
            cap_role, p["reviewer_mask"], p["contributor_map"]
        )
        replicator = get_proportional_split(
            cap_role, p["replicator_mask"], p["contributor_map"]
        )
        assert author + reviewer + replicator == pytest.approx(
            1.0, rel=1e-9
        )

    def test_result_in_unit_interval(self, cap_role, role_params):
        p = role_params
        for mask in (
            p["author_mask"],
            p["reviewer_mask"],
            p["replicator_mask"],
        ):
            result = get_proportional_split(
                cap_role, mask, p["contributor_map"]
            )
            assert 0.0 <= result <= 1.0

    def test_zero_capital_returns_zero(self, role_params):
        # If all capital is zero, function returns 0.0 (no division)
        p = role_params
        zero = sparse.csr_matrix((p["M"], p["total_cols"]))
        result = get_proportional_split(
            zero,
            mask_by_role=p["author_mask"],
            contributor_index_map_subset=p["contributor_map"],
        )
        assert result == 0.0

    def test_non_sparse_raises_type_error(self, role_params):
        p = role_params
        dense = np.ones((p["M"], p["total_cols"]))
        with pytest.raises(TypeError):
            get_proportional_split(
                dense,
                mask_by_role=p["author_mask"],
                contributor_index_map_subset=p["contributor_map"],
            )

    def test_single_role_only_split_is_one(self, role_params):
        # A matrix with capital only in the author block -> author split = 1.0
        p = role_params
        data = [8.0, 4.0]
        rows = [0, 1]
        cols = [2, 2]   # author column only
        author_only = sparse.csr_matrix(
            (data, (rows, cols)), shape=(p["M"], p["total_cols"])
        )
        result = get_proportional_split(
            author_only,
            mask_by_role=p["author_mask"],
            contributor_index_map_subset=p["contributor_map"],
        )
        assert result == pytest.approx(1.0, rel=1e-9)


# ===========================================================================
# get_proportional_split -- large invariant tests
# ===========================================================================

@pytest.mark.large
class TestGetProportionalSplitLarge:

    def test_reviewer_split_in_range(
        self, large_data, large_contributor_subset
    ):
        result = get_proportional_split(
            capital=large_data.capital,
            mask_by_role=large_data.reviewers_slice,
            contributor_index_map_subset=large_contributor_subset,
        )
        assert 0.0 <= result <= 1.0

    def test_replicator_split_in_range(
        self, large_data, large_contributor_subset
    ):
        result = get_proportional_split(
            capital=large_data.capital,
            mask_by_role=large_data.replicators_slice,
            contributor_index_map_subset=large_contributor_subset,
        )
        assert 0.0 <= result <= 1.0

    def test_author_split_in_range(
        self, large_data, large_contributor_subset
    ):
        result = get_proportional_split(
            capital=large_data.capital,
            mask_by_role=large_data.authors_slice,
            contributor_index_map_subset=large_contributor_subset,
        )
        assert 0.0 <= result <= 1.0

    def test_all_splits_sum_to_one(
        self, large_data, large_contributor_subset
    ):
        # The three role slices partition all contributor columns, so
        # their proportional splits must sum exactly to 1.0.
        cap = large_data.capital
        cmap = large_contributor_subset
        author = get_proportional_split(
            cap, large_data.authors_slice, cmap
        )
        reviewer = get_proportional_split(
            cap, large_data.reviewers_slice, cmap
        )
        replicator = get_proportional_split(
            cap, large_data.replicators_slice, cmap
        )
        assert author + reviewer + replicator == pytest.approx(
            1.0, abs=1e-6
        )

    def test_result_is_finite(
        self, large_data, large_contributor_subset
    ):
        result = get_proportional_split(
            capital=large_data.capital,
            mask_by_role=large_data.reviewers_slice,
            contributor_index_map_subset=large_contributor_subset,
        )
        assert math.isfinite(result)

    def test_scaling_invariance(
        self, large_data, large_contributor_subset
    ):
        # Multiplying the entire capital matrix by a positive scalar k
        # does not change any proportional split (numerator and denominator
        # both scale by k).
        k = 3.7
        cap = large_data.capital
        cap_scaled = cap.multiply(k)
        cmap = large_contributor_subset
        original = get_proportional_split(
            cap, large_data.reviewers_slice, cmap
        )
        scaled = get_proportional_split(
            cap_scaled, large_data.reviewers_slice, cmap
        )
        assert original == pytest.approx(scaled, rel=1e-9)


# ===========================================================================
# get_diversification_ratio -- small correctness tests
# ===========================================================================

class TestGetDiversificationRatioValidation:
    """Input-validation tests that do not reach the body computation."""

    def test_single_entry_raises_value_error(
        self, cap_role, role_params
    ):
        p = role_params
        with pytest.raises(ValueError):
            get_diversification_ratio([cap_role], p["contributor_map"])

    def test_empty_history_raises_value_error(
        self, role_params
    ):
        p = role_params
        with pytest.raises(ValueError):
            get_diversification_ratio([], p["contributor_map"])


class TestGetDiversificationRatioCorrectness:
    """
    Correctness tests.

    DR = (weighted sum of per-manuscript volatilities) / portfolio volatility.

    For a two-manuscript portfolio scaled uniformly over time, all per-
    manuscript returns equal the portfolio return, so DR -> 1.0.
    """

    def test_uniform_scaling_dr_equals_one(
        self, cap_role, cap_role_t1, role_params
    ):
        # Both manuscripts grow by the same factor: per-manuscript
        # volatility == portfolio volatility => DR = 1.0
        p = role_params
        history = [cap_role, cap_role_t1]
        result = get_diversification_ratio(history, p["contributor_map"])
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_result_is_positive(
        self, cap_role, cap_role_t1, role_params
    ):
        p = role_params
        result = get_diversification_ratio(
            [cap_role, cap_role_t1], p["contributor_map"]
        )
        assert result >= 0.0

    def test_longer_history_accepted(
        self, cap_role, cap_role_t1, role_params
    ):
        p = role_params
        cap_t2 = cap_role_t1.multiply(0.8)
        result = get_diversification_ratio(
            [cap_role, cap_role_t1, cap_t2], p["contributor_map"]
        )
        assert math.isfinite(result)


# ===========================================================================
# get_diversification_ratio -- large invariant tests
# ===========================================================================

@pytest.mark.large
class TestGetDiversificationRatioLarge:

    def test_result_is_nonnegative(
        self, large_data, large_contributor_subset
    ):
        cap = large_data.capital
        cmap = large_contributor_subset
        result = get_diversification_ratio(
            [cap, cap.multiply(1.5)], cmap
        )
        assert result >= 0.0

    def test_result_is_finite(
        self, large_data, large_contributor_subset
    ):
        cap = large_data.capital
        result = get_diversification_ratio(
            [cap, cap.multiply(1.2)], large_contributor_subset
        )
        assert math.isfinite(result)

    def test_uniform_growth_dr_equals_one(
        self, large_data, large_contributor_subset
    ):
        # Uniform scaling: every manuscript grows by the same factor.
        # All per-manuscript volatilities equal portfolio volatility => DR=1.
        cap = large_data.capital
        cmap = large_contributor_subset
        result = get_diversification_ratio(
            [cap, cap.multiply(2.0), cap.multiply(3.0)], cmap
        )
        assert result == pytest.approx(1.0, rel=1e-5)

    def test_longer_history_accepted(
        self, large_data, large_contributor_subset
    ):
        cap = large_data.capital
        history = [
            cap,
            cap.multiply(1.1),
            cap.multiply(1.3),
            cap.multiply(0.9),
        ]
        result = get_diversification_ratio(
            history, large_contributor_subset
        )
        assert math.isfinite(result)
