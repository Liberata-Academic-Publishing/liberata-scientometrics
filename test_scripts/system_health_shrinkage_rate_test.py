# Tests for get_reviewer_shrinkage_rate and get_replicator_shrinkage_rate.
import pytest
from scipy import sparse

from liberata_metrics.metrics.system_health_metrics import (
    get_reviewer_shrinkage_rate,
    get_replicator_shrinkage_rate,
)


class TestReviewerShrinkageRateCorrectness:

    def test_shrinking_fmp(self, cap_role, cap_role_t1, role_params):
        result = get_reviewer_shrinkage_rate(
            [cap_role_t1, cap_role], role_params["contributor_map"]
        )
        assert result == 2.5

    def test_growing_fmp(self, cap_role, cap_role_t1, role_params):
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role_t1], role_params["contributor_map"]
        )
        assert result == -2.5

    def test_flat_returns_zero(self, cap_role, role_params):
        result = get_reviewer_shrinkage_rate(
            [cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == 0.0

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_reviewer_shrinkage_rate(
                [cap_role], role_params["contributor_map"]
            )

    def test_empty_history_raises(self, role_params):
        with pytest.raises(ValueError):
            get_reviewer_shrinkage_rate([], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        with pytest.raises(TypeError):
            get_reviewer_shrinkage_rate(
                [cap_role, cap_role.toarray()],
                role_params["contributor_map"],
            )


class TestReplicatorShrinkageRateCorrectness:

    #We make a simple matrix since the fixture does not have replicator data!
    def test_shrinking_fmp(self, role_params):
        p = role_params
        cap_t0 = sparse.csr_matrix(
            ([10.0], ([0], [4])), shape=(p["M"], p["total_cols"])
        )
        cap_t1 = sparse.csr_matrix(
            ([6.0], ([0], [4])), shape=(p["M"], p["total_cols"])
        )
        result = get_replicator_shrinkage_rate(
            [cap_t0, cap_t1], p["contributor_map"]
        )
        assert result == 4.0

    def test_growing_fmp(self, role_params):
        p = role_params
        cap_t0 = sparse.csr_matrix(
            ([6.0], ([0], [4])), shape=(p["M"], p["total_cols"])
        )
        cap_t1 = sparse.csr_matrix(
            ([10.0], ([0], [4])), shape=(p["M"], p["total_cols"])
        )
        result = get_replicator_shrinkage_rate(
            [cap_t0, cap_t1], p["contributor_map"]
        )
        assert result == -4.0

    def test_flat_returns_zero(self, cap_role, role_params):
        result = get_replicator_shrinkage_rate(
            [cap_role, cap_role], role_params["contributor_map"]
        )
        assert result == 0.0

    def test_too_few_entries_raises(self, cap_role, role_params):
        with pytest.raises(ValueError):
            get_replicator_shrinkage_rate(
                [cap_role], role_params["contributor_map"]
            )

    def test_empty_history_raises(self, role_params):
        with pytest.raises(ValueError):
            get_replicator_shrinkage_rate([], role_params["contributor_map"])

    def test_non_sparse_raises(self, cap_role, role_params):
        with pytest.raises(TypeError):
            get_replicator_shrinkage_rate(
                [cap_role, cap_role.toarray()],
                role_params["contributor_map"],
            )


@pytest.mark.large
class TestShrinkageRateLarge:

    def test_reviewer_flat_returns_zero(self, large_data):
        cap = large_data.capital
        result = get_reviewer_shrinkage_rate(
            [cap, cap, cap], large_data.contributor_index_map
        )
        assert result == 0.0

    def test_replicator_flat_returns_zero(self, large_data):
        cap = large_data.capital
        result = get_replicator_shrinkage_rate(
            [cap, cap, cap], large_data.contributor_index_map
        )
        assert result == 0.0

    def test_reviewer_growing_fmp_is_negative(self, large_data):
        cap = large_data.capital
        result = get_reviewer_shrinkage_rate(
            [cap, cap.multiply(1.5)], large_data.contributor_index_map
        )
        assert result < 0.0

    def test_replicator_growing_fmp_is_negative(self, large_data):
        cap = large_data.capital
        result = get_replicator_shrinkage_rate(
            [cap, cap.multiply(1.5)], large_data.contributor_index_map
        )
        assert result < 0.0
