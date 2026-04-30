import pytest
import numpy as np
from scipy import sparse

from liberata_metrics.metrics.system_health_metrics import (
    get_regional_hhi,
    get_gini_per_capita,
    get_gini_per_contributor,
    get_gini_per_gdp,
)


@pytest.fixture
def region_maps():
    return {
        "A": {"Alice": 0},
        "B": {"Bob": 1},
    }


class TestRegionalHhiCorrectness:

    def test_equal_shares_is_half(self, small_params):
        p = small_params
        equal_cap = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])), shape=(p["M"], p["total_cols"])
        )
        result = get_regional_hhi(
            equal_cap,
            {"Alice": 0, "Bob": 1},
            {"A": {"Alice": 0}, "B": {"Bob": 1}},
        )
        assert result == 0.5

    def test_monopoly_is_one(self, small_params):
        p = small_params
        mono_cap = sparse.csr_matrix(
            ([10.0], ([0], [2])), shape=(p["M"], p["total_cols"])
        )
        result = get_regional_hhi(
            mono_cap,
            {"Alice": 0, "Bob": 1},
            {"A": {"Alice": 0}, "B": {"Bob": 1}},
        )
        assert result == 1.0

    def test_exact_unequal(self, cap_t0, region_maps):
        result = get_regional_hhi(cap_t0, {"Alice": 0, "Bob": 1}, region_maps)
        assert result == pytest.approx(5.0 / 9.0, rel=1e-9)

    def test_zero_capital_raises(self, small_params):
        p = small_params
        zero_cap = sparse.csr_matrix((p["M"], p["total_cols"]))
        with pytest.raises(ValueError):
            get_regional_hhi(
                zero_cap,
                {"Alice": 0, "Bob": 1},
                {"A": {"Alice": 0}, "B": {"Bob": 1}},
            )

    def test_non_sparse_raises(self):
        with pytest.raises(TypeError):
            get_regional_hhi(
                np.array([[1, 0, 0, 0]]),
                {"Alice": 0},
                {"A": {"Alice": 0}},
            )


class TestGiniPerCapitaCorrectness:

    def test_equal_per_capita_is_zero(self, small_params):
        p = small_params
        equal_cap = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])), shape=(p["M"], p["total_cols"])
        )
        result = get_gini_per_capita(
            equal_cap,
            {"A": {"Alice": 0}, "B": {"Bob": 1}},
            {"A": 1.0, "B": 1.0},
        )
        assert result == 0.0

    def test_unequal_per_capita_positive(self, cap_t0, region_maps):
        result = get_gini_per_capita(cap_t0, region_maps, {"A": 1.0, "B": 1.0})
        assert result > 0.0

    def test_missing_population_raises(self, cap_t0, region_maps):
        with pytest.raises(ValueError):
            get_gini_per_capita(cap_t0, region_maps, {"A": 1.0})

    def test_non_sparse_raises(self, region_maps):
        with pytest.raises(TypeError):
            get_gini_per_capita(
                np.array([[1, 0, 0, 0]]), region_maps, {"A": 1.0, "B": 1.0}
            )


class TestGiniPerContributorCorrectness:

    def test_equal_per_contributor_is_zero(self, small_params):
        p = small_params
        equal_cap = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])), shape=(p["M"], p["total_cols"])
        )
        result = get_gini_per_contributor(
            equal_cap,
            {"A": {"Alice": 0}, "B": {"Bob": 1}},
            {"A": 1, "B": 1},
        )
        assert result == 0.0

    def test_unequal_per_contributor_positive(self, cap_t0, region_maps):
        result = get_gini_per_contributor(cap_t0, region_maps, {"A": 1, "B": 1})
        assert result > 0.0

    def test_missing_count_raises(self, cap_t0, region_maps):
        with pytest.raises(ValueError):
            get_gini_per_contributor(cap_t0, region_maps, {"A": 1})

    def test_non_sparse_raises(self, region_maps):
        with pytest.raises(TypeError):
            get_gini_per_contributor(
                np.array([[1, 0, 0, 0]]), region_maps, {"A": 1, "B": 1}
            )


class TestGiniPerGdpCorrectness:

    def test_equal_per_gdp_is_zero(self, small_params):
        p = small_params
        equal_cap = sparse.csr_matrix(
            ([10.0, 10.0], ([0, 1], [2, 3])), shape=(p["M"], p["total_cols"])
        )
        result = get_gini_per_gdp(
            equal_cap,
            {"A": {"Alice": 0}, "B": {"Bob": 1}},
            {"A": 1.0, "B": 1.0},
        )
        assert result == 0.0

    def test_unequal_per_gdp_positive(self, cap_t0, region_maps):
        result = get_gini_per_gdp(cap_t0, region_maps, {"A": 1.0, "B": 1.0})
        assert result > 0.0

    def test_missing_gdp_raises(self, cap_t0, region_maps):
        with pytest.raises(ValueError):
            get_gini_per_gdp(cap_t0, region_maps, {"A": 1.0})

    def test_non_sparse_raises(self, region_maps):
        with pytest.raises(TypeError):
            get_gini_per_gdp(
                np.array([[1, 0, 0, 0]]), region_maps, {"A": 1.0, "B": 1.0}
            )


@pytest.mark.large
class TestHhiGiniLarge:

    @pytest.fixture
    def two_region_maps(self, large_contributor_subset):
        keys = list(large_contributor_subset.keys())
        mid = len(keys) // 2
        return {
            "A": {k: large_contributor_subset[k] for k in keys[:mid]},
            "B": {k: large_contributor_subset[k] for k in keys[mid:]},
        }

    def test_hhi_in_range(self, large_data, large_contributor_subset, two_region_maps):
        result = get_regional_hhi(
            large_data.capital, large_contributor_subset, two_region_maps
        )
        assert 0.0 < result <= 1.0

    def test_hhi_finite(self, large_data, large_contributor_subset, two_region_maps):
        result = get_regional_hhi(
            large_data.capital, large_contributor_subset, two_region_maps
        )
        assert np.isfinite(result)

    def test_gini_per_capita_in_range(self, large_data, two_region_maps):
        result = get_gini_per_capita(
            large_data.capital, two_region_maps, {"A": 1000.0, "B": 2000.0}
        )
        assert 0.0 <= result <= 1.0

    def test_gini_per_contributor_in_range(self, large_data, two_region_maps):
        result = get_gini_per_contributor(
            large_data.capital, two_region_maps, {"A": 10, "B": 20}
        )
        assert 0.0 <= result <= 1.0

    def test_gini_per_gdp_in_range(self, large_data, two_region_maps):
        result = get_gini_per_gdp(
            large_data.capital, two_region_maps, {"A": 1e6, "B": 2e6}
        )
        assert 0.0 <= result <= 1.0
