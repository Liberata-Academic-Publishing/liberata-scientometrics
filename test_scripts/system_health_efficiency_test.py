import pytest
import numpy as np

from liberata_metrics.metrics.system_health_metrics import (
    get_funding_efficiency,
    get_gdp_efficiency,
    get_ppp_efficiency,
    get_time_efficiency,
)


# cap_t0 total capital = 30.0


class TestFundingEfficiencyCorrectness:

    def test_exact_value(self, cap_t0):
        assert get_funding_efficiency(cap_t0, 10.0) == 3.0

    def test_larger_spending_smaller_result(self, cap_t0):
        assert get_funding_efficiency(cap_t0, 30.0) == 1.0

    def test_zero_spending_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_funding_efficiency(cap_t0, 0.0)

    def test_negative_spending_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_funding_efficiency(cap_t0, -1.0)

    def test_non_sparse_raises(self):
        with pytest.raises(TypeError):
            get_funding_efficiency(np.array([[1, 0, 0, 0]]), 1.0)


class TestGdpEfficiencyCorrectness:

    def test_exact_value(self, cap_t0):
        # funding_efficiency = 3.0, gdp = 2.0 -> 6.0
        assert get_gdp_efficiency(cap_t0, 10.0, 2.0) == 6.0

    def test_zero_gdp_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_gdp_efficiency(cap_t0, 10.0, 0.0)

    def test_negative_gdp_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_gdp_efficiency(cap_t0, 10.0, -1.0)

    def test_zero_spending_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_gdp_efficiency(cap_t0, 0.0, 1.0)


class TestPppEfficiencyCorrectness:

    def test_exact_value(self, cap_t0):
        # funding_efficiency = 3.0, ppp = 4.0 -> 12.0
        assert get_ppp_efficiency(cap_t0, 10.0, 4.0) == 12.0

    def test_zero_ppp_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_ppp_efficiency(cap_t0, 10.0, 0.0)

    def test_negative_ppp_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_ppp_efficiency(cap_t0, 10.0, -1.0)

    def test_zero_spending_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_ppp_efficiency(cap_t0, 0.0, 1.0)


class TestTimeEfficiencyCorrectness:

    def test_exact_value(self, cap_t0):
        assert get_time_efficiency(cap_t0, 5.0) == 6.0

    def test_larger_time_smaller_result(self, cap_t0):
        assert get_time_efficiency(cap_t0, 30.0) == 1.0

    def test_zero_delta_t_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_time_efficiency(cap_t0, 0.0)

    def test_negative_delta_t_raises(self, cap_t0):
        with pytest.raises(ValueError):
            get_time_efficiency(cap_t0, -1.0)

    def test_non_sparse_raises(self):
        with pytest.raises(TypeError):
            get_time_efficiency(np.array([[1, 0, 0, 0]]), 1.0)


@pytest.mark.large
class TestEfficiencyLarge:

    def test_funding_efficiency_positive(self, large_data):
        result = get_funding_efficiency(large_data.capital, 1000.0)
        assert result > 0.0

    def test_funding_efficiency_finite(self, large_data):
        result = get_funding_efficiency(large_data.capital, 1000.0)
        assert np.isfinite(result)

    def test_gdp_efficiency_scales_with_gdp(self, large_data):
        cap = large_data.capital
        r1 = get_gdp_efficiency(cap, 1000.0, 1.0)
        r2 = get_gdp_efficiency(cap, 1000.0, 2.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-9)

    def test_ppp_efficiency_scales_with_ppp(self, large_data):
        cap = large_data.capital
        r1 = get_ppp_efficiency(cap, 1000.0, 1.0)
        r2 = get_ppp_efficiency(cap, 1000.0, 3.0)
        assert r2 == pytest.approx(3.0 * r1, rel=1e-9)

    def test_time_efficiency_positive(self, large_data):
        result = get_time_efficiency(large_data.capital, 1.0)
        assert result > 0.0

    def test_time_efficiency_finite(self, large_data):
        result = get_time_efficiency(large_data.capital, 1.0)
        assert np.isfinite(result)
