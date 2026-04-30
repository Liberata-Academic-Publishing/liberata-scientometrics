"""
Tests for shares graph Laplacian functions (graph.py).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.graph import (
    get_shares_laplacian_spectrum,
    get_shares_fiedler_value,
    get_shares_connected_components,
    get_shares_clusters,
)




def _make_shares() -> sparse.csr_matrix:
    M, C = 2, 2
    rows = [0, 1, 0, 1]
    cols = [M + 0, M + C + 1, M + C, M + 1]
    data = [0.8, 0.6, 0.2, 0.4]
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, M + 3 * C))


class TestSharesLaplacianSpectrum:

    def test_smallest_eigenvalue_is_zero(self):
        """The smallest eigenvalue of any graph Laplacian is always 0"""
        shares = _make_shares()
        eigenvalues, _ = get_shares_laplacian_spectrum(shares, k=3)
        assert eigenvalues[0] < 1e-6

    def test_eigenvalues_sorted_ascending(self):
        """Eigenvalues must be returned in ascending order."""
        shares = _make_shares()
        eigenvalues, _ = get_shares_laplacian_spectrum(shares, k=4)
        assert np.all(np.diff(eigenvalues) >= -1e-10)

    def test_eigenvectors_shape(self):
        """Eigenvectors must have shape (N, k) where N = M + 3C."""
        shares = _make_shares()
        k = 3
        _, eigenvectors = get_shares_laplacian_spectrum(shares, k=k)
        assert eigenvectors.shape == (shares.shape[1], k)

    def test_raises_type_error(self):
        """Must raise TypeError if shares is not sparse."""
        import pytest
        with pytest.raises(TypeError):
            get_shares_laplacian_spectrum(np.zeros((2, 8)), k=3)

    def test_raises_value_error_bad_k(self):
        """Must raise ValueError if k >= N."""
        import pytest
        shares = _make_shares()
        with pytest.raises(ValueError):
            get_shares_laplacian_spectrum(shares, k=shares.shape[1])


class TestSharesFiedlerValue:

    def test_fiedler_near_zero_disconnected(self):
        """Fiedler value ~ 0 because graph has isolated replicator nodes."""
        shares = _make_shares()
        fiedler = get_shares_fiedler_value(shares)
        assert fiedler < 1e-6

    def test_fiedler_nonnegative(self):
        """Fiedler value is always >= 0 up to floating point noise."""
        shares = _make_shares()
        assert get_shares_fiedler_value(shares) >= -1e-10

    def test_raises_type_error(self):
        import pytest
        with pytest.raises(TypeError):
            get_shares_fiedler_value(np.zeros((2, 8)))


class TestSharesConnectedComponents:

    def test_four_components(self):
        """2 manuscript components + 2 isolated replicator nodes = 4 components."""
        shares = _make_shares()
        assert get_shares_connected_components(shares, k=5) == 4

    def test_returns_int(self):
        shares = _make_shares()
        result = get_shares_connected_components(shares, k=5)
        assert isinstance(result, int)

    def test_raises_type_error(self):
        import pytest
        with pytest.raises(TypeError):
            get_shares_connected_components(np.zeros((2, 8)), k=3)


class TestSharesClusters:

    def test_labels_shape(self):
        """Labels must have shape (N,) where N = M + 3C."""
        shares = _make_shares()
        labels = get_shares_clusters(shares, n_clusters=4, seed=0)
        assert labels.shape == (shares.shape[1],)

    def test_correct_number_of_clusters(self):
        """Number of unique labels must equal n_clusters."""
        shares = _make_shares()
        labels = get_shares_clusters(shares, n_clusters=4, seed=0)
        assert len(np.unique(labels)) == 4

    def test_labels_are_integers(self):
        shares = _make_shares()
        labels = get_shares_clusters(shares, n_clusters=4, seed=0)
        assert labels.dtype in (np.int32, np.int64, int)

    def test_raises_type_error(self):
        import pytest
        with pytest.raises(TypeError):
            get_shares_clusters(np.zeros((2, 8)), n_clusters=2)

    def test_raises_value_error_bad_n_clusters(self):
        import pytest
        shares = _make_shares()
        with pytest.raises(ValueError):
            get_shares_clusters(shares, n_clusters=shares.shape[1])
