"""
pytest configuration and shared fixtures for liberata-metrics tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

# Make data_loading importable from this directory without installation
sys.path.insert(0, str(Path(__file__).parent))
from data_loading import load_data, select_contributor_subset

# Most recent large dataset. REPLACE THIS WITH THE NAME OF YOUR OWN DATA
LARGE_DATA_DIR = (
    Path(__file__).parent / "output" / "m800_c1200_20260412_051528"
)


def _csr(M: int, total_cols: int, data, rows, cols) -> sparse.csr_matrix:
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, total_cols))

#Small matrix fixtures

@pytest.fixture(scope="session")
def small_params():
    """
    Shape parameters for 2-manuscript, 2-contributor synthetic matrices.

    Capital matrix layout (shape = (2, 4)):
        column 0-1: manuscript zero-block
        column 2:   Alice's capital
        column 3:   Bob's capital
    """
    return {
        "M": 2,
        "total_cols": 4,
        "contributor_map": {"Alice": 0, "Bob": 1},
    }


@pytest.fixture(scope="session")
def cap_t0(small_params):
    """Alice=10, Bob=20  ->  total capital = 30."""
    p = small_params
    return _csr(p["M"], p["total_cols"], [10.0, 20.0], [0, 1], [2, 3])


@pytest.fixture(scope="session")
def cap_t1(small_params):
    """Alice=15, Bob=30  ->  total capital = 45  (+50% from T0)."""
    p = small_params
    return _csr(p["M"], p["total_cols"], [15.0, 30.0], [0, 1], [2, 3])


@pytest.fixture(scope="session")
def cap_t2(small_params):
    """Alice=7.5, Bob=15  ->  total capital = 22.5  (-50% from T1)."""
    p = small_params
    return _csr(p["M"], p["total_cols"], [7.5, 15.0], [0, 1], [2, 3])


@pytest.fixture(scope="session")
def cap_new(small_params):
    """Alice=30, Bob=60  ->  total capital = 90  (+100% from T1)."""
    p = small_params
    return _csr(p["M"], p["total_cols"], [30.0, 60.0], [0, 1], [2, 3])


@pytest.fixture(scope="session")
def cap_zero(small_params):
    """All-zero sparse matrix -- used for divide-by-zero guard tests."""
    p = small_params
    return sparse.csr_matrix((p["M"], p["total_cols"]))


@pytest.fixture(scope="session")
def role_params():
    """Parameters for a 2-manuscript, 1-contributor-per-role matrix."""
    M = 2
    C_per_role = 1
    total_cols = M + 3 * C_per_role
    return {
        "M": M,
        "total_cols": total_cols,
        "num_contributors": C_per_role,
        "contributor_map": {"C0": 0},
        "author_mask": np.s_[:, M: M + C_per_role],
        "reviewer_mask": np.s_[
            :, M + C_per_role: M + 2 * C_per_role
        ],
        "replicator_mask": np.s_[
            :, M + 2 * C_per_role: total_cols
        ],
    }


@pytest.fixture(scope="session")
def cap_role(role_params):
    """Three-role matrix at T0.  Author=15, Reviewer=5, Total=20."""
    p = role_params
    data = [10.0, 5.0, 3.0, 2.0]
    rows = [0, 1, 0, 1]
    cols = [2, 2, 3, 3]      # col 2 = author, col 3 = reviewer
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(p["M"], p["total_cols"])
    )


@pytest.fixture(scope="session")
def cap_role_t1(role_params):
    """Three-role matrix at T1, scaled x1.5.  Author=22.5, Reviewer=7.5."""
    p = role_params
    data = [15.0, 7.5, 4.5, 3.0]
    rows = [0, 1, 0, 1]
    cols = [2, 2, 3, 3]
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(p["M"], p["total_cols"])
    )


#Large matrix fixtures

@pytest.fixture(scope="session")
def large_data():
    if not LARGE_DATA_DIR.exists():
        pytest.skip(f"Large matrix data not found: {LARGE_DATA_DIR}")
    return load_data(str(LARGE_DATA_DIR))


@pytest.fixture(scope="session")
def large_contributor_subset(large_data) -> dict:
    """First 50 contributors from the large dataset as a portfolio subset."""
    return select_contributor_subset(large_data.contributor_index_map, n_first=50)

@pytest.fixture(scope="session")
def cap_new(small_params) -> sparse.csr_matrix:
    """Alice=30, Bob=60 → total capital = 90."""
    p = small_params
    return _csr(p["M"], p["total_cols"], [30.0, 60.0], [0, 1], [2, 3])
