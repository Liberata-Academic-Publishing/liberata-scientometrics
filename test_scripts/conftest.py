"""
pytest configuration and shared fixtures for liberata-metrics tests.

Fixture strategy
----------------
- Small synthetic matrices: 2 manuscripts × 2 contributors, hand-verifiable values.
  Used for correctness / exact-value unit tests.
- Large pre-generated matrices: 800 manuscripts × 1200 contributors, loaded from disk.
  Used for scalability, invariant, and property-based tests.
  Skipped automatically when the data directory is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

# Make data_loading importable from this directory without installation
sys.path.insert(0, str(Path(__file__).parent))
from data_loading import load_data, select_contributor_subset  # noqa: E402

# Most recent large dataset; update when a new run is generated
LARGE_DATA_DIR = (
    Path(__file__).parent / "output" / "m800_c1200_20260213_154126"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _csr(M: int, total_cols: int, data, rows, cols) -> sparse.csr_matrix:
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, total_cols))

# ── Small synthetic fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_params():
    """
    Shape parameters for 2-manuscript, 2-contributor synthetic matrices.

    Capital matrix layout (shape = (2, 4)):
        column 0-1: manuscript zero-block (unused in simple case)
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


# ── Small three-role fixtures ──────────────────────────────────────────────────
#
# Shape: (M=2, total_cols=5)
# Column layout:
#   [0-1] manuscript zero-block
#   [2]   author capital for C0
#   [3]   reviewer capital for C0
#   [4]   replicator capital for C0
#
# Fixture values (cap_role):
#   Author:     m0=10, m1=5  -> total 15
#   Reviewer:   m0=3,  m1=2  -> total  5
#   Replicator: 0            -> total  0
#   Grand total = 20

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


# ── Large matrix fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def large_data():
    """
    Load the pre-generated 800x1200 capital matrix from disk.

    Session-scoped so the expensive IO happens at most once per test run.
    Tests that depend on this fixture are automatically skipped when the
    data directory does not exist (CI environments without large assets).
    """
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
