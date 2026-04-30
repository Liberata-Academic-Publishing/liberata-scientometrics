"""
Tests for spanning tree ratio functions (graph.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.graph import (
    get_spanning_tree_ratio,
    get_weighted_spanning_tree_ratio,
    get_relative_spanning_tree_ratio,
)


def _complete_bipartite_shares(M: int, C: int) -> sparse.csr_matrix:
    """Complete bipartite graph: every manuscript connected to all 3C non-manuscript
    nodes with equal weight (1 / 3C per edge, summing to 1 per manuscript)."""
    N_other = 3 * C
    rows, cols, data = [], [], []
    for m in range(M):
        for k in range(N_other):
            rows.append(m)
            cols.append(M + k)
            data.append(1.0 / N_other)
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, M + N_other))


def _sparse_shares(M: int, C: int) -> sparse.csr_matrix:
    """Each manuscript connected to exactly one non-manuscript node."""
    rows = list(range(M))
    cols = [M + i for i in range(M)]
    data = [1.0] * M
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, M + 3 * C))


# ---------------------------------------------------------------------------
# get_spanning_tree_ratio
# ---------------------------------------------------------------------------

class TestGetSpanningTreeRatio:

    def test_returns_float(self):
        shares = _complete_bipartite_shares(2, 1)
        assert isinstance(get_spanning_tree_ratio(shares), float)

    def test_complete_bipartite_equals_one(self):
        shares = _complete_bipartite_shares(3, 2)
        assert pytest.approx(get_spanning_tree_ratio(shares), rel=1e-6) == 1.0

    def test_sparse_graph_less_than_one(self):
        shares = _sparse_shares(3, 2)
        assert get_spanning_tree_ratio(shares) < 1.0

    def test_positive(self):
        shares = _complete_bipartite_shares(2, 1)
        assert get_spanning_tree_ratio(shares) > 0

    def test_raises_on_dense_matrix(self):
        with pytest.raises(TypeError):
            get_spanning_tree_ratio(np.ones((2, 8)))


# ---------------------------------------------------------------------------
# get_weighted_spanning_tree_ratio
# ---------------------------------------------------------------------------

class TestGetWeightedSpanningTreeRatio:

    def test_returns_float(self):
        shares = _complete_bipartite_shares(2, 1)
        assert isinstance(get_weighted_spanning_tree_ratio(shares), float)

    def test_complete_bipartite_equal_weights_equals_one(self):
        # Equal weights on complete bipartite: STRw = 1
        shares = _complete_bipartite_shares(3, 2)
        assert pytest.approx(get_weighted_spanning_tree_ratio(shares), rel=1e-6) == 1.0

    def test_positive(self):
        shares = _complete_bipartite_shares(2, 1)
        assert get_weighted_spanning_tree_ratio(shares) > 0

    def test_raises_on_dense_matrix(self):
        with pytest.raises(TypeError):
            get_weighted_spanning_tree_ratio(np.ones((2, 8)))


    def test_uneven_weights_lower_than_equal(self):
        # Skewed weights should give lower STRw than equal weights on same topology
        M, C = 3, 2
        N_other = 3 * C
        equal = _complete_bipartite_shares(M, C)
        rows, cols, data = [], [], []
        for m in range(M):
            for k in range(N_other):
                rows.append(m)
                cols.append(M + k)
                # Put 90% on first node, rest split
                data.append(0.9 if k == 0 else 0.1 / (N_other - 1))
        uneven = sparse.csr_matrix((data, (rows, cols)), shape=(M, M + N_other))
        assert get_weighted_spanning_tree_ratio(uneven) < get_weighted_spanning_tree_ratio(equal)


# ---------------------------------------------------------------------------
# get_relative_spanning_tree_ratio
# ---------------------------------------------------------------------------

class TestGetRelativeSpanningTreeRatio:

    def test_returns_float(self):
        shares = _complete_bipartite_shares(2, 1)
        assert isinstance(get_relative_spanning_tree_ratio(shares), float)

    def test_complete_equal_weights_equals_one(self):
        shares = _complete_bipartite_shares(3, 2)
        assert pytest.approx(get_relative_spanning_tree_ratio(shares), rel=1e-6) == 1.0

    def test_equals_strw_over_str(self):
        shares = _complete_bipartite_shares(3, 2)
        str_ = get_spanning_tree_ratio(shares)
        strw = get_weighted_spanning_tree_ratio(shares)
        rstr = get_relative_spanning_tree_ratio(shares)
        assert pytest.approx(rstr, rel=1e-9) == strw / str_

    def test_raises_on_dense_matrix(self):
        with pytest.raises(TypeError):
            get_relative_spanning_tree_ratio(np.ones((2, 8)))

