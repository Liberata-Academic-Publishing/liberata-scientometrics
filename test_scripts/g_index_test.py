"""
Tests for get_g_index (legacy_metric.py).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.legacy_metric import (
    get_g_index,
    get_h_index,
)



def _make_capital(
    num_manuscripts: int,
    num_contributors: int,
    author_col: int,
    authored_rows: list,
) -> sparse.csr_matrix:
    """Capital matrix with 1.0 at (row, M+author_col) for each authored row."""
    total_cols = num_manuscripts + num_contributors
    cols = [num_manuscripts + author_col] * len(authored_rows)
    data = [1.0] * len(authored_rows)
    return sparse.csr_matrix(
        (data, (authored_rows, cols)),
        shape=(num_manuscripts, total_cols),
    )


class TestGetGIndexCorrectness:

    def test_case_a_g3(self):
        """
        3 papers, citation counts [9, 4, 1] -> g = 3.
        cumsum [9, 13, 14] >= rank^2 [1, 4, 9] at all ranks.
        """
        m = 15
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0, 1, 2])

        refs_dense = np.zeros((m, m), dtype=float)
        for col in range(9):
            refs_dense[0, col] = 1.0
        for col in range(4):
            refs_dense[1, col] = 1.0
        refs_dense[2, 0] = 1.0
        references = sparse.csr_matrix(refs_dense)

        assert get_g_index(capital, references, author_col) == 3

    def test_case_b_g2(self):
        """
        4 papers, citation counts [3, 1, 0, 0] -> g = 2.
        cumsum [3, 4] >= rank^2 [1, 4]; cumsum[2]=4 < 9.
        """
        m = 10
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0, 1, 2, 3])

        refs_dense = np.zeros((m, m), dtype=float)
        for col in range(3):
            refs_dense[0, col] = 1.0
        refs_dense[1, 0] = 1.0
        references = sparse.csr_matrix(refs_dense)

        assert get_g_index(capital, references, author_col) == 2

    def test_case_c_g1(self):
        """
        2 papers, citation counts [2, 0] -> g = 1.
        cumsum[0]=2 >= 1; cumsum[1]=2 < 4.
        """
        m = 5
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0, 1])

        refs_dense = np.zeros((m, m), dtype=float)
        refs_dense[0, 0] = 1.0
        refs_dense[0, 1] = 1.0
        references = sparse.csr_matrix(refs_dense)

        assert get_g_index(capital, references, author_col) == 1

    def test_case_d_no_papers_g0(self):
        """Author with no capital entries -> g = 0."""
        m = 5
        # capital has entries only for contributor_col=1, not 0
        capital = _make_capital(m, 2, author_col=1, authored_rows=[0, 1])
        references = sparse.csr_matrix((m, m))

        assert get_g_index(capital, references, contributor_col=0) == 0

    def test_case_e_zero_citations_g0(self):
        """
        Author has 2 papers but both have 0 citations -> g = 0.
        cumsum[0]=0 < 1 -> loop exits immediately.
        """
        m = 5
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0, 1])
        references = sparse.csr_matrix((m, m))

        assert get_g_index(capital, references, author_col) == 0

    def test_g_index_geq_h_index(self):
        """
        g-index is always >= h-index for the same author.
        For citations [9, 4, 1]: h=2 (rank2: 4>=2, rank3: 1<3), g=3.
        """
        m = 15
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0, 1, 2])

        refs_dense = np.zeros((m, m), dtype=float)
        for col in range(9):
            refs_dense[0, col] = 1.0
        for col in range(4):
            refs_dense[1, col] = 1.0
        refs_dense[2, 0] = 1.0
        references = sparse.csr_matrix(refs_dense)

        g = get_g_index(capital, references, author_col)
        h = get_h_index(capital, references, author_col)
        assert g >= h

    def test_single_paper_exact_square_citations(self):
        """
        1 paper with exactly 1 citation: cumsum[0]=1 >= 1^2=1 -> g=1.
        """
        m = 5
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0])

        refs_dense = np.zeros((m, m), dtype=float)
        refs_dense[0, 1] = 1.0
        references = sparse.csr_matrix(refs_dense)

        assert get_g_index(capital, references, author_col) == 1

    def test_single_paper_no_citations_g0(self):
        """1 paper with 0 citations -> g = 0."""
        m = 5
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0])
        references = sparse.csr_matrix((m, m))

        assert get_g_index(capital, references, author_col) == 0

    def test_multiple_contributors_isolation(self):
        """
        Two contributors share the same capital matrix.
        Contributor 0 has papers [0,1] with citations [3,1] -> g=2.
        Contributor 1 has paper  [2]   with citations [0]   -> g=0.
        Each is computed independently.
        """
        m = 10
        c = 2
        total_cols = m + c

        rows = [0, 1, 2]
        cols = [m + 0, m + 0, m + 1]
        data = [1.0, 1.0, 1.0]
        capital = sparse.csr_matrix(
            (data, (rows, cols)), shape=(m, total_cols)
        )

        refs_dense = np.zeros((m, m), dtype=float)
        for col in range(3):
            refs_dense[0, col] = 1.0
        refs_dense[1, 0] = 1.0
        references = sparse.csr_matrix(refs_dense)

        assert get_g_index(capital, references, contributor_col=0) == 2
        assert get_g_index(capital, references, contributor_col=1) == 0

    def test_returns_int(self):
        """Return type must be int."""
        m = 5
        author_col = 0
        capital = _make_capital(m, 1, author_col, [0])
        references = sparse.csr_matrix((m, m))

        result = get_g_index(capital, references, author_col)
        assert isinstance(result, int)
