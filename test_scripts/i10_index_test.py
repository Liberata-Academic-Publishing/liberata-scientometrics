import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.legacy_metric import (  # noqa: E402
    get_i10_index,
)



def _make_capital(
    num_manuscripts: int,
    num_contributors: int,
    author_col: int,
    manuscript_rows: list,
) -> sparse.csr_matrix:
    """Capital matrix with the given author on manuscript_rows."""
    total_cols = num_manuscripts + num_contributors
    col = num_manuscripts + author_col
    data = [1.0] * len(manuscript_rows)
    cols = [col] * len(manuscript_rows)
    return sparse.csr_matrix(
        (data, (manuscript_rows, cols)),
        shape=(num_manuscripts, total_cols),
    )


def _make_references(
    num_manuscripts: int, citations: dict
) -> sparse.csr_matrix:
    """
    Build an (N x N) references matrix where citations[i] is the number of
    papers citing paper i.

    get_citation_counts binarizes the matrix before summing, so each citation
    must occupy a distinct column.  The matrix must be square (M x M), so
    num_manuscripts must be >= max citation count for any paper.  Each test
    is responsible for passing a large enough num_manuscripts.
    """
    arr = np.zeros((num_manuscripts, num_manuscripts), dtype=np.int8)
    for paper_row, count in citations.items():
        for j in range(count):
            arr[paper_row, j] = 1
    return sparse.csr_matrix(arr)



def test_i10_basic():
    """3 papers: 2 have >=10 citations, 1 has fewer. Expect i10=2."""
    # M must be >= max citation count (15); use 20 for headroom
    num_manuscripts = 20
    author_col = 0
    manuscript_rows = [0, 1, 2]

    capital = _make_capital(num_manuscripts, 1, author_col, manuscript_rows)
    # paper 0: 15 cites, paper 1: 10 cites, paper 2: 5 cites
    references = _make_references(num_manuscripts, {0: 15, 1: 10, 2: 5})

    assert get_i10_index(capital, references, author_col) == 2


def test_i10_all_qualify():
    """All 4 papers have exactly 10 citations. Expect i10=4."""
    num_manuscripts = 15  # >= 10
    author_col = 0
    manuscript_rows = [0, 1, 2, 3]

    capital = _make_capital(num_manuscripts, 1, author_col, manuscript_rows)
    references = _make_references(
        num_manuscripts, {0: 10, 1: 10, 2: 10, 3: 10}
    )

    assert get_i10_index(capital, references, author_col) == 4


def test_i10_none_qualify():
    """All papers have fewer than 10 citations. Expect i10=0."""
    num_manuscripts = 15
    author_col = 0
    manuscript_rows = [0, 1, 2]

    capital = _make_capital(num_manuscripts, 1, author_col, manuscript_rows)
    references = _make_references(num_manuscripts, {0: 9, 1: 5, 2: 0})

    assert get_i10_index(capital, references, author_col) == 0


def test_i10_no_manuscripts():
    """Author with no manuscripts. Expect i10=0."""
    num_manuscripts = 5

    capital = _make_capital(num_manuscripts, 1, 0, [])
    references = _make_references(num_manuscripts, {})

    assert get_i10_index(capital, references, 0) == 0


def test_i10_boundary_exactly_ten():
    """Boundary: exactly 10 citations qualifies. Expect i10=1."""
    num_manuscripts = 15  # >= 10
    author_col = 0

    capital = _make_capital(num_manuscripts, 1, author_col, [0])
    references = _make_references(num_manuscripts, {0: 10})

    assert get_i10_index(capital, references, author_col) == 1


def test_i10_boundary_nine_does_not_qualify():
    """Boundary: 9 citations does not qualify. Expect i10=0."""
    num_manuscripts = 15
    author_col = 0

    capital = _make_capital(num_manuscripts, 1, author_col, [0])
    references = _make_references(num_manuscripts, {0: 9})

    assert get_i10_index(capital, references, author_col) == 0


# -- Multi-author isolation ---------------------------------------------------

def test_i10_only_counts_own_papers():
    """
    Two authors share the same reference matrix.
    Author 0: papers 0 and 1 (15 and 12 citations) -> i10=2.
    Author 1: paper 2 (3 citations) -> i10=0.
    """
    num_manuscripts = 20  # >= 15
    num_contributors = 2
    total_cols = num_manuscripts + num_contributors

    # Author 0 on papers 0 and 1; Author 1 on paper 2
    data = [1.0, 1.0, 1.0]
    rows = [0, 1, 2]
    cols = [num_manuscripts, num_manuscripts, num_manuscripts + 1]
    capital = sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_manuscripts, total_cols)
    )
    references = _make_references(
        num_manuscripts, {0: 15, 1: 12, 2: 3}
    )

    assert get_i10_index(capital, references, 0) == 2
    assert get_i10_index(capital, references, 1) == 0


# -- Zero citation matrix -----------------------------------------------------

def test_i10_zero_references():
    """Author has manuscripts but no citations at all. Expect i10=0."""
    num_manuscripts = 5
    author_col = 0

    capital = _make_capital(num_manuscripts, 1, author_col, [0, 1, 2])
    references = sparse.csr_matrix((num_manuscripts, num_manuscripts))

    assert get_i10_index(capital, references, author_col) == 0


# -- Return type --------------------------------------------------------------

def test_i10_returns_int():
    """get_i10_index must return a Python int."""
    num_manuscripts = 15
    author_col = 0

    capital = _make_capital(num_manuscripts, 1, author_col, [0, 1])
    references = _make_references(num_manuscripts, {0: 11, 1: 3})

    result = get_i10_index(capital, references, author_col)
    assert isinstance(result, int)
