from datetime import date, datetime, timedelta
from typing import Optional
import numpy as np
from scipy import sparse


def _rng(seed: Optional[int]) -> np.random.RandomState:
    '''create RandomState'''
    return np.random.RandomState(seed)


def random_date(rng: np.random.RandomState, start: date, end: date) -> date:
    '''draw a random date in given range'''
    days = (end-start).days
    if days <= 0:
        return start
    return start + timedelta(days=int(rng.randint(0, days + 1)))


def sparse_divide(
        divisor: sparse.spmatrix, 
        dividend: sparse.spmatrix,
        ) -> sparse.spmatrix:
    """
    Element-wise division of two sparse matrices with zero-division handling.
    Performs element-wise division by computing the multiplicative inverse of the
    divisor and multiplying with the dividend. Division by zero is implicitly
    handled by sparse matrix structure - zero elements in the divisor remain zero
    in the result.
    Parameters
    ----------
    divisor : sparse.spmatrix
        Sparse matrix to divide by. Must have non-zero elements at positions
        where division is desired.
    dividend : sparse.spmatrix
        Sparse matrix to be divided. Must have compatible shape with divisor.
    Returns
    -------
    sparse.spmatrix
        Sparse matrix containing the element-wise division result (dividend / divisor).
        The result maintains the sparsity pattern of the dividend.
    Notes
    -----
    - Zero elements in the divisor are treated as undefined divisions and do not
        appear in the output due to sparse matrix structure.
    - Both input matrices must have compatible shapes for element-wise operations.
    - The result is returned as a sparse CSR array format.
    Examples
    --------
    >>> from scipy import sparse
    >>> divisor = sparse.csr_matrix([[2, 0], [0, 4]])
    >>> dividend = sparse.csr_matrix([[4, 0], [0, 8]])
    >>> result = sparse_divide(divisor, dividend)
    >>> result.toarray()
    array([[2., 0.],
            [0., 2.]])
    """

    if divisor.shape != dividend.shape:
        raise ValueError("Divisor and dividend must have the same shape for element-wise division.")
    
    try:
        inv_divisor_data = 1.0 / divisor.data
        inv_divisor = sparse.csr_array((inv_divisor_data, divisor.indices, divisor.indptr), shape=divisor.shape)
    except TypeError as te:
        if "unsupported operand type(s) for /: 'float' and 'memoryview'" in str(te):
            inv_divisor = 1.0 / divisor
    

    return dividend.multiply(inv_divisor)
