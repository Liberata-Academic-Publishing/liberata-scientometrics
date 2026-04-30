from scipy import sparse
import numpy as np

def hhi_discrepancy(
    shares_matrix: sparse.spmatrix,
    mask_portfolio: slice,
) -> float:
    """
    Compute the discrepancy between the HHI of the field and that of the manuscript or portfolio.
    Parameters
    ----------
    shares_matrix : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents an author.
        Entries are the shares each author has for each given manuscript.
        This matrix should have all manuscripts in the field/discipline.
    indices : np.array
        Array of indices of manuscripts in the shares_matrix that are to be considered in the HHI calculation.
    Returns
    -------
    float
        The HHID value, a measure of discrepancy between a portfolio and the industry. Higher values indicate an anomaly that 
        should be further investigated.
    Examples
    --------
    >>> hhi_discrepancy(shares_matrix, [1,2,5,6,7])
    0.375
    """
    hhi_field = share_splits_inequality(shares_matrix)

    #Take submatrix of shares_matrix corresponding to the given slice
    hhi_portfolio = share_splits_inequality(shares_matrix[mask_portfolio, :])
    
    return abs(hhi_field - hhi_portfolio)


def share_splits_inequality(
    portfolio: sparse.spmatrix,
) -> float:
    """
    Compute the Herfindahl-Hirschman Index (HHI) of a portfolio of manuscripts.
    Parameters
    ----------
    portfolio : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents an author.
        Entries are the shares each author has for each given manuscript.
    Returns
    -------
    float
        The mean HHI value, a measure of the typical concentration in the portfolio. Returns 0.0 if the portfolio is empty.
    ------
    TypeError
        If `portfolio` is not a scipy sparse matrix.
    Notes
    -----
    - The share splits inequality is calculated as the mean HHI of all provided manuscripts.
    - The HHI for a single manuscript is calculated as the sum of the squares of the share splits for that manuscript.
    Examples
    --------
    >>> share_splits_inequality(portfolio)
    0.375
    """
    if not sparse.issparse(portfolio):
        raise TypeError('Portfolio matrix must be a scipy sparse matrix')
    
    # Square the portfolio shares (element wise), then sum across authors for each manuscript, 
    # and finally take the mean across the manuscripts
    portfolio_squared = portfolio.multiply(portfolio)
    return float(portfolio_squared.sum(axis=1).mean())
