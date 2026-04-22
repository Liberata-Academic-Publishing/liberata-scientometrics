"""
System health metrics computation module.

This module provides classes and functions for computing and analyzing
system health metrics, such as growth rates and shrinkage rates, and various
properties related to the health of a portfiolio.
"""

from typing import Dict, List
from scipy import sparse

from liberata_metrics.metrics.portfolio_metrics import academic_capital


def get_academic_capital_growth_rate(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
) -> float:
    """
    Compute the average Academic Capital Growth Rate of a portfolio. This is the 
    rate at which the academic capital of a portfolio has grown over a historical
    sequence of capital matrices.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse capital matrices
        representing portfolio states over time. Consecutive entries are
        should be a year apart
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column
        index in the capital matrices.

    Returns
    -------
    float
        The compound growth rate per time period.
        Returns 0.0 if the starting capital is zero

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than two entries.

    Notes
    -----
    """
    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute growth rate')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    cap_start = academic_capital(capital_history[0], contributor_index_map_subset)
    cap_end = academic_capital(capital_history[-1], contributor_index_map_subset)

    if cap_start <= 0.0:
        return 0.0

    T = len(capital_history) - 1
    return float((cap_end / cap_start) ** (1.0 / T) - 1.0)


def total_fair_market_price(
    capital: sparse.spmatrix,
    contributor_index_map: Dict[str, int],
    is_reviewer: bool,
) -> float:
    """
    Compute the total fair market price for either reviewers or replicators across all manuscripts.
    This is the total capital across all reveiwers or replicators.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix of shape (M, M + C), where M is the number of
        manuscripts and C = num_contributors * 3.
    contributor_index_map : Dict[str, int]
        Mapping from contributor identifier to base column index.
    is_reviewer : bool
        If True, sums the reviewer block (second role block).
        If False, sums the replicator block (third role block).

    Returns
    -------
    float
        Total fair market price across all manuscripts for the specified role.
    """
    #Note: This is an internal helper function
    M = capital.shape[0]
    C = int(capital.shape[1]) - M

    num_contributors = C // 3
    indices = list(contributor_index_map.values())

    if is_reviewer:
        col_indices = [M + num_contributors + i for i in indices]
    else:
        col_indices = [M + 2 * num_contributors + i for i in indices]

    return float(capital[:, col_indices].sum())


def get_reviewer_shrinkage_rate(
    capital_history: List[sparse.spmatrix],
    contributor_index_map: Dict[str, int],
) -> float:
    """
    Compute the shrinkage rate of the global fair market price for reviewers.
    This is just the negative rate of change in the total FMP for reviewers.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse capital matrices.
    contributor_index_map : Dict[str, int]
        Full contributor map (all contributors, not a subset).

    Returns
    -------
    float
        Per-period shrinkage rate for reviewer FMP

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than two entries.
    """
    if len(capital_history) < 2:
        raise ValueError(
            'capital_history must contain at least two entries '
            'to compute reviewer shrinkage rate'
        )

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError(
                'All elements in capital_history must be scipy sparse matrices'
            )


    T = float(len(capital_history) - 1)

    return (total_fair_market_price(capital_history[0], contributor_index_map, is_reviewer=True)
            - total_fair_market_price(capital_history[-1], contributor_index_map, is_reviewer=True)) / T


def get_replicator_shrinkage_rate(
    capital_history: List[sparse.spmatrix],
    contributor_index_map: Dict[str, int],
) -> float:
    """
    Compute the shrinkage rate of the global fair market price for replicators.
    This is just the negative rate of change in the total FMP for replicators.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse capital matrices.
    contributor_index_map : Dict[str, int]
        Full contributor map (all contributors, not a subset).

    Returns
    -------
    float
        Per-period shrinkage rate for replicator FMP.

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than two entries.
    """
    if len(capital_history) < 2:
        raise ValueError(
            'capital_history must contain at least two entries '
            'to compute replicator shrinkage rate'
        )

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError(
                'All elements in capital_history must be scipy sparse matrices'
            )

    T = float(len(capital_history) - 1)

    return (total_fair_market_price(capital_history[0], contributor_index_map, is_reviewer=False) 
             - total_fair_market_price(capital_history[-1], contributor_index_map, is_reviewer=False)) / T