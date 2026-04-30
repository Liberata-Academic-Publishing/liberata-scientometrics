"""
System health metrics computation module.

This module provides classes and functions for computing and analyzing
system health metrics, such as growth rates and shrinkage rates, and various
properties related to the health of a portfiolio.
"""

from typing import Dict, List
import numpy as np
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

    try:
        cap_sub = capital[:, col_indices]
    except Exception:
        cap_sub = sparse.hstack([capital.getcol(c) for c in col_indices], format="csr")

    
    return float(cap_sub.sum())


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

    return -(total_fair_market_price(capital_history[-1], contributor_index_map, is_reviewer=False)
             - total_fair_market_price(capital_history[0], contributor_index_map, is_reviewer=False)) / T


def get_reviewer_fmp_volatility(
    capital_history: List[sparse.spmatrix],
    contributor_index_map: Dict[str, int],
) -> float:
    """
    Compute the volatility of the global fair market price for reviewers
    over a time period of n steps.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse capital matrices.
    contributor_index_map : Dict[str, int]
        Full contributor map (all contributors, not a subset).

    Returns
    -------
    float
        Volatility of reviewer FMP.

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than two entries.
    """
    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute volatility')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    n = len(capital_history)
    M = capital_history[0].shape[0]
    C = int(capital_history[0].shape[1]) - M
    num_contributors = C // 3
    indices = list(contributor_index_map.values())
    col_indices = [M + num_contributors + i for i in indices]

    prices = np.array([float(cap[:, col_indices].sum()) for cap in capital_history])
    return np.std(prices, ddof=0) * np.sqrt(n)


def get_replicator_fmp_volatility(
    capital_history: List[sparse.spmatrix],
    contributor_index_map: Dict[str, int],
) -> float:
    """
    Compute the volatility of the global fair market price for replicators
    over a time period of n steps.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse capital matrices.
    contributor_index_map : Dict[str, int]
        Full contributor map (all contributors, not a subset).

    Returns
    -------
    float
        Volatility of replicator FMP.

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than two entries.
    """
    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute volatility')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    n = len(capital_history)
    M = capital_history[0].shape[0]
    C = int(capital_history[0].shape[1]) - M
    num_contributors = C // 3
    indices = list(contributor_index_map.values())
    col_indices = [M + 2 * num_contributors + i for i in indices]

    prices = np.array([float(cap[:, col_indices].sum()) for cap in capital_history])
    return np.std(prices, ddof=0) * np.sqrt(n)


def get_funding_efficiency(
    capital: sparse.spmatrix,
    total_spending: float,
) -> float:
    """
    Compute global research funding efficiency, which is the academic capital produced per unit of research spending.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    total_spending : float
        Total global research spending ($Θ).

    Returns
    -------
    float
        Returns the funding efficiency.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If total_spending is not positive.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    if total_spending <= 0.0:
        raise ValueError('total_spending must be positive')

    return float(capital.sum()) / total_spending


def get_gdp_efficiency(
    capital: sparse.spmatrix,
    total_spending: float,
    gdp: float,
) -> float:
    """
    Compute global research GDP efficiency, which is funding efficiency scaled by GDP.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    total_spending : float
        Total global research spending ($Θ).
    gdp : float
        Global GDP (GDP_Θ).

    Returns
    -------
    float
        Returns the GDP efficiency.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If total_spending or gdp is not positive.
    """
    if gdp <= 0.0:
        raise ValueError('gdp must be positive')

    return get_funding_efficiency(capital, total_spending) * gdp


def get_ppp_efficiency(
    capital: sparse.spmatrix,
    total_spending: float,
    ppp: float,
) -> float:
    """
    Compute global research PPP efficiency, which is funding efficiency scaled by PPP.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    total_spending : float
        Total global research spending ($Θ).
    ppp : float
        Global purchasing power parity (PPP_Θ).

    Returns
    -------
    float
        Returns the PPP efficiency.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If total_spending or ppp is not positive.
    """
    if ppp <= 0.0:
        raise ValueError('ppp must be positive')

    return get_funding_efficiency(capital, total_spending) * ppp


def get_time_efficiency(
    capital: sparse.spmatrix,
    delta_t: float,
) -> float:
    """
    Compute global research time efficiency, which is the academic capital produced per unit time.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    delta_t : float
        Time elapsed (Δt_Θ).

    Returns
    -------
    float
        Returns the time efficiency.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If delta_t is not positive.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    if delta_t <= 0.0:
        raise ValueError('delta_t must be positive')

    return float(capital.sum()) / delta_t


def get_regional_academic_capital(
    capital: sparse.spmatrix,
    region_contributor_index_maps: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """
    Compute total academic capital per geographic region.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    region_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from region identifier to contributor index map for that region.

    Returns
    -------
    Dict[str, float]
        Mapping from region identifier to total academic capital.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    """

    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    return {
        region: academic_capital(capital, contributor_map)
        for region, contributor_map in region_contributor_index_maps.items()
    }


def get_field_capital_shares(
    capital: sparse.spmatrix,
    region_contributor_index_map: Dict[str, int],
    field_contributor_index_maps: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """
    Compute the proportionate contribution of each academic field to the total
    academic capital of a region.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix.
    region_contributor_index_map : Dict[str, int]
        Contributor index map for the full region.
    field_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from field identifier to contributor index map for that field
        within the region.

    Returns
    -------
    Dict[str, float]
        Mapping from field identifier to its proportionate share of regional academic capital.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If total regional academic capital is zero.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    
    total = academic_capital(capital, region_contributor_index_map)

    if total == 0.0:
        raise ValueError('Total regional academic capital is zero, cannot compute field shares')

    return {
        field: academic_capital(capital, contributor_map) / total
        for field, contributor_map in field_contributor_index_maps.items()
    }


def get_regional_hhi(
    capital: sparse.spmatrix,
    region_contributor_index_map: Dict[str, int],
    field_contributor_index_maps: Dict[str, Dict[str, int]],
) -> float:
    """
    Compute the HHI of academic capital concentration across fields for a region.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix.
    region_contributor_index_map : Dict[str, int]
        Contributor index map for the full region.
    field_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from field identifier to contributor index map for that field
        within the region.

    Returns
    -------
    float
        HHI value.

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If total regional academic capital is zero.
    """

    #Errors raised in the function it calls
    shares = get_field_capital_shares(capital, region_contributor_index_map, field_contributor_index_maps)
    return sum(s ** 2 for s in shares.values())


def _gini_weighted(
    capital: sparse.spmatrix,
    region_contributor_index_maps: Dict[str, Dict[str, int]],
    weights: Dict[str, float],
) -> float:
    """
    Compute the Gini coefficient using weighted pairwise absolute differences
    of regional academic capital.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix.
    region_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from region identifier to contributor index map.
    weights : Dict[str, float]
        Per-region weights (e.g. population, contributor count, GDP).

    Returns
    -------
    float
        Gini coefficient.

    Raises
    ------
    ValueError
        If total capital is zero
    """
    regions = list(region_contributor_index_maps.keys())
    caps = np.array([academic_capital(capital, region_contributor_index_maps[r]) for r in regions])
    wts = np.array([weights[r] for r in regions])

    total_capital = caps.sum()
    if total_capital == 0.0:
        raise ValueError('Total capital is zero')

    per_unit = np.where(wts > 0, caps / wts, 0.0)
    n = len(regions)
    diff_sum = float(np.sum(np.abs(per_unit[:, None] - per_unit[None, :])))
    mean_per_unit = per_unit.mean()

    return diff_sum / (2 * n ** 2 * mean_per_unit)


def get_gini_per_capita(
    capital: sparse.spmatrix,
    region_contributor_index_maps: Dict[str, Dict[str, int]],
    regional_populations: Dict[str, float],
) -> float:
    """
    Compute the Gini coefficient of per capita academic capital inequality
    across regions.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    region_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from region identifier to contributor index map for that region.
    regional_populations : Dict[str, float]
        Mapping from region identifier to population.

    Returns
    -------
    float
        Gini coefficient in [0, 1].

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If a region present in region_contributor_index_maps is missing from regional_populations.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    for region in region_contributor_index_maps:
        if region not in regional_populations:
            raise ValueError(f'Population missing for region: {region}')

    return _gini_weighted(capital, region_contributor_index_maps, regional_populations)


def get_gini_per_contributor(
    capital: sparse.spmatrix,
    region_contributor_index_maps: Dict[str, Dict[str, int]],
    regional_contributor_counts: Dict[str, int],
) -> float:
    """
    Compute the Gini coefficient of per contributor academic capital inequality across regions.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    region_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from region identifier to contributor index map for that region.
    regional_contributor_counts : Dict[str, int]
        Mapping from region identifier to number of contributors in that region.

    Returns
    -------
    float
        Gini coefficient in [0, 1].

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If a region present in region_contributor_index_maps is missing from regional_contributor_counts.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    for region in region_contributor_index_maps:
        if region not in regional_contributor_counts:
            raise ValueError(f'Contributor count missing for region: {region}')

    weights = {r: float(c) for r, c in regional_contributor_counts.items()}
    return _gini_weighted(capital, region_contributor_index_maps, weights)


def get_gini_per_gdp(
    capital: sparse.spmatrix,
    region_contributor_index_maps: Dict[str, Dict[str, int]],
    regional_gdps: Dict[str, float],
) -> float:
    """
    Compute the Gini coefficient of per GDP academic capital inequality across regions.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix covering all manuscripts globally.
    region_contributor_index_maps : Dict[str, Dict[str, int]]
        Mapping from region identifier to contributor index map for that region.
    regional_gdps : Dict[str, float]
        Mapping from region identifier to GDP.

    Returns
    -------
    float
        Gini coefficient in [0, 1].

    Raises
    ------
    TypeError
        If capital is not a scipy sparse matrix.
    ValueError
        If a region present in region_contributor_index_maps is missing from regional_gdps.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    for region in region_contributor_index_maps:
        if region not in regional_gdps:
            raise ValueError(f'GDP missing for region: {region}')

    return _gini_weighted(capital, region_contributor_index_maps, regional_gdps)
