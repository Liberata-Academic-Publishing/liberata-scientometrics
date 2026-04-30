"""
Portfolio metrics computation module.

This module provides classes and functions for computing and analyzing
portfolio performance metrics, including returns, risk, correlation,
and spectral properties.
"""
from typing import Dict, Iterable, List, Union, Tuple, Optional
from typing import Dict, Iterable, List, Union, Tuple, Optional
import numpy as np
from scipy import sparse
import warnings
from liberata_metrics.utils import sparse_divide



def academic_capital(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    """
    Compute total academic capital for a subset of contributors.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row is a manuscript and each column is a contributor.
        Entries represent capital allocated from a contributor to a manuscript.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`.
        Only columns referenced by this mapping are included in the total.

    Returns
    -------
    float
        Total academic capital aggregated across the specified contributor columns.
        Returns 0.0 if the mapping is empty.

    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.

    Notes
    -----
    The function sums the selected columns of the sparse matrix and returns the scalar total.
    """

    # validation
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    
    M = int(capital.shape[0])
    C = int(capital.shape[1])-M
    num_contributors = C // 3 if C % 3 == 0 else C

    indices = np.array(list(contributor_index_map_subset.values()))
    if len(indices) == 0:
        return 0.0
    
    col_sums = np.array(capital.sum(axis=0)).ravel()
    if num_contributors < C:
        sums = col_sums[M + indices] + col_sums[M + num_contributors + indices] + col_sums[M + 2*num_contributors + indices]  # include manuscript self-citations
    else:
        sums = col_sums[M + indices]
    total = float(sums.sum())

    return total


def get_per_manuscript_cap(
        capital: sparse.spmatrix,
        contributor_index_map_subset: Dict[str, int]                                    
) -> sparse.spmatrix:
    

    # Compute the AC per manuscript for a subset of contributors
    """
    Extract per-manuscript capital for a subset of contributors.
    Parameters
    ---------- 
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital allocated from a contributor to a manuscript.
    contributor_index_map_subset : Dict[str, int]   
        Mapping from contributor identifier to the corresponding column index in `capital`.
        Only the columns specified by the values of this mapping are considered.
    Returns
    -------
    scipy.sparse.spmatrix
        Sparse matrix containing the per-manuscript capital aggregated over the specified contributor columns.
    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    ValueError
        If `contributor_index_map_subset` is empty.
    Notes
    -----
    - The returned matrix has the same number of rows as `capital` and a single column,
        where each entry represents the total capital for that manuscript from the specified contributors.
    - Implementation details: the function first tries to slice `capital` using the provided column indices.
      If slicing fails, it falls back to horizontally stacking individual columns via `capital.getcol`. 
    Examples
    --------
    >>> # capital: scipy CSR matrix with shape (num_manuscripts, num_contributors)
    >>> # contributor_index_map_subset = {'alice': 0, 'bob': 2}
    >>> get_per_manuscript_cap(capital, {'alice': 0, 'bob   : 2})
    <2x1 sparse matrix of type '<class 'numpy.float64'>' 
        with 3 stored elements in Compressed Sparse Row format>
    """
    
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    if not contributor_index_map_subset:
        raise ValueError('Contributor to index mapping subset cannot be empty')

    col_indices = list(contributor_index_map_subset.values())
    try:
        cap_sub = capital[:, col_indices]
    except Exception:
        cap_sub = sparse.hstack([capital.getcol(c) for c in col_indices], format="csr")
    
    return cap_sub


def get_col_indices(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int],
) -> np.ndarray:
    """
    Find the column indices corresponding to a subset of contributors,
    accounting for where contributor columns are repeated three times.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse capital matrix of shape (M, M + C), where M is the number of manuscripts
        and C is the total number of contributor columns.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to their base column index

    Returns
    -------
    np.ndarray
        Array of absolute column indices for the given contributors,
        spanning all role blocks if the layout is role-blocked.
    """

    M = int(capital.shape[0])
    C = int(capital.shape[1]) - M

    num_contributors = C // 3 if C % 3 == 0 else C

    indices = np.array(list(contributor_index_map_subset.values()))

    if num_contributors < C:
        return np.concatenate([
            M + indices,
            M + num_contributors + indices,
            M + 2 * num_contributors + indices,
        ])
    
    return M + indices



def allocation_weights(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> Dict[int, float]:
    """
    Compute allocation weights for each manuscript in a portfolio, i.e., the fraction of total portfolio capital
    contributed due to each manuscript.
    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital allocated from a contributor to a manuscript. This
        function requires a scipy sparse matrix input and will raise TypeError if given a non-sparse array.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`. Only the
        columns specified by the values of this mapping are considered. If this mapping is empty, the
        function returns an empty dictionary.
    Returns
    -------
    Dict[int, float]
        A dictionary mapping manuscript row indices (int) to their allocation weight (float). Each weight
        equals the manuscript's aggregated capital from the selected contributors divided by the total
        capital across the portfolio. Manuscripts with zero aggregated capital are omitted. If the total
        portfolio capital is zero, an empty dictionary is returned.
    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    IndexError, ValueError, or other slicing-related exceptions
        May be raised if provided column indices are out of bounds or otherwise invalid. The function
        attempts a direct column slice and falls back to constructing the submatrix by stacking individual
        columns.
    Notes
    -----
    - The returned weights sum to 1.0 up to floating-point precision when non-empty.
    - Implementation details: the function first tries to slice `capital` using the provided column indices.
      If slicing fails, it falls back to horizontally stacking individual columns via `capital.getcol`.
    - Output contains only manuscripts with non-zero aggregated capital.
    Examples
    --------
    >>> # capital: scipy CSR matrix with shape (num_manuscripts, num_contributors)
    >>> # contributor_index_map_subset = {'alice': 0, 'bob': 2}
    >>> allocation_weights(capital, {'alice': 0, 'bob': 2})
    {0: 0.5, 3: 0.25, 7: 0.25}
    """

    # if not sparse.issparse(capital):
    #     raise TypeError('Capital matrix must be a scipy sparse matrix')
    # if not contributor_index_map_subset:
    #     return {}
    
    # col_indices = list(contributor_index_map_subset.values())
    # try:
    #     cap_sub = capital[:, col_indices]
    # except Exception:
    #     cap_sub = sparse.hstack([capital.getcol(c) for c in col_indices], format="csr")


    cap_sub = capital.tocsr()[:, get_col_indices(capital, contributor_index_map_subset)]

    per_manuscript_cap = np.asarray(cap_sub.sum(axis=1)).ravel()
    total_portfolio_capital = float(per_manuscript_cap.sum())

    if total_portfolio_capital == 0.0:
        return {}
    
    nz_idx = np.nonzero(per_manuscript_cap)[0]
    weights = {m: float(per_manuscript_cap[m]/total_portfolio_capital) for m in nz_idx}

    return weights



def portfolio_hhi(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    
    """
    Compute the Herfindahl-Hirschman Index (HHI) of a portfolio.
    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital allocated from a contributor to a manuscript.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`.
    Returns
    -------
    float
        The HHI value, a measure of concentration in the portfolio. Returns 0.0 if the portfolio is empty
        or if all weights are zero.
    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    Notes
    -----
    - The HHI is calculated as the sum of the squares of the allocation weights.
    Examples
    --------
    >>> portfolio_hhi(capital, {'alice': 0, 'bob': 2})
    0.375
    """

    '''compute HHI of portfolio'''
    weights = allocation_weights(capital, contributor_index_map_subset)
    if not weights:
        return 0.0
    
    weight_vals = np.array(list(weights.values()), dtype=float)
    if np.allclose(weight_vals, 0.0):
        return 0.0
    
    return float(np.sum(weight_vals*weight_vals))



def portfolio_gini(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    
    """
    Compute the Gini coefficient of a portfolio.
    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital allocated from a contributor to a manuscript.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`.
    Returns
    -------
    float
        The Gini coefficient, a measure of inequality in the portfolio. Returns 0.0 if the portfolio is empty
        or if there is only one manuscript.
    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    Notes
    -----
    - The Gini coefficient is calculated based on the allocation weights.
    Examples
    --------
    >>> portfolio_gini(capital, {'alice': 0, 'bob': 2})
    0.25
    """

    # validation
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    weights = allocation_weights(capital, contributor_index_map_subset)
    if not weights:
        return 0.0

    wvals = np.array(list(weights.values()), dtype=float)
    n = wvals.size
    if n <= 1:
        return 0.0

    diffs = np.abs(wvals[:, None] - wvals[None, :])
    gini = float(diffs.sum() / (2.0 * float(n)))
    return gini



def portfolio_normalized_entropy(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    
    """
    Compute the normalized entropy of a portfolio.
    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital allocated from a contributor to a manuscript.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`.
    Returns
    -------
    float
        The normalized entropy, a measure of diversity in the portfolio. Returns 0.0 if the portfolio is empty
        or if there is only one manuscript.
    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    Notes
    -----
    - The normalized entropy is calculated based on the allocation weights.
    Examples
    --------
    >>> portfolio_normalized_entropy(capital, {'alice': 0, 'bob': 2})
    0.85
    """

    # validation
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    weights = allocation_weights(capital, contributor_index_map_subset)
    if not weights:
        return 0.0

    wvals = np.array(list(weights.values()), dtype=float)
    n = wvals.size
    if n <= 1:
        return 0.0

    entropy = -float(np.sum(wvals * np.log(wvals)))
    h_norm = float(entropy / np.log(float(n)))
    return h_norm

def get_manuscript_memberships_matrix(
    manuscript_tags: Dict[str, sparse.sparray],         # Map of primary tags to OHE sparse array of manuscript memberships (easier to iterate over len(tags) vs len(manuscripts))
) -> Tuple[sparse.spmatrix, List[str]]:
    
    """
    Build a stacked membership matrix from a tag-to-membership mapping.

    Parameters
    ----------
    manuscript_tags : Dict[str, sparse.sparray]
        Mapping from tag name to a sparse one-hot row array of length
        num_manuscripts indicating which manuscripts belong to that tag.

    Returns
    -------
    manuscript_memberships : scipy.sparse.spmatrix
        CSR matrix of shape (num_tags, num_manuscripts) where rows are tags
        and columns are manuscripts. Entry [i, j] is 1 if manuscript j
        belongs to tag i.
    keys : List[str]
        Ordered list of tag names corresponding to the rows of the matrix.
    """

    keys = list(manuscript_tags.keys())
    manuscript_memberships = sparse.vstack([manuscript_tags[k] for k in keys], format='csr')

    return manuscript_memberships, keys # return as csr


def mix_by_tag(capital_sub: sparse.sparray,                   
    manuscript_memberships: sparse.spmatrix, 
    ) -> sparse.sparray:

    """
    Find capital by tag and contributor

    Parameters
    ----------
    capital_sub : sparse.sparray (num_manuscripts, num_contributors)
        A capital matrix where entry [i, j] is the capital of contributor j on manuscript i

    manuscript_memberships : scipy.sparse.spmatrix (num_tags, num_manuscripts)
        Matrix where entry [t, i] is 1 if manuscript i belongs to tag t and 0 otherwise

    Returns
    -------
    sparse.sparray (num_tags, num_contributors)
        Entry [t, j] is the total capital of contributor j across all manuscripts belonging to tag t
    """

    return manuscript_memberships @ capital_sub

def mix_by_role(capital,
                contributor_index_map_subset,
                mask_by_role,
                manuscript_memberships,
                ):
    
    """
    Compute total capital and tag-wise capital mix for a specific role.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Full capital matrix
    
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to column index

    mask_by_role : slice
        Slice selecting the columns in the capital matrix that correspond to the role
        (Example: np.s_[:, 10:60] selects all rows and columns 10-59)

    manuscript_memberships : scipy.sparse.spmatrix (num_tags, num_manuscripts)
        Entry [t, i] is 1 if manuscript i belongs to tag t

    Returns
    -------
    rolewise_cap_tot : float
        Total capital attributed to this role across all manuscripts and contributors

    rolewise_capmix_by_tag : sparse.sparray (num_tags, num_contributors)
        Entry [t, j] is the total capital of contributor j in manuscripts belonging to tag t
    """
    
    # role_masked_capital = mask_by_role.multiply(capital)
    role_masked_capital = capital.tocsc()[mask_by_role]
    rolewise_cap_sub = get_per_manuscript_cap(role_masked_capital, contributor_index_map_subset)

    # compute total capital based on role mask
    rolewise_cap_tot = float(rolewise_cap_sub.sum())

    # compute tag-wise capital based on role mask
    rolewise_capmix_by_tag = mix_by_tag(rolewise_cap_sub, manuscript_memberships)

    return  rolewise_cap_tot, rolewise_capmix_by_tag

def query_portfolio_mix(
    capital: sparse.spmatrix,
    mask_authors: slice,
    mask_reviewers: slice,
    mask_replicators: slice,
    manuscript_memberships: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int],
    # num_contributors: int,
    # size_zero_block: int = 0,
    by: str = 'role',

) -> Tuple[List[float], List[sparse.sparray]]:
    
    # Add docstring
    """
    Query the portfolio mix by specified criteria.
    This function computes allocation weights across a portfolio of manuscripts
    and aggregates them based on the specified grouping criterion (by role or by tag).
    Args:
        capital (sparse.spmatrix): A scipy sparse matrix representing capital allocation.
        contributor_index_map_subset (Dict[str, int]): A dictionary mapping contributor names
            to their indices, representing a subset of all contributors.
        manuscript_index_map (Dict[str, int]): A dictionary mapping manuscript names to their indices.
        manuscript_tags (Dict[str, sparse.spmatrix]): A dictionary mapping tag names to sparse
            binary matrices indicating manuscript membership in each tag.
        by (str, optional): The criterion for grouping the portfolio mix. Either 'role' or 'tag'.
            Defaults to 'role'.
    Returns:
        dict[str, float]: A dictionary mapping role or tag names to their corresponding
            allocation weights in the portfolio.
    Raises:
        TypeError: If capital is not a scipy sparse matrix.
    Notes:
        - When by='tag': For each tag, a sparse mask identifies manuscripts belonging to that tag,
          and weights are intersected with the mask.
        - When by='role': Capital is subsampled by role, and allocation weights are computed
          for each role separately.
        - Returns an empty dictionary if no weights can be computed or if n <= 1.
    
    """

    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    # if not sparse.issparse(mask_authors):
    #     raise TypeError('Authors mask must be a scipy sparse matrix')
    # if not sparse.issparse(mask_reviewers):
    #     raise TypeError('Reviewers mask must be a scipy sparse matrix')
    # if not sparse.issparse(mask_replicators):
    #     raise TypeError('Replicators mask must be a scipy sparse matrix')
    
    """
    Logic: 
        
        If by = 'tag'
        For each tag, get a sparse mask array indicating the manuscripts belonging to the tag, 
        then use the mask to find the manuscript_index_map. Take an intersection of weights with 
        the mask - see if this should be modularized away

        If by = 'role'
        select only the column defined by the role, i.e., subsample capital based on each role and
        compute the allocation weights - see if this should be modularized away
    """
    
    if by == 'tag':
        rolewise_cap_sub = get_per_manuscript_cap(capital, contributor_index_map_subset)
        return [], [mix_by_tag(rolewise_cap_sub, manuscript_memberships)]
    
    # mask_authors = np.s_[:,size_zero_block : size_zero_block + num_contributors]
    # mask_reviewers = np.s_[:, size_zero_block + num_contributors: size_zero_block + 2*num_contributors]
    # mask_replicators = np.s_[:, size_zero_block + 2*num_contributors:] 
    
    if by == 'role':
        author_role_cap, author_mix_by_tag = mix_by_role(capital=capital, 
                                                         contributor_index_map_subset=contributor_index_map_subset,
                                                         mask_by_role=mask_authors,
                                                         manuscript_memberships=manuscript_memberships)
        reviewer_role_cap, reviewer_mix_by_tag = mix_by_role(capital=capital, 
                                                         contributor_index_map_subset=contributor_index_map_subset,
                                                         mask_by_role=mask_reviewers,
                                                         manuscript_memberships=manuscript_memberships)
        replicator_role_cap, replicator_mix_by_tag = mix_by_role(capital=capital, 
                                                         contributor_index_map_subset=contributor_index_map_subset,
                                                         mask_by_role=mask_replicators,
                                                         manuscript_memberships=manuscript_memberships)
        
        return [author_role_cap, reviewer_role_cap, replicator_role_cap], [author_mix_by_tag, reviewer_mix_by_tag, replicator_mix_by_tag]

def role_based_proportional_loss(
                            capital: sparse.spmatrix,
                            retractions_capital: sparse.spmatrix,
                            mask_by_role: slice,
                            contributor_index_map_subset: Dict[str, int],
                            ) -> float:
    """
    Compute the proportion of capital loss for a specific role in the portfolio.
    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
        Entries represent the amount of capital accrued from a manuscript by a contributor.
    retraction_capital : scipy.sparse.spmatrix
        Sparse matrix of the same shape as `capital`, representing capital associated with retracted manuscripts.
    mask_by_role : scipy.sparse.spmatrix
        Sparse binary matrix indicating which entries in `capital` correspond to the specified role.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in `capital`.
    Returns
    -------
    float
        The proportion of capital loss for the specified role in the portfolio.
        Returns 0.0 if the total capital is zero.
    Raises
    ------
    TypeError
        If `capital` or `mask_by_role` is not a scipy sparse matrix.
    Notes
    -----
    - The function computes the total capital for the specified role and compares it to the overall
      total capital to determine the proportion of loss.
    """
    # role_masked_capital = mask_by_role.multiply(capital)
    role_masked_capital = capital.tocsc()[mask_by_role] 
    # role_masked_retraction_capital = mask_by_role.multiply(retractions_capital)
    role_masked_retraction_capital = retractions_capital.tocsc()[mask_by_role]
    
    rolewise_cap_sub = get_per_manuscript_cap(role_masked_capital, contributor_index_map_subset).sum()
    rolewise_retraction_cap_sub = get_per_manuscript_cap(role_masked_retraction_capital, contributor_index_map_subset).sum()
    
    rolewise_proportional_loss = rolewise_retraction_cap_sub / rolewise_cap_sub if rolewise_cap_sub > 0 else 1.0 if rolewise_retraction_cap_sub > 0 else 0.0

    return rolewise_proportional_loss

def get_proportional_split(
    capital: sparse.spmatrix,
    mask_by_role: slice,
    contributor_index_map_subset: Dict[str, int],
) -> float:
    """
    Compute the Proportional Split metric.

    This metric measures the proportion of academic capital a contributor (or group)
    derives from a specific role (e.g., peer review or replication) relative to their
    total academic capital.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column represents a contributor.
    mask_by_role : slice or sparse matrix mask
        A mask or slice that isolates the specific role's capital within the capital matrix.
        Should be compatible with `capital.tocsc()[mask_by_role]`.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to the corresponding column index in `capital`.

    Returns
    -------
    float
        The proportional split value (0.0 to 1.0). Returns 0.0 if total capital is zero.

    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    total_capital = academic_capital(capital, contributor_index_map_subset)

    if total_capital == 0.0:
        return 0.0

    role_masked_capital = capital.tocsc()[mask_by_role]
    task_capital = get_per_manuscript_cap(role_masked_capital, contributor_index_map_subset).sum()

    return float(task_capital / total_capital)

def get_diversification_ratio(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
) -> float:
    """
    Compute the Diversification Ratio of a portfolio. The Diversification Ratio is defined as 
    the weighted average of the volatilities of the individual assets divided by the volatility 
    of the entire portfolio.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing
        portfolio capital states over time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to their corresponding column
        indices in the capital matrices.

    Returns
    -------
    float
        The Diversification Ratio. Returns 0.0 if the portfolio volatility is zero

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.

    ValueError
        If `capital_history` contains fewer than two entries.
    
    Returns 1.0 if the portfolio consists of a single asset or perfectly correlated assets.

    """

    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute diversification ratio')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    weights = allocation_weights(capital_history[-1], contributor_index_map_subset)

    if not weights:
        return 0.0

    col_indices = get_col_indices(capital_history[-1], contributor_index_map_subset)

    #A matrix with size (T, M) where each index represents the volatility of a manuscript within a portfolio at time t
    caps = np.array([
        cap.tocsr()[:, col_indices].sum(axis=1).A1
        for cap in capital_history
    ])

    #Here we're getting the returns from each interval by subtracting two matrices with differing times
    c0 = caps[:-1]
    c1 = caps[1:]
    
    #Here I'm making sure c0 is not 0
    with np.errstate(invalid='ignore', divide='ignore'):
        ms_returns = np.where(np.abs(c0) > 1e-10, (c1 - c0) / c0, 0.0)

    # Portfolio volatility from summed caps
    portfolio_caps = caps.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        portfolio_returns = np.where(np.abs(portfolio_caps[:-1]) > 1e-10,
                                     np.diff(portfolio_caps) / portfolio_caps[:-1], 0.0)
    portfolio_volatility = float(np.std(portfolio_returns, ddof=0))

    if portfolio_volatility == 0:
        return 1.0

    ms_vol = np.std(ms_returns, axis=0, ddof=0)

    manuscript_indices = np.array(list(weights.keys()))
    weight_vals = np.array(list(weights.values()))
    weighted_sigma_sum = float(weight_vals @ ms_vol[manuscript_indices])

    dr = weighted_sigma_sum / portfolio_volatility

    return dr

def get_proportional_return(
    capital_start: sparse.spmatrix,
    capital_end: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    """
    Compute the returns over a given time interval, defined by the proportional increase in academic capital over an interval

    Parameters
    ----------
    capital_start : scipy.sparse.spmatrix
        Sparse matrix representing the academic capital at the start of the interval.
    capital_end : scipy.sparse.spmatrix
        Sparse matrix representing the academic capital at the end of the interval.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index.
        Only columns referenced by this mapping are included in the calculation.
    
    Returns
    -------
    float
        The proportional return during the interval

    Raises
    ------
    TypeError
        If either `capital_start` or `capital_end` is not a scipy sparse matrix.
    ValueError
        If the starting capital is zero, which would make return calculation undefined.
    
    Notes
    -----
    - This function relies on two distinct sparse capital matrices
    """

    if not sparse.issparse(capital_start) or not sparse.issparse(capital_end):
        raise TypeError('Capital matrices must be scipy sparse matrices')

    cap_t0 = academic_capital(capital_start, contributor_index_map_subset)
    cap_t1 = academic_capital(capital_end, contributor_index_map_subset)

    if cap_t0 == 0.0:
        raise TypeError('Starting capital cannot be zero for return calculation')
    
    return (cap_t1 - cap_t0) / cap_t0

def get_returns(
    capital_start: sparse.spmatrix,
    capital_end: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int],
    time_interval: float = 1,
) -> float:
    """
    Compute the returns, defined as the change in returns over the change in time.

    Parameters
    ----------
    capital_start : scipy.sparse.spmatrix
        Sparse matrix representing the academic capital at the start of the interval.
    capital_end : scipy.sparse.spmatrix
        Sparse matrix representing the academic capital at the end of the interval.
    time_interval : float
        The time interval over which to compute returns. Must be positive.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index.

    Returns
    -------
    float
        The return per unit time during the interval.
    
    Raises
    ------
    TypeError
        If either `capital_start` or `capital_end` is not a scipy sparse matrix.
    ValueError
        If `time_interval` is not positive, which would make return calculation undefined.
    """
    if time_interval <= 0.0:
        raise ValueError('Time interval must be positive for return calculation')
    
    cap_t0 = academic_capital(capital_start, contributor_index_map_subset)
    cap_t1 = academic_capital(capital_end, contributor_index_map_subset)

    return (cap_t1 - cap_t0) / time_interval

def get_expected_proportional_returns(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int]
)-> float:
    """
    Compute the expected proportional returns based on a historical sequence of sparse capital matrices

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing the portfolio states over time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index.

    Returns
    -------
    float
        The expected return, calculated as the arithmetic mean of the historical returns

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.
    ValueError
        If `capital_history` contains fewer than 2 entries, which is insufficient to compute returnss
    """
    if len(capital_history) < 2:
        raise ValueError("capital_history must have at least 2 entries")

    
    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    returns = []
    for i in range(len(capital_history) - 1):
        r = get_proportional_return(capital_history[i], capital_history[i+1], contributor_index_map_subset)
        returns.append(r)

    return float(np.mean(returns))

def get_expected_returns(
    capital_history: List[sparse.spmatrix],
    time_history: List[float],
    contributor_index_map_subset: Dict[str, int]
) -> float:
    """
    Compute the expected returns based on a historical sequence of sparse capital matrices and corresponding time intervals

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing the portfolio states over time.
    time_history : List[float]
        A list of positive floats representing the time points at which the capital states are recorded. Must have length equal to `capital_history`.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index.

    Returns
    -------
    float
        The expected return per unit time, calculated as the arithmetic mean of the historical returns per unit time.

    Raises
    ------
    TypeError
        If any element in capital_history is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than 2 entries, or if any time interval is not positive.
    """
    
    if len(capital_history) < 2:
        raise ValueError("capital_history must have at least 2 entries")
    
    if len(time_history) != len(capital_history):
        raise ValueError("time_history must have the same length as capital_history")
    
    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    caps = np.array([academic_capital(matrixattime, contributor_index_map_subset) for matrixattime in capital_history])
    returns = np.diff(caps) / np.diff(time_history)

    return np.mean(returns)

def get_volatility(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
) -> tuple[float, float]:
    """
    Compute the volatility of portfolio returns, defined as the population 
    standard deviation of returns, from a historical sequence of sparse 
    academic-capital matrices.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing
        portfolio capital states over time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to their corresponding column
        indices in the capital matrices.

    Returns
    -------
    tuple[float, float]
        A tuple of (volatility, expected_returns) where volatility is the
        population standard deviation of portfolio returns.

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.
    ValueError
        If capital_history contains fewer than 2 entries
    """
    if len(capital_history) < 2:
        raise ValueError('At least two capital states are required to compute volatility')
    
    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')
    
    caps = np.array([academic_capital(m, contributor_index_map_subset) for m in capital_history])
    returns = np.diff(caps) / caps[:-1]

    expected_returns = float(returns.mean())

    return np.sqrt(np.sum((returns - expected_returns) ** 2) / (len(capital_history) - 1)), expected_returns

def get_sharpe_ratio(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
) -> float:
    """
    Compute the Sharpe ratio of a portfolio from a historical sequence
    of sparse academic-capital matrices.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing
        portfolio capital states over time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to their corresponding column
        indices in the capital matrices.

    Returns
    -------
    float
        The Sharpe ratio of the portfolio. Returns 0.0 if volatility is zero
        or if `capital_history` contains fewer than two entries.

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.
    """
    if len(capital_history) < 2:
        raise ValueError('At least two capital states are required to compute Sharpe ratio')
    
    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    volatility, expected_returns = get_volatility(capital_history, contributor_index_map_subset)

    if abs(volatility) < 1e-10:
        return 0.0

    return expected_returns/volatility

def get_arc(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
) -> float:
    """
    Compute the Academic Returns-to-Capital ratio (ARC) for the most recent return period
    divided by the current academic capital of the portfolio.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        Chronological list of sparse academic-capital matrices.
        The final two entries are used to compute the most recent return.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to column indices.

    Returns
    -------
    float
        Phase-sensitive ARC value. Returns 0.0 if insufficient history
        or if current capital is zero.
    """
    if len(capital_history) < 2:
        raise ValueError('At least two capital states are required to compute ARC')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError("All elements in capital_history must be scipy sparse matrices")

    recent_return = get_proportional_return(
        capital_history[-2],
        capital_history[-1],
        contributor_index_map_subset,
    )

    current_capital = academic_capital(
        capital_history[-1],
        contributor_index_map_subset,
    )

    if current_capital == 0.0:
        return 0.0

    return recent_return / current_capital


def get_risk_asymmetry(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
    expected_return: Optional[float] = None,
    volatility: Optional[float] = None,
) -> float:
    """
    Compute the risk asymmetry of a portfolio from a historical sequence
    of sparse academic-capital matrices.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing
        portfolio capital states over time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to their corresponding column
        indices in the capital matrices.

    Returns
    -------
    float
        The risk asymmetry of the portfolio.

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.

        Returns 0.0 if `capital_history` contains fewer than two entries.
    """
    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute risk asymmetry')
    
    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')
    
    if volatility is None:
        volatility = get_volatility(capital_history, contributor_index_map_subset, expected_return)
    
    if expected_return is None:
        expected_return = get_expected_returns(capital_history, contributor_index_map_subset)
    
    sce = 0.0
    for i in range(len(capital_history)-1):
        r = get_returns(capital_history[i], capital_history[i+1], contributor_index_map_subset)
        sce += (r - expected_return)**3
    
    return sce / ((len(capital_history)-1) * (volatility**3)) if volatility != 0.0 else 0.0


def get_funding_efficiency(
    capital: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int],
    funding: float,
) -> float:
    """
    Compute the funding efficiency of a portfolio.

    Funding efficiency measures the academic capital generated per unit of
    external funding, analogous to return on investment (ROI) for research grants.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Sparse matrix where each row represents a manuscript and each column
        represents a contributor.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index
        in `capital`.
    funding : float
        Total funding associated with the portfolio. Must be non-negative.

    Returns
    -------
    float
        The funding efficiency ratio (academic capital / funding).
        Returns 0.0 if funding is zero.

    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    ValueError
        If `funding` is negative.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')
    if funding < 0:
        raise ValueError('Funding must be non-negative')
    if funding == 0.0:
        return float('nan')

    total_capital = academic_capital(capital, contributor_index_map_subset)
    return total_capital / funding

def get_time_efficiency(
    capital_history: List[sparse.spmatrix],
    contributor_index_map_subset: Dict[str, int],
    time_period: Optional[float] = None,
) -> float:
    """
    Compute the time efficiency of a portfolio.

    Time efficiency measures the average rate of academic capital accumulation
    per time period, computed as the total change in academic capital divided
    by the number of elapsed time periods.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        A chronological list of at least two sparse matrices representing
        portfolio capital states over time. Consecutive entries are assumed
        to be uniformly spaced in time.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifiers to their corresponding column
        indices in the capital matrices.

    Returns
    -------
    float
        The average academic capital accumulated per time period.
        Returns 0.0 if `capital_history` contains fewer than two entries.

    Raises
    ------
    TypeError
        If any element in `capital_history` is not a scipy sparse matrix.
    """
    if len(capital_history) < 2:
        raise ValueError('capital_history must contain at least two entries to compute time efficiency')

    for cap in capital_history:
        if not sparse.issparse(cap):
            raise TypeError('All elements in capital_history must be scipy sparse matrices')

    cap_start = academic_capital(capital_history[0], contributor_index_map_subset)
    cap_end = academic_capital(capital_history[-1], contributor_index_map_subset)

    if time_period is not None:
        return (cap_end - cap_start) / time_period
    
    return (cap_end - cap_start) / float(len(capital_history) - 1)