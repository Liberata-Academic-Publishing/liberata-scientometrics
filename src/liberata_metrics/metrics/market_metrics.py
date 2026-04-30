from typing import Dict, Union, Tuple, List
import numpy as np
from scipy import sparse
import warnings

from liberata_metrics.metrics import get_per_manuscript_cap, mix_by_tag
from liberata_metrics.metrics.distribution_metrics import hhi_discrepancy, share_splits_inequality
from liberata_metrics.metrics.portfolio_metrics import allocation_weights, get_volatility, get_expected_returns
from liberata_metrics.utils import sparse_divide

def _time_series_csr(time_series: List[sparse.spmatrix]) -> List[sparse.spmatrix]:
    return [snap.tocsr() if not sparse.isspmatrix_csr(snap) else snap for snap in time_series]

def _manuscript_capital_history(capital_history_csr: List[sparse.spmatrix], manuscript_index: int) -> sparse.spmatrix:
    return [snapshot[manuscript_index] for snapshot in capital_history_csr]

def _domain_capital_history(capital_history_csr: List[sparse.spmatrix], manuscript_tags: np.ndarray, memberships_csr: sparse.spmatrix) -> List[sparse.spmatrix]:
    return [
        sparse.csr_matrix(np.asarray(mix_by_tag(snapshot, memberships_csr)[manuscript_tags == 1].sum(axis=0)))
        for snapshot in capital_history_csr
    ]

def compute_fair_marketprice(
        capital: sparse.spmatrix,
        mask_reviewers: slice,
        mask_replicators: slice,
        manuscript_memberships: sparse.spmatrix,
        contributor_index_map_subset: Dict[str, int],
        num_contributors: int,
        # size_zero_block: int = 0,
) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    

    # Add docstring
    """
    Compute tag-level fair market prices (FMP) for reviewers and replicators based on
    the capital allocated to these roles across all manuscripts. The function performs the following steps:
    1. Masks the global capital allocation by reviewer and replicator roles using the provided
        mask matrices (mask_reviewers and mask_replicators).
    2. Computes the per-manuscript capital allocated to reviewers and replicators using
        get_per_manuscript_cap, restricted to the specified subset of contributors.
    3. Aggregates the per-manuscript capital up to tags using the manuscript_memberships matrix.   
    4. Divides the aggregated capital per tag by the number of manuscripts associated with each tag
        to yield the fair market price (FMP) for reviewers and replicators per tag.
    Parameters 
    ----------
    capital : sparse.spmatrix
        Contributor-level capital matrix. Shape: (n_manuscripts, n_contributors).   
    mask_reviewers : sparse.spmatrix
        Manuscript-by-contributor mask indicating reviewer involvement. Shape:
        (n_manuscripts, n_contributors). Used to select reviewers' share of capital per manuscript. 
    mask_replicators : sparse.spmatrix
        Manuscript-by-contributor mask indicating replicator involvement. Shape:
        (n_manuscripts, n_contributors). Used to select replicators' share of capital per manuscript.
    manuscript_memberships : sparse.spmatrix
        Tag-by-manuscript membership matrix (shape: n_tags x n_manuscripts). Used to aggregate
        per-manuscript capital up to tags. Membership entries may be binary or weighted.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier (string) to column index in the mask matrices
        for the subset of contributors of interest. The function uses the mapped indices to
        select columns from mask_reviewers and mask_replicators.
    Returns
    -------
    Tuple[sparse.spmatrix, sparse.spmatrix]
        A tuple containing two sparse arrays (each of length n_tags):
        - reviewer_fmp: tagwise fair market price for reviewers.
        - replicator_fmp: tagwise fair market price for replicators.
    Raises  
    ------
    ValueError
        If input matrix/vector shapes are incompatible (e.g., contributor counts, manuscript
        counts, or tag counts do not align) the function may raise errors propagated from
        underlying sparse operations or from get_per_manuscript_cap.
    KeyError    
        If contributor_index_map_subset contains indices that do not correspond to columns
        in the provided mask matrices, column selection may fail.
    Notes
    -----
    - The function attempts a fast column-slicing of mask_reviewers and mask_replicators
        for the contributor subset and falls back to an explicit hstack/getcol approach if slicing is not supported.
    - Elementwise multiplications and sums assume broadcasting consistent with sparse matrix
        shapes: reviewer/replicator masks are treated as per-manuscript indicators and are multiplied with
        per-manuscript capital vectors produced by get_per_manuscript_cap.
    - manuscript_memberships is expected to map manuscripts to tags (rows correspond to tags).
    - The returned fair market price values are not normalized by the number of reviewers
        or replicators unless such normalization is performed earlier (e.g., in get_per_manuscript_cap).
    - All inputs are expected to be sparse-compatible to avoid dense expansion; be mindful of
        memory usage when converting between sparse and dense formats.
    Examples    
    --------
    Shape conventions (illustrative):
    - mask_reviewers / mask_replicators: (n_manuscripts, n_contributors)
    - capital: (n_manuscripts, n_contributors)
    - contributor_index_map_subset: subset of contributor -> column index mappings
    - get_per_manuscript_cap(masked_capital, subset_map) -> returns array of length n_manuscripts
    - manuscript_memberships: (n_tags, n_manuscripts)
    """
    
    # Check contributor subset is actually the whole set
    # This is needed since we want to compute the averages over whole domains
    # num_manuscripts = capital.shape[0]
    # num_contributors = capital.shape[1] - num_manuscripts
    if len(contributor_index_map_subset.keys()) < num_contributors:
        raise ValueError("The contributor index map is a subset. Was expecting the whole set")
    
    # mask_authors = np.s_[:,size_zero_block : size_zero_block + num_contributors]
    # mask_reviewers = np.s_[:, size_zero_block + num_contributors: size_zero_block + 2*num_contributors]
    # mask_replicators = np.s_[:, size_zero_block + 2*num_contributors:] 

    capital = capital.tocsc()
    reviewer_capital    = capital[mask_reviewers]
    replicator_capital  = capital[mask_replicators]

    reviewer_cap_sub    = get_per_manuscript_cap(reviewer_capital, contributor_index_map_subset)
    replicator_cap_sub  = get_per_manuscript_cap(replicator_capital, contributor_index_map_subset)

    review_cost_per_tag = mix_by_tag(reviewer_cap_sub, manuscript_memberships).sum(axis=1)
    repplication_cost_per_tag = mix_by_tag(replicator_cap_sub, manuscript_memberships).sum(axis=1)

    num_manuscripts_per_tag = manuscript_memberships.sum(axis=1)

    # sparse divide to get average cost of review per tag
    review_fmp = np.divide(review_cost_per_tag, num_manuscripts_per_tag)
    replication_fmp = np.divide(repplication_cost_per_tag, num_manuscripts_per_tag)

    return review_fmp, replication_fmp

def compute_risk_premiums(
        
        capital: sparse.spmatrix,
        mask_authors: slice,
        mask_reviewers: slice,
        mask_replicators: slice,
        contributor_index_map_subset: Dict[str, int],
        reviewer_fmp: sparse.sparray,
        replicator_fmp: sparse.sparray,
        manuscript_memberships: sparse.spmatrix,
        # num_contributors: int,
        # size_zero_block: int = 0,
) -> Tuple[sparse.sparray, sparse.sparray]:

    """
    Compute tag-level risk premiums for reviewers and replicators corresponding to manuscripts
    authored by a given subset of contributors.
    This function calculates, for each tag, the difference between the capital that reviewers
    (or replicators) have obtained from manuscripts authored by the specified contributor subset
    and the fair market price (FMP) for that tag. The computation proceeds in these steps:
    1. Select the subset of author columns from mask_authors and create a per-manuscript mask
        indicating which manuscripts were authored by selected contributors.
    2. Mask the global capital allocation by reviewer/replicator role (mask_reviewers
        / mask_replicators) and compute the per-manuscript capital given for QC services, i.e.,
        peer review and replication.
    3. Restrict reviewer/replicator per-manuscript capital to manuscripts authored by the
        selected contributors.
    4. Aggregate per-manuscript capital up to tags via manuscript_memberships (tags x manuscripts).
    5. Subtract the provided tag-level fair market price vectors (reviewer_fmp, replicator_fmp)
        to produce tag-level risk premiums.
    Parameters
    ----------
    capital : sparse.spmatrix
            Contributor-level capital vector or sparse column matrix. Expected length (or number
            of rows) equals the number of contributors (n_contributors). This represents the
            capital each contributor has available/allocated (units consistent with FMP).
    mask_authors : sparse.spmatrix
            Manuscript-by-contributor binary (or weighted) mask indicating authorship. Shape:
            (n_manuscripts, n_contributors). Nonzero entry indicates that the contributor
            authored (or contributed to) the manuscript.
    mask_reviewers : sparse.spmatrix
            Manuscript-by-contributor mask indicating reviewer involvement. Shape:
            (n_manuscripts, n_contributors). Used to select reviewers' share of capital per manuscript.
    mask_replicators : sparse.spmatrix
            Manuscript-by-contributor mask indicating replicator involvement. Shape:
            (n_manuscripts, n_contributors). Used to select replicators' share of capital per manuscript.
    contributor_index_map_subset : Dict[str, int]
            Mapping from contributor identifier (string) to column index in the mask matrices
            for the subset of contributors of interest. The function uses the mapped indices to
            select columns from mask_authors, mask_reviewers and mask_replicators.
    reviewer_fmp : sparse.sparray
            Tag-level fair market price vector for reviewers. Shape: (n_tags,). Units must match
            those of capital after aggregation.
    replicator_fmp : sparse.sparray
            Tag-level fair market price vector for replicators. Shape: (n_tags,). Units must match
            those of capital after aggregation.
    manuscript_memberships : sparse.spmatrix
            Tag-by-manuscript membership matrix (shape: n_tags x n_manuscripts). Used to aggregate
            per-manuscript capital up to tags. Membership entries may be binary or weighted.
    Returns
    -------
    Tuple[sparse.sparray, sparse.sparray]
            A tuple containing two sparse arrays (each of length n_tags):
            - reviewer_risk_premium: tagwise (aggregated) reviewer capital on manuscripts authored
            by the selected contributors minus reviewer_fmp.
            - replicator_risk_premium: tagwise (aggregated) replicator capital on manuscripts authored
            by the selected contributors minus replicator_fmp.
    Raises
    ------
    ValueError
            If input matrix/vector shapes are incompatible (e.g., contributor counts, manuscript
            counts, or tag counts do not align) the function may raise errors propagated from
            underlying sparse operations or from get_per_manuscript_cap.
    KeyError
            If contributor_index_map_subset contains indices that do not correspond to columns
            in the provided mask matrices, column selection may fail.
    Notes
    -----
    - The function attempts a fast column-slicing of mask_authors for the contributor subset
        and falls back to an explicit hstack/getcol approach if slicing is not supported.
    - Elementwise multiplications and sums assume broadcasting consistent with sparse matrix
        shapes: author masks are treated as per-manuscript indicators and are multiplied with
        per-manuscript capital vectors produced by get_per_manuscript_cap.
    - manuscript_memberships is expected to map manuscripts to tags (rows correspond to tags).
    - The returned risk premium values are not normalized by the number of authors, reviewers,
        or replicators unless such normalization is performed earlier (e.g., in get_per_manuscript_cap).
    - All inputs are expected to be sparse-compatible to avoid dense expansion; be mindful of
        memory usage when converting between sparse and dense formats.
    Examples
    --------
    Shape conventions (illustrative):
    - mask_authors: (n_manuscripts, n_contributors)
    - mask_reviewers / mask_replicators: (n_manuscripts, n_contributors)
    - capital: (n_contributors,) or (n_contributors, 1)
    - contributor_index_map_subset: subset of contributor -> column index mappings
    - get_per_manuscript_cap(masked_capital, subset_map) -> returns array of length n_manuscripts
    - manuscript_memberships: (n_tags, n_manuscripts)
    - reviewer_fmp / replicator_fmp: (n_tags,)
    """

    # mask_authors = np.s_[:,size_zero_block : size_zero_block + num_contributors]
    # mask_reviewers = np.s_[:, size_zero_block + num_contributors: size_zero_block + 2*num_contributors]
    # mask_replicators = np.s_[:, size_zero_block + 2*num_contributors:] 
    
    # Get a mask where the given contributor subset has authored manuscripts
    col_indices = list(contributor_index_map_subset.values())
    # try:
    #     author_mask_subset = sparse.csr_array(mask_authors[:, col_indices].sum(axis=1))
    # except Exception as e:
    #     author_mask_subset = sparse.csr_array(sparse.hstack([mask_authors.getcol(c) for c in col_indices], format="csr").sum(axis=1))
    capital = capital.tocsc()
    author_mask_subset = capital[mask_authors][:, col_indices]
    row_indices = np.unique(author_mask_subset.nonzero()[0])
    authored_mask = np.zeros(capital.shape[0], dtype=bool)
    authored_mask[row_indices] = True
    sp_authored_mask = authored_mask[:, np.newaxis]
    
    # reviewer_masked_capital = mask_reviewers.multiply(capital)
    reviewer_masked_capital = capital[mask_reviewers]
    reviewer_cap_sub = get_per_manuscript_cap(reviewer_masked_capital, contributor_index_map_subset)

    # replicator_masked_capital = mask_replicators.multiply(capital)
    replicator_masked_capital = capital[mask_replicators]
    replicator_cap_sub = get_per_manuscript_cap(replicator_masked_capital, contributor_index_map_subset)

    # import pdb; pdb.set_trace()

    reviewer_cap_on_authored = reviewer_cap_sub.multiply(sp_authored_mask)            # Capital of reviewers on the manuscripts authored by selected contributors 
    replicator_cap_on_authored = replicator_cap_sub.multiply(sp_authored_mask)        # Capital of replicators on the manuscripts authored by selected contributors

    # TODO: everything above this line should be put in a separate function so that it may be reused
    tagwise_reviewer_cap_on_authored = (manuscript_memberships @ reviewer_cap_on_authored).sum(axis=1)
    tagwise_replicator_cap_on_authored = (manuscript_memberships @ replicator_cap_on_authored).sum(axis=1)

    # risk premium is capital on authored manuscripts minus fair market price per tag
    reviewer_risk_premium = tagwise_reviewer_cap_on_authored - reviewer_fmp
    replicator_risk_premium = tagwise_replicator_cap_on_authored - replicator_fmp
    

    return reviewer_risk_premium, replicator_risk_premium

def compute_utility_function(
    capital_history: List[sparse.spmatrix],
    time_history: List[float],
    contributor_index_map_subset: Dict[str, int],
    risk_willingness: float
) -> np.ndarray:
    """
    Compute mean-variance utility for a contributor subset over time.

    This function derives expected returns from the provided capital history,
    estimates volatility over the same history, and combines the two using a
    standard quadratic utility formulation.

    The computation proceeds in these steps:
    1. Compute expected returns for the selected contributors using
    ``get_expected_returns``.
    2. Compute volatility for the same contributor subset using
    ``get_volatility``.
    3. Combine expected return and volatility with the risk willingness
    coefficient to produce a utility vector.

    Parameters
    ----------
    capital_history : List[sparse.spmatrix]
        Sequence of sparse capital matrices, ordered over time. Each matrix is
        expected to contain manuscript-by-contributor capital allocations for a
        single time point.
    time_history : List[float]
        Time values corresponding to ``capital_history``. These are passed to
        ``get_expected_returns`` when computing the expected return series.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to the corresponding column index in
        the capital matrices. Only the contributors in this subset are included
        in the utility calculation.
    risk_willingness : float
        Risk preference coefficient used in the mean-variance utility formula.
        Larger values penalize volatility more strongly.

    Returns
    -------
    np.ndarray
        A 1D NumPy array containing utility values for the selected
        contributors. The result is computed as:

        ``utility = expected_returns - 0.5 * risk_willingness * volatility``

    Raises
    ------
    ValueError
        If the input histories are incompatible in length or if the expected
        return / volatility helpers reject the provided data.
    TypeError
        If any history element is not sparse-compatible in a way that the
        downstream helper functions can process.

    Notes
    -----
    - The function assumes the helper routines return aligned vectors for the
    same contributor subset.
    - No normalization is performed here beyond the risk-adjusted utility
    expression.
    - The returned values are higher when expected returns increase and lower
    when volatility increases.

    Examples
    --------
    >>> utility = compute_utility_function(capital_history, time_history, contributor_index_map_subset, 1.5)
    >>> utility.shape
    (n_contributors,)
    """
    expected_returns = get_expected_returns(capital_history, time_history, contributor_index_map_subset)
    volatility = get_volatility(capital_history, contributor_index_map_subset, expected_returns)

    return expected_returns - 0.5 * risk_willingness * volatility

def compute_relative_performance(
    shares: sparse.spmatrix,
    capital_history: List[sparse.spmatrix],
    time_history: List[float],
    manuscript_memberships: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    """
    Compute portfolio relative performance against manuscript-domain benchmarks.

    This metric evaluates the weighted quality/performance of manuscripts held by
    a contributor-defined portfolio. For each manuscript in the portfolio, the
    manuscript expected return is divided by the expected return of its domain
    (as defined by manuscript tag memberships), then aggregated using portfolio
    manuscript weights.

    Conceptually, the function computes:

        rho_Pi = (1 / sum_{m in Pi} s_m) * sum_{m in Pi} s_m * (mu_m / mu_d(m))

    where:
    - ``Pi`` is the set of manuscripts held by the portfolio,
    - ``s_m`` is the portfolio weight/share in manuscript ``m``,
    - ``mu_m`` is the manuscript expected return,
    - ``mu_d(m)`` is the expected return of manuscript ``m``'s domain/tag.

    Processing overview
    -------------------
    1. Infer portfolio manuscript weights with ``allocation_weights`` using the
    provided ``shares`` matrix and contributor subset.
    2. For each manuscript in the inferred portfolio:
    - Estimate manuscript expected return from manuscript-level slices of
        ``capital_history`` via ``get_expected_returns``.
    - Identify the manuscript's tag/domain membership from
        ``manuscript_memberships``.
    - Build domain-level capital history using ``mix_by_tag`` and estimate
        domain expected return via ``get_expected_returns``.
    - Accumulate the weighted ratio ``s_m * (mu_m / mu_d(m))``.
    3. Normalize by total portfolio shares/weights.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Sparse manuscript-by-contributor matrix of shares/weights used to infer
        the portfolio composition for ``contributor_index_map_subset``. Must be
        compatible with ``allocation_weights`` indexing conventions.
    capital_history : List[scipy.sparse.spmatrix]
        Chronological sequence of sparse capital matrices used to estimate
        returns over time. Each entry should be aligned to the same manuscript
        and contributor indexing scheme.
    time_history : List[float]
        Time points corresponding to ``capital_history``. Must be the same
        length as ``capital_history`` and strictly increasing for return-per-time
        calculations in ``get_expected_returns``.
    manuscript_memberships : scipy.sparse.spmatrix
        Sparse tag-by-manuscript membership matrix of shape
        ``(n_tags, n_manuscripts)`` where entry ``[t, m]`` indicates membership
        of manuscript ``m`` in tag/domain ``t``.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to contributor column index for the
        portfolio entity being evaluated.

    Returns
    -------
    float
        Relative performance score for the selected portfolio. Values above 1.0
        indicate average outperformance versus domain baselines, while values
        below 1.0 indicate underperformance.

    Raises
    ------
    TypeError
        If ``shares`` is not a scipy sparse matrix.
    ValueError
        If ``contributor_index_map_subset`` is empty, or if called helper
        functions reject invalid input shapes/history lengths.

    Notes
    -----
    - Portfolio manuscript membership is inferred from non-zero allocation
    weights returned by ``allocation_weights``.
    - Domain return extraction depends on ``manuscript_memberships`` and assumes
    consistent manuscript row indexing across all inputs.
    - The implementation currently propagates edge-case behavior from helper
    functions (for example, domain-return zero handling) rather than
    explicitly applying a zero-safe division at this level.

    Examples
    --------
    >>> rp = compute_relative_performance(
    ...     shares=shares,
    ...     capital_history=capital_history,
    ...     time_history=time_history,
    ...     manuscript_memberships=manuscript_memberships,
    ...     contributor_index_map_subset=subset_map,
    ... )
    >>> isinstance(rp, float)
    True
    """

    if not sparse.issparse(shares):
        raise TypeError("shares must be a scipy sparse matrix")
    if not contributor_index_map_subset:
        raise ValueError("contributor_index_map_subset cannot be empty")

    weights = allocation_weights(shares, contributor_index_map_subset)
    capital_history_csr = _time_series_csr(capital_history)
    memberships_csr = manuscript_memberships.tocsr() if not sparse.isspmatrix_csr(manuscript_memberships) else manuscript_memberships

    total_shares = sum(weights.values())
    numerator = 0.0

    for manuscript in weights.keys():
        manuscript_returns = get_expected_returns(
            _manuscript_capital_history(capital_history_csr, manuscript),
            time_history,
            contributor_index_map_subset,
        )
        manuscript_tags = memberships_csr[:, manuscript].toarray().ravel()
        domain_capital_history = [
            sparse.csr_matrix(np.asarray(mix_by_tag(snapshot, memberships_csr)[manuscript_tags == 1].sum(axis=0)))
            for snapshot in capital_history_csr
        ]
        domain_returns = get_expected_returns(domain_capital_history, time_history, contributor_index_map_subset)

        # Avoid division by zero: skip manuscripts with no domain returns
        if abs(domain_returns) > 1e-10:
            numerator += weights[manuscript] * manuscript_returns / domain_returns
        elif abs(manuscript_returns) < 1e-10:
            # Both manuscript and domain had near-zero returns - treat as neutral (1.0) performance
            numerator += weights[manuscript] * 1.0
    return numerator / total_shares

def compute_sensitivity(
    manuscript_expected_returns: float,
    domain_expected_returns: float,
) -> float:
    """
    Compute sensitivity (beta) of manuscript returns to domain returns.

    This function returns the ratio of covariance between manuscript expected
    returns and corresponding domain expected returns to the variance of domain
    expected returns:

        beta = Cov(mu_m, mu_d(m)) / Var(mu_d(m))

    Parameters
    ----------
    manuscript_expected_returns : float
        Manuscript expected return value(s). While typed as ``float`` in the
        current signature, this implementation delegates to NumPy covariance and
        variance routines and is typically meaningful when array-like values are
        provided.
    domain_expected_returns : float
        Domain expected return value(s) aligned with
        ``manuscript_expected_returns``.

    Returns
    -------
    float
        Sensitivity (beta) estimate. Returns 0.0 if domain variance is zero or
        very close to zero (within numerical epsilon).

    Notes
    -----
    - The implementation uses ``np.var(..., ddof=0)``, i.e., population variance.
    This is appropriate when working with return series.
    - Returns 0.0 when domain variance is near zero to avoid division by zero.

    Examples
    --------
    >>> beta = compute_sensitivity(np.array([3.0, 5.0, 7.0, 9.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    >>> isinstance(beta, float)
    True
    """
    domain_var = np.var(domain_expected_returns, ddof=0)
    cov = float(np.cov(manuscript_expected_returns, domain_expected_returns, ddof=0)[0, 1])
    
    # Avoid division by zero
    if np.abs(domain_var) < 1e-10 and np.abs(cov) < 1e-10:
        return 1.0  # If both covariance and variance are effectively zero, treat as perfect sensitivity
    elif np.abs(domain_var) < 1e-10:
        return 0.0  # If variance is zero but covariance is not, return zero

    return float(cov / domain_var)

def compute_risk_adjusted_excess_return(
    capital_history: List[sparse.spmatrix],
    time_history: List[float],
    manuscript_memberships: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int],
    manuscript_index: int,
) -> np.ndarray:
    """
    Compute manuscript-level risk-adjusted excess return (alpha).

    For the selected manuscript, this function estimates:

        alpha_m = mu_m - beta_m * mu_d(m)

    where manuscript and domain expected returns are derived from
    ``capital_history`` and ``time_history`` using ``get_expected_returns``, and
    ``beta_m`` is computed via ``compute_sensitivity``.

    Domain returns are constructed by:
    1. extracting the manuscript's tag/domain memberships from
    ``manuscript_memberships``;
    2. aggregating capital history by tag with ``mix_by_tag``; and
    3. selecting/summing the relevant domain rows for the manuscript.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        Chronological list of sparse capital matrices. Each matrix should use
        consistent manuscript and contributor indexing.
    time_history : List[float]
        Time points corresponding to ``capital_history`` used to compute
        expected returns per unit time.
    manuscript_memberships : scipy.sparse.spmatrix
        Sparse tag-by-manuscript matrix indicating manuscript-domain
        membership.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to contributor column index for the
        portfolio subset used in return calculations.
    manuscript_index : int
        Row index of the manuscript for which to compute risk-adjusted excess
        return.

    Returns
    -------
    np.ndarray
        Risk-adjusted excess return (alpha) for the specified manuscript.

    Raises
    ------
    IndexError
        If ``manuscript_index`` is out of bounds for any matrix in
        ``capital_history`` or for ``manuscript_memberships``.
    ValueError
        Propagated from helper functions when histories are invalid (for
        example, length mismatches or insufficient points).
    TypeError
        Propagated from helper functions when non-sparse inputs are supplied
        where sparse matrices are expected.

    Notes
    -----
    - This routine computes alpha for a single manuscript index.
    - If domain expected returns are constant/degenerate, sensitivity estimation
    may be unstable depending on NumPy behavior.

    Examples
    --------
    >>> alpha = compute_risk_adjusted_excess_return(
    ...     capital_history=capital_history,
    ...     time_history=time_history,
    ...     manuscript_memberships=manuscript_memberships,
    ...     contributor_index_map_subset=subset_map,
    ...     manuscript_index=10,
    ... )
    """
    capital_history_csr = _time_series_csr(capital_history)
    memberships_csr = manuscript_memberships.tocsr()

    manuscript_returns = get_expected_returns(
        _manuscript_capital_history(capital_history_csr, manuscript_index),
        time_history,
        contributor_index_map_subset,
    )

    manuscript_tags = memberships_csr[:, manuscript_index].toarray().ravel()
    domain_capital_history = _domain_capital_history(capital_history_csr, manuscript_tags, memberships_csr)
    domain_returns = get_expected_returns(domain_capital_history, time_history, contributor_index_map_subset)

    sensitivity = compute_sensitivity(manuscript_returns, domain_returns)
    return manuscript_returns - sensitivity * domain_returns

def compute_risk_adjusted_relative_performance(
    capital_history: List[sparse.spmatrix],
    time_history: List[float],
    manuscript_memberships: sparse.spmatrix,
    contributor_index_map_subset: Dict[str, int]
) -> float:
    """
    Compute risk-adjusted relative performance for an entire portfolio.

    This function extends manuscript-level risk-adjusted excess return to a
    portfolio-level quantity by evaluating every manuscript row in the first
    capital matrix. For each manuscript, it computes the risk-adjusted excess
    return ``alpha_m`` and then normalizes it by the manuscript's domain-level
    expected return ``mu_d(m)``. The final result is the sum of these
    manuscript-wise contributions.

    Conceptually, the metric follows:

        RARP = \sum_{m \in \Pi} \alpha_m / \mu_d(m)

    where:
    - ``Pi`` is the set of manuscripts in the portfolio,
    - ``alpha_m`` is the risk-adjusted excess return for manuscript ``m``,
    - ``mu_d(m)`` is the expected return of the manuscript's domain/tag.

    The manuscript-level ``alpha_m`` values are computed by calling
    ``compute_risk_adjusted_excess_return`` for each manuscript index. Domain returns
    are then reconstructed from ``capital_history`` using ``mix_by_tag`` and
    the manuscript's tag membership in ``manuscript_memberships``.

    Parameters
    ----------
    capital_history : List[scipy.sparse.spmatrix]
        Chronological sequence of sparse capital matrices. The first matrix is
        used to determine the manuscript rows iterated over, and all matrices
        should share the same manuscript indexing convention.
    time_history : List[float]
        Time points corresponding to ``capital_history``. These are passed to
        ``get_expected_returns`` when computing manuscript and domain expected
        returns.
    manuscript_memberships : scipy.sparse.spmatrix
        Sparse tag-by-manuscript membership matrix of shape
        ``(n_tags, n_manuscripts)``. Entry ``[t, m]`` indicates whether
        manuscript ``m`` belongs to tag/domain ``t``.
    contributor_index_map_subset : Dict[str, int]
        Mapping from contributor identifier to contributor column index for the
        subset of contributors used in the portfolio calculation.

    Returns
    -------
    float
        Risk-adjusted relative performance score for the portfolio.

    Raises
    ------
    ValueError
        If ``contributor_index_map_subset`` is empty.
    TypeError
        May be raised by helper functions if the provided histories are not
        sparse-compatible.
    IndexError
        May be raised if manuscript indices are out of bounds for the supplied
        matrices.

    Notes
    -----
    - The loop iterates over ``range(capital_history[0].shape[0])``.
    - The current implementation accumulates ``alpha / domain_returns`` for
    every manuscript and returns the total.
    - Division by zero is not handled explicitly here; any zero-domain-return
    behavior is inherited from the underlying arithmetic and helper calls.

    Examples
    --------
    >>> rarp = compute_risk_adjusted_relative_performance(
    ...     capital_history=capital_history,
    ...     time_history=time_history,
    ...     manuscript_memberships=manuscript_memberships,
    ...     contributor_index_map_subset=subset_map,
    ... )
    >>> isinstance(rarp, float)
    True
    """
    if not contributor_index_map_subset:
        raise ValueError("contributor_index_map_subset cannot be empty")

    capital_history_csr = _time_series_csr(capital_history)
    memberships_csr = manuscript_memberships.tocsr() if not sparse.isspmatrix_csr(manuscript_memberships) else manuscript_memberships
    total = 0.0

    for manuscript in range(capital_history_csr[0].shape[0]):
        alpha = compute_risk_adjusted_excess_return(capital_history_csr, time_history, memberships_csr, contributor_index_map_subset, manuscript)
        
        manuscript_tags = memberships_csr[:, manuscript].toarray().ravel()
        domain_capital_history = _domain_capital_history(capital_history_csr, manuscript_tags, memberships_csr)
        domain_returns = get_expected_returns(domain_capital_history, time_history, contributor_index_map_subset)
        

        # Avoid division by zero: skip manuscripts with no domain returns
        if abs(domain_returns) > 1e-10:
            total += alpha / domain_returns
    return total

# Removed as duplicates in distribution_metrics.py

# def compute_share_splits_inequality(
#     shares: sparse.spmatrix,
# ) -> float:
#     """
#     Compute the Herfindahl-Hirschman Index (HHI) of a portfolio of manuscripts.
#     Parameters
#     ----------
#     shares : scipy.sparse.spmatrix
#         Sparse matrix where each row represents a manuscript and each column represents an author.
#         Entries are the shares each author has for each given manuscript.
#     Returns
#     -------
#     float
#         The mean HHI value, a measure of the typical concentration in the portfolio. Returns 0.0 if the portfolio is empty.
#     ------
#     TypeError
#         If `shares` is not a scipy sparse matrix.
#     Notes
#     -----
#     - The share splits inequality is calculated as the mean HHI of all provided manuscripts.
#     - The HHI for a single manuscript is calculated as the sum of the squares of the share splits for that manuscript.
#     Examples
#     --------
#     >>> share_splits_inequality(shares)
#     0.375
#     """
#     return share_splits_inequality(shares)

# def compute_hhi_discrepancy(
#     shares: sparse.spmatrix,
#     mask_portfolio: slice,
# ) -> float:
#     """
#     Compute the discrepancy between the HHI of the field and that of the manuscript or portfolio.
#     Parameters
#     ----------
#     shares_matrix : scipy.sparse.spmatrix
#         Sparse matrix where each row represents a manuscript and each column represents an author.
#         Entries are the shares each author has for each given manuscript.
#         This matrix should have all manuscripts in the field/discipline.
#     indices : np.array
#         Array of indices of manuscripts in the shares_matrix that are to be considered in the HHI calculation.
#     Returns
#     -------
#     float
#         The HHID value, a measure of discrepancy between a portfolio and the industry. Higher values indicate an anomaly that 
#         should be further investigated.
#     Examples
#     --------
#     >>> hhi_discrepancy(shares_matrix, [1,2,5,6,7])
#     0.375
#     """
#     return hhi_discrepancy(shares, mask_portfolio)
