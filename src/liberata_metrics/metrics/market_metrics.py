from typing import Dict, Iterable, List, Union, Tuple, Any
import numpy as np
from scipy import sparse
import warnings

from liberata_metrics.metrics import get_per_manuscript_cap, mix_by_tag
from liberata_metrics.utils import sparse_divide



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

    
    
