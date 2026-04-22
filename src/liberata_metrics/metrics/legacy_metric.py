"""
This code implements legacy metrics such as the h-index, i-10 index, and g-index.
"""
from typing import Dict

import numpy as np
from scipy import sparse


def get_citation_counts(
    references: sparse.spmatrix,
    manuscript_rows: list[int],
) -> Dict[int, int]:
    """
    Returns the number of incoming citations for each manuscript in the subset.

    references[i, j] encodes a citation from paper j (citing) to paper i (cited),
    so the number of papers citing manuscript i is the number of non-zero entries
    in row i.

    Args:
        references: Sparse citation matrix of shape (M, M).
        manuscript_row: List of row indices.

    Returns:
        Dict mapping each manuscript index to its incoming citation count (int).
    """

    # Slice to selected manuscript rows; binarize to get integer citation counts
    sub = references[manuscript_rows].astype(bool)

    #Basically sum along the axes horizontally to get the per manuscript citations
    counts = np.asarray(sub.sum(axis=1)).ravel()

    return dict(zip(manuscript_rows, counts.tolist()))


def get_h_index(
    capital: sparse.spmatrix,
    references: sparse.spmatrix,
    contributor_col: int,
) -> int:
    """
    Computes the h-index for a single author.

    h is the largest integer such that h of the author's papers each have
    at least h incoming citations.

    Args:
        capital: Sparse matrix of shape (M, M + 3*C). Used to find which
            manuscripts this author contributed to (author block: cols M to M+C).
        references: Sparse citation matrix of shape (M, M).
        contributor_col: Column index of the author in the contributor space
            (0-indexed, before the M offset). Pass contributor_index_map[id]
            at the call site.

    Returns:
        The h-index as an integer.
    """
    citation_counts = get_author_citations(capital, references, contributor_col)
    sorted_counts = sorted(citation_counts.values(), reverse=True)

    # Find the first paper in the sorted list with fewer citations than the count of papers that precede it
    h = 0
    for rank, count in enumerate(sorted_counts, start=1):
        if count >= rank:
            h = rank
        else:
            break

    return h

def get_author_citations(
    capital: sparse.spmatrix,
    references: sparse.spmatrix,
    contributor_col: int,
) -> Dict[int, int]:
    """
    For a contributor, returns authored manuscripts and corresponding citations

    Args:
        capital: Sparse matrix of shape (M, M + 3*C). Used to find which
            manuscripts this author contributed to.
        references: Sparse citation matrix of shape (M, M).
        contributor_col: Column index of the author in the contributor space
            (0-indexed, before the M offset). Pass contributor_index_map[id]
            at the call site.

    Returns:
        Dict mapping each manuscript index to its citation count.
    """

    #Find the author's index in the capital matrix
    M = capital.shape[0]
    author_col = M + contributor_col

    #Find the indices correspoinding to the author
    manuscript_indices = capital.getcol(author_col).nonzero()[0].tolist()

    #By definition, if someone has no papers, the return value would be 0, 
    #since this person could be listed as a peer reviewer and exist in the capital matrix
    if not manuscript_indices:
        return {}

    #Get the citations for the manuscripts
    citations = get_citation_counts(references, manuscript_indices)

    return citations



def get_i10_index(
    capital: sparse.spmatrix,
    references: sparse.spmatrix,
    contributor_col: int,
) -> int:
    """
    Computes the i-10 index

    Args:
        capital: Sparse matrix of shape (M, M + 3*C). Used to find which
            manuscripts this author contributed to.
            The author block are the cols M to M+C because the capital matrix
            columns are blocked by author, peer reviewer, and replicator, with each
            contributor located in indentical indices within each block.

        references: Sparse citation matrix of shape (M, M).
        contributor_col: Column index of the author in the contributor space
            (0-indexed, before the M offset). Pass contributor_index_map[id]
            at the call site.

    Returns:
        The i-10 index as an integer
    """

    citations = get_author_citations(capital, references, contributor_col)

    i10 = 0
    for cit in citations.values():
        if cit>=10:
            i10+=1

    return i10



def get_g_index(
    capital: sparse.spmatrix,
    references: sparse.spmatrix,
    contributor_col: int,
) -> int:
    """
    Computes the g-index

    Args:
        capital: Sparse matrix of shape (M, M + 3*C). Used to find which
            manuscripts this author contributed to (author block: cols M to M+C).
        references: Sparse citation matrix of shape (M, M).
        contributor_col: Column index of the author in the contributor space
            (0-indexed, before the M offset). Pass contributor_index_map[id]
            at the call site.

    Returns:
        The g-index as an integer
    """
    citations = get_author_citations(capital, references, contributor_col)

    sorted_citations = sorted(citations.values(), reverse=True)

    # Find the first paper in the sorted list where the sum of the entries squared is smaller than the rank squared
    g = 0
    sum_sq = 0
    for rank, count in enumerate(sorted_citations, start=1):
        sum_sq+=count
        if sum_sq >= rank**2:
            g = rank
        else:
            break

    return g
