"""
Graph algorithms for liberata
"""
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans




def _bipartite_adjacency(shares: sparse.spmatrix) -> sparse.csr_matrix:

    M = shares.shape[0]
    N = shares.shape[1]  # M + 3C
    B = shares.tocsr()[:, M:].astype(float).tocoo()  # (M, 3C)
    rows = np.concatenate([B.row, B.col + M])
    cols = np.concatenate([B.col + M, B.row])
    data = np.concatenate([B.data, B.data])
    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N))


def _laplacian(shares: sparse.spmatrix, normalised: bool) -> sparse.csr_matrix:
    G = _bipartite_adjacency(shares)
    degrees = np.asarray(G.sum(axis=1)).ravel()
    if normalised:
        degrees[degrees == 0] = 1.0
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees), format='csr')
        N = G.shape[0]
        return (sparse.eye(N, format='csr') - d_inv_sqrt @ G @ d_inv_sqrt).tocsr()
    D = sparse.diags(degrees, format='csr')
    return (D - G).tocsr()


def get_shares_laplacian_spectrum(
    shares: sparse.spmatrix,
    k: int = 10,
    normalised: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the k smallest eigenvalues and corresponding eigenvectors of the
    shares graph Laplacian.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C)
    k : int
        Number of eigenvalues eigenvector pairs to compute
    normalised : bool
        If True, use the symmetric normalised Laplacian.
        If False (default), use the unnormalised Laplacian.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
        Sorted ascending. First value(s) are ~0.
    eigenvectors : np.ndarray, shape (M + 3C, k)
        Columns are the corresponding eigenvectors. Rows 0..M-1 correspond
        to manuscripts; rows M..M+3C-1 correspond to contributors.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    ValueError
        If k < 1 or k >= M + 3C.
    """

    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')
    
    N = shares.shape[1]
    if k < 1 or k >= N:
        raise ValueError(f'k must satisfy 1 <= k < N={N}, got k={k}')

    L = _laplacian(shares, normalised)
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def get_shares_fiedler_value(
    shares: sparse.spmatrix,
    normalised: bool = False,
) -> float:
    """
    Return the Fiedler value (second smallest eigenvalue) of the
    shares graph Laplacian. A nonzero Fiedler value means the shares graph is connected.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).
    normalised : bool
        If True, use the symmetric normalised Laplacian. If False (default),
        use the unnormalised Laplacian.

    Returns
    -------
    float
        The fiedler value of the shares graph Laplacian

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    ValueError
        If M + 3C < 2.
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')
    if shares.shape[1] < 2:
        raise ValueError('Need at least 2 nodes to compute the Fiedler value')

    eigenvalues, _ = get_shares_laplacian_spectrum(shares, k=2, normalised=normalised)
    return float(eigenvalues[1])


def get_shares_connected_components(
    shares: sparse.spmatrix,
    k: int = 10,
    normalised: bool = False,
) -> int:
    """
    Estimate the number of connected components in the shares graph by counting
    eigenvalues of the Laplacian that are approximately zero.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).
    k : int
        Number of smallest eigenvalues to inspect. Should be at least as large
        as the expected number of connected components.
    normalised : bool
        If True, use the symmetric normalised Laplacian. If False (default),
        use the unnormalised Laplacian.

    Returns
    -------
    int
        Number of eigenvalues <= 1e-8, i.e. the estimated number of connected
        components.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.

    Examples
    --------
    >>> n_components = get_shares_connected_components(shares, k=10)
    >>> n_components
    2
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')

    eigenvalues, _ = get_shares_laplacian_spectrum(shares, k=k, normalised=normalised)
    return int(np.sum(eigenvalues <= 1e-8))


def get_shares_clusters(
    shares: sparse.spmatrix,
    n_clusters: int,
    normalised: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Cluster nodes of the shares graph using spectral k-means on the k smallest
    eigenvectors of the Laplacian. The eigenvectors are row-normalised before k-means.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).
    n_clusters : int
        Number of clusters. Must satisfy 1 <= n_clusters < M + 3C.
    normalised : bool
        If True, use the normalised Laplacian for the spectral embedding.
        If False (default), use the unnormalised Laplacian.
    seed : int, optional
        Random seed for k-means initialisation.

    Returns
    -------
    np.ndarray of int, shape (M + 3C,)
        Cluster index for each node

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    ValueError
        If n_clusters < 1 or n_clusters >= M + 3C.
    """

    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')
    N = shares.shape[1]
    if n_clusters < 1 or n_clusters >= N:
        raise ValueError(f'n_clusters must satisfy 1 <= n_clusters < N={N}')

    _, eigenvectors = get_shares_laplacian_spectrum(
        shares, k=n_clusters, normalised=normalised
    )

    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = eigenvectors / norms

    return KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit_predict(embedding)



def get_references_gram_matrix(
    references: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Compute the Gram matrix of the references graph. For the unweighted references matrix, 
    entry (i,j) counts the number of references manuscripts i and j have in common.

    Parameters
    ----------
    references : scipy.sparse.spmatrix
        References matrix of shape (M, M).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (M, M)
        Gram matrix G^T G. Symmetric, with non-negative entries.

    Raises
    ------
    TypeError
        If `references` is not a scipy sparse matrix.
    """
    if not sparse.issparse(references):
        raise TypeError('References matrix must be a scipy sparse matrix')

    G = references.tocsr().astype(float)
    return (G.T @ G).tocsr()


def get_references_transpose_gram_matrix(
    references: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Compute the transpose Gram matrix of the references graph. For the unweighted references matrix, 
    entry (i,j) counts the number of manuscripts that cite both manuscript i and manuscript j

    Parameters
    ----------
    references : scipy.sparse.spmatrix
        References matrix of shape (M, M). Entry [i, j] is nonzero if
        manuscript j cites manuscript i.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (M, M)
        Transpose Gram matrix G G^T. Symmetric, with non-negative entries.

    Raises
    ------
    TypeError
        If `references` is not a scipy sparse matrix.
    """
    if not sparse.issparse(references):
        raise TypeError('References matrix must be a scipy sparse matrix')

    G = references.tocsr().astype(float)
    return (G @ G.T).tocsr()


def get_shares_two_step_graph(
    shares: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Compute the two-step shares graph.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (M + 3C, M + 3C)
        Two-step graph adjacency matrix G2S = GS^T @ GS.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')

    GS = _bipartite_adjacency(shares)
    return (GS.T @ GS).tocsr()


def get_capital_two_step_graph(
    capital: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Compute the two-step capital graph.

    Parameters
    ----------
    capital : scipy.sparse.spmatrix
        Capital matrix of shape (M, M + 3C).

    Returns
    -------
    scipy.sparse.csr_matrix
        Two-step capital graph adjacency matrix

    Raises
    ------
    TypeError
        If `capital` is not a scipy sparse matrix.
    """
    if not sparse.issparse(capital):
        raise TypeError('Capital matrix must be a scipy sparse matrix')

    GK = _bipartite_adjacency(capital)
    return (GK.T @ GK).tocsr()


def get_references_power(
    references: sparse.spmatrix,
    n: int,
) -> sparse.csr_matrix:
    """
    Compute the n-th matrix power of the weighted references graph.
    This gives gives the academic capital manuscript y contributes to
    manuscript x through n step citation chains. Row sums give the 
    total impact of x on manuscripts n generations later.

    Parameters
    ----------
    references : scipy.sparse.spmatrix
        Weighted references matrix of shape (M, M).
    n : int
        Number of steps. Must be >= 1.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (M, M)
        n-th matrix power of the references graph.

    Raises
    ------
    TypeError
        If `references` is not a scipy sparse matrix.
    """
    if not sparse.issparse(references):
        raise TypeError('References matrix must be a scipy sparse matrix')
    

    G = references.tocsr().astype(float)
    result = sparse.eye(G.shape[0], format='csr', dtype=float)
    while n:
        if n & 1:
            result = result @ G
        G = G @ G
        n >>= 1
    return result.tocsr()


def _log_kirchhoff_spanning_trees(adjacency: sparse.spmatrix) -> float:

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    active = np.where(degrees > 0)[0]
    A_sub = adjacency.tocsr()[active, :][:, active].toarray()

    #Find the degree matrix
    d_sub = A_sub.sum(axis=1)

    #Compute Laplacian
    L = np.diag(d_sub) - A_sub
    
    #Compute the 0,0 cofactor
    cofactor = L[1:, 1:]

    #Find the log of the determinant
    _, logabsdet = np.linalg.slogdet(cofactor)
    return logabsdet


def get_spanning_tree_ratio(
    shares: sparse.spmatrix,
) -> float:
    """
    Compute the unweighted spanning tree ratio of the shares graph, which is the
    ratio of the log of the maximum spanning tree count to the log of the actual count.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).

    Returns
    -------
    float
        Unweighted spanning tree ratio. Values closer to 1 indicate a denser
        collaboration graph.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')

    M = shares.shape[0]
    c_total = shares.shape[1] - M

    log_tau_c = (c_total - 1) * np.log(M) + (M - 1) * np.log(c_total)

    A = _bipartite_adjacency(shares)
    a_bin = A.astype(bool).astype(float)
    log_tau_k = _log_kirchhoff_spanning_trees(a_bin)

    return log_tau_c / log_tau_k


def get_weighted_spanning_tree_ratio(
    shares: sparse.spmatrix,
) -> float:
    """
    Compute the weighted spanning tree ratio of the shares graph, which is the
    ratio of the log of the maximum weighted spanning tree count to the log of 
    the actual count of the weighted Laplacian.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).

    Returns
    -------
    float
        Weighted spanning tree ratio.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')

    M = shares.shape[0]
    c_total = shares.shape[1] - M
    s = 1.0 / c_total

    log_tau_cw = (M + c_total - 1) * np.log(s) + (c_total - 1) * np.log(M) + (M - 1) * np.log(c_total)

    A = _bipartite_adjacency(shares)
    log_tau_kw = _log_kirchhoff_spanning_trees(A)

    return log_tau_cw / log_tau_kw


def get_relative_spanning_tree_ratio(
    shares: sparse.spmatrix,
) -> float:
    """
    Compute the relative spanning tree ratio of the shares graph, which is
    the ratio of the weighted spanning tree ratio to the unweighted ratio.
    When RSTR < 1, share distributions are more uneven.
    RSTR closer to 1 indicates uniform contribution splits.

    Parameters
    ----------
    shares : scipy.sparse.spmatrix
        Shares matrix of shape (M, M + 3C).

    Returns
    -------
    float
        Relative spanning tree ratio.

    Raises
    ------
    TypeError
        If `shares` is not a scipy sparse matrix.
    """
    if not sparse.issparse(shares):
        raise TypeError('Shares matrix must be a scipy sparse matrix')

    return get_weighted_spanning_tree_ratio(shares) / get_spanning_tree_ratio(shares)