import numpy as np
from fast_nuclear_norm import nuclear_norm_slq

def nuclear_norm(
    M: np.ndarray
) -> float:
    """
    Compute nuclear norm (sum of singular values) of a matrix
    Runtime: O(nd^2) for nxd matrix M
    """
    S = np.linalg.svd(M, compute_uv=False)
    return np.sum(S)


def _resolve_nuclear_norm_fn(nuclear_norm_fn):
    """Map selector to an implementation of the nuclear norm."""
    if nuclear_norm_fn in (None, 'exact'):
        return nuclear_norm
    if nuclear_norm_fn == 'fast':
        return nuclear_norm_slq
    if callable(nuclear_norm_fn):
        return nuclear_norm_fn
    raise ValueError(f"Unknown nuclear_norm_fn: {nuclear_norm_fn}")


def clustering_to_transformation(
    X: np.ndarray,
    Y: np.ndarray,
    cutoffs: list[int],
    order: list[int]
) -> tuple[list[np.ndarray], float]:
    """Given two sets of vectors in R^d and a clustering, find the best Procrustes alignment along each cluster and retur the sum-of-squares cost"""
    n = X.shape[0]
    cutoffs = [0] + cutoffs + [n]
    total_cost = 0
    transformations = []
    
    for i in range(len(cutoffs) - 1):
        start, end = cutoffs[i], cutoffs[i+1]
        cluster_i = order[start:end]
        X_cluster = X[cluster_i]
        Y_cluster = Y[cluster_i]

        # compute optimal Q for this cluster
        M = X_cluster.T @ Y_cluster
        U, _, Vt = np.linalg.svd(M)
        Q_opt = U @ Vt

        # compute cost for this cluster
        diff = X_cluster @ Q_opt - Y_cluster
        cluster_cost = np.linalg.norm(diff)**2
        total_cost += cluster_cost
    return total_cost


def piecewise_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    order: list[int],
    k: int = 1,
    *,
    n_samples = None,
    nuclear_norm_fn = 'exact'
) -> tuple[list[np.ndarray], list[int]]:
    """Given two sets of vectors in R^d, find the best k-wise clustering (along a given order) that minimizes the Procrustes distance"""
    n, d = X.shape
    assert len(order) == n

    norm_fn = _resolve_nuclear_norm_fn(nuclear_norm_fn)
    
    outer_prods = np.array([Y[order[i]].T @ X[order[i]] for i in range(n)])
    M_prefix = np.cumsum(outer_prods, axis=0)
    M_prefix = np.concatenate([np.zeros((1,d,d)), M_prefix], axis=0)
    
    DP_array = np.zeros((n+1,k+1))
    backpointers = {}

    # Precompute all nuclear norms
    norm = np.zeros((n+1,n+1))
    for x in range(1,n+1):
        for y in range(1, x+1):
            norm[x, y] = norm_fn(M_prefix[x] - M_prefix[y-1])

    # Base case: r=1 (one cluster)
    for j in range(1, n+1):
        DP_array[j,1] = norm[j,1]
        backpointers[(j,1)] = 0

    for r in range(2, k+1):
        for j in range(r,n+1):
            max_val = -np.inf
            best_i = -1
            for i in range(r-1, j):
                val = norm[j, i+1] + DP_array[i, r-1]
                if val > max_val:
                    max_val = val
                    best_i = i
            DP_array[j, r] = max_val
            backpointers[(j,r)] = best_i
    # use backpointers to find optimal clustering
    cutoffs = []
    j = n
    r = k
    while r > 1:
        i = backpointers[(j, r)]
        cutoffs.append(i)
        j = i
        r -= 1
    cutoffs.reverse()
    
    return clustering_to_transformation(X, Y, cutoffs, order), cutoffs
