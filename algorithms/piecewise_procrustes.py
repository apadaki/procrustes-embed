import warnings
from functools import partial
import inspect
from typing import Optional, Sequence

import numpy as np

from .baseline_procrustes import baseline_procrustes

try:
    from .fast_nuclear_norm import nuclear_norm_slq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    nuclear_norm_slq = None

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
        if nuclear_norm_slq is None:
            warnings.warn(
                "fast_nuclear_norm is not installed; falling back to the exact nuclear norm.",
                RuntimeWarning,
            )
            return nuclear_norm
        return nuclear_norm_slq
    if callable(nuclear_norm_fn):
        return nuclear_norm_fn
    raise ValueError(f"Unknown nuclear_norm_fn: {nuclear_norm_fn}")


def _wrap_nuclear_norm_fn(norm_fn, nuclear_norm_kwargs):
    """Attach keyword arguments to the nuclear norm function when supported."""
    if not nuclear_norm_kwargs:
        return norm_fn

    sig = inspect.signature(norm_fn)
    allowed = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    extras = set(nuclear_norm_kwargs) - allowed
    if extras:
        warnings.warn(
            f"Ignoring unsupported nuclear_norm_kwargs keys: {sorted(extras)}",
            RuntimeWarning,
        )
    applicable_kwargs = {k: v for k, v in nuclear_norm_kwargs.items() if k in allowed}
    if not applicable_kwargs:
        return norm_fn
    return partial(norm_fn, **applicable_kwargs)


def clustering_to_transformation(
    X: np.ndarray,
    Y: np.ndarray,
    cutoffs: list[int],
    order: list[int]
) -> tuple[list[np.ndarray], float]:
    """Given two sets of vectors in R^d and a clustering, find the best Procrustes alignment along each cluster and retur the sum-of-squares cost"""
    n = X.shape[0]
    cutoffs = [0] + cutoffs + [n]
    total_cost = 0.0
    transformations: list[np.ndarray] = []
    
    for i in range(len(cutoffs) - 1):
        start, end = cutoffs[i], cutoffs[i+1]
        if start == end:
            continue
        cluster_i = order[start:end]
        X_cluster = X[cluster_i]
        Y_cluster = Y[cluster_i]

        Q_opt = baseline_procrustes(X_cluster, Y_cluster)
        transformations.append(Q_opt)
        diff = X_cluster @ Q_opt - Y_cluster
        cluster_cost = float(np.linalg.norm(diff)**2)
        total_cost += cluster_cost
    return transformations, total_cost


def piecewise_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    order: list[int],
    k: int = 1,
    *,
    n_samples = None,
    nuclear_norm_fn = 'exact',
    nuclear_norm_kwargs: Optional[dict] = None,
    candidate_boundaries: Optional[Sequence[int]] = None,
) -> tuple[list[np.ndarray], float, list[int]]:
    """Given two sets of vectors in R^d, find the best k-wise clustering (along a given order) that minimizes the Procrustes distance

    Args:
        X: source matrix of shape (n, d)
        Y: target matrix of shape (n, d)
        order: ordering of the rows (as indices into X/Y)
        k: number of clusters
        n_samples: optional down-sampling of candidate boundary locations along the provided order
        candidate_boundaries: optional additional boundary indices (in the ordered data) that must be included in the DP grid
        nuclear_norm_fn: selector for the nuclear norm implementation
        nuclear_norm_kwargs: optional kwargs forwarded to the nuclear norm function (e.g., n_vectors, lanczos_steps for 'fast')
    """
    n_full, d = X.shape
    assert len(order) == n_full

    norm_fn_raw = _resolve_nuclear_norm_fn(nuclear_norm_fn)
    norm_fn = _wrap_nuclear_norm_fn(norm_fn_raw, nuclear_norm_kwargs)

    X_ordered = X[order]
    Y_ordered = Y[order]

    # Down-sample only the candidate boundary locations; prefix sums still use all rows.
    if n_samples is None or n_samples >= n_full:
        prefix_positions = np.arange(n_full + 1, dtype=int)
    else:
        n_samples_int = int(n_samples)
        if n_samples_int <= 0:
            raise ValueError("n_samples must be positive when provided.")
        target_samples = min(n_full, max(n_samples_int, k))
        sampled_positions = np.linspace(0, n_full - 1, num=target_samples, dtype=int)
        sampled_positions = np.unique(np.concatenate([[0, n_full - 1], sampled_positions]))
        prefix_positions = np.unique(np.concatenate([[0], sampled_positions + 1])).astype(int)
        if prefix_positions[-1] != n_full:
            prefix_positions = np.append(prefix_positions, n_full)

    if candidate_boundaries:
        extra = np.asarray(candidate_boundaries, dtype=int)
        valid = extra[(extra >= 0) & (extra <= n_full)]
        if valid.size > 0:
            prefix_positions = np.unique(np.concatenate([prefix_positions, valid]))
    if prefix_positions[0] != 0:
        prefix_positions = np.insert(prefix_positions, 0, 0)
    if prefix_positions[-1] != n_full:
        prefix_positions = np.append(prefix_positions, n_full)

    n_work = len(prefix_positions) - 1

    # Build prefix sums only at the candidate boundary locations to avoid
    # materializing the full n x d x d tensor of outer products.
    acc_dtype = np.result_type(X_ordered.dtype, Y_ordered.dtype)
    prefix_sums = np.zeros((len(prefix_positions), d, d), dtype=acc_dtype)
    cumulative = np.zeros((d, d), dtype=acc_dtype)
    next_prefix = 1
    for row_idx in range(n_full):
        cumulative += np.outer(X_ordered[row_idx], Y_ordered[row_idx])
        while next_prefix < len(prefix_positions) and (row_idx + 1) == prefix_positions[next_prefix]:
            prefix_sums[next_prefix] = cumulative
            next_prefix += 1
            if next_prefix == len(prefix_positions):
                break
        if next_prefix == len(prefix_positions):
            break
    if next_prefix != len(prefix_positions):
        raise RuntimeError("Failed to build prefix sums for all candidate boundaries.")
    
    DP_array = np.zeros((n_work + 1, k + 1))
    backpointers: dict[tuple[int, int], int] = {}

    # Precompute all nuclear norms

    norm = np.zeros((n_work + 1, n_work + 1))
    total_norms = n_work * (n_work + 1) // 2

    def _progress_bar(computed: int) -> None:
        if total_norms <= 0:
            return
        width = 28
        frac = min(1.0, computed / total_norms)
        filled = int(width * frac)
        bar = '=' * filled + ' ' * (width - filled)
        print(f'\rnuclear norm computations [{bar}] {computed}/{total_norms}', end='', flush=True)

    update_interval = max(1, total_norms // 200)
    computed_norms = 0
    for x in range(1, n_work + 1):
        for y in range(1, x + 1):
            norm[x, y] = norm_fn(prefix_sums[x] - prefix_sums[y - 1])
            computed_norms += 1
            if computed_norms % update_interval == 0 or computed_norms == total_norms:
                _progress_bar(computed_norms)
    if total_norms > 0:
        print()
    # Base case: r=1 (one cluster)
    for j in range(1, n_work + 1):
        DP_array[j,1] = norm[j,1]
        backpointers[(j,1)] = 0

    for r in range(2, k+1):
        for j in range(r, n_work + 1):
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
    j = n_work
    r = k
    while r > 1:
        i = backpointers[(j, r)]
        boundary = int(prefix_positions[i])
        if 0 < boundary < n_full:
            cutoffs.append(boundary)
        j = i
        r -= 1
    cutoffs = sorted(set(cutoffs))

    transformations, total_cost = clustering_to_transformation(X, Y, cutoffs, order)
    return transformations, total_cost, cutoffs
