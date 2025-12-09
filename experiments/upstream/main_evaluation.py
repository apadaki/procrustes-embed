import argparse
import sys
from pathlib import Path

import numpy as np
from typing import Any, Optional, Sequence
from sklearn.cluster import KMeans

# Ensure the project root is on PYTHONPATH when running from the experiments directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.baseline_procrustes import baseline_procrustes
from algorithms.piecewise_procrustes import piecewise_procrustes
from experiment_framework import (
    AlgorithmOutcome,
    CompetitorAlgorithm,
    ExperimentData,
    ExperimentParams,
    ExperimentRunner,
    ExperimentResult,
    NuclearNormApproximation,
)
parser = argparse.ArgumentParser()
parser.add_argument(
    '--nvecs',
    action='store',
    type=int,
    help='Number of random vectors used in nuclear norm approximation',
)
parser.add_argument(
    '--steps',
    action='store',
    type=int,
    help='Number of Lanczos steps used in nuclear norm approximation',
)

DEFAULT_FAST_NUCLEAR_NORM = {
    "nuclear_norm_fn": "fast",
    "nuclear_norm_kwargs": {
        "n_vectors": 15,
        "lanczos_steps": 5,
    }
}
# Central place to tweak the fast nuclear norm approximation.

PIECEWISE_COMPETITOR_NAME = 'piecewise'

def kmeans_procrustes_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
    """
    Cluster X via k-means, solve a Procrustes problem inside each cluster, and
    return the aggregate squared alignment error.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of rows.")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = kmeans.fit_predict(X)

    d = X.shape[1]
    X_aligned = np.zeros_like(X)
    transformations: list[np.ndarray] = []
    total_cost = 0.0

    for cluster_id in range(n_clusters):
        cluster_idx = np.where(labels == cluster_id)[0]
        if cluster_idx.size == 0:
            transformations.append(np.eye(d))
            continue

        X_cluster = X[cluster_idx]
        Y_cluster = Y[cluster_idx]
        Q_cluster = baseline_procrustes(X_cluster, Y_cluster)

        aligned = X_cluster @ Q_cluster
        X_aligned[cluster_idx] = aligned
        diff = aligned - Y_cluster
        total_cost += float(np.linalg.norm(diff) ** 2)
        transformations.append(Q_cluster)

    return labels, transformations, X_aligned, total_cost

def piecewise_align(
    X: np.ndarray,
    Y: np.ndarray,
    order: list[int],
    Qs: list[np.ndarray],
    cutoffs: list[int],
    *,
    assign_best: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Apply the learned piecewise transformations and compute the alignment cost.

    If assign_best is True, each point is aligned with whichever transformation in Qs
    minimizes its individual squared error with the corresponding Y point.
    """
    n = X.shape[0]
    total_cost = 0.0
    aligned_chunks: list[np.ndarray] = []

    if assign_best:
        for idx in order:
            x_vec = X[idx]
            y_vec = Y[idx]

            best_aligned = None
            best_cost = np.inf
            for Q in Qs:
                aligned = x_vec @ Q
                cost = float(np.linalg.norm(aligned - y_vec) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_aligned = aligned

            aligned_chunks.append(best_aligned)
            total_cost += best_cost

        X_aligned = np.vstack(aligned_chunks) if aligned_chunks else np.empty_like(X)
        return X_aligned, total_cost

    cutoffs = [0] + cutoffs + [n]

    for i in range(len(cutoffs) - 1):
        start, end = cutoffs[i], cutoffs[i + 1]
        cluster_indices = order[start:end]
        if len(cluster_indices) == 0:
            continue

        X_cluster = X[cluster_indices]
        Y_cluster = Y[cluster_indices]

        Q = Qs[i]
        X_cluster_aligned = X_cluster @ Q
        aligned_chunks.append(X_cluster_aligned)

        diff = X_cluster_aligned - Y_cluster
        total_cost += float(np.linalg.norm(diff) ** 2)

    X_aligned = np.vstack(aligned_chunks) if aligned_chunks else np.empty_like(X)
    return X_aligned, total_cost


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _apply_rotation_perturbation(
    base: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scale <= 0:
        return base
    dim = base.shape[0]
    noise = rng.normal(size=(dim, dim))
    skew = noise - noise.T
    delta = np.eye(dim) + scale * skew
    q_delta, r = np.linalg.qr(delta)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q_delta *= signs
    if np.linalg.det(q_delta) < 0:
        q_delta[:, 0] *= -1
    return base @ q_delta


def generate_temporal_shift_data(
    *,
    n_samples: int = 1800,
    dim: int = 32,
    n_pieces: int = 3,
    intracluster_drift: float = 0.01,
    intercluster_jump: float = 0.5,
    seed: int = 0,
) -> ExperimentData:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if n_pieces <= 0 or n_pieces > n_samples:
        raise ValueError("n_pieces must be in [1, n_samples].")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, dim))
    Y = np.empty_like(X)

    base_lengths = np.full(n_pieces, n_samples // n_pieces, dtype=int)
    base_lengths[: n_samples % n_pieces] += 1

    true_cutoffs: list[int] = []
    Q_current = _random_rotation(dim, rng)
    idx = 0

    for piece_idx, seg_len in enumerate(base_lengths):
        for _ in range(seg_len):
            Q_current = _apply_rotation_perturbation(Q_current, intracluster_drift, rng)
            Y[idx] = X[idx] @ Q_current
            idx += 1

        if piece_idx < n_pieces - 1:
            true_cutoffs.append(idx)
            Q_current = _apply_rotation_perturbation(Q_current, intercluster_jump, rng)

    metadata = {
        'true_cutoffs': true_cutoffs,
        'n_pieces': n_pieces,
        'intracluster_drift': intracluster_drift,
        'intercluster_jump': intercluster_jump,
        'seed': seed,
    }
    return ExperimentData(X=X, Y=Y, metadata=metadata)


def _global_procrustes_competitor(X: np.ndarray, Y: np.ndarray, _: ExperimentParams) -> AlgorithmOutcome:
    Q_global = baseline_procrustes(X, Y)
    cost = float(np.linalg.norm(X @ Q_global - Y) ** 2)
    return AlgorithmOutcome(cost=cost)


def _kmeans_competitor_factory(n_clusters: int) -> CompetitorAlgorithm:
    def _fn(X: np.ndarray, Y: np.ndarray, _: ExperimentParams) -> AlgorithmOutcome:
        _, _, _, cost = kmeans_procrustes_baseline(
            X,
            Y,
            n_clusters=n_clusters,
        )
        return AlgorithmOutcome(cost=cost, details={'n_clusters': n_clusters})

    return CompetitorAlgorithm('kmeans', _fn)


def _piecewise_competitor(X: np.ndarray, Y: np.ndarray, params: ExperimentParams) -> AlgorithmOutcome:
    if params.order is None:
        raise ValueError("Piecewise competitor requires an explicit order.")
    order = list(params.order)
    transformations, piecewise_cost, cutoffs = piecewise_procrustes(
        X,
        Y,
        order,
        k=params.k,
        n_samples=params.dp_samples,
        nuclear_norm_fn=params.nuclear_norm.fn,
        nuclear_norm_kwargs=params.nuclear_norm.kwargs,
    )
    details = {
        'transformations': transformations,
        'cutoffs': cutoffs,
    }
    return AlgorithmOutcome(cost=piecewise_cost, details=details)


def temporal_shift_experiment(
    *,
    n_samples: int = 1800,
    dim: int = 32,
    n_pieces: int = 3,
    kmeans_clusters: Optional[int] = None,
    intracluster_drift: float = 0.01,
    intercluster_jump: float = 0.5,
    nuclear_norm_config: Optional[dict[str, Any]] = None,
    assign_best: bool = False,
    competitors: Optional[Sequence[CompetitorAlgorithm]] = None,
    dp_samples: Optional[int] = 150,
    metrics: Sequence[str] = ('runtime', 'cost'),
    seed: int = 0,
) -> ExperimentResult:
    if n_pieces <= 0:
        raise ValueError("n_pieces must be positive.")

    if kmeans_clusters is None:
        kmeans_clusters = n_pieces
    if kmeans_clusters <= 0:
        raise ValueError("kmeans_clusters must be positive.")

    norm_config = nuclear_norm_config or DEFAULT_FAST_NUCLEAR_NORM
    nuclear_norm = NuclearNormApproximation(
        fn=norm_config.get('nuclear_norm_fn', 'exact'),
        kwargs=dict(norm_config.get('nuclear_norm_kwargs', {})),
    )

    params = ExperimentParams(
        k=n_pieces,
        n_samples=n_samples,
        nuclear_norm=nuclear_norm,
        dp_samples=dp_samples,
    )

    runner = ExperimentRunner(generate_temporal_shift_data)

    default_competitors: list[CompetitorAlgorithm] = [
        CompetitorAlgorithm('global_procrustes', _global_procrustes_competitor),
        _kmeans_competitor_factory(kmeans_clusters),
        CompetitorAlgorithm(PIECEWISE_COMPETITOR_NAME, _piecewise_competitor),
    ]
    competitor_list = list(competitors) if competitors is not None else default_competitors

    result = runner.run(
        params=params,
        competitors=competitor_list,
        metrics=metrics,
        data_kwargs={
            'dim': dim,
            'n_pieces': n_pieces,
            'intracluster_drift': intracluster_drift,
            'intercluster_jump': intercluster_jump,
            'seed': seed,
        },
    )

    if assign_best:
        piecewise_measurement = result.competitor_results.get(PIECEWISE_COMPETITOR_NAME)
        if piecewise_measurement:
            transformations = piecewise_measurement.details.get('transformations')
            cutoffs = piecewise_measurement.details.get('cutoffs')
            if transformations is not None and cutoffs is not None:
                X = result.data.X
                Y = result.data.Y
                order = list(result.params.order or range(X.shape[0]))
                _, best_cost = piecewise_align(
                    X,
                    Y,
                    order,
                    transformations,
                    cutoffs,
                    assign_best=True,
                )
                piecewise_measurement.details['assign_best_cost'] = best_cost

    return result


def main() -> None:
    args = parser.parse_args()

    nuclear_kwargs = dict(DEFAULT_FAST_NUCLEAR_NORM['nuclear_norm_kwargs'])
    if args.nvecs is not None:
        nuclear_kwargs['n_vectors'] = int(args.nvecs)
    if args.steps is not None:
        nuclear_kwargs['lanczos_steps'] = int(args.steps)

    nuclear_config = {
        'nuclear_norm_fn': DEFAULT_FAST_NUCLEAR_NORM['nuclear_norm_fn'],
        'nuclear_norm_kwargs': nuclear_kwargs,
    }

    result = temporal_shift_experiment(
        nuclear_norm_config=nuclear_config,
        dim=100,
        n_pieces=4,
        intracluster_drift=1e-5,
        intercluster_jump=1,
        assign_best=True,
    )

    print('Temporal shift experiment metrics:')
    for name, measurement in result.competitor_results.items():
        metric_parts = [f'{metric}: {value:.6f}' for metric, value in measurement.metrics.items()]
        metrics_str = ', '.join(metric_parts)
        print(f'{name}: {metrics_str}')
        if 'assign_best_cost' in measurement.details:
            print(f"  assign_best_cost: {measurement.details['assign_best_cost']:.6f}")

    piecewise_details = result.competitor_results.get(PIECEWISE_COMPETITOR_NAME)
    if piecewise_details:
        print(f"Estimated cutoffs: {piecewise_details.details.get('cutoffs')}")
    print(f"True cutoffs: {result.data.metadata.get('true_cutoffs')}")

if __name__ == '__main__':
    main()
