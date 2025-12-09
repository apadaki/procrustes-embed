import numpy as np


def baseline_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Compute the optimal orthogonal matrix Q that minimizes ||XQ - Y||_F.
    Uses the classic SVD-based Procrustes solution.
    """
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt
