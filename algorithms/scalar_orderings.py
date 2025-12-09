import numpy as np
from sklearn.decomposition import PCA


def get_direction(
    X: np.ndarray,
    *,
    method: str = 'pca',
) -> np.ndarray:
    if method == 'pca':
        pca = PCA(n_components=1)
        pca.fit(X)
        direction = pca.components_[0]
    elif method == 'random':
        direction = np.random.randn(X.shape[1])
    else:
        raise ValueError(f"Unknown direction method: {method}")
    return direction / np.linalg.norm(direction)


def proj_ordering(
    X: np.ndarray,
    *,
    direction_method: str = 'pca',
    normalize: bool = False
) -> np.ndarray:
    direction = get_direction(X, method=direction_method)
    if normalize:
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        X_normalized = X / row_norms
    X_proj = X_normalized @ direction
    return np.argsort(X_proj)
    
    
