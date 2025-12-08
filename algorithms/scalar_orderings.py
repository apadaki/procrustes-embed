import numpy as np
from sklearn.decomposition import PCA


def get_direction(
    X: np.ndarray,
    *,
    method = 'pca'
) -> np.ndarray:
    if method == 'pca':
        pca = PCA(n_components=1)
        pca.fit(X)
        direction = pca.components_[0]
    if method = 'random':
        direction = np.random.randn(X.shape[1])
    return direction / np.linalg.norm(direction)

def ordering_from_direction(
    X: np.ndarray,
    order: list[int]
    direction_method = 'pca'
) -> np.ndarray:
     direction = get_direction(X, method=direction_method)
     X_proj = X @ direction
     return np.argsort(X_proj)
    
    
