import numpy as np


class _LinearOperator:
    """Minimal linear-operator wrapper used for Lanczos without SciPy."""

    def __init__(self, shape, matvec, dtype):
        self.shape = shape
        self.dtype = dtype
        self._matvec = matvec

    def matvec(self, x):
        return self._matvec(x)

def lanczos_tridiag(A, v, m):
    """
    Performs m steps of Lanczos iteration on matrix A with starting vector v.
    Returns the tridiagonal matrix T (diagonals alpha, off-diagonals beta).
    """
    n = len(v)
    alpha = np.zeros(m)
    beta = np.zeros(m-1)
    
    # Normalize start vector
    v_curr = v / np.linalg.norm(v)
    v_prev = np.zeros(n)
    
    for j in range(m):
        # Matrix-vector product w = A * v_curr
        # In our case A is passed as a LinearOperator or function
        w = A.matvec(v_curr)
        
        # Orthogonalize
        alpha[j] = np.dot(w, v_curr)
        w = w - alpha[j] * v_curr - (beta[j-1] * v_prev if j > 0 else 0)
        
        if j < m - 1:
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-10: # Breakdown check
                break
            v_prev = v_curr.copy()
            v_curr = w / beta[j]
            
    # Construct T
    T = np.diag(alpha) + np.diag(beta, k=1) + np.diag(beta, k=-1)
    return T

def nuclear_norm_slq(A, n_vectors=10, lanczos_steps=20):
    """
    Approximates nuclear norm ||A||_* using Stochastic Lanczos Quadrature.

    Args:
        A: numpy array or scipy sparse matrix
        n_vectors: number of random probe vectors used for Hutchinson estimation
        lanczos_steps: number of Lanczos iterations per probe vector
    """
    m, n = A.shape
    
    # Define LinearOperator for M = A.T @ A
    # This avoids forming the dense matrix M
    def mv(x):
        return A.T @ (A @ x)
    
    M_op = _LinearOperator((n, n), mv, A.dtype)

    results = []
    
    for _ in range(n_vectors):
        # 1. Random vector (Rademacher +/- 1 is common and low variance)
        v = np.random.choice([-1, 1], size=n)
        v_norm_sq = np.dot(v, v)
        
        # 2. Lanczos Tridiagonalization
        T = lanczos_tridiag(M_op, v, lanczos_steps)
        
        # 3. Compute f(T) = sqrt(T)
        # T is small, so we use dense eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Apply function to eigenvalues: sqrt(lambda)
        # Clamp negative small values to 0 (numerical noise)
        func_eigvals = np.sqrt(np.maximum(eigvals, 0))
        
        # Reconstruct: f(T) = Q * f(D) * Q.T
        # We only need the first element of f(T) * e1, which is (Q * f(D) * Q.T)[0,0]
        # This simplifies to: sum( (Q[0,j]^2) * f(lambda_j) )
        est = np.sum((eigvecs[0, :]**2) * func_eigvals)
        
        results.append(est * v_norm_sq)
        
    return np.mean(results)
"""
# --- Example Usage ---
# Create a low-rank matrix
N = 1000
U = np.random.randn(N, 10)
V = np.random.randn(10, N)
A_dense = U @ V

# True Nuclear Norm (sum of singular values)
s_true = np.linalg.svd(A_dense, compute_uv=False)
print(f"True Nuclear Norm: {np.sum(s_true):.4f}")

# Approximated
approx = nuclear_norm_slq(A_dense, n_vectors=15, lanczos_steps=30)
print(f"SLQ Approximation: {approx:.4f}")
"""
