import numpy as np
import logging

from scipy import sparse
from scipy.sparse.linalg import spsolve

from .base_solver import Solver, SolveResult

logger = logging.getLogger(__name__)

# For Jacobians larger than this threshold (n × n), skip dense SVD/LU and
# instead convert to sparse CSC before solving.  Gas-network Jacobians are
# very sparse (degree ≈ 2–4 per node) so spsolve is fast and reliable.
_SPARSE_THRESHOLD = 500


def _linear_solve(J, r):
    """Solve J·dx = r.

    * If J is already a scipy sparse matrix → spsolve directly.
    * If J is a large dense matrix (n ≥ _SPARSE_THRESHOLD) → convert to
      sparse CSC and use spsolve (avoids dense SVD failure on large systems).
    * Otherwise → np.linalg.solve (fast LU), falling back to lstsq only
      if the matrix is singular.
    """
    if sparse.issparse(J):
        return spsolve(J.tocsc(), r)
    n = J.shape[0]
    if n >= _SPARSE_THRESHOLD:
        return spsolve(sparse.csc_matrix(J), r)
    try:
        return np.linalg.solve(J, r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, r, rcond=None)[0]


class NewtonRaphsonSolver(Solver):
    """
    Hand-rolled Newton-Raphson solver with optional under-relaxation.

    At each iteration:
      1. Evaluate Jacobian J(x) and residual F(x)   (one _compute call)
      2. Solve the linear system: J dx = F
      3. Update: x ← x + dx / underrelaxation_factor

    Convergence is declared when max|F(x)| ≤ tol before a step is taken.
    """

    def __init__(self, underrelaxation_factor: float = 2.0):
        self.underrelaxation_factor = underrelaxation_factor

    def solve(self, residual_fn, jacobian_fn, x0, tol, max_iter):
        x = x0.copy()

        for n_iter in range(max_iter + 1):
            # Jacobian is evaluated first so the formulation can cache both
            # J and F from the single _compute(x) call, making residual_fn
            # a free cache hit.
            J = jacobian_fn(x)
            r = residual_fn(x)

            err = float(np.max(np.abs(r)))
            logger.debug(f"Newton iteration {n_iter}: max|F| = {err:.3e}")

            if err <= tol:
                return SolveResult(
                    x=x,
                    converged=True,
                    n_iter=n_iter,
                    message=f"Converged in {n_iter} iteration(s)",
                )

            if n_iter == max_iter:
                break

            dx = _linear_solve(J, r) / self.underrelaxation_factor
            x = x + dx

        return SolveResult(
            x=x,
            converged=False,
            n_iter=max_iter,
            message=f"Did not converge in {max_iter} iteration(s)",
        )
