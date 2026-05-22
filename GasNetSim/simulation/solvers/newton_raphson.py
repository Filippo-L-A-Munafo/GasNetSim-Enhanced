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

# Relative regularization added to the diagonal when spsolve returns NaN/inf.
# Small enough to be negligible when J is well-conditioned, large enough to
# prevent breakdown on near-singular Jacobians (e.g. when many short segments
# have nearly equal inlet/outlet pressures).
_REG_FACTOR = 1e-8

# Maximum fractional pressure change allowed per Newton step (relative to the
# initial pressure vector).  Caps explosive divergence in ill-conditioned
# systems (e.g. large networks with near-flat initial pressure and high
# demand imbalance) without preventing convergence once the iterate is near
# the solution.  Value 0.3 → each junction pressure can change by at most
# 30 % of its starting value per raw Newton step (before URF is applied).
_MAX_STEP_FRAC = 0.3

# Absolute minimum pressure (Pa) enforced after every Newton step.
# Physical pressures are always positive; clamping prevents the Weymouth /
# Panhandle equations from receiving negative absolute pressures, which would
# make the squared-pressure terms behave erratically.
_P_FLOOR_PA = 1.0


def _linear_solve(J, r):
    """Solve J·dx = r.

    * If J is already a scipy sparse matrix → spsolve directly.
    * If J is a large dense matrix (n ≥ _SPARSE_THRESHOLD) → convert to
      sparse CSC and use spsolve (avoids dense SVD failure on large systems).
    * If spsolve returns NaN/inf (near-singular or ill-conditioned Jacobian,
      common when many short segments have near-equal pressures) → retry with
      Tikhonov diagonal regularisation (eps·I added), then fall back to the
      iterative minimum-norm solver lsmr.
    * Otherwise → np.linalg.solve (fast LU), falling back to lstsq only
      if the matrix is singular.
    """
    if sparse.issparse(J):
        J_csc = J.tocsc()
    elif J.shape[0] >= _SPARSE_THRESHOLD:
        J_csc = sparse.csc_matrix(J)
    else:
        try:
            return np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(J, r, rcond=None)[0]

    # --- sparse direct solve ---
    dx = spsolve(J_csc, r)

    if not np.isfinite(dx).all():
        # Near-singular Jacobian.  Add a small diagonal regularisation term
        # eps·I to stabilise the system without changing the solution
        # significantly when J is well-conditioned.
        data_max = np.abs(J_csc.data).max() if J_csc.nnz > 0 else 1.0
        eps = data_max * _REG_FACTOR
        n = J_csc.shape[0]
        J_reg = J_csc + sparse.diags(np.full(n, eps), format='csc')
        dx = spsolve(J_reg, r)
        logger.debug("spsolve returned NaN — retried with Tikhonov reg eps=%.2e", eps)

    if not np.isfinite(dx).all():
        # Still not finite: use iterative minimum-norm least-squares (lsmr).
        # Handles rank-deficient Jacobians the same way SVD pseudo-inverse does.
        from scipy.sparse.linalg import lsmr
        dx = lsmr(J_csc, r, atol=1e-8, btol=1e-8)[0]
        logger.debug("Tikhonov reg still NaN — fell back to lsmr")

    return dx


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

        # Per-component step cap: prevents explosive divergence in
        # ill-conditioned systems.  The cap is fixed to the initial state so
        # it stays meaningful even after x drifts far from x0.
        _max_dp = np.abs(x0) * _MAX_STEP_FRAC
        _max_dp = np.where(_max_dp < _P_FLOOR_PA, _P_FLOOR_PA, _max_dp)

        # Track the best (lowest-residual) iterate so that the non-converged
        # return value is the closest we ever got to the solution, not the
        # final (possibly diverged) state.
        best_err = float("inf")
        best_x = x.copy()

        for n_iter in range(max_iter + 1):
            # Jacobian is evaluated first so the formulation can cache both
            # J and F from the single _compute(x) call, making residual_fn
            # a free cache hit.
            J = jacobian_fn(x)
            r = residual_fn(x)

            err = float(np.max(np.abs(r)))
            logger.debug(f"Newton iteration {n_iter}: max|F| = {err:.3e}")

            if err < best_err:
                best_err = err
                best_x = x.copy()

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

            # Cap each component to ±_max_dp to prevent blow-up.
            dx = np.clip(dx, -_max_dp, _max_dp)

            x = x + dx

            # Enforce pressure positivity: gas absolute pressures cannot
            # be negative.  Clamping keeps the Weymouth equations physical.
            x = np.where(x < _P_FLOOR_PA, _P_FLOOR_PA, x)

        logger.warning(
            "Newton-Raphson did not converge in %d iteration(s). "
            "Best residual: %.3e (at iter where best was stored). "
            "Returning best iterate.",
            max_iter, best_err,
        )
        return SolveResult(
            x=best_x,
            converged=False,
            n_iter=max_iter,
            message=f"Did not converge in {max_iter} iteration(s)",
        )
