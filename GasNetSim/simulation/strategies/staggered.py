import copy
import logging

import numpy as np

from .base_strategy import Strategy
from .direct import DirectStrategy

logger = logging.getLogger(__name__)


class StaggeredStrategy(Strategy):
    """
    Staggered (outer-Picard) coupling strategy.

    Alternates between two sub-steps until both pressure and composition
    converge:

    1. **Pressure step** — freeze composition, run a full direct Newton
       solve (``DirectStrategy`` with ``tracking_method="no_mixing"``).
    2. **Composition step** — freeze pressure, propagate compositions
       topologically (``network.composition_initialization``), then apply
       relaxed blending toward the new target.

    Parameters
    ----------
    solver : Solver, optional
        Inner Solver passed to the DirectStrategy pressure step.
        Defaults to NewtonRaphsonSolver.
    coupling_max_iter : int
        Maximum number of outer Picard iterations.
    coupling_tol : float
        Convergence tolerance on the composition change (max absolute
        difference across all nodal compositions).
    outer_pressure_tol : float, optional
        Convergence tolerance on the pressure change between outer
        iterations (Pa).  Defaults to ``max(10.0, tol * 1e5)``.
    composition_relaxation_factor : float
        Under-relaxation factor applied when blending compositions
        between outer iterations (1.0 = no relaxation).
    """

    def __init__(
        self,
        solver=None,
        coupling_max_iter=20,
        coupling_tol=1e-4,
        outer_pressure_tol=None,
        composition_relaxation_factor=0.5,
        aitken_acceleration=False,
    ):
        self.solver = solver
        self.coupling_max_iter = coupling_max_iter
        self.coupling_tol = coupling_tol
        self.outer_pressure_tol = outer_pressure_tol
        # Fattore di rilassamento della composizione. Con aitken_acceleration
        # è solo il valore INIZIALE: da lì Aitken (Irons–Tuck) lo adatta a ogni
        # iterazione. Con aitken_acceleration=False resta fisso (comportamento
        # classico).
        self.composition_relaxation_factor = composition_relaxation_factor
        self.aitken_acceleration = aitken_acceleration

    def solve(
        self,
        network,
        *,
        max_iter,
        tol,
        underrelaxation_factor,
        use_cuda,
        sparse_matrix,
        tracking_method,
        time_step,
        warm_start=False,
    ):
        # no_mixing needs no composition coupling — delegate directly.
        if tracking_method == "no_mixing":
            return DirectStrategy(solver=self.solver).solve(
                network,
                max_iter=max_iter,
                tol=tol,
                underrelaxation_factor=underrelaxation_factor,
                use_cuda=use_cuda,
                sparse_matrix=sparse_matrix,
                tracking_method=tracking_method,
                time_step=time_step,
                warm_start=warm_start,
            )

        if self.coupling_max_iter < 1:
            raise ValueError("coupling_max_iter must be at least 1.")

        outer_pressure_tol = self.outer_pressure_tol
        if outer_pressure_tol is None:
            outer_pressure_tol = max(10.0, tol * 1e5)

        inner_strategy = DirectStrategy(solver=self.solver)
        common = dict(
            max_iter=max_iter,
            tol=tol,
            underrelaxation_factor=underrelaxation_factor,
            use_cuda=use_cuda,
            sparse_matrix=sparse_matrix,
            time_step=time_step,
            warm_start=warm_start,
        )

        # Save and restore network initialisation flags across outer iters.
        original_run_initialization = network.run_initialization
        original_pressure_prev = copy.deepcopy(network.pressure_prev)
        original_run_composition_initialization = (
            network.run_composition_initialization
        )

        previous_pressure = None

        # Stato per l'accelerazione di Aitken (Irons–Tuck) sulla composizione.
        aitken_prev_r = None
        aitken_omega = float(self.composition_relaxation_factor)

        try:
            for outer_iter in range(1, self.coupling_max_iter + 1):
                if outer_iter > 1:
                    network.run_initialization = False
                    network.pressure_prev = network.save_pressure_values()

                # --- Pressure step (composition frozen) ---
                # run_composition_initialization only gates
                # network.composition_initialization().  It does NOT stop
                # update_pipeline_parameters() from re-picking each edge's gas
                # mixture from the sign of its flow rate on every residual
                # evaluation, which makes the residual discontinuous in x and
                # stalls the line search.  freeze_mixture_during_newton is what
                # actually delivers the "composition frozen" promised here.
                network.run_composition_initialization = False
                network.freeze_mixture_during_newton = True
                try:
                    inner_strategy.solve(
                        network, tracking_method="no_mixing", **common
                    )
                finally:
                    network.freeze_mixture_during_newton = False

                current_pressure = np.array(
                    network.save_pressure_values(), copy=True
                )
                previous_composition = network._snapshot_nodal_eos_compositions()

                # --- Composition step (pressure frozen) ---
                network.run_composition_initialization = True
                if tracking_method == "batch_tracking" and network.pipelines:
                    cached_batch_information = {
                        i: (
                            p.batch_location_history.copy(),
                            p.composition_history.copy(),
                        )
                        for i, p in network.pipelines.items()
                    }
                else:
                    cached_batch_information = {}
                network.composition_initialization(
                    tracking_method=tracking_method,
                    time_step=time_step,
                    cached_batch_information=cached_batch_information,
                )
                target_composition = network._snapshot_nodal_eos_compositions()

                # --- Rilassamento della composizione ---------------------------
                # Fisso (composition_relaxation_factor) oppure ADATTIVO di Aitken
                # (Irons–Tuck): stima a ogni iterazione il fattore di
                # rilassamento ottimale dai residui successivi, accelerando una
                # convergenza altrimenti lineare (rate 0.5 → ~15 iterazioni).
                #
                # omega resta in [0.1, 1.0]: MAI sovra-rilassamento. La
                # composizione è una variabile vincolata (frazioni molari ≥ 0,
                # somma 1); a omega=1 si prende la composizione propagata (già
                # valida), per omega<1 una combinazione convessa di due
                # composizioni valide (ancora valida). Con omega>1 si
                # estrapolerebbe fuori dal simplesso → frazioni negative.
                omega = self.composition_relaxation_factor
                if self.aitken_acceleration:
                    _nonj = set(network.non_junction_nodes)
                    _parts = [
                        target_composition[nid] - previous_composition[nid]
                        for nid in network.nodes
                        if nid not in _nonj
                    ]
                    r = (
                        np.concatenate(_parts)
                        if _parts
                        else np.zeros(0)
                    )
                    if aitken_prev_r is not None and r.size:
                        dr = r - aitken_prev_r
                        denom = float(dr @ dr)
                        if denom > 1e-30:
                            aitken_omega = (
                                -aitken_omega
                                * float(aitken_prev_r @ dr)
                                / denom
                            )
                        aitken_omega = min(max(aitken_omega, 0.1), 1.0)
                    omega = aitken_omega
                    aitken_prev_r = r

                composition_delta = network._apply_relaxed_nodal_compositions(
                    previous_composition,
                    target_composition,
                    omega,
                )

                pressure_delta = (
                    np.inf
                    if previous_pressure is None
                    else float(np.max(np.abs(current_pressure - previous_pressure)))
                )

                logger.info(
                    "Staggered coupling iteration %d/%d: "
                    "pressure delta %.3f Pa, composition delta %.3e, omega %.3f",
                    outer_iter,
                    self.coupling_max_iter,
                    pressure_delta,
                    composition_delta,
                    omega,
                )

                if (
                    previous_pressure is not None
                    and pressure_delta <= outer_pressure_tol
                    and composition_delta <= self.coupling_tol
                ):
                    logger.info(
                        "Staggered coupling converges in %d iteration(s).",
                        outer_iter,
                    )
                    return network

                previous_pressure = current_pressure

        finally:
            network.run_initialization = original_run_initialization
            network.pressure_prev = original_pressure_prev
            network.run_composition_initialization = (
                original_run_composition_initialization
            )

        raise RuntimeError(
            f"Staggered coupling not converged in {self.coupling_max_iter} "
            "iteration(s)!"
        )
