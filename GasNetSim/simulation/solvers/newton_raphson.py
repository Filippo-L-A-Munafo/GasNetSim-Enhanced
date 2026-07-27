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

# Maximum number of backtracking halvings in the line search.
#
# This must be generous, because the per-component clip to _max_dp MASKS the
# backtracking: while step_scale * |dx_raw| still exceeds the cap, halving
# step_scale changes nothing — the component comes out clipped at the cap
# either way.  The step only starts to shrink once
#     step_scale < _max_dp / |dx_raw|.
# Near an ill-conditioned solution |dx_raw| can reach 1e9-1e10 Pa while
# _max_dp is ~2e6 Pa, so roughly 12-13 halvings are consumed before the line
# search does anything at all.  The old limit of 12 ran out exactly there and
# reported a false "no step reduces the residual".  2^-40 ~ 1e-12 leaves ample
# margin past the clip.
_MAX_LS_ITER = 40


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

    def __init__(self, underrelaxation_factor: float = 2.0,
                 n_pressure_vars: int | None = None,
                 flow_step_cap: float | None = None):
        self.underrelaxation_factor = underrelaxation_factor
        # Cap assoluto per le componenti di PORTATA della formulazione
        # aumentata. Il cap ordinario e' |x0|*_MAX_STEP_FRAC, cioe' relativo
        # allo stato iniziale: per una pressione (~7e6 Pa) e' generoso, ma per
        # una portata di compressore stimata male puo' essere minuscolo, e la
        # variabile resta strozzata a passi infinitesimi mentre le pressioni
        # si muovono liberamente. Il risultato e' uno stallo che sembra una
        # direzione non di discesa ma e' solo una componente bloccata.
        self.flow_step_cap = flow_step_cap
        # Callable opzionale i -> str che dà un nome leggibile alla i-esima
        # equazione. Impostata da DirectStrategy quando il problema la
        # espone; serve a dire QUALE equazione blocca la convergenza.
        self.residual_labeller = None
        # Numero di componenti iniziali di x che sono PRESSIONI, e alle quali
        # va applicato il pavimento di positivita'.  None = tutte (caso
        # classico, x contiene solo pressioni).
        #
        # Con la formulazione aumentata dei compressori x diventa
        # [pressioni | portate]: le portate possono essere piccole o negative
        # (controflusso) e clamparle a _P_FLOOR_PA le falserebbe.
        self.n_pressure_vars = n_pressure_vars

    def _apply_pressure_floor(self, x_new: np.ndarray) -> np.ndarray:
        n_p = self.n_pressure_vars
        if n_p is None or n_p >= x_new.shape[0]:
            return np.where(x_new < _P_FLOOR_PA, _P_FLOOR_PA, x_new)
        testa = np.where(x_new[:n_p] < _P_FLOOR_PA, _P_FLOOR_PA, x_new[:n_p])
        return np.concatenate([testa, x_new[n_p:]])

    def solve(self, residual_fn, jacobian_fn, x0, tol, max_iter):
        x = x0.copy()

        # Per-component step cap: prevents explosive divergence in
        # ill-conditioned systems.  The cap is fixed to the initial state so
        # it stays meaningful even after x drifts far from x0.
        _max_dp = np.abs(x0) * _MAX_STEP_FRAC
        _max_dp = np.where(_max_dp < _P_FLOOR_PA, _P_FLOOR_PA, _max_dp)

        # Le componenti di portata ricevono un cap assoluto proprio, slegato
        # dalla loro stima iniziale.
        n_p = self.n_pressure_vars
        if (self.flow_step_cap is not None and n_p is not None
                and n_p < x0.shape[0]):
            _max_dp[n_p:] = float(self.flow_step_cap)
            logger.debug(
                "Cap portate compressore fissato a %.3e (le %d componenti "
                "finali di x)", self.flow_step_cap, x0.shape[0] - n_p,
            )

        # Track the best (lowest-residual) iterate so that the non-converged
        # return value is the closest we ever got to the solution, not the
        # final (possibly diverged) state.
        best_err = float("inf")
        best_x = x.copy()
        
        # --- Under-relaxation ADATTIVO -------------------------------------
        # L'URF scelto dall'utente diventa il PAVIMENTO: il passo piu'
        # aggressivo consentito (step_scale = 1/urf_floor). Quando la
        # convergenza fatica (molto backtracking nella line search, o residuo
        # che scende poco) urf_eff SALE verso urf_ceil, smorzando; quando
        # recupera, RIENTRA verso il pavimento. Simmetrico e limitato: evita
        # la deriva monotona del vecchio schema adattivo (che saliva a ~25 su
        # run_249 e si impiantava). La line search resta la sicurezza
        # per-iterazione; qui si adatta solo il PUNTO DI PARTENZA del passo,
        # cioe' la traiettoria fra un'iterazione e l'altra.
        urf_floor = float(self.underrelaxation_factor)
        urf_ceil = max(urf_floor * 4.0, 8.0)
        current_urf = urf_floor
        prev_err = float("inf")

        # Stop anticipato per convergenza lineare troppo lenta.
        #
        # Non basta chiedere "il residuo migliora?": ai gradini fini
        # dell'affinamento migliora davvero, ma solo dell'~1-2% per iterazione
        # (rapporto di riduzione ~0.98), e a quel ritmo servirebbero centinaia
        # di iterazioni per raggiungere tol. E' il regime lineare con rapporto
        # che tende a 1, dove Newton non aiuta piu'.
        #
        # La domanda giusta e': "con la velocita' di riduzione attuale,
        # raggiungero' tol entro le iterazioni che mi restano?". Si stima il
        # rapporto medio di riduzione su una finestra recente e si proietta il
        # numero di iterazioni necessarie; se supera il budget residuo (con un
        # margine), ci si ferma e si restituisce il miglior punto trovato.
        from collections import deque as _deque
        _err_window = _deque(maxlen=6)   # residui recenti per stimare il rapporto
        _MIN_ITERS_BEFORE_STALL = 8      # non giudicare prima di aver preso il ritmo

        for n_iter in range(max_iter + 1):
            # Jacobian is evaluated first so the formulation can cache both
            # J and F from the single _compute(x) call, making residual_fn
            # a free cache hit.
            J = jacobian_fn(x)
            r = residual_fn(x)

            err = float(np.max(np.abs(r)))
            logger.debug(f"Newton iteration {n_iter}: max|F| = {err:.3e}, URF = {current_urf:.2f}")

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

            # Stop per convergenza lineare troppo lenta (vedi nota sopra).
            _err_window.append(err)
            # Il giudizio di "convergenza lineare lenta" vale solo quando NON
            # stiamo smorzando di proposito: un urf_eff alto rende il passo
            # piccolo e la riduzione per-iterazione naturalmente lenta, ma e'
            # una scelta di stabilita', non uno stallo. Si giudica quindi solo
            # vicino al pavimento (traiettoria aggressiva).
            if (n_iter >= _MIN_ITERS_BEFORE_STALL and len(_err_window) >= 4
                    and err > tol and current_urf <= urf_floor * 1.5):
                err_start = _err_window[0]
                n_span = len(_err_window) - 1
                # Rapporto medio di riduzione per iterazione sulla finestra.
                if err_start > 0 and err > 0:
                    ratio = (err / err_start) ** (1.0 / n_span)
                else:
                    ratio = 1.0

                # Un Newton sano vicino alla soluzione riduce il residuo di
                # almeno la meta' per iterazione (comportamento quadratico,
                # rapporto <= 0.5). Un rapporto >= _SLOW_RATIO e' il regime
                # lineare degenere — Newton sta convergendo verso un punto
                # fisso che NON e' uno zero di F, e non accelerera' piu'.
                # Continuare a macinare centinaia di iterazioni per limare
                # qualche punto percentuale del residuo non vale il tempo:
                # meglio fermarsi e restituire il miglior iterato.
                _SLOW_RATIO = 0.90
                if ratio >= _SLOW_RATIO:
                    logger.debug(
                        "Convergenza lineare lenta a iter %d: max|F| = %.3e, "
                        "rapporto di riduzione %.4f/iter (>= %.2f). Newton non "
                        "accelerera' piu' verso tol %.3e; interrompo e "
                        "restituisco il miglior iterato (%.3e).",
                        n_iter, err, ratio, _SLOW_RATIO, tol, best_err,
                    )
                    return SolveResult(
                        x=best_x,
                        converged=False,
                        n_iter=n_iter,
                        message=(
                            f"Slow linear convergence at iter {n_iter}: "
                            f"max|F| = {best_err:.3e}, ratio {ratio:.4f}/iter, "
                            f"tol {tol:.3e} not worth pursuing"
                        ),
                    )
            prev_err = err

            # Calculate standard Newton step direction
            dx_raw = _linear_solve(J, r)

            # --- TRUST REGION ---
            # Cap the raw Newton direction ONCE, before the line search.
            #
            # This ordering is essential.  The previous code clipped *inside*
            # the loop, i.e. clip(dx_raw * step_scale).  While
            # step_scale * |dx_raw| still exceeded the cap, every component
            # came out clipped at exactly the same value regardless of
            # step_scale, so halving the step changed NOTHING: the line search
            # was silently inert for its first
            #     log2(|dx_raw| / _max_dp)
            # iterations.  With an ill-conditioned Jacobian |dx_raw| can reach
            # 1e10 Pa against a cap of ~2e6 Pa, which burns ~12 halvings doing
            # nothing — exactly the old limit — producing a spurious "no step
            # reduces the residual" abort while still far from the solution.
            #
            # Capping first makes the line search behave as intended: each
            # halving genuinely halves the step, and step_scale -> 0 really
            # does drive dx -> 0.
            dx_capped = np.clip(dx_raw, -_max_dp, _max_dp)

            # --- BACKTRACKING LINE SEARCH ---
            # Try to step along dx_capped. If the residual grows, halve the
            # step (up to _MAX_LS_ITER times) until the error decreases or we
            # run out of backtracking attempts.
            step_scale = 1.0 / current_urf

            step_accepted = False
            best_ls_err = float("inf")
            ls_halvings = 0

            for ls_iter in range(_MAX_LS_ITER):
                dx = dx_capped * step_scale

                x_new = x + dx

                # Enforce pressure positivity (clamping) — solo sulle
                # componenti di pressione, mai sulle portate dei compressori.
                x_new = self._apply_pressure_floor(x_new)

                # Evaluate residual at tentative new point
                r_new = residual_fn(x_new)
                new_err = float(np.max(np.abs(r_new)))
                best_ls_err = min(best_ls_err, new_err)

                # Accept the step only if it actually reduces the residual.
                if new_err < err:
                    step_accepted = True
                    ls_halvings = ls_iter
                    if ls_iter > 0:
                        logger.debug(f"      Line Search: step halved {ls_iter} times. new_err = {new_err:.3e}")
                    break

                # Otherwise, halve step and try again
                step_scale *= 0.5

            if not step_accepted:
                # VERIFICA DEL JACOBIANO ALLE DIFFERENZE FINITE.
                #
                # Se nessun passo riduce il residuo mentre il residuo E'
                # continuo, l'unica spiegazione restante e' che la direzione
                # di Newton sia sbagliata, cioe' che J non sia la derivata di
                # F. Invece di dedurlo, lo si misura: si perturba una alla
                # volta le variabili delle equazioni piu' fuori bilancio e si
                # confronta (F(x+h)-F(x))/h con la riga corrispondente di J.
                try:
                    peggiori = np.argsort(-np.abs(r))[:3]
                    logger.error("  Verifica Jacobiano (analitico vs differenze finite):")
                    for eq in peggiori:
                        eq = int(eq)
                        for var in {eq, x.shape[0] - 1}:
                            if var >= x.shape[0]:
                                continue
                            h = max(abs(x[var]) * 1e-7, 1e-3)
                            xp = x.copy()
                            xp[var] += h
                            rp = residual_fn(xp)
                            num = (rp[eq] - r[eq]) / h
                            ana = float(J[eq, var]) if not sparse.issparse(J) \
                                else float(J.tocsr()[eq, var])
                            rap = (ana / num) if num != 0 else float("inf")
                            logger.error(
                                "    dF[%d]/dx[%d]:  analitico=%+.4e  "
                                "numerico=%+.4e  rapporto=%.3f",
                                eq, var, ana, num, rap,
                            )
                except Exception as exc:
                    logger.error("  verifica Jacobiano non riuscita: %s", exc)

            if not step_accepted and self.residual_labeller is not None:
                # Chi non scende? Le prime componenti del residuo per modulo,
                # con il nome dell'equazione e quanto si e' mossa ciascuna
                # sotto il passo di Newton grezzo.
                ordine = np.argsort(-np.abs(r))[:6]
                logger.error("  Equazioni con residuo maggiore allo stallo:")
                for pos in ordine:
                    try:
                        nome = self.residual_labeller(int(pos))
                    except Exception:
                        nome = f"equazione {int(pos)}"
                    logger.error(
                        "    r[%4d] = %+.4e   dx = %+.4e   %s",
                        int(pos), float(r[pos]), float(dx_raw[pos]), nome,
                    )

            if not step_accepted:
                # No step length along the Newton direction reduces the
                # residual.  For a smooth F this is impossible as step_scale
                # -> 0, so reaching this branch means the residual is NOT a
                # continuous function of x at this point.  Typical causes:
                #  * a component mutates x during residual evaluation
                #    (e.g. a pressure-controlled compressor overwriting its
                #    outlet node pressure);
                #  * a non-differentiable switch (max()/min()/if) in a
                #    component model that is chattering on its boundary;
                #  * a Jacobian inconsistent with the residual.
                # Committing the step anyway (the old `ls_iter == 11`
                # behaviour) freezes the error and burns every remaining
                # iteration, so we abort immediately with a diagnosis.
                # The smallest step actually tried, in Pa: this is the number
                # that matters.  If it is tiny and the residual STILL did not
                # improve, the discontinuity is real.
                smallest_step_inf = float(np.max(np.abs(dx_capped))) * step_scale * 2.0
                logger.error(
                    "Newton-Raphson stalled at iteration %d: no step reduced "
                    "max|F| = %.3e (best line-search attempt %.3e, last %.3e) "
                    "after %d halvings. "
                    "|dx_raw|_inf = %.3e Pa, cap = %.3e Pa, "
                    "|dx_capped|_inf = %.3e Pa, smallest step tried = %.3e Pa. "
                    "The step direction was capped %s. "
                    "Since the step was driven down to ~%.3e Pa without "
                    "improving the residual, F is not continuous in x here — "
                    "check compressor / valve switching logic, composition "
                    "tracking on flow reversal, and any component that writes "
                    "node pressures during residual evaluation.",
                    n_iter, err, best_ls_err, new_err, _MAX_LS_ITER,
                    float(np.max(np.abs(dx_raw))), float(np.max(_max_dp)),
                    float(np.max(np.abs(dx_capped))), smallest_step_inf,
                    "(trust region active)"
                    if float(np.max(np.abs(dx_raw))) > float(np.max(_max_dp))
                    else "(no capping needed)",
                    smallest_step_inf,
                )
                return SolveResult(
                    x=best_x,
                    converged=False,
                    n_iter=n_iter,
                    message=(
                        f"Stalled at iteration {n_iter}: line search failed, "
                        f"max|F| = {err:.3e}. Residual appears discontinuous "
                        f"in x (non-smooth component model or state mutation "
                        f"during residual evaluation)."
                    ),
                )

            # Commit the accepted step
            x = x_new

            # --- Adattamento dell'under-relaxation (simmetrico, limitato) ---
            #
            # Si guarda come e' andato il passo appena accettato:
            #   * ls_halvings = quante volte la line search ha dovuto dimezzare
            #     (quanto ha faticato a partire da 1/current_urf);
            #   * accepted_ratio = new_err/err = quanto e' sceso il residuo.
            # e si sposta urf_eff dentro la banda [urf_floor, urf_ceil].
            #
            # STORIA: un vecchio adattivo era stato rimosso perche' derivava
            # monotono verso l'alto (saliva a ~25 su run_249 e si impiantava):
            # memoria illimitata, asimmetrico, e SOMMATO alla line search.
            # Questa versione corregge i tre difetti:
            #   1. banda LIMITATA [urf_floor, urf_ceil];
            #   2. simmetrica, con RIENTRO verso il pavimento appena recupera;
            #   3. adatta solo il PUNTO DI PARTENZA del passo (la traiettoria),
            #      lasciando alla line search la sicurezza per-iterazione.
            # Cosi' una rete "gentile" resta a urf_floor (passo pieno, veloce)
            # e una rete "dura" fa salire urf_eff da sola, ritrovando la
            # traiettoria smorzata che prima si otteneva solo alzando l'URF a
            # mano dal pannello.
            accepted_ratio = (new_err / err) if err > 0 else 0.0
            if ls_halvings == 0 and accepted_ratio < 0.5:
                # progresso sano gia' dal passo di partenza: allenta (piu'
                # aggressivo, verso il passo pieno di Newton)
                current_urf = max(urf_floor, current_urf / 1.5)
            elif ls_halvings >= 2 or accepted_ratio > 0.8:
                # ha faticato (molto backtracking o riduzione debole): smorza
                current_urf = min(urf_ceil, current_urf * 1.5)
            # zona intermedia: urf_eff invariato (isteresi, niente oscillazioni)


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
