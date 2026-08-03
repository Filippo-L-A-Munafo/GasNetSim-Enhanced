import numpy as np
import logging

from .base_problem import Problem
from .composition_tracker import CompositionTracker
from GasNetSim.components.utils.cuda_support import list_to_array

logger = logging.getLogger(__name__)


class PressureProblem(Problem):
    """
    Steady-state gas network problem with nodal pressures as unknowns.

    State vector x
    --------------
    Junction node pressures (Pa), in simulation-index order, excluding
    non-junction (reference / shortpipe-inlet) nodes whose pressures are fixed.

    Residual F(x)
    -------------
    Nodal flow balance at each junction node:
        F_i = f_target_i - sum_j Q_ij(p)
    where Q_ij is the flow from node i to node j computed from the current
    pressure state.

    Jacobian J(x)
    -------------
    ∂F/∂p assembled by Pipeline/Resistance/Compressor via
    Network.jacobian_matrix().

    Caching
    -------
    Each call to residual(x) or jacobian(x) triggers _compute(x), which does
    the full evaluation: property updates → jacobian_matrix() → composition
    tracking → residual assembly. Results are cached by x so that the second
    call (same x) is a free lookup. This matters for Newton-Raphson which
    calls jacobian then residual for the same x every iteration.
    """

    def __init__(
        self,
        network,
        use_cuda: bool = False,
        sparse_matrix: bool = False,
        tracking_method: str = "simple_mixing",
        time_step: int = 3600,
        run_composition_tracking: bool = True,
        warm_start: bool = False,
    ):
        self.network = network
        self.use_cuda = use_cuda
        self.sparse_matrix = sparse_matrix
        self.tracking_method = tracking_method
        self.time_step = time_step
        # When False, composition is frozen during Newton iterations and
        # initial_state skips the topological composition seeding. Used by
        # the staggered coupler, which owns composition updates.
        self.run_composition_tracking = run_composition_tracking
        self.warm_start = warm_start

        # Derived index lists (computed once from the fixed network topology)
        non_junction_ids = set(network.non_junction_nodes)
        n = network.get_simulation_node_count()
        self._junction_sim_indices = [
            i for i in range(n)
            if network.simulation_node_index_to_node_id(i) not in non_junction_ids
        ]
        self._non_junction_ids = non_junction_ids

        # Internal state updated during iterations
        self._f_target = None
        self._t = None

        # Batch tracking snapshot — taken once so CompositionTracker can
        # restore pipeline history before each Newton iteration.
        if tracking_method == "batch_tracking" and network.pipelines is not None:
            cached_batch_information = {
                i: (p.batch_location_history.copy(), p.composition_history.copy())
                for i, p in network.pipelines.items()
            }
        else:
            cached_batch_information = {}

        self._composition_tracker = CompositionTracker(
            network=network,
            tracking_method=tracking_method,
            time_step=time_step,
            cached_batch_information=cached_batch_information,
        )

        # Cache for last _compute(x) result
        self._cache_key: bytes = b""
        self._cache_r: np.ndarray = None
        self._cache_J = None

        # Connection → gas_mixture assignment held fixed for the whole Newton
        # solve when network.freeze_mixture_during_newton is set.
        self._frozen_mixtures: dict | None = None

        # ------------------------------------------------------------------
        # Formulazione aumentata per i compressori
        # ------------------------------------------------------------------
        # Un compressore interno non puo' essere descritto dalle sole pressioni
        # nodali: la sua portata non e' determinata da una relazione
        # pressione-portata come in una condotta, ma dal vincolo di controllo
        # (rapporto o pressione di mandata). Va quindi trattato come un
        # sistema algebrico-differenziale:
        #
        #   incognite  : pressioni di giunzione  +  UNA PORTATA PER COMPRESSORE
        #   equazioni  : bilanci nodali          +  UN VINCOLO PER COMPRESSORE
        #
        # con vincolo   p_out - ratio * p_in = 0        (control_mode="ratio")
        #        oppure p_out - p_out_target = 0        (control_mode="pressure")
        #
        # Cosi il sistema resta quadrato, la portata attraversa realmente la
        # stazione (la massa si conserva) e nessuna pressione viene piu'
        # sovrascritta durante la valutazione del residuo.
        #
        # Nota: con questa formulazione il nodo di uscita NON deve essere un
        # reference. Renderlo reference (il vecchio "PCM") lo trasforma in una
        # sorgente a portata libera in mezzo alla rete, che e' un modello mal
        # posto su uno snodo di transito, e per giunta rende degenere la riga
        # di vincolo.
        comps = getattr(network, "compressors", None)
        self._compressor_ids = sorted(comps.keys()) if comps else []
        self._n_c = len(self._compressor_ids)
        self._augment = self._n_c > 0

        # Scala per rendere il residuo di vincolo (in Pa) commensurabile con i
        # bilanci nodali (in Sm3/s): senza questo il vincolo dominerebbe
        # max|F| di sei ordini di grandezza e la convergenza sarebbe
        # irraggiungibile. Si normalizza sulla pressione di riferimento e si
        # riporta alla scala delle portate.
        # Le scale devono essere INVARIANTI: si ricavano dai dati di progetto
        # (pressione dei reference, domande nodali dai CSV), mai dallo stato
        # corrente della rete. Prendendole dallo stato, dopo un solve fallito
        # si leggono pressioni prive di senso — nei log comparivano 142 bar su
        # una rete a 70 — e la riga di vincolo veniva riscalata diversamente a
        # ogni tentativo: il sistema cambiava fra un tentativo e l'altro.
        self._p_scale = 1.0
        self._q_scale = 1.0
        if self._augment:
            # Il flag va alzato QUI, non in initial_state(): quest'ultimo
            # chiama update_compressor_parameters() prima di poterlo
            # impostare, e quella chiamata percorrerebbe ancora il ramo
            # vecchio, ricalcolando flow_rate con l'euristica della media e
            # (in modalita' rapporto) sovrascrivendo la pressione di mandata.
            network.compressor_flow_from_state = True

            riferimenti = set(getattr(network, "reference_nodes", []) or [])

            # p_scale: pressione dei nodi reference (dato di progetto, fisso),
            # con fallback sul target dei compressori.
            p_ref = [
                float(network.nodes[n].pressure)
                for n in riferimenti
                if getattr(network.nodes.get(n), "pressure", None)
            ]
            if not p_ref:
                p_ref = [
                    float(c.p_out_target)
                    for c in (network.compressors or {}).values()
                    if getattr(c, "p_out_target", None)
                ]
            self._p_scale = max(p_ref) if p_ref else 7e6

            # q_scale: somma delle domande nodali, ESCLUSI i reference — la
            # loro portata e' calcolata dal solver, quindi vale 0 al primo
            # giro e ~ -somma_domande dopo, facendo raddoppiare la scala.
            portate = [
                abs(float(n.volumetric_flow))
                for node_id, n in network.nodes.items()
                if node_id not in riferimenti
                and getattr(n, "volumetric_flow", None)
            ]
            self._q_scale = max(sum(portate), 1.0)

        # Diagnostica esplicita: senza questa riga non c'e' modo di sapere dal
        # log se la formulazione aumentata sia davvero attiva, quale vincolo
        # stia applicando e con che scala. Tutte informazioni che finora
        # abbiamo dovuto dedurre dai sintomi.
        n_j = len(self._junction_sim_indices)
        if self._augment:
            logger.info(
                "Formulazione AUMENTATA attiva: %d giunzioni + %d portate "
                "compressore = sistema %dx%d  (p_scale=%.3e Pa, "
                "q_scale=%.3e Sm3/s)",
                n_j, self._n_c, n_j + self._n_c, n_j + self._n_c,
                self._p_scale, self._q_scale,
            )
            for cid in self._compressor_ids:
                c = network.compressors[cid]
                logger.info(
                    "  compressore %s: %s->%s  control_mode=%r  "
                    "p_out_target=%r  ratio=%.6f",
                    cid, c.inlet_index, c.outlet_index,
                    getattr(c, "control_mode", None),
                    getattr(c, "p_out_target", None),
                    float(getattr(c, "compression_ratio", 0.0)),
                )
        else:
            logger.info(
                "Formulazione STANDARD (nessun compressore): sistema %dx%d",
                n_j, n_j,
            )

    # ------------------------------------------------------------------
    # Problem interface
    # ------------------------------------------------------------------

    def initial_state(self) -> np.ndarray:
        """
        Initialise network parameters and return junction pressures as x0.
        """
        network = self.network
        init_f, init_p, init_t = network.newton_raphson_initialization(warm_start=self.warm_start)

        self._t = list_to_array(init_t, use_cuda=self.use_cuda)
        self._f_target = list_to_array(init_f, use_cuda=self.use_cuda)

        p_full = list_to_array(init_p, use_cuda=self.use_cuda)
        network.update_node_parameters(
            pressure=p_full, flow=self._f_target, temperature=self._t
        )
        if network.pipelines is not None:
            network.update_pipeline_parameters()
        if network.resistances is not None:
            network.update_resistance_parameters()
        if network.compressors is not None:
            network.update_compressor_parameters()

        # Safe to run here: this is outside the Newton loop, so it cannot make
        # the residual discontinuous.  It self-disables via
        # network.run_composition_initialization, which the staggered coupler
        # clears during the pressure step so the relaxed composition survives.
        if self.run_composition_tracking:
            network.composition_initialization(
                tracking_method=self.tracking_method,
                time_step=self.time_step,
                cached_batch_information=self._composition_tracker.cached_batch_information,
            )

        # Snapshot the connection→gas_mixture assignment decided here, at the
        # start of the Newton solve.  _compute() will restore it after every
        # update_*_parameters() call so that the assignment cannot flip
        # mid-solve.  See _mixture_frozen() for why.
        if self._mixture_frozen():
            self._frozen_mixtures = {
                cid: conn.gas_mixture
                for cid, conn in network.connections.items()
            }

        x = self._full_p_to_x(p_full)

        if self._augment:
            # (il flag compressor_flow_from_state e' gia' alzato in __init__)
            x = np.concatenate([x, self._initial_compressor_flows()])

        return x

    # ------------------------------------------------------------------
    # Compressori: helper della formulazione aumentata
    # ------------------------------------------------------------------

    def _initial_compressor_flows(self) -> np.ndarray:
        """Stima iniziale della portata di ciascun compressore.

        Si usa la portata che le condotte di monte gia' convogliano verso il
        nodo di ingresso: e' la quantita' che fisicamente deve attraversare la
        stazione, quindi un punto di partenza molto migliore di zero (che
        renderebbe la rete di valle scollegata alla prima iterazione).
        """
        network = self.network
        q0 = np.zeros(self._n_c)
        pipelines = list(network.pipelines.values()) if network.pipelines else []

        for k, cid in enumerate(self._compressor_ids):
            comp = network.compressors[cid]
            try:
                tot_in, _, tot_out, _ = comp.calculate_incoming_flows_and_derivatives(
                    pipelines
                )
                stima = 0.5 * (abs(tot_in) + abs(tot_out))
            except Exception:
                stima = 0.0
            if not np.isfinite(stima) or stima == 0.0:
                stima = self._q_scale * 0.01
            q0[k] = stima
        logger.info(
            "  portate iniziali compressori: %s Sm3/s",
            ", ".join(f"{v:.1f}" for v in q0),
        )
        return q0

    def describe_residual(self, i: int) -> str:
        """Nome leggibile della i-esima equazione del sistema.

        Serve al solver per dire QUALE equazione non scende quando stalla,
        invece di lasciarlo dedurre dai sintomi.
        """
        n_j = len(self._junction_sim_indices)
        if i < n_j:
            sim_idx = self._junction_sim_indices[i]
            node_id = self.network.simulation_node_index_to_node_id(sim_idx)
            node = self.network.nodes[node_id]
            p = node.pressure
            return (
                f"bilancio nodo {node_id} "
                f"(p={p/1e5:.3f} bar)" if p else f"bilancio nodo {node_id}"
            )
        k = i - n_j
        if k < len(self._compressor_ids):
            cid = self._compressor_ids[k]
            c = self.network.compressors[cid]
            modo = getattr(c, "control_mode", "ratio")
            return (
                f"VINCOLO compressore {cid} ({c.inlet_index}->{c.outlet_index}, "
                f"modo={modo})"
            )
        return f"equazione {i}"

    def _junction_row(self, node_id):
        """Indice di riga/colonna di un nodo nel sistema ridotto, o None se il
        nodo e' un reference (escluso dalle incognite)."""
        if node_id in self._non_junction_ids:
            return None
        sim_idx = self.network.node_id_to_simulation_node_index(node_id)
        try:
            return self._junction_sim_indices.index(sim_idx)
        except ValueError:
            return None

    def _augment_jacobian(self, j_mat):
        """Borda il Jacobiano delle pressioni con i blocchi dei compressori.

        Struttura del sistema (n_j pressioni, n_c portate):

            | dBil/dp    dBil/dq |   | dp |     | residuo bilanci |
            |                    | * |    |  =  |                 |
            | dVinc/dp   dVinc/dq|   | dq |     | residuo vincoli |

        * dBil/dq : il bilancio del nodo di monte contiene -q, quello di valle
          +q, e il residuo e' (target - flusso), quindi le derivate valgono
          rispettivamente +1 e -1.
        * dVinc/dp : per il rapporto,  d(p_out - r*p_in)/dp_out = 1,
          d/dp_in = -r; per la pressione di mandata solo d/dp_out = 1.
          Entrambe riscalate come il residuo di vincolo.
        * dVinc/dq : nulla, il vincolo non dipende dalla portata.

        Il blocco dBil/dq ha rango pieno finche' ogni compressore tocca almeno
        una giunzione, quindi il sistema bordato resta non singolare.
        """
        network = self.network
        j = np.asarray(
            j_mat.todense() if hasattr(j_mat, "todense") else j_mat,
            dtype=float,
        )
        n_j = j.shape[0]
        n_c = self._n_c
        fattore = self._q_scale / self._p_scale

        out = np.zeros((n_j + n_c, n_j + n_c), dtype=float)
        out[:n_j, :n_j] = j

        for k, cid in enumerate(self._compressor_ids):
            comp = network.compressors[cid]
            col = n_j + k

            r_in = self._junction_row(comp.inlet_index)
            r_out = self._junction_row(comp.outlet_index)

            # ATTENZIONE AL SEGNO. Il solver risolve J*dx = r e poi fa
            # x += dx, quindi perche' sia Newton (dx = -(dF/dx)^-1 * F) la
            # matrice deve essere J = -dF/dx, NON +dF/dx. Il ramo delle
            # condotte lo fa esplicitamente:
            #     jacobian_mat[i][i] += -flow_rate_first_order_derivative(...)
            # I blocchi bordati devono seguire la stessa convenzione: qui si
            # scrive percio' l'OPPOSTO delle derivate calcolate a mano.
            #
            # Con il segno diretto Newton muoveva la portata del compressore
            # nel verso sbagliato: il residuo cresceva e nessuna lunghezza di
            # passo poteva ridurlo — lo stallo osservato dai run 231 in poi.

            # dBilancio/dq:  dr_in/dq = +1,  dr_out/dq = -1  ->  cambiati di segno
            if r_in is not None:
                out[r_in, col] = -1.0
            if r_out is not None:
                out[r_out, col] = +1.0

            # dVincolo/dp, anch'esse cambiate di segno
            riga = n_j + k
            modo = getattr(comp, "control_mode", "ratio")
            if modo == "pressure" and getattr(comp, "p_out_target", None) is not None:
                # vincolo: p_out - target   ->  d/dp_out = +fattore
                if r_out is not None:
                    out[riga, r_out] = -fattore
            else:
                # vincolo: p_out - ratio*p_in
                ratio = float(comp.compression_ratio)
                if r_out is not None:
                    out[riga, r_out] = -fattore
                if r_in is not None:
                    out[riga, r_in] = +ratio * fattore

        return out

    def _split_state(self, x: np.ndarray):
        """Separa x in (pressioni di giunzione, portate dei compressori)."""
        if not self._augment:
            return x, np.empty(0)
        n_j = len(self._junction_sim_indices)
        return x[:n_j], x[n_j:]

    def _constraint_residuals(self, q_c: np.ndarray) -> np.ndarray:
        """Residuo del vincolo di controllo, riscalato come una portata.

        ratio    :  p_out - ratio * p_in
        pressure :  p_out - p_out_target

        diviso per p_scale e moltiplicato per q_scale, cosi convive nello
        stesso max|F| dei bilanci nodali.
        """
        network = self.network
        r = np.zeros(self._n_c)
        fattore = self._q_scale / self._p_scale

        for k, cid in enumerate(self._compressor_ids):
            comp = network.compressors[cid]
            p_in = float(network.nodes[comp.inlet_index].pressure or 0.0)
            p_out = float(network.nodes[comp.outlet_index].pressure or 0.0)

            if getattr(comp, "control_mode", "ratio") == "pressure" \
                    and getattr(comp, "p_out_target", None) is not None:
                r[k] = (p_out - float(comp.p_out_target)) * fattore
            else:
                r[k] = (p_out - float(comp.compression_ratio) * p_in) * fattore
        return r

    def _mixture_frozen(self) -> bool:
        """True when the pipe/compressor gas-mixture assignment must be held
        fixed for the whole Newton solve.

        Pipeline.update_gas_mixture() (and its Resistance / ShortPipe /
        Compressor twins) pick the edge's gas mixture from the SIGN of the
        current flow rate:

            self.gas_mixture = (self.inlet if flow_rate >= 0
                                else self.outlet).gas_mixture

        update_pipeline_parameters() re-runs that choice on every single
        residual evaluation.  When compositions differ across the network, a
        pipe sitting near zero flow flips its mixture as soon as the iterate
        nudges it across the boundary — and specific_gravity, compressibility
        and density jump with it, so the flow coefficient and hence the
        residual jump too.  F is then DISCONTINUOUS in x: no step length can
        reduce it, and the line search correctly reports a stall.

        The staggered coupler already assumes this cannot happen — it labels
        its inner Newton solve "Pressure step (composition frozen)" — but the
        flag it sets (network.run_composition_initialization) governs
        network.composition_initialization(), not this assignment, so nothing
        was actually frozen.  We freeze on the coupler's behalf and let it own
        the direction/composition refresh between outer iterations, which is
        exactly the staggered design.
        """
        return bool(
            getattr(self.network, "freeze_mixture_during_newton", False)
        )

    def residual(self, x: np.ndarray) -> np.ndarray:
        # I tentativi della line search NON devono aggiornare _f_target:
        # vedi la nota in _compute().
        r, _ = self._compute(x, refresh_target=False)
        return r

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        # Il solver chiama jacobian(x) una sola volta per iterazione, sul
        # punto ACCETTATO: e' li' che ha senso riallineare _f_target.
        _, J = self._compute(x, refresh_target=True)
        return J

    def unpack(self, x: np.ndarray) -> None:
        """Write converged junction pressures back and update derived quantities."""
        network = self.network

        x_p, q_c = self._split_state(x)

        # Update all node pressures
        for node_id in network.nodes:
            if node_id not in self._non_junction_ids:
                sim_idx = network.node_id_to_simulation_node_index(node_id)
                k = self._junction_sim_indices.index(sim_idx)
                network.nodes[node_id].pressure = x_p[k]

        # Portate dei compressori: sono incognite risolte, si riscrivono
        # sull'oggetto e si aggiorna il rapporto effettivo raggiunto (utile
        # per il calcolo della potenza assorbita).
        if self._augment:
            for k, cid in enumerate(self._compressor_ids):
                comp = network.compressors[cid]
                comp.flow_rate = float(q_c[k])
                p_in = float(network.nodes[comp.inlet_index].pressure or 0.0)
                p_out = float(network.nodes[comp.outlet_index].pressure or 0.0)
                # Il rapporto RAGGIUNTO va tenuto separato da quello di
                # CONSEGNA: compression_ratio e' il dato di controllo che il
                # vincolo usa a ogni iterazione, e sovrascriverlo qui lo
                # farebbe derivare a ogni iterazione esterna dello staggered
                # (il modello cambierebbe sotto i piedi del solver).
                comp.achieved_ratio = (p_out / p_in) if p_in > 0 else None
                std = getattr(comp.gas_mixture, "standard_density", 0.8)
                comp.mass_flow_rate = comp.flow_rate * std

        # Recompute flow matrix to get supply/demand at non-junction nodes
        _, f_mat = network.jacobian_matrix(
            use_cuda=self.use_cuda, sparse_matrix=self.sparse_matrix
        )
        if self.use_cuda:
            import cupy as cp
            nodal_flow = cp.sum(f_mat, axis=1)
        else:
            nodal_flow = np.sum(f_mat, axis=1)

        for node_id in self._non_junction_ids:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
            network.nodes[node_id].volumetric_flow = nodal_flow[sim_idx]
            node = network.nodes[node_id]
            if node.flow_type == "volumetric":
                node.convert_volumetric_to_energy_flow()
            elif node.flow_type == "energy":
                node.convert_energy_to_volumetric_flow()

        # Finalise gas composition (promote tmp → permanent)
        for node in network.nodes.values():
            node.gas_mixture.eos_composition = node.gas_mixture.eos_composition_tmp
            node.gas_mixture.convert_eos_composition_to_dictionary()

        network.update_connection_flow_rate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self, x: np.ndarray, refresh_target: bool = True):
        """
        Core evaluation: update network state for junction pressures x, then
        return (residual, jacobian). Results are cached by x.tobytes() so
        that the second call with the same x is free.
        """
        # La chiave include refresh_target: il punto accettato viene gia'
        # valutato durante la line search con refresh=False, e senza questa
        # distinzione la chiamata a jacobian() dell'iterazione successiva
        # ripescherebbe quella voce dalla cache saltando l'aggiornamento di
        # _f_target. Costa una rivalutazione per iterazione, non di piu'.
        cache_key = x.tobytes() + (b"\x01" if refresh_target else b"\x00")
        if cache_key == self._cache_key:
            return self._cache_r, self._cache_J

        network = self.network

        x_p, q_c = self._split_state(x)

        # 1. Write junction pressures into network nodes and update all
        #    gas-mixture properties (compressibility, viscosity, etc.)
        p_full = self._x_to_full_p(x_p)
        network.update_node_parameters(
            pressure=p_full, flow=self._f_target, temperature=self._t
        )

        # Le portate dei compressori vengono dallo stato, PRIMA di costruire la
        # matrice dei flussi: cosi jacobian_matrix() le include gia'
        # correttamente nei bilanci dei due nodi, con segni opposti, e la massa
        # attraversa davvero la stazione.
        if self._augment:
            for k, cid in enumerate(self._compressor_ids):
                network.compressors[cid].flow_rate = float(q_c[k])
        # Refresh flow rates at the current pressures *before* re-assigning
        # gas-mixture nodes (inlet vs outlet).  For pipelines whose pressure
        # drive nearly equals the slope correction (p1²−p2²−sc ≈ 0), the
        # inlet/outlet choice depends on the sign of that tiny residual.
        # Using stale flow rates from the previous iteration can flip the sign,
        # leading to a self-sustaining oscillation.  This one-line refresh
        # mirrors the legacy loop, which called update_connection_flow_rate()
        # at the updated pressures before calling update_pipeline_parameters().
        network.update_connection_flow_rate()
        if network.pipelines is not None:
            network.update_pipeline_parameters()
        if network.resistances is not None:
            network.update_resistance_parameters()
        if network.compressors is not None:
            network.update_compressor_parameters()

        # Undo any flow-sign-driven gas-mixture reassignment performed by the
        # update_*_parameters() calls above.  Holding this fixed is what makes
        # the residual a CONTINUOUS function of x within a Newton solve.
        if self._frozen_mixtures is not None:
            for cid, gas_mixture in self._frozen_mixtures.items():
                connection = network.connections.get(cid)
                if connection is not None:
                    connection.gas_mixture = gas_mixture

        for connection in network.connections.values():
            connection.inlet = network.nodes[connection.inlet_index]
            connection.outlet = network.nodes[connection.outlet_index]

        # 2. Assemble Jacobian and flow matrix
        j_mat, f_mat = network.jacobian_matrix(
            use_cuda=self.use_cuda, sparse_matrix=self.sparse_matrix
        )

        # 3. Composition tracking (single-pass, cached topological order)
        network.update_connection_flow_rate()
        # Skipped while frozen: the tracker rebuilds its topological order from
        # flow signs and re-propagates compositions, so running it inside the
        # Newton loop reintroduces exactly the discontinuity we just removed.
        # The staggered coupler performs the composition update between outer
        # iterations instead.
        if self.run_composition_tracking and self._frozen_mixtures is None:
            self._composition_tracker.update()

        # 4. Compute nodal flow balance residual using the target flow from the
        #    start of the iteration. The legacy solver only refreshed
        #    volumetric targets after forming the residual so the next Newton
        #    step sees the updated demand/supply state.
        if self.use_cuda:
            import cupy as cp
            nodal_flow = cp.sum(f_mat, axis=1)
        else:
            nodal_flow = np.sum(f_mat, axis=1)

        delta_flow = self._f_target - nodal_flow
        residual = list_to_array(
            [
                delta_flow[i]
                for i in range(len(delta_flow))
                if network.simulation_node_index_to_node_id(i)
                not in self._non_junction_ids
            ],
            use_cuda=self.use_cuda,
        )

        # 4b. Sistema aumentato: si aggiungono le righe di vincolo dei
        #     compressori e le bordature del Jacobiano.
        if self._augment:
            residual = np.concatenate(
                [np.asarray(residual), self._constraint_residuals(q_c)]
            )
            j_mat = self._augment_jacobian(j_mat)

        # Riallineamento di _f_target alle proprieta' del gas allo stato
        # corrente — SOLO sul punto accettato, mai sui tentativi della line
        # search.
        #
        # _f_target e' il termine noto del residuo (r = f_target - flusso).
        # Aggiornarlo a ogni valutazione significa che ogni tentativo della
        # line search viene confrontato con un BERSAGLIO DIVERSO: il test
        # "new_err < err" mette a paragone residui non omogenei. Il passo
        # intero appare sempre peggiore e solo passi molto piccoli sembrano
        # migliorare, perche' spostano meno il bersaglio. Nei log si vedeva
        # "step halved 3 times" a ogni iterazione e una riduzione del residuo
        # del 5% per volta: non era smorzamento, era il bersaglio che si
        # muoveva sotto la line search.
        #
        # Il solver chiama jacobian(x) una volta per iterazione sul punto
        # accettato e residual(x) molte volte durante la ricerca: si aggiorna
        # quindi solo nel primo caso (refresh_target=True).
        if refresh_target:
            for node in network.nodes.values():
                if node.flow_type == "volumetric":
                    node.convert_volumetric_to_energy_flow()
                elif node.flow_type == "energy":
                    node.convert_energy_to_volumetric_flow()

            self._f_target = list_to_array(
                [
                    node.volumetric_flow if node.volumetric_flow is not None else 0.0
                    for node in network.nodes.values()
                ],
                use_cuda=self.use_cuda,
            )

        self._cache_key = cache_key
        self._cache_r = residual
        self._cache_J = j_mat

        return residual, j_mat

    def _full_p_to_x(self, p_full) -> np.ndarray:
        """Extract junction pressures from the full pressure vector."""
        return np.array([p_full[i] for i in self._junction_sim_indices])

    def _x_to_full_p(self, x: np.ndarray) -> np.ndarray:
        """Build the full pressure vector from junction pressures x."""
        network = self.network
        n = network.get_simulation_node_count()
        p_full = np.zeros(n)
        for k, sim_idx in enumerate(self._junction_sim_indices):
            p_full[sim_idx] = x[k]
        # Fixed pressures for non-junction nodes
        for node_id in self._non_junction_ids:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
            p_full[sim_idx] = network.nodes[node_id].pressure
        return p_full
