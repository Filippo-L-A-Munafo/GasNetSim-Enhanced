# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## UNRELEASED

### Changed
- Add modular `Problem`/`Solver` interfaces for simulation internals
- Add an optional `staggered` coupling strategy for composition-sensitive cases (`coupling_strategy="staggered"`)

### Fixed
- `components/pipeline.py`: `flow_rate_first_order_derivative()` sollevava `inf` quando `p1 == p2`
  (es. inizializzazione quasi-piatta su reti grandi) perché calcolava `(0.0)**(-0.5)`. Fix: clampare
  `delta = |p1²-p2²-slope_corr|` a `_DELTA_P2_MIN = 1.0 Pa²`. Corretto anche
  `Resistance.determine_flow_direction()` per restituire `0` invece di `ValueError` quando `p1 == p2`.
  *(commit `61bd626`)*
- `simulation/solvers/newton_raphson.py`: aggiunto solver sparse (`scipy.sparse.linalg.spsolve`) per
  reti con n ≥ 500 nodi, evitando `LinAlgError: SVD did not converge` con matrici dense grandi.
  *(commit `caf34d3`)*
- `simulation/solvers/newton_raphson.py`: aggiunto doppio fallback in `_linear_solve()` — regularizzazione
  Tikhonov diagonale poi `lsmr` — per gestire Jacobiani mal condizionati che restituiscono NaN da `spsolve`.
  *(commit `ace6334`)*
- `simulation/solvers/newton_raphson.py`: divergenza Newton-Raphson su reti con carico fortemente
  asimmetrico (es. nodo export nazionale): step esplosivo alla prima iterazione → pressioni a -2×10⁹ bar.
  Fix: step-cap `_MAX_STEP_FRAC=0.3` (max 30% variazione per passo), pressure floor `_P_FLOOR_PA=1.0 Pa`,
  e best-iterate tracking (restituisce il punto a residuo minimo in caso di mancata convergenza).
  *(commit `0edebe3`)*

### Added — non ancora committato (working tree, scoperto durante audit del 2026-08-03)
- **Formulazione aumentata per i compressori** (`simulation/formulations/pressure_problem.py`,
  `simulation/strategies/direct.py`, `components/network.py`): il vettore di stato di Newton viene
  esteso con una portata per compressore oltre alle pressioni nodali; nuove equazioni di vincolo
  bordate (`p_out - ratio*p_in = 0` in modo `ratio`, `p_out - p_out_target = 0` in modo `pressure`)
  con Jacobiano bordato dedicato (`_augment_jacobian`). Sostituisce il precedente calcolo euristico
  di `Compressor.update_flow_rate()` quando `network.compressor_flow_from_state = True`.
- `components/compressor.py`: nuovi parametri `control_mode` (`"ratio"` | `"pressure"`) e
  `p_out_target` sul costruttore; `update_flow_rate()` ora distingue i casi in cui inlet/outlet
  sono nodi reference (bilancio non vincolato nel sistema assemblato) dal caso standard.
- `components/utils/create_network.py` (`read_compressors`): le colonne CSV `control_mode` e
  `p_out_target` erano lette ma mai passate al costruttore `Compressor(...)` — bug che rendeva il
  controllo a pressione di mandata non configurabile via CSV. Ora vengono estratte, validate e
  propagate, con fallback a `"ratio"` e warning se `p_out_target` manca o non è un float positivo.
- `components/network.py` (`_propagate_through_compressors`, nuovo): le strategie di
  inizializzazione pressione camminavano solo su `pipelines`, lasciando la sotto-rete a valle dei
  compressori inizializzata come scollegata. Propaga ora la stima di pressione anche attraverso gli
  archi compressore (fino a 8 passate smorzate 0.7/0.3, soglia di stop 1000 Pa).
- `components/network.py`: supporto **warm-start** (`newton_raphson_initialization(warm_start=...)`,
  `simulation(warm_start=...)`) per riusare la pressione nodale corrente invece di ricalcolarla da
  zero; nuovo metodo `update_demands(csv_file)` per rileggere la domanda da un `sf_nodes.csv`
  aggiornato (utile per run sequenziali/orarie).
- `components/gas_mixture/typical_mixture_composition.py`: aggiunte le composizioni `SNAM_MAZARA_GAS`
  (gas algerino via Tunisia) e `SNAM_GELA_GAS` (gas libico, Greenstream) a `COMMON_GAS_COMPOSITIONS`.
- `simulation/strategies/staggered.py`: meccanismo di relax adattivo alternativo ad Aitken (attivo
  solo quando `aitken_acceleration=False`) — dimezza `relax_current` (floor 0.05, mai risalito) se il
  delta di composizione risale di oltre il 5% da un'iterazione all'altra, per smorzare cicli-limite
  da inversione di flusso su una tubazione.

### Fixed — non ancora committato (working tree, scoperto durante audit del 2026-08-03)
- `simulation/formulations/pressure_problem.py`: la scelta di gas-mixture per-connessione (basata sul
  segno del flusso) veniva ricalcolata a ogni valutazione del residuo durante un singolo solve di
  Newton, rendendo il residuo discontinuo vicino a flusso zero e bloccando la line search. Fix:
  "congelamento" della miscela (`_mixture_frozen`) per l'intera risoluzione quando
  `network.freeze_mixture_during_newton` è impostato dallo staggered coupler.
- `simulation/formulations/pressure_problem.py`: `_f_target` veniva ricalcolato a ogni valutazione
  del residuo, anche durante i tentativi della line search — ogni step veniva quindi confrontato con
  un bersaglio diverso, causando falsi "step halved" ripetuti. Fix: `_f_target` si aggiorna solo sul
  punto accettato (`refresh_target=True` solo da `jacobian()`, mai da `residual()`).
- `components/utils/pipeline_function/flow_rate.py` (`calculate_stable_flow_rate`): la variazione
  percentuale di portata generava `nan` (RuntimeWarning) quando la portata precedente era zero. Fix:
  guardia esplicita (0.0 se anche la portata corrente è zero, altrimenti `inf` per forzare un'altra
  iterazione).
- `components/network.py` (`jacobian_matrix`, ramo pipeline/resistenza standard): rimossa una riga
  morta che calcolava `(0.0) ** (-0.5)` inutilizzata, fonte di un `RuntimeWarning` rumoroso.

### Note tecniche da chiudere prima del commit (dall'audit del 2026-08-03)
- [ ] `components/utils/utils.py.bak` (626 righe) è finito nello staging git: è un file di backup,
  va rimosso dall'indice (`git restore --staged`) e cancellato, non va committato.
- [ ] `components/network.py` (`newton_raphson_initialization`): i due rami `if warm_start` / `else`
  calcolano lo stesso identico `pressure_init` — probabile refactor incompleto, il ramo `else` è
  dead code (il valore viene comunque sovrascritto subito dopo).
- [ ] `components/network.py` (`update_demands`): nuovo metodo pubblico non richiamato da nessun
  altro file di questo diff — verificare se orfano o usato da script esterni non ancora committati.
- [ ] `network.compressor_flow_from_state` è un attributo dinamico (mai definito in
  `Network.__init__`, letto con `getattr(..., False)`) impostato "da fuori" da `PressureProblem` —
  accoppiamento implicito e fragile fra le due classi, da formalizzare nell'interfaccia di `Network`.
- [ ] Costanti magiche non configurabili aggiunte in questa serie di modifiche: `num_passes=8` e
  damping `0.7/0.3` in `_propagate_through_compressors`; `p_scale` fallback `7e6 Pa` in
  `pressure_problem.py`; `flow_step_cap = q_scale * 0.5` in `direct.py`; floor `0.05`/soglia `1.05`
  in `staggered.py`. Da valutare se esporle come parametri prima di usare il motore su reti diverse
  da quella siciliana di test.
- [ ] `components/pipeline.py`: un `print(...)` di debug in `determine_flow_direction` è stato
  commentato ma non rimosso.
- [ ] Nessuna delle modifiche sopra ha test dedicati in questo audit — in particolare la formulazione
  aumentata dei compressori (467 righe, la parte più a rischio) non risulta coperta da nuovi test.


## v0.2.2

### Fixed

- Fixed node and pipeline indexing
- Added an if-condition for `no_mixing` in `gas_composition_tracking()`

## v0.2.1

### Changed

- Use caches for gas mixture properties calculation to accelerate imports
- Update tutorials/02_gas_mixture_properties.ipynb to show the import speed improvement

## v0.2.0 - 2025-12-27

### Added

- Tutorials implemented using Binder
- Function to create network using files
- Use `networkx` for topological sort of pipelines and nodes
- Different reference temperature values for gas mixture heating value calculation (15 or 25 degrees Celsius)
- Reference pressure, temperature and composition for gas mixture initialization
- pre-commit hooks for automatically updating requirements.txt
- Compressor model and corresponding examples and tests

### Changed

- Changed the packaging tool from `setuptools` / `poetry` to `uv`
- Changed reference temperature for GasMixture class (wrt. density and combustion properties)
- Changed run_time_series() to use profiles instead of profile file path
- Improved results saving of time series simulations (support different output file formats)
- Simulation solver uses individual indexing system for nodes and pipelines

### Deprecated

- Renamed create_network_from_csv() to create_network_from_folder() for clarity and add FutureWarning

## v0.1.0

### Added

- Added support for parallel pipelines
- Test for pipeline outlet temperature calculation function
- Test and validation with original [C++ implementation](https://github.com/usnistgov/AGA8/tree/master/AGA8CODE/C) of
  GERG-2008 EOS
- Added support of heating value calculation for GERG2008 gas mixture class
- Tests and validations for the implementation of the heating values calculation
- Tests and validations of gas mixture properties calculation using the _thermo_ package and the GERG-2008 EOS

### Changed

- Change `Cantera` dependency to `v3.0.0`
- Fixed pipeline outlet temperature calculation
- Fixed GERG-2008 implementation
- Set default gas mixture EOS for simulation to GERG-2008

### Removed

- Local adapted implementation of `thermo` package