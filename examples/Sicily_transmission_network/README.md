# Sicilian Regional Natural Gas Transmission Network (340 Nodes, 350 Pipelines)

This directory provides the complete benchmark dataset and standalone simulation workflow for the high-pressure natural gas transmission grid of Sicily (~2,316 km of pipelines), enriched with green hydrogen blending.

This benchmark accompanies the research manuscript:
> **"Techno-Economic and Hydraulic Feasibility of Green Hydrogen Blending in Regional Transmission Networks: A 3D Digital Twin Analysis of the Sicilian Grid"**  
> *Submitted to Applied Energy (Elsevier)*

---

## 1. Overview and Key Dataset Features

* **Network Scale:** 340 nodes, 350 pipelines, spanning the entire island of Sicily;
* **Diameters and Lengths:** Nominal pipe diameters from 152.4 mm (6") up to 1,219.2 mm (48"); pipe lengths between 16.7 m and 123.6 km;
* **Orography & 3D Altimetry:** Ground surface elevations georeferenced from the **Copernicus DEM** (smoothed at 250 m), ranging from sea level (1.0 m a.s.l.) to the central mountain passes near Enna (1,280.0 m a.s.l.);
* **Gas Composition:** Algerian natural gas baseline enriched with **13.92 vol% green hydrogen**, corresponding to the binding regulatory limit for relative density ($d \ge 0.555$ pursuant to Italian DM 24/11/1984 and EN 16726);
* **Primary Import Terminal:** Mazara del Vallo (Node 8118, Transmed pipeline from Algeria) operating at 75.0 barg (76.013 bar abs);
* **Offtake Demands:** Reconstructed from hourly sectoral delivery profiles (Terna and Snam 2025/2026 data) covering residential, industrial, and thermoelectric power demands (~1.364 MSm³/h at h12).

---

## 2. Multi-Solver Cross-Validation Benchmark

This configuration has been cross-validated node-by-node against two independent, external simulation engines:
1. **SHIMMER++** (Politecnico di Torino — DISMA/DENERG, C++ linearized 1D Navier-Stokes solver);
2. **pandapipes** (Fraunhofer IEE / University of Kassel, Python graph-based Newton-Raphson solver).

### Validation Statistics Across All 340 Nodes (15 Nov 2025, h12 — 13.92% H₂)

| Statistical Metric | SHIMMER++ vs GasNetSim | pandapipes vs GasNetSim | SHIMMER++ vs pandapipes (Inter-Code Reference) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **0.082 bar** | **0.084 bar** | **0.048 bar** |
| **Relative MAE ($\text{MAE}_{\%}$)** | **0.11 %** | **0.12 %** | **0.07 %** |
| **Root Mean Square Error (RMSE)** | **0.125 bar** | **0.130 bar** | **0.060 bar** |
| **Mean Bias ($\mu = P_{\text{ref}} - P_{\text{GNS}}$)** | **+0.017 bar** | **+0.063 bar** | **−0.046 bar** |
| **Maximum Absolute Error** | **0.931 bar** | **0.948 bar** | **0.169 bar** |

The microscopic inter-code deviation (< 0.05 bar between SHIMMER++ and pandapipes) and the exceptional agreement of GasNetSim with both external references (< 0.12% MAE) confirm the accuracy of the model without requiring empirical SCADA tuning.

---

## 3. Algorithmic Architecture: How the Simulation Works

The simulation in GasNetSim couples real-gas thermodynamics, multi-component gas quality tracking, and non-linear network hydraulics through a **staggered coupling architecture**:

```
                       [ Start / Input Ingestion ]
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │  create_network_from_folder   │
                   │  - Ingest sf_nodes.csv        │
                   │  - Ingest sf_pipelines.csv    │
                   │  - ReMi boundary: P_ref       │
                   │  - Demand nodes: V_dot        │
                   └───────────────┬───────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │     OUTER PICARD ITERATION    │
                   │ (Multicomponent Tracking)     │
                   └───────────────┬───────────────┘
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│  Hydrogen Streamline Advec.  │               │   Dynamic Viscosity Cache    │
│  x_H2 tracked from Mazara    │               │   mu(T, x) memoized via      │
│  downstream via mass mixing  │               │   Herning-Zipperer mixing    │
└──────────────┬───────────────┘               └───────────────┬──────────────┘
               └───────────────────────────┬───────────────────┘
                                           │
                                           ▼
                   ┌───────────────────────────────────────────┐
                   │      INNER NEWTON-RAPHSON SOLVER          │
                   │      (Hydraulic Nodal Pressures)          │
                   └───────────────────────┬───────────────────┘
                                           │
  ┌────────────────────────────────────────┼────────────────────────────────────────┐
  ▼                                        ▼                                        ▼
┌────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
│ Nodal Mass Balance     │       │ Frictional Closure     │       │ Real-Gas EOS           │
│ Sum(m_dot_ij) = m_dem  │       │ Explicit Chen (1979)   │       │ GERG-2008 (ISO 20765-2)│
│                        │       │ C1 smoothstep Re-trans │       │ Local Z(P,T) and rho   │
└───────────┬────────────┘       └───────────┬────────────┘       └───────────┬────────────┘
            └──────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
                   ┌───────────────────────────────────────────┐
                   │  Sparse Jacobian Assembly (CSC Format)    │
                   │  Direct Sparse Solve (scipy spsolve)      │
                   │  - Step capping: |dP| <= dP_max           │
                   │  - Pressure floor: P >= 1.0 bar           │
                   └───────────────────────┬───────────────────┘
                                           │
                                           ▼
                   ┌───────────────────────────────────────────┐
                   │        Adaptive Relaxation Strategy       │
                   │  Check delta_P and delta_x_comp           │
                   │  - If oscillation detected -> halve omega │
                   │  - Damps period-2 limit cycles            │
                   └───────────────────────┬───────────────────┘
                                           │
                                  Converged?
                                   │       │
                          No ──────┘       └──────► Yes
                                                     │
                                                     ▼
                                      [ Output Results & Summary ]
```

### Detailed Algorithmic Steps:

1. **Topology & Ingestion (`gns.create_network_from_folder`)**:
   Reads `sf_nodes.csv` and `sf_pipelines.csv`. It assigns boundary conditions:
   * ReMi station (node 8118, Mazara del Vallo): fixed pressure $P_{\text{set}} = 76.013\text{ bar abs}$ ($75.0\text{ barg} + P_{\text{atm}}$);
   * Demand nodes: mass or standard volumetric offtakes $\dot{m}_{\text{demand}} = \rho_{\text{std}} \cdot \dot{V}_{\text{std}}$;
   * Pipelines: internal diameter $D$, length $L$, absolute pipe roughness $\epsilon = 12\,\mu\text{m}$, and elevations $z_{\text{in}}, z_{\text{out}}$.

2. **Outer Picard Loop (Hydrogen Quality Tracking)**:
   * Flow directions across all 350 pipelines are inferred from current pressure gradients;
   * Hydrogen mole fractions $x_{\text{H}_2}$ are advected downstream along flow paths, applying complete mixing at junctions;
   * Gas mixture properties (molar mass $M_{\text{mix}}$, density, and dynamic viscosity $\mu(T, x)$) are dynamically recalculated and cached.

3. **Inner Newton-Raphson Solver (Nonlinear Hydraulic Pressures)**:
   Solves the nonlinear system of mass balance equations across all non-reference nodes:
   $$F_i(P) = \sum_{j \in \mathcal{N}(i)} \dot{m}_{ij}(P_i, P_j) - \dot{m}_{\text{demand}, i} = 0$$
   * **Momentum Equation & Frictional Losses:** Evaluated via the Darcy-Weisbach formulation:
     $$P_i^2 - P_j^2 = \frac{16 f_{ij} L_{ij}}{\pi^2 D_{ij}^5} \frac{Z_{\text{avg}} R T}{M_{\text{mix}}} \dot{m}_{ij} |\dot{m}_{ij}| + 2 g \rho_{\text{avg}}^2 (z_j - z_i)$$
   * **Transitional Friction Factor:** GasNetSim applies the explicit formula by **Chen (1979)**, accurately matching Colebrook-White in the transitional regime ($Re \approx 10^5 \div 5\cdot 10^6$), regularized with a $C^1$-smoothstep at low Reynolds numbers;
   * **Real-Gas EOS (GERG-2008):** Local compressibility $Z(P, T)$ is dynamically evaluated node-by-node;
   * **Sparse Linear Algebra:** The Jacobian matrix $\mathbf{J} = \partial F / \partial P$ is assembled directly in Compressed Sparse Column (CSC) format and inverted with direct sparse LU factorization (`scipy.sparse.linalg.spsolve`), yielding a >100× acceleration over dense solvers;
   * **Safeguarded Line Search:** Step capping limits maximum pressure increments ($|\Delta P| \le \Delta P_{\max}$), while strict pressure flooring ($P \ge 1.0\text{ bar}$) prevents non-physical negative pressures from exiting the thermodynamic domain of GERG-2008.

4. **Oscillation Damping & Adaptive Underrelaxation**:
   Between outer Picard updates, state vectors are relaxed: $x^{(k+1)} = x^{(k)} + \omega \Delta x$. If the solver detects recurring oscillations (such as the 2-cycle periodic oscillation observed during high-load winter peaks), $\omega$ is automatically reduced from 0.5 to 0.25, dampening the limit cycle and restoring monotonic convergence.

5. **Convergence Criteria**:
   Convergence is achieved when both:
   * Maximum nodal pressure residual: $\|F(P)\|_\infty \le \text{tol}$ (default 0.01 bar / 1000 Pa);
   * Gas quality consistency: $\|\Delta x_{\text{comp}}\|_\infty \le \text{tol}_x$ (default $10^{-4}$).

---

## 4. Headless Execution (Console-Only — No Graphics Needed)

> **Note:** The simulation runs **100% headless** directly in your terminal, PowerShell, command prompt, or SSH session. **No graphical desktop, X11 server, Qt window, or matplotlib GUI is required.**

### Running the Example

From this directory:

```bash
# 1. Run the standard Autumn transit case (15 Nov 2025, h12 — 13.92% H2 blend):
python run_sicily_simulation.py

# 2. Run the high-load Winter peak snapshot (08 Jan 2026, h18):
python run_sicily_simulation.py --winter-peak
```

### Command-Line Arguments and Options

The runner script supports flexible flags to simulate any of the 19 representative days or hours:

```bash
# 1. Run the default baseline benchmark (15 Nov 2025, h12)
python examples/Sicily_transmission_network/run_sicily_simulation.py

# 2. Run the extreme winter peak snapshot (08 Jan 2026, h18)
python examples/Sicily_transmission_network/run_sicily_simulation.py --winter

# 3. List all 19 available representative day datasets
python examples/Sicily_transmission_network/run_sicily_simulation.py --list-days

# 4. Simulate a specific day and hour (e.g. 15 Aug 2025, hour 20 CCGT duck-curve peak)
python examples/Sicily_transmission_network/run_sicily_simulation.py --day 20250815 --hour 20

# 5. Simulate all 24 hours of a specific day in sequence
python examples/Sicily_transmission_network/run_sicily_simulation.py --day 20260108

# 6. Simulate any custom folder path
python examples/Sicily_transmission_network/run_sicily_simulation.py --folder path/to/input_folder
```

### What the Console Output Displays

When you run the command, it outputs a clean, structured text report directly to stdout:

```text
============================================================================
  GasNetSim Benchmark: Sicilian Regional Transmission Network
  Folder: Sicily_transmission_network
============================================================================
Topology loaded successfully:
  * Nodes     : 340
  * Pipelines : 350

Running steady-state hydraulic simulation (staggered coupling)...
  - Outer loop : composition & dynamic viscosity tracking (simple mixing)
  - Inner loop : Newton-Raphson on sparse pressure system (CSC/spsolve)
  - Friction   : Chen (1979) with C1-smoothstep Reynolds transition
  - EOS        : GERG-2008 (ISO 20765-2) real gas formulation

[OK] Simulation completed in 4.82 s

----------------------------------------------------------------------------
Node ID    | Location / Role                            | Pressure [bar abs]
----------------------------------------------------------------------------
8118       | Mazara del Vallo (Transmed Import Point)   |        76.0132 bar
28         | Enna Central Network Junction              |        72.8163 bar
1727       | Priolo Gargallo Petrochemical Pole         |        73.3811 bar
4073       | Caltanissetta Delivery Station             |        72.2389 bar
8891       | Caltagirone Regional Branch Junction       |        72.9304 bar
25         | Gela Coastal Delivery Station              |        70.5861 bar
12         | Catania Delivery Station                   |        71.5627 bar
----------------------------------------------------------------------------

Network Global Extremes:
  * Minimum Pressure : 69.8412 bar abs (well above 60 bar regulatory threshold)
  * Maximum Pressure : 76.0132 bar abs (at Mazara del Vallo injection)
  * Maximum Velocity : 7.14 m/s (well below 16 m/s API RP 14E limit)

[OK] Numerical results exported to CSV:
  * examples/Sicily_transmission_network/results/results_nodes.csv
  * examples/Sicily_transmission_network/results/results_pipelines.csv
============================================================================
```

### Exported Numerical Tables

The runner script automatically saves the full state solution into the `./results/` folder:
* `results_nodes.csv`: Complete table of all 340 nodes (`node_id`, `pressure_bar`, `volumetric_flow_sm3s`, `altitude_m`);
* `results_pipelines.csv`: Complete table of all 350 pipes (`pipeline_id`, `inlet_id`, `outlet_id`, `diameter_m`, `length_m`, `velocity_m_s`).

---

## 5. Directory Contents

| File / Folder | Purpose |
| :--- | :--- |
| [`run_sicily_simulation.py`](run_sicily_simulation.py) | Standalone, headless Python runner script with CLI flags (`--day`, `--hour`, `--winter`, `--list-days`). |
| [`sf_nodes.csv`](sf_nodes.csv) | 340-node table: coordinates, Copernicus DEM elevations, boundary pressures, and consumption flows (Autumn reference h12). |
| [`sf_pipelines.csv`](sf_pipelines.csv) | 350-pipeline table: internal diameters, topological lengths, roughness, and friction models. |
| [`params.json`](params.json) | Simulation configuration: tolerances, max iterations, and 12-component Algerian gas blend with 13.92% H₂. |
| [`winter_peak_0108_h18/`](winter_peak_0108_h18/) | Subfolder with the winter peak high-demand dataset (08 Jan 2026, h18) for stress-testing and limit-cycle analysis. |
| [`all_19_days/`](all_19_days/) | **Complete annual campaign archive (19 representative days, 456 hourly snapshots)** covering all 12 monthly archetypes and the full winter peak week. |
| [`results/`](results/) | Output directory containing generated `results_nodes.csv` and `results_pipelines.csv`. |
| `README.md` | This technical documentation and user guide. |

