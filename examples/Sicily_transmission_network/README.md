# Sicilian Regional Transmission Network (340 Nodes, 350 Pipelines)

This example provides the complete, georeferenced dataset for the high-pressure
natural gas transmission network of Sicily (~2,316 km of transmission corridors),
developed for the research article:

> **"Techno-Economic and Hydraulic Feasibility of Green Hydrogen Blending in Regional Transmission Networks: A 3D Digital Twin Analysis of the Sicilian Grid"**  
> *Submitted to Applied Energy (Elsevier)*

## 1. Network Characteristics

* **Topological Complexity:** 340 nodes, 350 pipelines;
* **Pipeline Diameters:** 152.4 mm (6") to 1,219.2 mm (48");
* **Terrain Altimetry:** Draped from Copernicus 3D DEM (elevations between 1.0 m and 1,280 m a.s.l.);
* **Gas Composition:** Algerian natural gas baseline with **13.92 vol% green hydrogen** (relative density $d \ge 0.555$);
* **Import Point:** Mazara del Vallo (Node 8118, Transmed pipeline from Algeria) at 75 barg (76.013 bar abs);
* **Demand Profiles:** Reconstructed from hourly sectoral delivery data (Terna / Snam) with realistic spatial clustering.

## 2. Multi-Solver Benchmark

This exact configuration was cross-validated against two independent external solvers:
* **SHIMMER++** (Politecnico di Torino, linearized 1D Navier-Stokes solver);
* **pandapipes** (Fraunhofer IEE / University of Kassel, graph-based Newton-Raphson).

Results across all 340 nodes:
* GasNetSim vs SHIMMER++: **MAE = 0.082 bar (0.11 %)**, RMSE = 0.125 bar;
* GasNetSim vs pandapipes: **MAE = 0.084 bar (0.12 %)**, RMSE = 0.130 bar;
* Inter-reference agreement (SHIMMER++ vs pandapipes): **MAE = 0.048 bar (0.07 %)**.

## 3. How to Run

From this directory:

```bash
# Run standard autumn transit snapshot (15 Nov 2025, h12)
python run_sicily_simulation.py

# Run winter peak snapshot (08 Jan 2026, h18)
python run_sicily_simulation.py --winter-peak
```

## 4. Folder Contents

* `sf_nodes.csv`: Nodal table with coordinates, elevations, boundary pressures, and consumption flows;
* `sf_pipelines.csv`: Pipeline table with diameters, topological lengths, roughness, and friction closure;
* `params.json`: Solver parameters and gas mixture composition;
* `run_sicily_simulation.py`: Executable demonstration script;
* `winter_peak_0108_h18/`: Subfolder containing the high-load winter peak case (used for studying convergence and staggered coupling dynamics).
