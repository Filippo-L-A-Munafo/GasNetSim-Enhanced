![](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim/-/raw/main/docs/GasNetSim_Logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/GasNetSim.svg?color=orange)](https://pypi.org/project/GasNetSim/)
![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![DOI:10.1109/OSMSES54027.2022.9769148](https://zenodo.org/badge/DOI/10.1109/OSMSES54027.2022.9769148.svg)](https://doi.org/10.1109/OSMSES54027.2022.9769148)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fjugit.fz-juelich.de%2Fiek-10%2Fpublic%2Fsimulation%2Fgasnetsim/HEAD)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Fork Notice

This repository is a **fork of [GasNetSim](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim)**,
originally developed by **ICE-1: Energy Systems Engineering, Forschungszentrum Jülich GmbH**
(maintainer: Yifei Lu, `yifei.lu@fz-juelich.de`). All credit for the original simulation engine —
the steady-state solver architecture, the GERG-2008 gas mixture modeling, and the overall package
design — belongs to the original authors. If you use this software, please cite the original work
(see the DOI badge above, or the [upstream repository](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim)
for the full citation).

This fork is maintained by **Filippo Munafo** (University of Palermo) as the simulation engine
behind the [GasNetSim_4Qgis](https://github.com/Filippo-L-A-Munafo/GasNetSim_4Qgis) QGIS plugin.
The full original commit history is preserved; on top of it, this fork adds:

- **Compressor support**: a new `Compressor` component with an *augmented* Newton–Raphson
  formulation (`simulation/formulations/pressure_problem.py`) that solves compressor flow rates
  and pressure ratios as first-class unknowns alongside nodal pressures, with two control modes
  (`ratio` and `pressure`/discharge-pressure target).
- **Solver robustness fixes**: sparse (`spsolve`)/regularized fallback for ill-conditioned
  Jacobians, adaptive under-relaxation, step-cap and pressure-floor safeguards, and a fix for the
  `inf`/`NaN` Jacobian entries that appeared when two connected nodes had (near-)equal pressure.
- **Staggered pressure/composition coupling improvements**: Aitken (Irons–Tuck) acceleration and
  an additional adaptive relaxation fallback to damp composition oscillations on flow-reversing
  pipelines.
- Two Sicilian gas supply compositions (`SNAM_MAZARA_GAS`, `SNAM_GELA_GAS`) added to
  `components/gas_mixture/typical_mixture_composition.py`.

See [CHANGELOG.md](CHANGELOG.md) for the detailed, dated list of changes. This fork is released
under the same license as upstream — MPL-2.0, see [LICENSE](LICENSE) — and every modified file
retains that license as required by MPL-2.0 §3.3.

---

# **GasNetSim**

*GasNetSim* is a simulation package designed for gas network steady-state simulation.
It supports the steady-state natural gas network simulations with different gas mixture
compositions, thus enabling accurate analysis of the impacts of hydrogen injection on the gas network.  
Moreover, users have the flexibility to modify this tool and implement their own desired
gas mixture modeling approaches.
Future work will be carried out to include gas storage units and to take into account
the dynamic behavior of the gas network so that short-term simulations can be performed.

## Installation

Currently, it is only supported using source files. Using following commands to install the package in editable mode.

- ``pip install -e .``
- ``pip install -r ./requirements.txt``

## License

The project is released under the terms of the [MPL 2.0](https://mozilla.org/MPL/2.0/).

## Dependencies

<!-- Dependencies -->

- ``numpy``>=1.19.2
- ``matplotlib``>=3.3.2
- ``scipy``>=1.5.2
- ``pandas``>=1.1.3
- ``pytest``>=6.2.5
- ``fluids``>=0.1.86
- ``pint``>=0.18
- ``setuptools``>=60.9.3
- ``requests``>=2.25.1
- ``pyparsing``~=3.0.7
- ``cantera``~=3.0.0
- ``thermo``~=0.2.23
- ``tqdm``>=4.64.1
- ``seaborn``>=0.12.2
- ``networkx``~=3.1

<!-- End Dependencies -->

For the ``thermo`` package, the version used in this repo is 0.1.40. Because there are some changes
and new features included in the newer versions. The source files of the `thermo` package is directly
stored in this repo. It will be updated in the future.

[//]: # (## Discussion)

[//]: # ()

[//]: # (You can connect with the community in a variety of ways...)

[//]: # ()

[//]: # (- [Mailing list]&#40;https://lists.lfenergy.org/g/xxxx-discussion&#41;)

[//]: # (- [#{{**PROJECT-NAME**}} channel on LF Energy Slack]&#40;https://slack.lfenergy.org&#41;)

[//]: # (- Other communication channels, e.g. Discord, Slack, Skype, Mattermost, FZJ Rocket Chat, ...)

[//]: # (## Contributing)

[//]: # ()

[//]: # (_**TODO** Provide contributing guidelines here or point to a_)

[//]: # (_[CONTRIBUTING.md]&#40;CONTRIBUTING.md&#41; file if the contributing guidelines require_)

[//]: # (_more than just a few lines._)

## Reporting Issues

To report a problem, you can open an
[issue](https://jugit.fz-juelich.de/iek-10/public/simulation/gasnetsim/-/issues)
in repository against a specific workflow. If the issue is sensitive in nature or
a security related issue, please do not report in the issue tracker but instead
email [Yifei Lu](yi.lu@fz-juelich.de).
