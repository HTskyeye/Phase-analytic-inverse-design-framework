# Phase analytic inverse design framework

## Overview

This codebase implements a **phase-analytic inverse-design framework** that quantitatively identifies three distinct physical regimes in nanophotonic inverse design:

- **Insufficient regime**: spatially constrained, physically limited designs  
- **Stable regime**: minimal yet robust designs (Minimal Stable Footprint, MSF)  
- **Redundant regime**: over-parameterized designs with degeneracy  

The framework explicitly links **design footprint scaling** to **geometric phase sensitivity**, and determines the MSF without relying on heuristic regularization.

A broadband **1×2 wavelength-division demultiplexer** is used as a representative example.

All simulations are performed using **Tidy3D** (GPU-accelerated FDTD) with adjoint-based continuous-material optimization.

---

## Scientific Scope

This repository is **not** intended as a general-purpose inverse-design toolbox.

Instead, it serves as a **methodological and numerical validation framework** to:

- Quantify how inverse-design performance depends on footprint scaling  
- Reveal the transition between insufficient, stable, and redundant regimes  
- Demonstrate a phase-based criterion for minimal stable design freedom  

All analysis procedures are directly aligned with the manuscript.  
No hidden heuristics or undocumented post-processing steps are used.

---

## Repository Structure

```text
phase-analytic-inverse-design-framework/
├── configs/
│   └── base.yaml                # Physical, numerical, and optimization settings
├── src/
│   ├── run.py                   # Main entry: simulation + optimization
│   └── tidy3d/
│       ├── simulation.py        # Geometry, materials, sources, monitors (Part 1 & 2)
│       ├── optimizer.py         # Adjoint-based optimization loop
│       └── postprocess.py       # Postprocessing and metric extraction
├── scripts/
│   ├── test_part1.py            # Geometry and grid sanity check
│   ├── test_part2.py            # Mode solver and monitor validation
│   ├── test_part3_objective.py
│   ├── test_part3_optimize_3steps.py
│   └── postprocess.py           # Command-line postprocessing
├── results/                     # Generated automatically (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```


**Design principle**:  
Each module corresponds one-to-one with a conceptual stage in the manuscript.  
The code structure mirrors the physical and analytical decomposition used in the paper.

---

## Methodology Summary

| Module | Role | Conceptual role in manuscript |
|------|------|-------------------------------|
| Geometry & Materials | Device parameterization | Physical model |
| Sources & Monitors | Mode excitation and readout | Electromagnetic simulation |
| Optimizer | Adjoint-based optimization | Inverse design |
| Postprocess | Phase / IL / XT analysis | Phase-analytic framework |

The design footprint is controlled by a single scalar parameter:

scale_factor ∈ [0.3, 1.2]


which uniformly rescales the inverse-design region.

---

## Regime Definition (Paper-Aligned)

The physical regime is automatically inferred from `scale_factor`:

| Scale factor range | Regime | Physical interpretation |
|-------------------|--------|-------------------------|
| 0.3 ≤ sf < 0.6 | Insufficient | Spatially constrained, strong impedance mismatch |
| 0.6 ≤ sf ≤ 0.8 | Stable | Minimal Stable Footprint (MSF) |
| sf > 0.8 | Redundant | Over-parameterized, degenerate solutions |

The upper bound `sf = 1.2` is included to explicitly demonstrate redundancy, as discussed in the manuscript.

---

## Quick Start (Reproducibility)

### 1. Install dependencies

pip install -r requirements.txt


A valid **Tidy3D license** and a **GPU-enabled environment** are required.

---

### 2. Run inverse design (simulation + optimization)

python -m src.run
--config configs/base.yaml
--scale_factor 0.70
--tag smoke
--init all05


Typical console output:

[OK] regime=stable sf=0.7
[0001/40] J=...
...
[DONE] Optimization finished.


All outputs are written automatically to:

results/<regime>/<timestamp>_sfXX_tag_init/


---

### 3. Postprocess results

python -m scripts.postprocess
--run_dir results/stable/<run_id>


---

## Postprocess Outputs

The postprocessing step generates the following files under:

results/<regime>/<run_id>/final/


- `density.png` — optimized design parameters  
- `eps.png` — final permittivity distribution  
- `flux_terminal_1.csv`, `flux_terminal_2.csv` — spectral transmission data  
- `transmission_db.png` — transmission spectra in dB  
- `metrics.csv`, `metrics.txt` — averaged insertion loss (IL) and crosstalk (XT)  

All spectral data are exported in CSV format for direct use in publication figures.

---

## Validation Strategy

**Numerical domain**  
- GPU-accelerated 2D FDTD simulations using Tidy3D  

**Experimental analogue**  
- Scaled microwave implementations based on drilled-dielectric effective media  

Experimental fabrication and measurements are reported exclusively in the manuscript.  
This repository focuses on the numerical and theoretical framework.

---

## Notes for Reviewers

- Emphasis on physical interpretability and reproducibility  
- No hidden regularization or tuning tricks  
- Continuous effective-permittivity parameterization  
- Regime boundaries derived from phase sensitivity, not heuristics  

---

## Citation

If you use this code, please cite:

Ge, X., Hu, X., Lv, L., Ma, S., Wang, C., Yang, Y., Qian, H., Ye, D.  
*Quantifying the Minimal Stable Footprint of Nanophotonic Inverse Design  
via Geometric Phase Sensitivity.*

---

## License


This project is released under the MIT License.

