
# Electrode Field Optimization Toolkit

Author: **Erdong Wang**

This repository provides tools for electrode geometry generation, electrostatic FEM field solving, multipole field analysis, and automatic geometry optimization using CMA‑ES.

The toolkit is designed for studying high‑uniformity electrostatic fields and optimizing electrode shapes for precision applications.

---

# Installation

Install dependencies:

pip install -r requirements.txt

---

# File Overview

## Geometry

electrode_geometry.yaml  
Main configuration file defining electrode geometry parameters.

electrode_geometry_example.yaml  
Example YAML configuration template.

genXXXX_evalYYYYYY_best.yaml  
Optimized geometry files produced during CMA‑ES runs.

---

# Field Solver

electrodes_field.py  
Main FEM solver that:

1. Builds electrode geometry
2. Generates FEM mesh
3. Solves electrostatic potential
4. Computes electric field (Ex, Ey)

Outputs:

- phi_map.png
- emag_density.png
- mesh_wireframe.png
- simulation_output.txt

---

electrodes_fast_v1.py  
A faster FEM implementation optimized for repeated evaluations during optimization.

---

# Field Analysis

field_analysis.py  
Computes multipole expansion coefficients from field data.

electrodes_field_analysis.py  
Runs the field solver and performs multipole analysis automatically.

Outputs a table:

n  name  Re(Cn)  Im(Cn)  |Cn|  phase(rad)  |Cn|*r_ref^(n-1)

Also calculates:

- Sum of higher‑order multipoles (n > 2)
- Radius where field deviation reaches 0.005

---

# Optimization

cma_es_optimize_serial.py  
Runs CMA‑ES optimization in serial mode.

cma_es_optimize_ll.py  
Advanced CMA‑ES optimization implementation.

electrodes_interp.cma.py  
Interpolation utilities used during optimization.

---

# Typical Workflow

1. Edit geometry

electrode_geometry.yaml

2. Run field solver

python electrodes_field.py electrode_geometry.yaml

3. Run field analysis

python electrodes_field_analysis.py electrode_geometry.yaml

4. Run optimization

python cma_es_optimize_serial.py

or

python cma_es_optimize_ll.py

---

# Summary

This toolkit provides a workflow for:

- Electrode geometry generation
- Electrostatic FEM simulation
- Multipole field analysis
- Automatic electrode optimization
