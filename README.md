# fieldanalysis — 2D Electric-Field Multipole Decomposition

This repository provides a Python tool to extract **2D multipole components** (dipole, quadrupole, sextupole, octupole, …) from a transverse electric field map containing **Ex(x,y)** and **Ey(x,y)**.

It computes complex multipole coefficients \(C_n\) using a circular Fourier method and generates:

- 2D maps of **Ex**, **Ey**, and **|E|** for total and each multipole order
- 1D cuts along **x (y≈0)** and **y (x≈0)** with total + multipole components
- a run report (`*_OPT.txt`) including \(C_n\), \(|C_n|\), phase, normalized amplitude, and field-quality radii (0.2%, 0.5%, 1%)

---

## Physics / Convention

Define

- $w = (x-x_0) + i(y-y_0)$
- $F = E_x - iE_y$

Multipole expansion:

$$
F(w) = \sum_{n\ge 1} C_n\, w^{n-1}
$$

Order meanings:

- \(n=1\) dipole
- \(n=2\) quadrupole
- \(n=3\) sextupole
- \(n=4\) octupole

Units:

- $E_x, E_y$ in V/m
- $C_n$ in V/m\(^n\)
- normalized amplitude column uses $|C_n| r_\mathrm{ref}^{n-1}$ (V/m)

---

## Input data format

A `.dat` (or `.txt` / `.csv`) file with a header and 4 columns:

X    Y    Ex    Ey

## Install

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ field.dat               # example input (NOT required to commit)
│  └─ .gitkeep
├─ outputs/
│  └─ .gitkeep                # outputs are generated at runtime
└─ src/
   └─ multipole_2d.py          # your script
