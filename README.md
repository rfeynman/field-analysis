# fieldanalysis — 2D Electric-Field Multipole Decomposition

This repository provides a Python tool to extract **2D multipole components** (dipole, quadrupole, sextupole, octupole, …) from a transverse electric field map containing **Ex(x,y)** and **Ey(x,y)**.

It computes complex multipole coefficients \(C_n\) using a circular Fourier method and generates:

- 2D maps of **Ex**, **Ey**, and **|E|** for total and each multipole order
- 1D cuts along **x (y≈0)** and **y (x≈0)** with total + multipole components
- a run report (`*_OPT.txt`) including \(C_n\), \(|C_n|\), phase, normalized amplitude, and field-quality radii (0.2%, 0.5%, 1%)

---

## Physics background (what the code is doing)

This tool treats your transverse 2D electric field map as approximately **source-free** in the region of interest (no space charge inside the aperture), so the electrostatic potential satisfies Laplace’s equation:

$$
\nabla^2 V(x,y) = 0,
\qquad
\mathbf{E}(x,y) = -\nabla V(x,y),
\qquad
\frac{\partial E_x}{\partial x} + \frac{\partial E_y}{\partial y} = 0.
$$

In a 2D Laplace region, the field can be represented by an analytic complex function. Define the complex coordinate and complex field:

$$
w = (x-x_0) + i (y-y_0),
$$

$$
F(w) = E_x(x,y) - i\,E_y(x,y).
$$

In a charge-free region, $F(w)$ is analytic and can be expanded as a power series about the reference center $(x_0,y_0)$:

$$
F(w) = \sum_{n=1}^{\infty} C_n\, w^{n-1}.
$$

The coefficients $C_n$ are the **multipole components**:

- $n=1$: dipole (uniform field)
- $n=2$: quadrupole (linear gradient)
- $n=3$: sextupole (quadratic nonlinearity)
- $n=4$: octupole (cubic nonlinearity)
- etc.

### Polar form and scaling with radius

Write

$$
w = r e^{i\theta}, \qquad C_n = |C_n| e^{i\phi_n}.
$$

Then the $n$th-order contribution is

$$
F_n(r,\theta) = C_n\, w^{n-1} = |C_n|\, r^{n-1}\, e^{i\left(\phi_n + (n-1)\theta\right)}.
$$

So the amplitude of each order scales with radius as:

$$
|F_n| \sim |C_n|\, r^{n-1}.
$$

This is why higher-order errors grow quickly as you move away from the center.

### Converting $F$ back to $(E_x,E_y)$

Because

$$
F = E_x - i E_y,
$$

the real field components are obtained by:

$$
E_x = \Re(F), \qquad E_y = -\Im(F).
$$

For each multipole order:

$$
E_x^{(n)}(x,y) = \Re\!\left(C_n w^{n-1}\right), \qquad
E_y^{(n)}(x,y) = -\Im\!\left(C_n w^{n-1}\right).
$$

### Units

If $E_x,E_y$ are in V/m and $w$ is in meters, then:

$$
[C_n] = \frac{\mathrm{V/m}}{\mathrm{m}^{n-1}} = \mathrm{V}\,\mathrm{m}^{-n}.
$$

The tool also reports a **normalized amplitude** at a reference radius $r_\mathrm{ref}$:

$$
A_n(r_\mathrm{ref}) = |C_n|\, r_\mathrm{ref}^{n-1},
$$

which has units of V/m and represents the characteristic magnitude of the $n$th multipole contribution at that radius. A simple nonlinearity estimate at $r_\mathrm{ref}$ is:

$$
\epsilon(r_\mathrm{ref}) \approx \frac{A_3(r_\mathrm{ref}) + A_4(r_\mathrm{ref}) + \cdots}{A_1}.
$$

### How the code extracts $C_n$ (circular Fourier method)

The coefficients are computed by sampling the field on a circle of radius $r$ about the reference center:

$$
x(\theta) = x_0 + r\cos\theta,\qquad
y(\theta) = y_0 + r\sin\theta.
$$

Form the complex field on that circle:

$$
F(\theta) = E_x(r,\theta) - i E_y(r,\theta).
$$

Using the orthogonality of $e^{ik\theta}$ on $[0,2\pi)$, the multipole coefficient is:

$$
C_n = \frac{1}{2\pi\, r^{n-1}} \int_{0}^{2\pi} F(\theta)\, e^{-i(n-1)\theta}\, d\theta.
$$

With $M$ uniform samples $\theta_k = 2\pi k/M$:

$$
C_n \approx \frac{1}{M\, r^{n-1}} \sum_{k=0}^{M-1} F(\theta_k)\, e^{-i(n-1)\theta_k}.
$$

If multiple radii are provided, the tool computes $C_n$ for each radius and averages them, which helps reduce sensitivity to interpolation noise.

### Field-quality radii (0.2%, 0.5%, 1% circles on $|E|$)

The total field magnitude is:

$$
|E(x,y)| = \sqrt{E_x^2(x,y) + E_y^2(x,y)}.
$$

Let $E_0$ be the magnitude at the nearest grid point to $(x_0,y_0)$:

$$
E_0 = |E(x_0,y_0)|.
$$

For each radius bin (thin annulus), the tool computes the **peak** relative deviation:

$$
d(r) = \max_{\text{points in annulus}} \frac{\left||E(x,y)| - E_0\right|}{E_0}.
$$

The 0.2%, 0.5%, and 1% radii are the smallest radii where:

$$
d(r) \ge 0.002,\quad d(r) \ge 0.005,\quad d(r) \ge 0.01.
$$

These radii are drawn as dashed circles on the total $|E|$ map and written into the `_OPT.txt` report.

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
