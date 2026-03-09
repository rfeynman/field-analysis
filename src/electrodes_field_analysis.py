#!/usr/bin/env python3
"""electrodes_field_analysis.py

Simplified multipole analysis.

- Does NOT read any data files.
- Runs electrodes_field.py to obtain `vals_small` (columns: x, y, Ex, Ey).
- Computes 2D multipole coefficients Cn using the convention:

    w = (x-x0) + i (y-y0)
    F = Ex - i Ey = sum_{n>=1} Cn * w^(n-1)

- Prints the table:

    n  name        Re(Cn)        Im(Cn)        |Cn|         phase(rad)   |Cn|*r_ref^(n-1)

- Returns:
    (1) sum_{n>2} |Cn|*r_ref^(n-1)
    (2) radius where field uniformity first exceeds 0.005 (0.5%) using
        d(r) = max_annulus | |E(r,θ)| - E0 | / E0

Edit the USER SETTINGS section as needed.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

# Multipole extraction center
X0 = 0.0
Y0 = 0.0

# Reference radius (meters) used for:
#  - sampling the circle to extract Cn
#  - the last column |Cn|*r_ref^(n-1)
R_REF = 0.029

# Fourier samples around the circle
M_THETA = 512

# Max multipole order to report
NMAX = 5

# Uniformity threshold (dimensionless). 0.005 == 0.5%
UNIFORMITY_TARGET = 0.005


# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _vals_small_to_grid(vals_small: np.ndarray):
    """Convert flattened [x,y,Ex,Ey] samples into (x1d,y1d,Ex2d,Ey2d)."""
    vals_small = np.asarray(vals_small, dtype=float)
    if vals_small.ndim != 2 or vals_small.shape[1] != 4:
        raise ValueError("vals_small must be an (N,4) array: [x,y,Ex,Ey].")

    x = np.unique(vals_small[:, 0])
    y = np.unique(vals_small[:, 1])
    nx, ny = x.size, y.size

    ix = np.searchsorted(x, vals_small[:, 0])
    iy = np.searchsorted(y, vals_small[:, 1])

    Ex = np.zeros((ny, nx), dtype=float)
    Ey = np.zeros((ny, nx), dtype=float)
    Ex[iy, ix] = vals_small[:, 2]
    Ey[iy, ix] = vals_small[:, 3]

    return x, y, Ex, Ey


def _bilinear_interp(x1d, y1d, F2d, xq, yq):
    """Bilinear interpolation on a regular tensor grid.

    F2d is indexed as F2d[jy, ix] = F(y[jy], x[ix]).
    Out-of-bounds queries return NaN.
    """
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    F2d = np.asarray(F2d)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    if x1d.size < 2 or y1d.size < 2:
        raise ValueError("Need at least 2 points in x and y for bilinear interpolation.")
    if not (np.all(np.diff(x1d) > 0) and np.all(np.diff(y1d) > 0)):
        raise ValueError("x and y arrays must be strictly increasing.")

    nx = x1d.size
    ny = y1d.size

    ix = np.searchsorted(x1d, xq) - 1
    iy = np.searchsorted(y1d, yq) - 1

    oob = (ix < 0) | (ix >= nx - 1) | (iy < 0) | (iy >= ny - 1)
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)

    x0 = x1d[ix]
    x1 = x1d[ix + 1]
    y0 = y1d[iy]
    y1 = y1d[iy + 1]

    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    f00 = F2d[iy, ix]
    f10 = F2d[iy, ix + 1]
    f01 = F2d[iy + 1, ix]
    f11 = F2d[iy + 1, ix + 1]

    fq = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
    fq = fq.astype(float, copy=False)
    fq[oob] = np.nan
    return fq


def _sample_circle(x1d, y1d, Ex2d, Ey2d, x0, y0, r, M):
    theta = np.linspace(0.0, 2.0 * np.pi, int(M), endpoint=False)
    xs = x0 + r * np.cos(theta)
    ys = y0 + r * np.sin(theta)

    Exs = _bilinear_interp(x1d, y1d, Ex2d, xs, ys)
    Eys = _bilinear_interp(x1d, y1d, Ey2d, xs, ys)

    F = Exs - 1j * Eys
    return theta, F


def _compute_Cn_on_radius(theta, Ftheta, r, nmax):
    """Discrete Fourier estimator for Cn on a single circle of radius r."""
    theta = np.asarray(theta)
    Ftheta = np.asarray(Ftheta)

    M = theta.size
    C = np.zeros(int(nmax), dtype=complex)
    for n in range(1, int(nmax) + 1):
        k = n - 1
        # Cn = (1/(2π)) ∫ F(θ) e^{-ikθ} dθ / r^k  -> discrete sum
        C[n - 1] = (1.0 / (M * (r ** k))) * np.nansum(Ftheta * np.exp(-1j * k * theta))
    return C


def _uniformity_radius(x1d, y1d, Ex2d, Ey2d, x0, y0, target=0.005):
    """Return r where d(r) first exceeds `target` (0.005 = 0.5%).

    d(r) is evaluated on annuli with thickness dr ~ grid spacing.
    """
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    Ex2d = np.asarray(Ex2d)
    Ey2d = np.asarray(Ey2d)

    Xg, Yg = np.meshgrid(x1d, y1d)
    R = np.sqrt((Xg - x0) ** 2 + (Yg - y0) ** 2)
    Emag = np.sqrt(Ex2d ** 2 + Ey2d ** 2)

    ix0 = int(np.argmin(np.abs(x1d - x0)))
    iy0 = int(np.argmin(np.abs(y1d - y0)))
    E0 = float(Emag[iy0, ix0])

    if not np.isfinite(E0) or E0 <= 0.0:
        # Fallback: average in a tiny disk
        r_small = max(np.min(np.diff(x1d)), np.min(np.diff(y1d)))
        m0 = R <= 1.5 * r_small
        E0 = float(np.nanmean(Emag[m0]))
        if not np.isfinite(E0) or E0 <= 0.0:
            return 0.0

    dr = float(min(np.min(np.diff(x1d)), np.min(np.diff(y1d))))
    rmax = float(np.nanmax(R))

    # Scan radii from 0 outward.
    r_centers = np.arange(0.0, rmax + 0.5 * dr, dr)
    last_ok = 0.0

    for r in r_centers:
        ann = (R >= (r - 0.5 * dr)) & (R < (r + 0.5 * dr))
        if not np.any(ann):
            continue
        dev = float(np.nanmax(np.abs(Emag[ann] - E0)) / E0)
        if np.isfinite(dev) and dev <= target:
            last_ok = float(r)
        elif np.isfinite(dev) and dev > target:
            break

    return last_ok


def _multipole_name(n: int) -> str:
    names = {
        1: "dipole",
        2: "quadrupole",
        3: "sextupole",
        4: "octupole",
        5: "decapole",
        6: "dodecapole",
        7: "14-pole",
        8: "16-pole",
    }
    return names.get(int(n), f"order{int(n)}")


# -----------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -----------------------------------------------------------------------------

def run_analysis(
    nmax: int = NMAX,
    x0: float = X0,
    y0: float = Y0,
    r_ref: float = R_REF,
    M: int = M_THETA,
    uniformity_target: float = UNIFORMITY_TARGET,
):
    """Run electrodes_field.py, compute multipoles, print the table.

    Returns:
        (sum_high_orders, r_uniform)

    where:
        sum_high_orders = Σ_{n>2} |Cn| * r_ref^(n-1)
        r_uniform       = radius where uniformity is still <= uniformity_target
    """

    # Import here so this file can be imported without triggering heavy deps.
    from electrodes_field import SolveConfig, run_simulation

    # Run the field solver and get vals_small.
    sim_cfg = SolveConfig(debug_plots=False, writefile=False)
    _max_result, vals_small = run_simulation(sim_cfg)

    x1d, y1d, Ex2d, Ey2d = _vals_small_to_grid(vals_small)

    # Multipoles on a circle
    theta, Ftheta = _sample_circle(x1d, y1d, Ex2d, Ey2d, x0, y0, float(r_ref), int(M))
    Cn = _compute_Cn_on_radius(theta, Ftheta, float(r_ref), int(nmax))

    # Print table
    print("n  name        Re(Cn)        Im(Cn)        |Cn|         phase(rad)   |Cn|*r_ref^(n-1)")
    for n in range(1, int(nmax) + 1):
        cn = Cn[n - 1]
        mag = float(np.abs(cn))
        ph = float(np.angle(cn))
        nm = _multipole_name(n)
        norm_amp = mag * (float(r_ref) ** (n - 1))
        print(
            f"{n:1d}  {nm:10s}  {cn.real:+.8e}  {cn.imag:+.8e}  {mag:.8e}  {ph:+.8f}  {norm_amp:.8e}"
        )

    # Sum higher orders (n > 2)
    n_arr = np.arange(1, int(nmax) + 1)
    amp = np.abs(Cn) * (float(r_ref) ** (n_arr - 1))
    sum_high = float(np.sum(amp[n_arr > 2]))

    # Uniformity radius for target (default 0.005)
    r_uni = float(_uniformity_radius(x1d, y1d, Ex2d, Ey2d, x0, y0, target=float(uniformity_target)))

    print("")
    print(f"Sum_{'{'}n>2{'}'} |Cn|*r_ref^(n-1) = {sum_high:.8e}  (V/m)")
    print(f"Uniformity radius for d(r) <= {uniformity_target:g} : r = {r_uni:.8e} m")

    return sum_high, r_uni


def main():
    run_analysis()


if __name__ == "__main__":
    main()
