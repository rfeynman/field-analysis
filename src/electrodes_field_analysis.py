#!/usr/bin/env python3
"""
electrodes_field_analysis.py
Author: Erdong Wang

Simplified 2D multipole analysis for electrode geometry.

This script runs the field solver (electrodes_field.py), extracts the
transverse electric field (Ex, Ey), and computes complex multipole
coefficients Cn using circular Fourier analysis.

---------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
---------------------------------------------------------------------------

1) Calls electrodes_field.py to generate the electric field map.
2) Converts the returned vals_small array into a strict rectangular grid.
3) Computes 2D multipole coefficients Cn using:

       w = (x - x0) + i (y - y0)
       F = Ex - i Ey
       F(w) = Σ_{n>=1} Cn * w^(n-1)

4) Prints a multipole table:

       n  name        Re(Cn)        Im(Cn)        |Cn|         phase(rad)   |Cn|*r_ref^(n-1)

5) Returns two quantities:

       (1) Sum_{n>2} |Cn| * r_ref^(n-1)
           (Total higher-order field amplitude at r_ref)

       (2) Uniformity radius:
           The first radius where the peak annulus deviation satisfies

               d(r) = max_annulus | |E| - E0 | / E0  >= UNIFORMITY_TARGET

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script does NOT read field data files directly.

Instead, it runs:

    electrodes_field.py

which generates vals_small = [x, y, Ex, Ey].

Optional input:

    geom_yaml

Path to a YAML file defining electrode geometry (e1..e4).
This file is passed directly to electrodes_field.py.

Geometry can be provided either:

    python electrodes_field_analysis.py geometry.yaml

or

    python electrodes_field_analysis.py --geom-yaml geometry.yaml

    python electrodes_field_analysis.py --geom-yaml geometry.yaml
If no YAML file is given, the default geometry in electrodes_field.py is used.

---------------------------------------------------------------------------
USER SETTINGS (edit inside the script)
---------------------------------------------------------------------------

X0, Y0
    Multipole expansion center (meters)

R_REF
    Reference radius (meters) used for:
        - sampling the extraction circle
        - computing |Cn|*r_ref^(n-1)

M_THETA
    Number of angular samples around the circle

NMAX
    Maximum multipole order to compute

UNIFORMITY_TARGET
    Field uniformity threshold (dimensionless)
    Example:
        0.005  →  0.5%

---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Console output only.

The script prints:

1) A formatted multipole table:

       n  name  Re(Cn)  Im(Cn)  |Cn|  phase(rad)  |Cn|*r_ref^(n-1)

2) The total higher-order field magnitude:

       Sum_{n>2} |Cn|*r_ref^(n-1)

3) The uniformity radius:

       r where d(r) >= UNIFORMITY_TARGET

These outputs are parsed by the CMA-ES optimizer script.

---------------------------------------------------------------------------
MATHEMATICAL BACKGROUND
---------------------------------------------------------------------------

On a circle of radius r_ref:

    F(θ) = Ex(r,θ) - i Ey(r,θ)

The multipole coefficient is computed via discrete Fourier projection:

    Cn = (1 / (2π r_ref^(n-1))) ∫ F(θ) e^{-i(n-1)θ} dθ

In discrete form:

    Cn ≈ (1 / (M r_ref^(n-1))) Σ F(θ_k) e^{-i(n-1)θ_k}

where θ_k = 2π k / M.

Higher-order components (n > 2) represent nonlinear field errors
(sextupole, octupole, etc.).

---------------------------------------------------------------------------
NOTES
---------------------------------------------------------------------------

- The field must form a complete rectangular grid.
- Missing grid points will raise an error.
- This script is designed to be called by optimization drivers,
  but can also be run independently for analysis.

---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from typing import Optional
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
    """Convert flattened [x,y,Ex,Ey] samples into (x1d,y1d,Ex2d,Ey2d).

    This matches the strict rectangular-grid expectation used in field_analysis.py:
    if any (x,y) grid point is missing, raise an error (no silent zero-fill).
    """
    vals_small = np.asarray(vals_small, dtype=float)
    if vals_small.ndim != 2 or vals_small.shape[1] != 4:
        raise ValueError("vals_small must be an (N,4) array: [x,y,Ex,Ey].")

    x = np.unique(vals_small[:, 0])
    y = np.unique(vals_small[:, 1])
    nx, ny = x.size, y.size

    ix = np.searchsorted(x, vals_small[:, 0])
    iy = np.searchsorted(y, vals_small[:, 1])

    Ex = np.full((ny, nx), np.nan, dtype=float)
    Ey = np.full((ny, nx), np.nan, dtype=float)
    Ex[iy, ix] = vals_small[:, 2]
    Ey[iy, ix] = vals_small[:, 3]

    if np.isnan(Ex).any() or np.isnan(Ey).any():
        missing = int(np.isnan(Ex).sum() + np.isnan(Ey).sum())
        raise ValueError(
            f"Data is not a complete rectangular grid (missing {missing} values). "
            "If sampling is irregular, we can switch to scattered interpolation."
        )

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
    """Return the *first* radius where peak annulus deviation reaches/exceeds `target`.

    This matches field_analysis.py (compute_threshold_radii_peak_annulus):
      - E0 is taken at the grid point nearest (x0,y0)
      - rel(x,y) = | |E| - E0 | / E0
      - annulus bins: [rlo, rhi) with dr = 0.5 * min(dx, dy)
      - d(r) = max_{points in annulus} rel
      - return r_center of the first annulus where d(r) >= target

    Returns:
        float radius (bin center) or None if the threshold is never reached.
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

    if E0 <= 0.0 or (not np.isfinite(E0)):
        return None

    rel = np.abs(Emag - E0) / E0

    dx = float(np.min(np.diff(x1d)))
    dy = float(np.min(np.diff(y1d)))
    dr = 0.5 * min(dx, dy)

    rmax = float(np.nanmax(R))
    rbins = np.arange(0.0, rmax + dr, dr)
    if rbins.size < 2:
        return None

    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    d_r = np.full_like(r_centers, np.nan, dtype=float)
    for i in range(r_centers.size):
        rlo, rhi = rbins[i], rbins[i + 1]
        mask = (R >= rlo) & (R < rhi)
        if np.any(mask):
            d_r[i] = float(np.nanmax(rel[mask]))

    idx = np.where(d_r >= target)[0]
    return None if idx.size == 0 else float(r_centers[idx[0]])


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
    geom_yaml: Optional[str] = None,
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
    _max_result, vals_small = run_simulation(sim_cfg, geom_yaml=geom_yaml)

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
    sum_high = float(np.sum(amp[n_arr > 1]))

    # Uniformity radius: first annulus where d(r) >= uniformity_target
    r_uni = _uniformity_radius(x1d, y1d, Ex2d, Ey2d, x0, y0, target=float(uniformity_target))

    print("")
    print(f"Sum_{{n>2}} |Cn|*r_ref^(n-1) = {sum_high:.8e}  (V/m)")
    if r_uni is None:
        print(f"Uniformity threshold d(r) >= {uniformity_target:g} was NOT reached within the grid extent.")
    else:
        print(f"Uniformity radius (first annulus where d(r) >= {uniformity_target:g}) : r = {r_uni:.8e} m")

    return sum_high, r_uni
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run multipole analysis using fields from electrodes_field.py. "
            "Geometry (e1..e4) can be provided via YAML."
        )
    )
    ap.add_argument(
        "geom_yaml",
        nargs="?",
        default=None,
        help="Path to YAML file defining electrode geometry (e1..e4).",
    )
    ap.add_argument(
        "--geom-yaml",
        "-g",
        dest="geom_yaml_flag",
        default=None,
        help="Same as positional geom_yaml; provided for convenience.",
    )
    args = ap.parse_args()

    geom_yaml = args.geom_yaml_flag if args.geom_yaml_flag is not None else args.geom_yaml
    run_analysis(geom_yaml=geom_yaml)


if __name__ == "__main__":
    main()
