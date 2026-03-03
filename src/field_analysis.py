#!/usr/bin/env python3
"""
2D electric-field multipole extraction (dipole/quadrupole/sextupole/octupole)
from Ex(x,y), Ey(x,y) on a rectangular grid.

Conventions used here:
  w = (x-x0) + i (y-y0)
  F(w) = Ex - i Ey = sum_{n>=1} Cn * w^(n-1)

So:
  n=1 dipole, n=2 quadrupole, n=3 sextupole, n=4 octupole

This script:
  1) loads x,y,Ex,Ey from a .npz (recommended) or from text (optional stub)
  2) samples Ex,Ey on a circle (or multiple radii) and computes Cn via Fourier
  3) reconstructs each order as a field map and plots:
       - 2D component maps (Ex and Ey for each order)
       - 1D cuts along x-axis (y=y0) and y-axis (x=x0)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Grid + bilinear interpolation
# ----------------------------
def bilinear_interp(x1d, y1d, F2d, xq, yq):
    """
    Bilinear interpolation for a regular tensor grid:
      x1d shape (nx,), y1d shape (ny,)
      F2d shape (ny, nx) with indexing F2d[jy, ix] = F(y1d[jy], x1d[ix])
    xq,yq can be arrays (same shape).
    Returns interpolated values with same shape as xq.
    Points outside grid are set to NaN.
    """
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    F2d = np.asarray(F2d)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    nx = x1d.size
    ny = y1d.size

    # Require monotonic increasing
    if not (np.all(np.diff(x1d) > 0) and np.all(np.diff(y1d) > 0)):
        raise ValueError("x and y arrays must be strictly increasing for this interpolator.")

    # Find x indices i such that x1d[i] <= xq < x1d[i+1]
    ix = np.searchsorted(x1d, xq) - 1
    iy = np.searchsorted(y1d, yq) - 1

    # Mask out-of-bounds
    oob = (ix < 0) | (ix >= nx - 1) | (iy < 0) | (iy >= ny - 1)
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)

    x0 = x1d[ix]
    x1 = x1d[ix + 1]
    y0 = y1d[iy]
    y1 = y1d[iy + 1]

    # Avoid divide by zero (shouldn't happen if strictly increasing)
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


# ----------------------------
# Multipole extraction
# ----------------------------
def sample_circle(x1d, y1d, Ex, Ey, x0, y0, r, M):
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    xs = x0 + r * np.cos(theta)
    ys = y0 + r * np.sin(theta)

    Exs = bilinear_interp(x1d, y1d, Ex, xs, ys)
    Eys = bilinear_interp(x1d, y1d, Ey, xs, ys)

    # Complex field per convention: F = Ex - i Ey
    F = Exs - 1j * Eys
    return theta, F


def compute_Cn_on_radius(theta, Ftheta, r, nmax):
    """
    Cn = (1/(2pi r^(n-1))) ∫ F(θ) e^{-i(n-1)θ} dθ
    Discrete uniform θ: Cn ≈ (1/(M r^(n-1))) Σ F_k e^{-i(n-1)θ_k}
    """
    M = theta.size
    C = np.zeros(nmax, dtype=complex)  # index 0 corresponds to n=1
    for n in range(1, nmax + 1):
        k = n - 1
        C[n - 1] = (1.0 / (M * (r ** k))) * np.nansum(Ftheta * np.exp(-1j * k * theta))
    return C


def compute_Cn_multiradius(x1d, y1d, Ex, Ey, x0, y0, radii, M, nmax):
    """
    Compute Cn for each radius, then average (ignoring NaNs).
    Returns:
      Cavg: (nmax,) complex
      Call: (nr, nmax) complex
    """
    Call = []
    for r in radii:
        theta, F = sample_circle(x1d, y1d, Ex, Ey, x0, y0, r, M)
        C = compute_Cn_on_radius(theta, F, r, nmax)
        Call.append(C)
    Call = np.vstack(Call)  # (nr, nmax)
    # Simple average across radii
    Cavg = np.nanmean(Call, axis=0)
    return Cavg, Call


# ----------------------------
# Field reconstruction for each order
# ----------------------------
def reconstruct_order_field(xg, yg, x0, y0, Cn, n):
    """
    Build Ex_n(x,y), Ey_n(x,y) for a single order n using:
      F_n = Ex_n - i Ey_n = Cn * w^(n-1), w = (x-x0) + i(y-y0)
    """
    w = (xg - x0) + 1j * (yg - y0)
    Fn = Cn * (w ** (n - 1))
    Exn = np.real(Fn)
    Eyn = -np.imag(Fn)
    return Exn, Eyn


def nearest_index(arr, val):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - val)))


# ----------------------------
# Plot helpers
# ----------------------------
def plot_component_maps(x1d, y1d, Exn, Eyn, title_prefix, out_prefix=None):
    extent = [x1d[0], x1d[-1], y1d[0], y1d[-1]]

    fig, ax = plt.subplots()
    im = ax.imshow(Exn, origin="lower", extent=extent, aspect="auto")
    ax.set_title(f"{title_prefix}: Ex")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)

    if out_prefix:
        fig.savefig(f"{out_prefix}_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    im = ax.imshow(Eyn, origin="lower", extent=extent, aspect="auto")
    ax.set_title(f"{title_prefix}: Ey")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)

    if out_prefix:
        fig.savefig(f"{out_prefix}_Ey.png", dpi=200, bbox_inches="tight")


def plot_1d_cuts(x1d, y1d, Ex_total, Ey_total, comps, x0, y0, out_prefix=None):
    """
    comps: list of dicts with keys: name, Ex, Ey
    """
    ix0 = nearest_index(x1d, x0)
    iy0 = nearest_index(y1d, y0)

    # Along x (y=y0)
    fig, ax = plt.subplots()
    ax.plot(x1d, Ex_total[iy0, :], label="Total Ex")
    for c in comps:
        ax.plot(x1d, c["Ex"][iy0, :], label=c["name"])
    ax.set_title("1D cut along x (y = y0): Ex")
    ax.set_xlabel("x")
    ax.set_ylabel("Ex")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_x_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(x1d, Ey_total[iy0, :], label="Total Ey")
    for c in comps:
        ax.plot(x1d, c["Ey"][iy0, :], label=c["name"])
    ax.set_title("1D cut along x (y = y0): Ey")
    ax.set_xlabel("x")
    ax.set_ylabel("Ey")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_x_Ey.png", dpi=200, bbox_inches="tight")

    # Along y (x=x0)
    fig, ax = plt.subplots()
    ax.plot(y1d, Ex_total[:, ix0], label="Total Ex")
    for c in comps:
        ax.plot(y1d, c["Ex"][:, ix0], label=c["name"])
    ax.set_title("1D cut along y (x = x0): Ex")
    ax.set_xlabel("y")
    ax.set_ylabel("Ex")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_y_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(y1d, Ey_total[:, ix0], label="Total Ey")
    for c in comps:
        ax.plot(y1d, c["Ey"][:, ix0], label=c["name"])
    ax.set_title("1D cut along y (x = x0): Ey")
    ax.set_xlabel("y")
    ax.set_ylabel("Ey")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_y_Ey.png", dpi=200, bbox_inches="tight")


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Input .npz containing x,y,Ex,Ey")
    p.add_argument("--x0", type=float, required=True, help="Reference center x0")
    p.add_argument("--y0", type=float, required=True, help="Reference center y0")
    p.add_argument("--r", type=float, default=None, help="Circle radius for extraction (single radius).")
    p.add_argument("--rmin", type=float, default=None, help="If set, use multiple radii: rmin..rmax.")
    p.add_argument("--rmax", type=float, default=None, help="If set, use multiple radii: rmin..rmax.")
    p.add_argument("--nr", type=int, default=5, help="Number of radii if using rmin/rmax.")
    p.add_argument("--M", type=int, default=512, help="Number of angular samples on circle.")
    p.add_argument("--nmax", type=int, default=4, help="Max order n to compute (4=octupole).")
    p.add_argument("--out", type=str, default="multipole", help="Output prefix for saved figures.")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = p.parse_args()

    data = np.load(args.npz)
    x = data["x"]
    y = data["y"]
    Ex = data["Ex"]
    Ey = data["Ey"]

    # Basic checks
    if Ex.shape != Ey.shape:
        raise ValueError("Ex and Ey must have same shape.")
    if Ex.shape != (y.size, x.size):
        raise ValueError("Expected Ex/Ey shape (ny,nx) matching y,x lengths.")

    x0, y0 = args.x0, args.y0
    nmax = args.nmax

    # Choose radii
    if args.rmin is not None and args.rmax is not None:
        radii = np.linspace(args.rmin, args.rmax, args.nr)
    elif args.r is not None:
        radii = np.array([args.r], dtype=float)
    else:
        raise ValueError("Provide either --r or both --rmin and --rmax.")

    # Compute coefficients
    Cavg, Call = compute_Cn_multiradius(x, y, Ex, Ey, x0, y0, radii, args.M, nmax)

    # Print results
    names = {1: "dipole", 2: "quadrupole", 3: "sextupole", 4: "octupole"}
    print("=== Multipole coefficients Cn (using F = Ex - i Ey) ===")
    print(f"Center (x0,y0)=({x0},{y0})")
    print(f"Radii used: {radii}")
    for n in range(1, nmax + 1):
        cn = Cavg[n - 1]
        mag = np.abs(cn)
        ph = np.angle(cn)
        nm = names.get(n, f"order{n}")
        print(f"n={n:2d} ({nm:10s}): Cn = {cn.real:+.6e} {cn.imag:+.6e}j   |Cn|={mag:.6e}  phase(rad)={ph:+.6f}")

    if radii.size > 1:
        print("\nRadius-consistency check (Cn per radius):")
        for ir, r in enumerate(radii):
            row = Call[ir]
            row_str = "  ".join([f"n={n}:{row[n-1].real:+.2e}{row[n-1].imag:+.2e}j" for n in range(1, nmax + 1)])
            print(f"  r={r:.6g}: {row_str}")

    # Build reconstruction grids
    X, Y = np.meshgrid(x, y)

    comps = []
    for n in range(1, nmax + 1):
        cn = Cavg[n - 1]
        Exn, Eyn = reconstruct_order_field(X, Y, x0, y0, cn, n)
        nm = names.get(n, f"order{n}")
        comps.append({"n": n, "name": nm, "Ex": Exn, "Ey": Eyn})

        plot_component_maps(x, y, Exn, Eyn, f"{nm} (n={n})", out_prefix=f"{args.out}_{nm}")

    # Optional: also plot total field maps for context
    plot_component_maps(x, y, Ex, Ey, "Total field", out_prefix=f"{args.out}_total")

    # 1D cuts along x and y through (x0,y0)
    plot_1d_cuts(x, y, Ex, Ey, comps, x0, y0, out_prefix=args.out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()