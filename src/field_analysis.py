#!/usr/bin/env python3
"""
fieldanalysis multipole tool (2D Ex/Ey)

Improvements implemented (per your requests):
1) No plt.show(); figures are only saved.
2) No auto-detect project root. You set PROJECT_ROOT as an absolute path string.
   You set FIELD_FILE manually; default suffix is ".dat" if not provided.
3) The 0.2/0.5/1% dashed circles are drawn ONLY on total_Emag.png (not on other maps),
   and use different colors.
4) All map plots use cmap="bwr".
5) Each run creates outputs/<field_stem>/<run_stamp>/ and saves everything there.
6) Creates two 2D output data files for 1D cuts:
     <field_stem>_x.dat  (cut at y≈y0)
     <field_stem>_y.dat  (cut at x≈x0)
   Each includes total + multipole components.
7) Creates <field_stem>_OPT.txt containing:
   - multipole definitions
   - center, radii
   - n=1..4: Cn, |Cn|, phase
   - 0.2%, 0.5%, 1% radii info from total |E| map

Input formats supported:
- Text table (.dat/.txt/.csv) with header columns: X Y Ex Ey
  (units: m, m, V/m, V/m)
- True NumPy .npz (zip) with arrays X,Y,Ex,Ey or x,y,Ex,Ey (optional)

Multipole convention used:
  w = (x-x0) + i (y-y0)
  F = Ex - i Ey = Σ_{n>=1} Cn * w^(n-1)
  n=1 dipole, n=2 quadrupole, n=3 sextupole, n=4 octupole
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS (DIRECT MODE)
# =============================================================================
PROJECT_ROOT = "/Users/wange/Coding/Python/fieldanalysis"

# You will manually put in field filename (can be absolute path, or relative to PROJECT_ROOT).
# If no suffix is given, ".dat" is assumed.
FIELD_FILE = "data/03032026_t commitwien.dat"  # examples: "data/field1", "/abs/path/to/field1.dat"

# Analysis parameters (direct mode defaults)
X0 = 0.0
Y0 = 0.0
# Choose either single radius:
R_SINGLE = 0.029
# Or multi-radius (set R_SINGLE=None and set rmin/rmax):
RMIN = None
RMAX = None
NR = 5

M_THETA = 512
NMAX = 4  # up to octupole by default
SAVE_DPI = 200

# Threshold circles on total |E| map:
THRESHOLDS = (0.002, 0.005, 0.01)  # 0.2%, 0.5%, 1.0%
CIRCLE_COLORS = ("gold", "lime", "cyan")  # distinct colors for the 3 circles


# =============================================================================
# DATA LOADING
# =============================================================================
def resolve_input_path(project_root: str, field_file: str, default_suffix: str = ".dat") -> Path:
    p = Path(field_file).expanduser()
    if not p.is_absolute():
        p = Path(project_root).expanduser().resolve() / p
    if p.suffix == "":
        p = p.with_suffix(default_suffix)
    return p.resolve()


def load_field_any(path: str | Path):
    """
    Load either:
      - text table file (.dat/.txt/.csv) with header columns: X Y Ex Ey
      - true NumPy .npz archive containing X,Y,Ex,Ey (or x,y,Ex,Ey)

    Returns:
      x (nx,), y (ny,), Ex (ny,nx), Ey (ny,nx)
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Cannot find field file: {p}")

    # True npz support (optional)
    if p.suffix.lower() == ".npz" and zipfile.is_zipfile(p):
        data = np.load(p)
        keys = {k.lower(): k for k in data.files}

        def pick(*names):
            for n in names:
                if n.lower() in keys:
                    return np.asarray(data[keys[n.lower()]])
            return None

        X = pick("x", "X")
        Y = pick("y", "Y")
        Ex = pick("ex", "Ex", "EX")
        Ey = pick("ey", "Ey", "EY")
        if X is None or Y is None or Ex is None or Ey is None:
            raise ValueError(f"NPZ missing required arrays. Found keys: {data.files}")

        if X.ndim == 1 and Y.ndim == 1:
            x, y = X, Y
            if Ex.shape != (y.size, x.size) or Ey.shape != (y.size, x.size):
                raise ValueError(f"Expected Ex/Ey shape (ny,nx)={(y.size,x.size)}, got {Ex.shape}, {Ey.shape}")
            return x, y, Ex, Ey

        if X.ndim == 2 and Y.ndim == 2:
            x = np.unique(X[0, :])
            y = np.unique(Y[:, 0])
            if Ex.shape != X.shape or Ey.shape != X.shape:
                raise ValueError("For 2D X,Y, Ex and Ey must match X,Y 2D shape.")
            return x, y, Ex, Ey

        raise ValueError("Unsupported NPZ shapes for X,Y. Use 1D axes or 2D grids.")

    # Text table case
    arr = np.genfromtxt(p, delimiter=None, names=True, dtype=float, encoding=None)
    if arr.dtype.names is None:
        raise ValueError("Could not read header. Ensure first line has: X  Y  Ex  Ey")

    col = {n.lower(): n for n in arr.dtype.names}

    def getcol(*names):
        for n in names:
            if n.lower() in col:
                return arr[col[n.lower()]]
        return None

    Xc = getcol("x", "X")
    Yc = getcol("y", "Y")
    Exc = getcol("ex", "Ex", "EX")
    Eyc = getcol("ey", "Ey", "EY")

    if Xc is None or Yc is None or Exc is None or Eyc is None:
        raise ValueError(f"Missing required columns. Found: {arr.dtype.names}")

    x = np.unique(Xc)
    y = np.unique(Yc)
    nx, ny = x.size, y.size

    ix = np.searchsorted(x, Xc)
    iy = np.searchsorted(y, Yc)

    Ex = np.full((ny, nx), np.nan, dtype=float)
    Ey = np.full((ny, nx), np.nan, dtype=float)
    Ex[iy, ix] = Exc
    Ey[iy, ix] = Eyc

    if np.isnan(Ex).any() or np.isnan(Ey).any():
        missing = int(np.isnan(Ex).sum() + np.isnan(Ey).sum())
        raise ValueError(
            f"Data is not a complete rectangular grid (missing {missing} values). "
            "If sampling is irregular, we can switch to scattered interpolation."
        )

    return x, y, Ex, Ey


# =============================================================================
# INTERPOLATION + MULTIPOLE
# =============================================================================
def bilinear_interp(x1d, y1d, F2d, xq, yq):
    """
    Bilinear interpolation for regular tensor grid.
    F2d is indexed as F2d[jy, ix] = F(y[jy], x[ix]).
    """
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    F2d = np.asarray(F2d)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    if not (np.all(np.diff(x1d) > 0) and np.all(np.diff(y1d) > 0)):
        raise ValueError("x and y arrays must be strictly increasing for bilinear interpolation.")

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


def sample_circle(x1d, y1d, Ex, Ey, x0, y0, r, M):
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    xs = x0 + r * np.cos(theta)
    ys = y0 + r * np.sin(theta)

    Exs = bilinear_interp(x1d, y1d, Ex, xs, ys)
    Eys = bilinear_interp(x1d, y1d, Ey, xs, ys)

    F = Exs - 1j * Eys  # convention
    return theta, F


def compute_Cn_on_radius(theta, Ftheta, r, nmax):
    M = theta.size
    C = np.zeros(nmax, dtype=complex)
    for n in range(1, nmax + 1):
        k = n - 1
        C[n - 1] = (1.0 / (M * (r ** k))) * np.nansum(Ftheta * np.exp(-1j * k * theta))
    return C


def compute_Cn_multiradius(x1d, y1d, Ex, Ey, x0, y0, radii, M, nmax):
    Call = []
    for r in radii:
        theta, F = sample_circle(x1d, y1d, Ex, Ey, x0, y0, r, M)
        Call.append(compute_Cn_on_radius(theta, F, r, nmax))
    Call = np.vstack(Call)
    Cavg = np.nanmean(Call, axis=0)
    return Cavg, Call


def reconstruct_order_field(xg, yg, x0, y0, Cn, n):
    w = (xg - x0) + 1j * (yg - y0)
    Fn = Cn * (w ** (n - 1))      # Fn = Ex - i Ey
    Exn = np.real(Fn)
    Eyn = -np.imag(Fn)
    return Exn, Eyn


def nearest_index(arr, val):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - val)))


# =============================================================================
# PLOTTING (SAVE ONLY)
# =============================================================================
def _save_imshow(data2d, extent, title, xlabel, ylabel, cbar_label, out_path: Path, symmetric=False):
    fig, ax = plt.subplots()
    if symmetric:
        vmax = float(np.nanmax(np.abs(data2d)))
        vmin = -vmax
    else:
        vmin, vmax = None, None

    im = ax.imshow(
        data2d,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_threshold_radii_peak_annulus(
    x1d, y1d, Emag, x0, y0, thresholds=(0.002, 0.005, 0.01)
):
    """
    Peak (worst-case) deviation in annulus:
      d(r) = max_{points in annulus} | |E| - E0 | / E0
    Return dict threshold->radius (or None).
    """
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    Xg, Yg = np.meshgrid(x1d, y1d)
    R = np.sqrt((Xg - x0) ** 2 + (Yg - y0) ** 2)

    ix0 = nearest_index(x1d, x0)
    iy0 = nearest_index(y1d, y0)
    E0 = float(Emag[iy0, ix0])

    if E0 <= 0 or not np.isfinite(E0):
        return {"E0": E0, "ix0": ix0, "iy0": iy0, "radii": {t: None for t in thresholds}}

    rel = np.abs(Emag - E0) / E0

    dx = float(np.min(np.diff(x1d)))
    dy = float(np.min(np.diff(y1d)))
    dr = 0.5 * min(dx, dy)

    rmax = float(np.nanmax(R))
    rbins = np.arange(0.0, rmax + dr, dr)
    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    d_r = np.full_like(r_centers, np.nan, dtype=float)
    for i in range(r_centers.size):
        rlo, rhi = rbins[i], rbins[i + 1]
        mask = (R >= rlo) & (R < rhi)
        if np.any(mask):
            d_r[i] = float(np.nanmax(rel[mask]))

    radii = {}
    for t in thresholds:
        idx = np.where(d_r >= t)[0]
        radii[t] = None if idx.size == 0 else float(r_centers[idx[0]])

    return {"E0": E0, "ix0": ix0, "iy0": iy0, "radii": radii}


def plot_component_maps_save_only(
    x1d, y1d, Ex2d, Ey2d, title_prefix: str, out_base: Path,
    save_emag: bool = True,
    overlay_threshold_circles: bool = False,
    x0: float | None = None,
    y0: float | None = None,
    thresholds=(0.002, 0.005, 0.01),
    circle_colors=("gold", "lime", "cyan"),
):
    """
    Saves Ex, Ey, and optionally |E| maps.
    Threshold circles are drawn ONLY if overlay_threshold_circles=True.
    Returns threshold_radii_info (dict) if circles computed, else None.
    """
    extent = [x1d[0], x1d[-1], y1d[0], y1d[-1]]

    _save_imshow(
        Ex2d, extent,
        title=f"{title_prefix}: Ex",
        xlabel="x (m)", ylabel="y (m)",
        cbar_label="Ex (V/m)",
        out_path=Path(str(out_base) + "_Ex.png"),
        symmetric=True
    )
    _save_imshow(
        Ey2d, extent,
        title=f"{title_prefix}: Ey",
        xlabel="x (m)", ylabel="y (m)",
        cbar_label="Ey (V/m)",
        out_path=Path(str(out_base) + "_Ey.png"),
        symmetric=True
    )

    info = None
    if save_emag:
        Emag = np.sqrt(Ex2d**2 + Ey2d**2)

        # Save magnitude map (with optional circles)
        fig, ax = plt.subplots()
        im = ax.imshow(
            Emag, origin="lower", extent=extent, aspect="auto",
            cmap="bwr"  # requested even for magnitude
        )
        ax.set_title(f"{title_prefix}: |E| = sqrt(Ex^2 + Ey^2)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, label="|E| (V/m)")

        if overlay_threshold_circles and (x0 is not None) and (y0 is not None):
            info = compute_threshold_radii_peak_annulus(x1d, y1d, Emag, x0, y0, thresholds=thresholds)

            # mark the exact grid point used for E0
            ix0 = info["ix0"]
            iy0 = info["iy0"]
            ax.plot([x1d[ix0]], [y1d[iy0]], marker="o", markersize=4, color="k")

            # circles
            for (t, c) in zip(thresholds, circle_colors):
                rt = info["radii"].get(t, None)
                if rt is None:
                    continue
                circ = plt.Circle((x0, y0), rt, fill=False, linestyle="--", linewidth=1.8, color=c)
                ax.add_patch(circ)
                ax.text(x0 + rt, y0, f"{100*t:.3g}%", va="bottom", ha="left", color=c)

        fig.savefig(Path(str(out_base) + "_Emag.png"), dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    return info


# =============================================================================
# OUTPUT FILES
# =============================================================================


def write_cuts_files_and_figures(run_dir: Path, stem: str, x, y, x0, y0, Ex_total, Ey_total, comps):
    """
    Create:
      - stem_x.dat: cut along x at y≈y0
      - stem_y.dat: cut along y at x≈x0
    And 1D figures:
      - stem_cut_x_Ex.png, stem_cut_x_Ey.png
      - stem_cut_y_Ex.png, stem_cut_y_Ey.png

    comps: list of dicts {name, Ex, Ey} where name is dipole/quadrupole/sextupole/octupole
    """
    ix0 = nearest_index(x, x0)
    iy0 = nearest_index(y, y0)

    # -------------------------
    # Along x (y ≈ y0)
    # -------------------------
    xline = x
    Ex_tot_x = Ex_total[iy0, :]
    Ey_tot_x = Ey_total[iy0, :]

    data_x = [xline, Ex_tot_x, Ey_tot_x]
    header_x = ["x(m)", "Ex_total(V/m)", "Ey_total(V/m)"]

    for c in comps:
        data_x.append(c["Ex"][iy0, :])
        data_x.append(c["Ey"][iy0, :])
        header_x.append(f"Ex_{c['name']}(V/m)")
        header_x.append(f"Ey_{c['name']}(V/m)")

    A = np.column_stack(data_x)
    out_x = run_dir / f"{stem}_x.dat"
    np.savetxt(out_x, A, header="  ".join(header_x), comments="")

    # 1D figure: Ex along x
    fig, ax = plt.subplots()
    ax.plot(xline, Ex_tot_x, label="Total")
    for c in comps:
        ax.plot(xline, c["Ex"][iy0, :], label=c["name"])
    ax.set_title(f"Ex along x at y≈{y[iy0]:.6g} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Ex (V/m)")
    ax.legend()
    fig.savefig(run_dir / f"{stem}_cut_x_Ex.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 1D figure: Ey along x
    fig, ax = plt.subplots()
    ax.plot(xline, Ey_tot_x, label="Total")
    for c in comps:
        ax.plot(xline, c["Ey"][iy0, :], label=c["name"])
    ax.set_title(f"Ey along x at y≈{y[iy0]:.6g} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Ey (V/m)")
    ax.legend()
    fig.savefig(run_dir / f"{stem}_cut_x_Ey.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Along y (x ≈ x0)
    # -------------------------
    yline = y
    Ex_tot_y = Ex_total[:, ix0]
    Ey_tot_y = Ey_total[:, ix0]

    data_y = [yline, Ex_tot_y, Ey_tot_y]
    header_y = ["y(m)", "Ex_total(V/m)", "Ey_total(V/m)"]

    for c in comps:
        data_y.append(c["Ex"][:, ix0])
        data_y.append(c["Ey"][:, ix0])
        header_y.append(f"Ex_{c['name']}(V/m)")
        header_y.append(f"Ey_{c['name']}(V/m)")

    B = np.column_stack(data_y)
    out_y = run_dir / f"{stem}_y.dat"
    np.savetxt(out_y, B, header="  ".join(header_y), comments="")

    # 1D figure: Ex along y
    fig, ax = plt.subplots()
    ax.plot(yline, Ex_tot_y, label="Total")
    for c in comps:
        ax.plot(yline, c["Ex"][:, ix0], label=c["name"])
    ax.set_title(f"Ex along y at x≈{x[ix0]:.6g} m")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Ex (V/m)")
    ax.legend()
    fig.savefig(run_dir / f"{stem}_cut_y_Ex.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 1D figure: Ey along y
    fig, ax = plt.subplots()
    ax.plot(yline, Ey_tot_y, label="Total")
    for c in comps:
        ax.plot(yline, c["Ey"][:, ix0], label=c["name"])
    ax.set_title(f"Ey along y at x≈{x[ix0]:.6g} m")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Ey (V/m)")
    ax.legend()
    fig.savefig(run_dir / f"{stem}_cut_y_Ey.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "cut_x_file": str(out_x),
        "cut_y_file": str(out_y),
        "x_cut_y_used": float(y[iy0]),
        "y_cut_x_used": float(x[ix0]),
        "fig_cut_x_Ex": str(run_dir / f"{stem}_cut_x_Ex.png"),
        "fig_cut_x_Ey": str(run_dir / f"{stem}_cut_x_Ey.png"),
        "fig_cut_y_Ex": str(run_dir / f"{stem}_cut_y_Ex.png"),
        "fig_cut_y_Ey": str(run_dir / f"{stem}_cut_y_Ey.png"),
    }

def write_opt_file(
    run_dir: Path,
    stem: str,
    input_path: Path,
    x0: float,
    y0: float,
    radii: np.ndarray,
    M: int,
    nmax: int,
    Cavg: np.ndarray,
    threshold_info: dict | None,
    r_ref: float,
):
    """
    Writes <stem>_OPT.txt into run_dir.

    Includes:
      - multipole convention / definitions
      - center, radii, M, nmax
      - table of n=1..nmax with: Re(Cn), Im(Cn), |Cn|, phase(rad), |Cn|*r_ref^(n-1)
      - threshold circle radii (0.2/0.5/1%) from total |E| map
    """
    names = {1: "dipole", 2: "quadrupole", 3: "sextupole", 4: "octupole"}
    out_opt = run_dir / f"{stem}_OPT.txt"

    r_ref = float(r_ref)

    lines = []
    lines.append("=== 2D Electric Field Multipole Analysis (Ex,Ey) ===")
    lines.append(f"Input file: {input_path}")
    lines.append("")
    lines.append("Definition / convention used:")
    lines.append("  w = (x - x0) + i (y - y0)")
    lines.append("  F = Ex - i Ey")
    lines.append("  F(w) = Σ_{n>=1} Cn * w^(n-1)")
    lines.append("  n=1 dipole, n=2 quadrupole, n=3 sextupole, n=4 octupole")
    lines.append("")
    lines.append("Units:")
    lines.append("  Ex,Ey: V/m")
    lines.append("  w^(n-1): m^(n-1)")
    lines.append("  => Cn: V/m^n")
    lines.append("  => |Cn| * r_ref^(n-1): V/m  (field amplitude scale at radius r_ref)")
    lines.append("")
    lines.append(f"Center used: x0 = {x0:.16g} m, y0 = {y0:.16g} m")
    lines.append(f"Angular samples on circle: M = {int(M)}")
    lines.append(f"nmax: {int(nmax)}")
    lines.append(f"Radii used (m): {', '.join([f'{float(r):.16g}' for r in radii])}")
    lines.append(f"r_ref for normalized amplitude (m): {r_ref:.16g}")
    lines.append("")
    lines.append("Multipole coefficients:")
    lines.append("  Columns:")
    lines.append("    n  name        Re(Cn)        Im(Cn)        |Cn|         phase(rad)   |Cn|*r_ref^(n-1)")
    lines.append("    -  ----        ------        ------        ----         ----------   ----------------")

    for n in range(1, int(nmax) + 1):
        cn = Cavg[n - 1]
        mag = float(np.abs(cn))
        ph = float(np.angle(cn))
        nm = names.get(n, f"order{n}")
        norm_amp = mag * (r_ref ** (n - 1))  # units: V/m
        lines.append(
            f"    {n:1d}  {nm:10s}  {cn.real:+.8e}  {cn.imag:+.8e}  {mag:.8e}  {ph:+.8f}  {norm_amp:.8e}"
        )

    lines.append("")
    lines.append("Total |E| map threshold-circle radii (peak deviation in annulus):")
    lines.append("  Deviation definition: d(r) = max_annulus | |E(r,θ)| - E0 | / E0")
    lines.append("  where E0 = |E| at nearest grid point to (x0,y0).")

    if threshold_info is None:
        lines.append("  (not computed)")
    else:
        E0 = threshold_info.get("E0", None)
        ix0 = threshold_info.get("ix0", None)
        iy0 = threshold_info.get("iy0", None)
        lines.append(f"  E0 = {float(E0):.8e} V/m")
        lines.append(f"  Center grid index used: ix0={ix0}, iy0={iy0}")

        radii_dict = threshold_info.get("radii", {})
        # Keep ordering 0.2%, 0.5%, 1.0% if present
        for t in sorted(radii_dict.keys()):
            rt = radii_dict[t]
            if rt is None:
                lines.append(f"  {100*t:.3g}% : not reached within map extent")
            else:
                lines.append(f"  {100*t:.3g}% : r = {float(rt):.8e} m")

    out_opt.write_text("\n".join(lines) + "\n")
    return str(out_opt)


# =============================================================================
# RUN CORE
# =============================================================================
@dataclass
class Config:
    project_root: str
    input_file: str
    x0: float
    y0: float
    r: float | None
    rmin: float | None
    rmax: float | None
    nr: int
    M: int
    nmax: int
    thresholds: tuple[float, float, float]
    circle_colors: tuple[str, str, str]

def run(cfg: Config):
    # Resolve + load
    input_path = resolve_input_path(cfg.project_root, cfg.input_file, default_suffix=".dat")
    x, y, Ex, Ey = load_field_any(input_path)

    # Choose radii
    if cfg.rmin is not None and cfg.rmax is not None:
        radii = np.linspace(float(cfg.rmin), float(cfg.rmax), int(cfg.nr))
    elif cfg.r is not None:
        radii = np.array([float(cfg.r)], dtype=float)
    else:
        raise ValueError("Provide either r (single) or (rmin,rmax).")

    # Reference radius for normalized amplitude column in OPT:
    # Use R_SINGLE if present, otherwise use RMAX (as you requested).
    if cfg.r is not None:
        r_ref = float(cfg.r)
    elif cfg.rmax is not None:
        r_ref = float(cfg.rmax)
    else:
        # fallback (shouldn't happen if above logic is correct)
        r_ref = float(radii[-1])

    # Output directory: outputs/<field_stem>/<run_stamp>/
    stem = input_path.stem
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.project_root).expanduser().resolve() / "outputs" / stem / run_stamp
    out_root.mkdir(parents=True, exist_ok=True)

    # Compute multipoles
    Cavg, Call = compute_Cn_multiradius(x, y, Ex, Ey, cfg.x0, cfg.y0, radii, cfg.M, cfg.nmax)

    names = {1: "dipole", 2: "quadrupole", 3: "sextupole", 4: "octupole"}
    Xg, Yg = np.meshgrid(x, y)

    comps = []
    for n in range(1, cfg.nmax + 1):
        cn = Cavg[n - 1]
        Exn, Eyn = reconstruct_order_field(Xg, Yg, cfg.x0, cfg.y0, cn, n)
        nm = names.get(n, f"order{n}")
        comps.append({"n": n, "name": nm, "Ex": Exn, "Ey": Eyn})

        # Save component maps (Ex/Ey/Emag) without threshold circles
        base = out_root / f"{stem}_{nm}"
        plot_component_maps_save_only(
            x, y, Exn, Eyn,
            title_prefix=f"{nm} (n={n})",
            out_base=base,
            save_emag=True,
            overlay_threshold_circles=False,
            x0=cfg.x0, y0=cfg.y0
        )

    # Save total maps with threshold circles ONLY on total Emag
    total_base = out_root / f"{stem}_total"
    threshold_info = plot_component_maps_save_only(
        x, y, Ex, Ey,
        title_prefix="Total field",
        out_base=total_base,
        save_emag=True,
        overlay_threshold_circles=True,
        x0=cfg.x0, y0=cfg.y0,
        thresholds=cfg.thresholds,
        circle_colors=cfg.circle_colors
    )

    # 1D cuts files + 1D figures
    cuts_info = write_cuts_files_and_figures(
        out_root, stem, x, y, cfg.x0, cfg.y0,
        Ex, Ey, comps
    )

    # OPT report file (now includes normalized amplitude using r_ref)
    opt_file = write_opt_file(
        out_root,
        stem,
        input_path,
        cfg.x0,
        cfg.y0,
        radii,
        cfg.M,
        cfg.nmax,
        Cavg,
        threshold_info=threshold_info,
        r_ref=r_ref,   # <-- key line: pass the reference radius in
    )

    # Short terminal summary
    print(f"Saved outputs to: {out_root}")
    print(f"  OPT:  {opt_file}")
    print(f"  Cuts: {cuts_info['cut_x_file']}  {cuts_info['cut_y_file']}")
    print(f"  1D figs: {cuts_info['fig_cut_x_Ex']}  {cuts_info['fig_cut_x_Ey']}  "
          f"{cuts_info['fig_cut_y_Ex']}  {cuts_info['fig_cut_y_Ey']}")
    if threshold_info is not None:
        for t, rt in threshold_info["radii"].items():
            print(f"  {100*t:.3g}% radius: {rt}")

# =============================================================================
# MAIN: CLI mode OR DIRECT mode
# =============================================================================
def main(mode=0):
    """
    mode=0: DIRECT mode (uses the variables at top of file)
    mode=1: CLI mode
    """
    if mode == 1:
        p = argparse.ArgumentParser()
        p.add_argument("--project_root", type=str, required=True, help="Absolute path to project root")
        p.add_argument("--input", type=str, required=True, help="Field file path (absolute or relative to project_root)")
        p.add_argument("--x0", type=float, required=True)
        p.add_argument("--y0", type=float, required=True)
        p.add_argument("--r", type=float, default=None)
        p.add_argument("--rmin", type=float, default=None)
        p.add_argument("--rmax", type=float, default=None)
        p.add_argument("--nr", type=int, default=5)
        p.add_argument("--M", type=int, default=512)
        p.add_argument("--nmax", type=int, default=4)
        args = p.parse_args()

        cfg = Config(
            project_root=args.project_root,
            input_file=args.input,
            x0=args.x0,
            y0=args.y0,
            r=args.r,
            rmin=args.rmin,
            rmax=args.rmax,
            nr=args.nr,
            M=args.M,
            nmax=args.nmax,
            thresholds=THRESHOLDS,
            circle_colors=CIRCLE_COLORS,
        )
        run(cfg)
        return

    # DIRECT mode
    cfg = Config(
        project_root=PROJECT_ROOT,
        input_file=FIELD_FILE,
        x0=X0,
        y0=Y0,
        r=R_SINGLE,
        rmin=RMIN,
        rmax=RMAX,
        nr=NR,
        M=M_THETA,
        nmax=NMAX,
        thresholds=THRESHOLDS,
        circle_colors=CIRCLE_COLORS,
    )
    run(cfg)


if __name__ == "__main__":
    # Default: direct mode (edit variables at top)
    main(mode=0)

    # If you want CLI mode instead, switch to:
    # main(mode=1)