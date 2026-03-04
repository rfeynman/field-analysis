#!/usr/bin/env python3
"""
multipole_2d.py

2D electric-field multipole extraction (dipole/quadrupole/sextupole/octupole)
from Ex(x,y), Ey(x,y) on a rectangular grid.

Project layout (recommended):
  project/
    data/field.dat      # or field.npz (true numpy npz archive)
    src/multipole_2d.py

Your field.dat format (text table):
  Header line:   X    Y    Ex    Ey
  Units:         m    m    V/m   V/m
  Data:          tab/space delimited, scientific notation allowed

Conventions used here:
  w = (x-x0) + i (y-y0)
  F(w) = Ex - i Ey = sum_{n>=1} Cn * w^(n-1)

So:
  n=1 dipole    (constant field)
  n=2 quadrupole
  n=3 sextupole
  n=4 octupole
  ...

Extraction method:
  Sample F(θ)=Ex - i Ey on a circle of radius r about (x0,y0)
  Cn ≈ (1/(M r^(n-1))) Σ_k F(θ_k) exp(-i(n-1)θ_k)

Outputs:
  - Prints Cn
  - Saves component maps (Ex and Ey) for each order and total field
  - Saves 1D cuts along x (y=y0) and y (x=x0)
  - Outputs go to project_root/outputs/ by default

Two run modes:
  mode=1 (default): command line parsing
  mode=0: direct mode; edit config dict inside main()

Usage examples:
  # From project root:
  python src/multipole_2d.py --x0 0 --y0 0 --r 0.01 --nmax 4 --out test --show

  # Explicit input:
  python src/multipole_2d.py --npz data/field.dat --x0 0 --y0 0 --rmin 0.005 --rmax 0.015 --nr 7 --out test
"""

import argparse
from pathlib import Path
import zipfile
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Paths (project layout helpers)
# ----------------------------
def get_project_paths(default_data_name: str = "field.dat") -> dict:
    """
    Assumes this script lives in project_root/src/.
    Returns paths dict including default input in project_root/data/.
    """
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    out_dir = project_root / "outputs"
    default_input = data_dir / default_data_name
    return {
        "script_path": script_path,
        "script_dir": script_dir,
        "project_root": project_root,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "default_input": default_input,
    }


# ----------------------------
# Data loading
# ----------------------------
def load_field_any(path):
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

    # True .npz archive support (optional)
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

        # 1D axes + 2D fields
        if X.ndim == 1 and Y.ndim == 1:
            x, y = X, Y
            if Ex.shape != (y.size, x.size) or Ey.shape != (y.size, x.size):
                raise ValueError(f"Expected Ex/Ey shape (ny,nx)={(y.size,x.size)}, got {Ex.shape}, {Ey.shape}")
            return x, y, Ex, Ey

        # 2D meshgrid + 2D fields
        if X.ndim == 2 and Y.ndim == 2:
            x = np.unique(X[0, :])
            y = np.unique(Y[:, 0])
            if Ex.shape != X.shape or Ey.shape != X.shape:
                raise ValueError("For 2D X,Y, Ex and Ey must match X,Y 2D shape.")
            return x, y, Ex, Ey

        raise ValueError("Unsupported NPZ shapes for X,Y. Use 1D axes or 2D grids.")

    # Text table case (your field.dat)
    arr = np.genfromtxt(p, delimiter=None, names=True, dtype=float, encoding=None)

    if arr.dtype.names is None:
        raise ValueError(
            "Could not read header. Ensure first line contains column names like: X  Y  Ex  Ey"
        )

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

    # Require complete rectangular grid for this version
    if np.isnan(Ex).any() or np.isnan(Ey).any():
        missing = int(np.isnan(Ex).sum() + np.isnan(Ey).sum())
        raise ValueError(
            f"Data is not a complete rectangular grid (missing {missing} values). "
            "If your sampling is irregular, tell me and I’ll switch to scattered interpolation."
        )

    return x, y, Ex, Ey


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

    if not (np.all(np.diff(x1d) > 0) and np.all(np.diff(y1d) > 0)):
        raise ValueError("x and y arrays must be strictly increasing for bilinear interpolation.")

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
    Call = []
    for r in radii:
        theta, F = sample_circle(x1d, y1d, Ex, Ey, x0, y0, r, M)
        C = compute_Cn_on_radius(theta, F, r, nmax)
        Call.append(C)
    Call = np.vstack(Call)  # (nr, nmax)
    Cavg = np.nanmean(Call, axis=0)
    return Cavg, Call


# ----------------------------
# Field reconstruction
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
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, label="Ex (V/m)")
    if out_prefix:
        fig.savefig(f"{out_prefix}_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    im = ax.imshow(Eyn, origin="lower", extent=extent, aspect="auto")
    ax.set_title(f"{title_prefix}: Ey")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, label="Ey (V/m)")
    if out_prefix:
        fig.savefig(f"{out_prefix}_Ey.png", dpi=200, bbox_inches="tight")


def plot_1d_cuts(x1d, y1d, Ex_total, Ey_total, comps, x0, y0, out_prefix=None):
    ix0 = nearest_index(x1d, x0)
    iy0 = nearest_index(y1d, y0)

    # Along x (y=y0)
    fig, ax = plt.subplots()
    ax.plot(x1d, Ex_total[iy0, :], label="Total Ex")
    for c in comps:
        ax.plot(x1d, c["Ex"][iy0, :], label=c["name"])
    ax.set_title(f"1D cut along x at y≈{y1d[iy0]:.6g} m: Ex")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Ex (V/m)")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_x_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(x1d, Ey_total[iy0, :], label="Total Ey")
    for c in comps:
        ax.plot(x1d, c["Ey"][iy0, :], label=c["name"])
    ax.set_title(f"1D cut along x at y≈{y1d[iy0]:.6g} m: Ey")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Ey (V/m)")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_x_Ey.png", dpi=200, bbox_inches="tight")

    # Along y (x=x0)
    fig, ax = plt.subplots()
    ax.plot(y1d, Ex_total[:, ix0], label="Total Ex")
    for c in comps:
        ax.plot(y1d, c["Ex"][:, ix0], label=c["name"])
    ax.set_title(f"1D cut along y at x≈{x1d[ix0]:.6g} m: Ex")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Ex (V/m)")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_y_Ex.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(y1d, Ey_total[:, ix0], label="Total Ey")
    for c in comps:
        ax.plot(y1d, c["Ey"][:, ix0], label=c["name"])
    ax.set_title(f"1D cut along y at x≈{x1d[ix0]:.6g} m: Ey")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Ey (V/m)")
    ax.legend()
    if out_prefix:
        fig.savefig(f"{out_prefix}_cut_y_Ey.png", dpi=200, bbox_inches="tight")


# ----------------------------
# Core run function (shared)
# ----------------------------
def run(config):
    x, y, Ex, Ey = load_field_any(config["input"])

    x0 = float(config["x0"])
    y0 = float(config["y0"])
    M = int(config["M"])
    nmax = int(config["nmax"])
    out = str(config["out"])
    show = bool(config["show"])

    # Radii selection
    if config.get("rmin") is not None and config.get("rmax") is not None:
        nr = int(config.get("nr", 5))
        radii = np.linspace(float(config["rmin"]), float(config["rmax"]), nr)
    elif config.get("r") is not None:
        radii = np.array([float(config["r"])], dtype=float)
    else:
        raise ValueError("Provide either r or (rmin and rmax).")

    # Output directory
    out_dir = Path(config["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute coefficients
    Cavg, Call = compute_Cn_multiradius(x, y, Ex, Ey, x0, y0, radii, M, nmax)

    names = {1: "dipole", 2: "quadrupole", 3: "sextupole", 4: "octupole"}
    print("=== Multipole coefficients Cn (F = Ex - i Ey) ===")
    print(f"Input: {Path(config['input']).expanduser().resolve()}")
    print(f"Center (x0,y0)=({x0},{y0})")
    print(f"Radii used: {radii}")
    for n in range(1, nmax + 1):
        cn = Cavg[n - 1]
        mag = np.abs(cn)
        ph = np.angle(cn)
        nm = names.get(n, f"order{n}")
        # Units: Cn has units (V/m) / m^(n-1) = V/m^n
        print(
            f"n={n:2d} ({nm:10s}): Cn = {cn.real:+.6e} {cn.imag:+.6e}j   "
            f"|Cn|={mag:.6e}  phase(rad)={ph:+.6f}"
        )

    if radii.size > 1:
        print("\nRadius-consistency check (Cn per radius):")
        for ir, r in enumerate(radii):
            row = Call[ir]
            row_str = "  ".join([f"n={n}:{row[n-1].real:+.2e}{row[n-1].imag:+.2e}j" for n in range(1, nmax + 1)])
            print(f"  r={r:.6g}: {row_str}")

    # Reconstruct and plot
    Xg, Yg = np.meshgrid(x, y)
    comps = []
    for n in range(1, nmax + 1):
        cn = Cavg[n - 1]
        Exn, Eyn = reconstruct_order_field(Xg, Yg, x0, y0, cn, n)
        nm = names.get(n, f"order{n}")
        comps.append({"n": n, "name": nm, "Ex": Exn, "Ey": Eyn})

        base = out_dir / f"{out}_{nm}"
        plot_component_maps(x, y, Exn, Eyn, f"{nm} (n={n})", out_prefix=str(base))

    # Total maps + 1D cuts
    plot_component_maps(x, y, Ex, Ey, "Total field", out_prefix=str(out_dir / f"{out}_total"))
    plot_1d_cuts(x, y, Ex, Ey, comps, x0, y0, out_prefix=str(out_dir / out))

    if show:
        plt.show()


# ----------------------------
# Main with two modes
# ----------------------------
def main(mode=1):
    """
    mode=1: command/CLI mode (parse argparse)
    mode=0: direct mode (set config dict here and run)
    """
    paths = get_project_paths(default_data_name="field.dat")

    if mode == 1:
        p = argparse.ArgumentParser()
        p.add_argument(
            "--input",
            default=str(paths["default_input"]),
            help=f"Field file path (default: {paths['default_input']}). "
                 f"Supports text table (X Y Ex Ey) or true .npz archive."
        )
        p.add_argument("--x0", type=float, required=True, help="Reference center x0 (m)")
        p.add_argument("--y0", type=float, required=True, help="Reference center y0 (m)")

        # Either single radius:
        p.add_argument("--r", type=float, default=None, help="Circle radius for extraction (m), single radius.")
        # Or multiple radii:
        p.add_argument("--rmin", type=float, default=None, help="Use multiple radii: rmin..rmax (m).")
        p.add_argument("--rmax", type=float, default=None, help="Use multiple radii: rmin..rmax (m).")
        p.add_argument("--nr", type=int, default=5, help="Number of radii if using rmin/rmax.")

        p.add_argument("--M", type=int, default=512, help="Number of angular samples on circle.")
        p.add_argument("--nmax", type=int, default=4, help="Max order n to compute (4=octupole).")

        p.add_argument("--out", type=str, default="multipole", help="Output prefix for saved figures.")
        p.add_argument(
            "--out_dir",
            type=str,
            default=str(paths["out_dir"]),
            help=f"Output directory for figures (default: {paths['out_dir']})",
        )
        p.add_argument("--show", action="store_true", help="Show plots interactively.")
        args = p.parse_args()
        run(vars(args))

    elif mode == 0:
        # -------------------------
        # DIRECT MODE: EDIT HERE
        # -------------------------
        config = {
            "input": str(paths["default_input"]),  # project_root/data/field.dat
            "x0": 0.0,
            "y0": 0.0,

            # Choose ONE:
            "r": 0.03,         # single radius (m)
            "rmin": None,
            "rmax": None,
            "nr": 5,

            # Multi-radius example:
            # "r": None,
            # "rmin": 0.005,
            # "rmax": 0.015,
            # "nr": 7,

            "M": 512,
            "nmax": 4,

            "out": "myrun",
            "out_dir": str(paths["out_dir"]),
            "show": True,
        }
        run(config)

    else:
        raise ValueError("mode must be 0 (direct) or 1 (command/CLI).")


if __name__ == "__main__":
    # Default: CLI mode
    main(mode=0)
    # For direct mode, switch to:
    # main(mode=0)