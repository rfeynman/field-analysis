"""
Fast 2D electrostatics solver (Mathematica -> Python port)

What it does (matches the notebook logic):
- Builds a 2D "air region" in a big rectangle [-Lx,Lx] x [-Ly,Ly]
- Inserts two electrodes (top and bottom) defined by 4 ellipses + smooth tangency
- Solves Laplace(φ)=0 in air with Dirichlet φ=Vt on top electrode boundary and φ=Vb on bottom electrode boundary
- Computes E = -∇φ
- Samples E on a uniform grid in [-0.1,0.1]^2 and returns:
    maxResult = [x_at_max, y_at_max, |E|_max]
    valsSmall = [[x,y,Ex,Ey], ...] clipped to |x|<=0.03 and |y|<=0.03

Outside the FEM/air region (holes, outside mesh, etc.) Ex and Ey are forced to 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import scale
from scipy.optimize import root
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


# -----------------------------
# Geometry helpers (ported from notebook)
# Ellipse parameterization:
#   p(θ)=c+[a cosθ, b sinθ]
#   t(θ)=[-a sinθ, b cosθ]
# Tangency constraints between ellipses e1,e2:
#   det([t1,t2])=0   (parallel tangents)
#   det([p2-p1,t1])=0 (common tangent line)
# -----------------------------

def _ellipse_point(c: np.ndarray, a: float, b: float, th: float) -> np.ndarray:
    return np.array([c[0] + a * math.cos(th), c[1] + b * math.sin(th)], dtype=float)


def _ellipse_tangent(a: float, b: float, th: float) -> np.ndarray:
    return np.array([-a * math.sin(th), b * math.cos(th)], dtype=float)


def _solve_ellipse_tangent(
    e1: Tuple[np.ndarray, float, float],
    e2: Tuple[np.ndarray, float, float],
    guess: Tuple[float, float],
):
    c1, a1, b1 = e1
    c2, a2, b2 = e2

    def fun(v):
        th1, th2 = float(v[0]), float(v[1])
        p1 = _ellipse_point(c1, a1, b1, th1)
        p2 = _ellipse_point(c2, a2, b2, th2)
        t1 = _ellipse_tangent(a1, b1, th1)
        t2 = _ellipse_tangent(a2, b2, th2)
        f1 = np.linalg.det(np.c_[t1, t2])           # det([t1,t2])=0
        f2 = np.linalg.det(np.c_[p2 - p1, t1])      # det([p2-p1,t1])=0
        return np.array([f1, f2], dtype=float)

    sol = root(fun, np.array(guess, dtype=float), method="hybr")
    if not sol.success:
        return None

    th1, th2 = map(float, sol.x)
    p1 = _ellipse_point(c1, a1, b1, th1)
    p2 = _ellipse_point(c2, a2, b2, th2)
    return p1, p2, th1, th2


def _solve_point_tangent(
    pt: np.ndarray,
    e: Tuple[np.ndarray, float, float],
    guess: float,
):
    c, a, b = e

    def fun(v):
        th = float(v[0])
        p = _ellipse_point(c, a, b, th)
        t = _ellipse_tangent(a, b, th)
        return np.array([np.linalg.det(np.c_[p - pt, t])], dtype=float)

    sol = root(fun, np.array([guess], dtype=float), method="hybr")
    if not sol.success:
        return None

    th = float(sol.x[0])
    p = _ellipse_point(c, a, b, th)
    return p, th


def _arc_points(
    e: Tuple[np.ndarray, float, float],
    th_start: float,
    th_end: float,
    n: int = 25,
) -> np.ndarray:
    # Force shortest-angle path (same as notebook)
    s = float(th_start)
    ee = float(th_end)
    while ee - s > math.pi:
        ee -= 2 * math.pi
    while s - ee > math.pi:
        ee += 2 * math.pi

    ts = np.linspace(s, ee, n + 1)
    c, a, b = e
    return np.column_stack([c[0] + a * np.cos(ts), c[1] + b * np.sin(ts)])


def _find_best_ellipse_tangent(
    e1: Tuple[np.ndarray, float, float],
    e2: Tuple[np.ndarray, float, float],
    side: str = "right",
):
    # Robustness vs unknown initial guesses: brute-force a small grid of guesses.
    guesses = np.linspace(-math.pi, math.pi, 7)
    sols = []
    for g1 in guesses:
        for g2 in guesses:
            out = _solve_ellipse_tangent(e1, e2, (float(g1), float(g2)))
            if out is None:
                continue
            p1, p2, th1, th2 = out
            if side == "right" and (p1[0] < 0 or p2[0] < 0):
                continue
            sols.append(out)

    if not sols:
        raise RuntimeError("No ellipse-ellipse tangent found. Try changing ellipse params or expanding guesses.")
    # Prefer the "outermost" tangent (largest average x)
    sols.sort(key=lambda s: (s[0][0] + s[1][0]), reverse=True)
    return sols[0]


def _find_best_point_tangent(
    pt: np.ndarray,
    e: Tuple[np.ndarray, float, float],
    side: str = "right",
):
    guesses = np.linspace(-math.pi, math.pi, 9)
    sols = []
    for g in guesses:
        out = _solve_point_tangent(pt, e, float(g))
        if out is None:
            continue
        p, th = out
        if side == "right" and p[0] < 0:
            continue
        sols.append(out)

    if not sols:
        raise RuntimeError("No point-ellipse tangent found. Try changing ellipse params or expanding guesses.")
    sols.sort(key=lambda s: s[0][0], reverse=True)
    return sols[0]


def generate_electrode(
    params: Sequence[Tuple[Tuple[float, float], float, float]],
    symmetry_x: float = 0.0,
    st: float = 0.035,
    ed: float = 0.045,
    n_arc: int = 25,
) -> Polygon:
    """
    Builds the same boundary the notebook builds: 4 ellipses connected by tangent arcs,
    then mirrored about x=symmetry_x to create a closed polygon.
    """
    e1, e2, e3, e4 = [(np.array(p[0], dtype=float), float(p[1]), float(p[2])) for p in params]

    top_pt = np.array([symmetry_x, ed], dtype=float)
    start_pt = np.array([symmetry_x, st], dtype=float)

    t12 = _find_best_ellipse_tangent(e1, e2, side="right")
    t23 = _find_best_ellipse_tangent(e2, e3, side="right")
    t34 = _find_best_ellipse_tangent(e3, e4, side="right")
    p4E, th4E = _find_best_point_tangent(start_pt, e4, side="right")

    right = []
    right.append(top_pt)
    right.append(np.array([e1[0][0], ed], dtype=float))  # forced flat top segment

    right.extend(_arc_points(e1, math.pi / 2, t12[2], n_arc)[1:])
    right.extend(_arc_points(e2, t12[3], t23[2], n_arc)[1:])
    right.extend(_arc_points(e3, t23[3], t34[2], n_arc)[1:])
    right.extend(_arc_points(e4, t34[3], th4E, n_arc)[1:])
    right.append(start_pt)

    right = np.array(right, dtype=float)

    mirrored = right.copy()
    mirrored[:, 0] = 2 * symmetry_x - mirrored[:, 0]
    full = np.vstack([right, mirrored[::-1]])

    # Remove near-duplicates
    cleaned = [full[0]]
    for p in full[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-8:
            cleaned.append(p)

    return Polygon(cleaned)


# -----------------------------
# FD "FEM-like" Laplace solver
# -----------------------------

@dataclass
class SolveConfig:
    Vt: float = 60000.0
    Vb: float = -60000.0
    Lx: float = 0.25
    Ly: float = 0.25
    # FD grid used for PDE solve (bigger -> more accurate, slower)
    nx_solve: int = 161
    ny_solve: int = 161

    # Sampling grid in ROI [-0.1,0.1]^2 (matches notebook)
    xmin: float = -0.10
    xmax: float = 0.10
    ymin: float = -0.10
    ymax: float = 0.10
    n_sample: int = 160

    roi_small: float = 0.03


def solve_laplace_fd(
    top_poly: Polygon,
    bottom_poly: Polygon,
    cfg: SolveConfig,
):
    """
    Solves Laplace(φ)=0 in "air", with Dirichlet inside the electrode polygons.
    Outer boundary uses Neumann(∂φ/∂n)=0 implemented via ghost=neighbor (=copy) trick.
    """
    xs = np.linspace(-cfg.Lx, cfg.Lx, cfg.nx_solve)
    ys = np.linspace(-cfg.Ly, cfg.Ly, cfg.ny_solve)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = shapely.points(np.c_[X.ravel(), Y.ravel()])

    top_mask = shapely.contains(top_poly, pts).reshape(cfg.ny_solve, cfg.nx_solve)
    bot_mask = shapely.contains(bottom_poly, pts).reshape(cfg.ny_solve, cfg.nx_solve)
    elec_mask = top_mask | bot_mask
    air_mask = ~elec_mask

    unknowns = np.argwhere(air_mask)
    idx = -np.ones((cfg.ny_solve, cfg.nx_solve), dtype=int)
    idx[air_mask] = np.arange(len(unknowns))
    N = len(unknowns)

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    diag = 2.0 * (cx + cy)

    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    phi_dir = np.zeros((cfg.ny_solve, cfg.nx_solve), dtype=float)
    phi_dir[top_mask] = cfg.Vt
    phi_dir[bot_mask] = cfg.Vb

    # Assemble sparse system
    for k, (j, i) in enumerate(unknowns):
        A[k, k] = diag
        for di, dj, coef in [(-1, 0, cx), (1, 0, cx), (0, -1, cy), (0, 1, cy)]:
            ii = i + di
            jj = j + dj
            if ii < 0 or ii >= cfg.nx_solve or jj < 0 or jj >= cfg.ny_solve:
                # Neumann 0: ghost value equals boundary node => subtract coef from diagonal
                A[k, k] -= coef
                continue

            if elec_mask[jj, ii]:
                b[k] += coef * phi_dir[jj, ii]
            else:
                kk = idx[jj, ii]
                A[k, kk] -= coef

    phi_air = spsolve(A.tocsr(), b)

    phi = np.array(phi_dir, copy=True)
    phi[air_mask] = phi_air
    return xs, ys, phi, air_mask


def compute_field(xs: np.ndarray, ys: np.ndarray, phi: np.ndarray):
    """E = -∇φ via finite differences on the solve grid."""
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]

    Ex[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * hx)
    Ey[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * hy)

    Ex[:, 0] = -(phi[:, 1] - phi[:, 0]) / hx
    Ex[:, -1] = -(phi[:, -1] - phi[:, -2]) / hx
    Ey[0, :] = -(phi[1, :] - phi[0, :]) / hy
    Ey[-1, :] = -(phi[-1, :] - phi[-2, :]) / hy
    return Ex, Ey


def sample_and_outputs(
    xs: np.ndarray,
    ys: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    air_mask: np.ndarray,
    cfg: SolveConfig,
):
    """Matches the notebook: sample in big ROI, then clip to small ROI; outside air -> Ex=Ey=0."""
    xi = np.linspace(cfg.xmin, cfg.xmax, cfg.n_sample)
    yi = np.linspace(cfg.ymin, cfg.ymax, cfg.n_sample)

    # interpolators expect (y, x) ordering
    Exi = RegularGridInterpolator((ys, xs), Ex_grid, bounds_error=False, fill_value=0.0)
    Eyi = RegularGridInterpolator((ys, xs), Ey_grid, bounds_error=False, fill_value=0.0)
    maski = RegularGridInterpolator((ys, xs), air_mask.astype(float), method="nearest", bounds_error=False, fill_value=0.0)

    big_vals: List[List[float]] = []
    for y in yi:
        pts = np.column_stack([np.full_like(xi, y), xi])
        ex = Exi(pts)
        ey = Eyi(pts)
        m = maski(pts)
        ex = np.where(m > 0.5, ex, 0.0)
        ey = np.where(m > 0.5, ey, 0.0)
        for x, e1, e2 in zip(xi, ex, ey):
            big_vals.append([float(x), float(y), float(e1), float(e2)])

    arr = np.array(big_vals, dtype=float)
    emag = np.hypot(arr[:, 2], arr[:, 3])
    imax = int(np.argmax(emag))
    max_result = [float(arr[imax, 0]), float(arr[imax, 1]), float(emag[imax])]

    small = arr[(np.abs(arr[:, 0]) <= cfg.roi_small) & (np.abs(arr[:, 1]) <= cfg.roi_small)]
    vals_small = small.tolist()
    return max_result, vals_small




def sample_field_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    air_mask: np.ndarray,
    cfg: SolveConfig,
):
    """
    Vectorized sampling on a uniform grid in [xmin,xmax]x[ymin,ymax].

    Returns:
      xi, yi: 1D arrays
      Ex2d, Ey2d, Emag2d: 2D arrays with shape (len(yi), len(xi))
    """
    xi = np.linspace(cfg.xmin, cfg.xmax, cfg.n_sample)
    yi = np.linspace(cfg.ymin, cfg.ymax, cfg.n_sample)

    Exi = RegularGridInterpolator((ys, xs), Ex_grid, bounds_error=False, fill_value=0.0)
    Eyi = RegularGridInterpolator((ys, xs), Ey_grid, bounds_error=False, fill_value=0.0)
    maski = RegularGridInterpolator(
        (ys, xs),
        air_mask.astype(float),
        method="nearest",
        bounds_error=False,
        fill_value=0.0,
    )

    X, Y = np.meshgrid(xi, yi, indexing="xy")
    pts = np.column_stack([Y.ravel(), X.ravel()])  # (y, x) order for interpolator
    ex = Exi(pts)
    ey = Eyi(pts)
    m = maski(pts)
    ex = np.where(m > 0.5, ex, 0.0)
    ey = np.where(m > 0.5, ey, 0.0)

    Ex2d = ex.reshape(len(yi), len(xi))
    Ey2d = ey.reshape(len(yi), len(xi))
    Emag2d = np.hypot(Ex2d, Ey2d)
    return xi, yi, Ex2d, Ey2d, Emag2d


def plot_emag_density(
    xi: np.ndarray,
    yi: np.ndarray,
    emag2d: np.ndarray,
    *,
    title: str = "|E|",
    savepath: str | None = None,
    show: bool = True,
):
    """
    Matplotlib equivalent of Mathematica DensityPlot for |E| on (x,y).

    Args:
      xi, yi: 1D coordinate arrays
      emag2d: 2D array shaped (len(yi), len(xi))
      savepath: if provided, saves the figure (e.g. "emag.png")
      show: if True, shows the window (plt.show())

    Returns:
      (fig, ax)
    """
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(xi, yi, indexing="xy")
    fig, ax = plt.subplots()
    # Use pcolormesh (closest to a 2D "DensityPlot" / heatmap)
    m = ax.pcolormesh(X, Y, emag2d, shading="auto")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(m, ax=ax, label="|E| (V/m)")

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def run_and_plot_emag(
    electrode_params: Sequence[Tuple[Tuple[float, float], float, float]] | None = None,
    cfg: SolveConfig | None = None,
    *,
    savepath: str | None = "emag.png",
    show: bool = True,
):
    """
    Convenience wrapper: runs the solve, samples |E| on the ROI grid, and plots a 2D heatmap.

    Returns:
      (maxResult, valsSmall, (xi, yi, emag2d))
    """
    if cfg is None:
        cfg = SolveConfig()

    st = 3.5e-2
    ed = st + 0.01

    if electrode_params is None:
        electrode_params = [
            ((6e-2, 4.2e-2), 0.3e-2, ed - 4.2e-2),
            ((6e-2, 2.5e-2), 0.5e-2, 0.5e-2),
            ((5.2e-2, 2.5e-2), 0.5e-2, 0.5e-2),
            ((4.5e-2, 3.2e-2), 0.3e-2, 0.3e-2),
        ]

    top = generate_electrode(electrode_params, symmetry_x=0.0, st=st, ed=ed, n_arc=25)
    bottom = scale(top, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))

    xs, ys, phi, air_mask = solve_laplace_fd(top, bottom, cfg)
    Ex_grid, Ey_grid = compute_field(xs, ys, phi)

    # Keep original outputs
    max_result, vals_small = sample_and_outputs(xs, ys, Ex_grid, Ey_grid, air_mask, cfg)

    # Plot data
    xi, yi, _, _, emag2d = sample_field_grid(xs, ys, Ex_grid, Ey_grid, air_mask, cfg)
    plot_emag_density(
        xi,
        yi,
        emag2d,
        title=f"|E| on [{cfg.xmin},{cfg.xmax}]×[{cfg.ymin},{cfg.ymax}]",
        savepath=savepath,
        show=show,
    )

    return max_result, vals_small, (xi, yi, emag2d)


# -----------------------------
# One-shot runner (defaults match notebook)
# -----------------------------

def run_simulation(
    electrode_params: Sequence[Tuple[Tuple[float, float], float, float]] | None = None,
    cfg: SolveConfig | None = None,
):
    """
    Returns:
      maxResult: [x, y, |E|_max]
      valsSmall: [[x,y,Ex,Ey], ...] within |x|<=roi_small and |y|<=roi_small
    """
    if cfg is None:
        cfg = SolveConfig()

    # Notebook defaults:
    # thick=0.01; st=3.5e-2; ed=st+thick
    st = 3.5e-2
    ed = st + 0.01

    if electrode_params is None:
        # exampleParams from the notebook
        electrode_params = [
            ((6e-2, 4.2e-2), 0.3e-2, ed - 4.2e-2),  # ellipse 1
            ((6e-2, 2.5e-2), 0.5e-2, 0.5e-2),       # ellipse 2
            ((5.2e-2, 2.5e-2), 0.5e-2, 0.5e-2),     # ellipse 3
            ((4.5e-2, 3.2e-2), 0.3e-2, 0.3e-2),     # ellipse 4
        ]

    top = generate_electrode(electrode_params, symmetry_x=0.0, st=st, ed=ed, n_arc=25)
    bottom = scale(top, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))  # mirror about y=0

    xs, ys, phi, air_mask = solve_laplace_fd(top, bottom, cfg)
    Ex_grid, Ey_grid = compute_field(xs, ys, phi)
    max_result, vals_small = sample_and_outputs(xs, ys, Ex_grid, Ey_grid, air_mask, cfg)
    return max_result, vals_small


if __name__ == "__main__":
    maxResult, valsSmall = run_simulation()
    print("maxResult =", maxResult)
    print("valsSmall length =", len(valsSmall))
