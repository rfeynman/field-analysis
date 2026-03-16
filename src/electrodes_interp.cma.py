#!/usr/bin/env python3
"""electrodes_interp_cma_gmsh_project_uniformity.py

Interpolation-driven electrode geometry + Gmsh/FEM field solve + multipole analysis + CMA-ES optimization.

Project layout (fixed):
  PROJECT_ROOT = /Users/wange/Coding/Python/fieldanalysis
  Script location: PROJECT_ROOT/src
  Outputs: PROJECT_ROOT/outputs

Best outputs (saved under PROJECT_ROOT/outputs/optruns_YYYYMMDD_HHMMSS/):
  - best.out.txt
  - best.points.txt
  - convergence.png
  - phi_map.png
  - mesh_wireframe.png
  - emag_density.png
  - best_simulation_output_map.txt

Per-evaluation outputs (saved under RUN_DIR/runs/):
  - genXXXX_evalYYYYYY_.yaml            (Npoint coordinates)
  - genXXXX_evalYYYYYY_out.txt         (per-eval report)

Uniformity:
  - SolveConfig.UNIFORMITY_TARGET = 0.005
  - Computes the "Uniformity radius" defined as:
      first annulus-bin center where d(r) >= UNIFORMITY_TARGET,
      where d(r) = max_{points in annulus} | |E(x,y)| - |E(0,0)| | / |E(0,0)|
    using annulus bins dr = 0.5 * min(dx, dy), with dx/dy from the regular grid.

Run:
  python electrodes_interp_cma_gmsh_project_uniformity.py --yaml geom.yaml --budget 500 --nproc 8

Notes:
- Terminal output prints progress: generation, eval/budget, current best objective.
- Gmsh is initialized/finalized inside each evaluation (safe for multiprocessing).
"""
from __future__ import annotations
import os

# ---- Force 1 thread per process for common math backends on macOS/Apple Silicon ----
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")   # Apple Accelerate/vecLib (most important on M-series)
os.environ.setdefault("OMP_NUM_THREADS", "1")          # OpenMP
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")     # OpenBLAS (if used)
os.environ.setdefault("MKL_NUM_THREADS", "1")          # MKL (usually not on Apple Silicon, but harmless)
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")      # numexpr (if used)

import argparse
import math
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import scale as shp_scale
from shapely.ops import unary_union

import gmsh

from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------
# Project paths (fixed)
# -----------------------------
PROJECT_ROOT = Path("/Users/wange/Coding/Python/fieldanalysis")
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# -----------------------------
# Config
# -----------------------------
@dataclass
class SolveConfig:
    # electrodes_field.py-aligned defaults where applicable
    Vt: float = 60000.0
    Vb: float = -60000.0
    thick: float = 0.016
    st: float = 0.035

    max_cell_area: float = 2e-5
    boundary_h_factor: float = 0.33

    xmin: float = -0.107
    xmax: float = 0.107
    ymin: float = -0.129
    ymax: float = 0.129

    n_sample: int = 160
    roi: float = 0.032

    # interpolation geometry
    Npoint: int = 8
    x0: float = 0.064
    circle_y_offset: float = 0.001
    # Control-point x distribution (0..1): fraction of points placed in [0, x0*0.7] line: 220
    # Example: split_point=0.3 and Npoint=25 -> 8 points in [0, x0*0.7], 17 points in (x0*0.7, x0]
    split_point: float = 0.25

    # multipole analysis
    cutrad: float = 0.029
    NMAX: int = 5
    M_theta: int = 512

    # uniformity
    UNIFORMITY_TARGET: float = 0.005

    # optimization
    outdir_opt: str = str(OUTPUTS_DIR)
    budget: int = 1200
    nproc: int = 10
    seed: int = 90

    # Initial CMA mean perturbation (only affects the starting mean, not the fixed endpoints).
    # If mean_range>0, interior y-control points are initialized uniformly in [st-mean_range, st+mean_range].
    mean_range: float = 0.003

    # bounds for optimization variables
    y_delta_max: float = 0.005

    @property
    def ed(self) -> float:
        return self.st + self.thick


# -----------------------------
# YAML load (optional)
# -----------------------------
def load_yaml_config(path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml is not installed. Install it or run without --yaml.") from e
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return data


def apply_overrides(cfg: SolveConfig, overrides: Dict) -> SolveConfig:
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# -----------------------------
# Geometry helpers
# -----------------------------
def tangent_points_point_to_circle(P: np.ndarray, C: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray]:
    v = P - C
    d = float(np.hypot(v[0], v[1]))
    if d <= r * (1.0 + 1e-12):
        raise ValueError(f"No tangents: point is inside/on circle (d={d}, r={r}).")
    u = v / d
    beta = math.acos(r / d)
    perp = np.array([-u[1], u[0]], dtype=float)
    T1 = C + r * (u * math.cos(beta) + perp * math.sin(beta))
    T2 = C + r * (u * math.cos(beta) - perp * math.sin(beta))
    return T1, T2


def choose_tangent_point(P: np.ndarray, C: np.ndarray, r: float, mode: str, x0: float, y0: float) -> np.ndarray:
    T1, T2 = tangent_points_point_to_circle(P, C, r)
    if mode == "y_gt_y0":
        cand = [T for T in (T1, T2) if T[1] > y0]
        return max(cand, key=lambda t: t[1]) if cand else (T1 if T1[1] >= T2[1] else T2)
    if mode == "x_gt_x0":
        cand = [T for T in (T1, T2) if T[0] > x0]
        return max(cand, key=lambda t: t[0]) if cand else (T1 if T1[0] >= T2[0] else T2)
    raise ValueError(f"Unknown mode: {mode}")


def _wrap_angle_0_2pi(th: float) -> float:
    return float(th) % (2.0 * math.pi)


def circle_arc_short(C: np.ndarray, r: float, Ta: np.ndarray, Tb: np.ndarray, n: int = 200) -> np.ndarray:
    a = _wrap_angle_0_2pi(math.atan2(Ta[1] - C[1], Ta[0] - C[0]))
    b = _wrap_angle_0_2pi(math.atan2(Tb[1] - C[1], Tb[0] - C[0]))

    d_ccw = (b - a) % (2.0 * math.pi)
    d_cw = (a - b) % (2.0 * math.pi)

    if d_ccw <= d_cw:
        thetas = np.linspace(a, a + d_ccw, n)
    else:
        thetas = np.linspace(a, a - d_cw, n)

    return np.column_stack([C[0] + r * np.cos(thetas), C[1] + r * np.sin(thetas)])


def x_control_points(cfg: SolveConfig) -> np.ndarray:
    """Return the Npoint x-coordinates for the interpolation control points.

    If cfg.split_point is in (0,1), allocate round(Npoint*split_point) points in [0, x0/2]
    and the rest in (x0/2, x0], with the midpoint x0/2 included only once (required by CubicSpline).
    """
    sp = float(cfg.split_point)
    sp = 0.0 if not np.isfinite(sp) else max(0.0, min(1.0, sp))

    n1 = int(round(cfg.Npoint * sp))
    # Ensure both halves have enough points and keep x strictly increasing.
    n1 = max(2, min(cfg.Npoint - 1, n1))

    x_half = 0.7 * cfg.x0 # Use 0.7*x0 as the "half" point to give more room for control points in the second half, which has more curvature.
    x1 = np.linspace(0.0, x_half, n1, endpoint=True)
    x2_full = np.linspace(x_half, cfg.x0, cfg.Npoint - n1 + 1, endpoint=True)
    x2 = x2_full[1:]  # drop midpoint to avoid duplicate
    x_ctrl = np.concatenate([x1, x2])

    # Safety: enforce strict monotonicity (CubicSpline requires increasing x)
    if not np.all(np.diff(x_ctrl) > 0):
        raise ValueError("x_control_points produced non-increasing x grid; adjust split_point/Npoint")
    return x_ctrl


def clamp_y_points(cfg: SolveConfig, y: np.ndarray) -> np.ndarray:
    y2 = y.astype(float).copy()
    y2[0] = cfg.st
    y2[-1] = cfg.st
    if cfg.Npoint > 2:
        lo = cfg.st - cfg.y_delta_max
        hi = cfg.st + cfg.y_delta_max
        y2[1:-1] = np.clip(y2[1:-1], lo, hi)
    return y2


def build_top_electrode_polygon(cfg: SolveConfig, y_points: np.ndarray) -> Polygon:
    y_points = clamp_y_points(cfg, y_points)
    x_ctrl = x_control_points(cfg)

    r = cfg.thick / 2.0
    C = np.array([cfg.x0, cfg.st + r + cfg.circle_y_offset], dtype=float)
    y0 = float(C[1])

    Ptop = np.array([0.0, cfg.ed], dtype=float)
    Pbot0 = np.array([0.0, cfg.st], dtype=float)
    Pbot = np.array([cfg.x0, cfg.st], dtype=float)

    T_top = choose_tangent_point(Ptop, C, r, mode="y_gt_y0", x0=cfg.x0, y0=y0)
    T_bot = choose_tangent_point(Pbot, C, r, mode="x_gt_x0", x0=cfg.x0, y0=y0)

    dx = float(T_bot[0] - Pbot[0])
    dy = float(T_bot[1] - Pbot[1])
    m_end = dy / dx if abs(dx) > 1e-14 else 0.0

    cs = CubicSpline(x_ctrl, y_points, bc_type=((1, 0.0), (1, m_end)))
    x_dense = np.linspace(0.0, cfg.x0, 400)
    y_dense = cs(x_dense)
    spline_pts = np.column_stack([x_dense, y_dense])

    pts_right: List[np.ndarray] = [Ptop, T_top]
    arc = circle_arc_short(C, r, T_top, T_bot, n=220)
    pts_right.extend(list(arc[1:]))
    pts_right.append(Pbot)

    spl_rev = spline_pts[::-1]
    if np.linalg.norm(spl_rev[0] - Pbot) > 1e-10:
        spl_rev[0] = Pbot
    if np.linalg.norm(spl_rev[-1] - Pbot0) > 1e-10:
        spl_rev[-1] = Pbot0
    pts_right.extend(list(spl_rev[1:]))

    right = np.asarray(pts_right, dtype=float)
    left = np.column_stack([-right[::-1, 0], right[::-1, 1]])
    full = np.vstack([right, left[1:]])

    poly = Polygon(full)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise RuntimeError("Generated top electrode polygon is invalid")
    return poly


# -----------------------------
# Meshing (Gmsh) and FEM solve
# -----------------------------
def _build_mesh_gmsh(top_poly: Polygon, bot_poly: Polygon, cfg: SolveConfig):
    h0 = math.sqrt(4.0 * cfg.max_cell_area / math.sqrt(3.0))
    h_fine = cfg.boundary_h_factor * h0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)

    x_half = 0.7 * cfg.x0 # Use 0.7*x0 as the "half" point to give more room for control points in the second half, which has more curvature.   gmsh.model.add("conformal")
    occ = gmsh.model.occ

    width = float(cfg.xmax - cfg.xmin)
    height = float(cfg.ymax - cfg.ymin)
    outer = occ.addRectangle(cfg.xmin, cfg.ymin, 0, width, height)

    def add_occ_poly(poly: Polygon):
        c = np.asarray(poly.exterior.coords)
        p_tags = [occ.addPoint(float(x), float(y), 0) for x, y in c[:-1]]
        l_tags = [occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
        return occ.addPlaneSurface([occ.addCurveLoop(l_tags)]), l_tags

    s1, l1 = add_occ_poly(top_poly)
    s2, l2 = add_occ_poly(bot_poly)

    occ.cut([(2, outer)], [(2, s1), (2, s2)])
    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 60)

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", l1 + l2)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h0)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.003)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.040)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    pts = coords.reshape(-1, 3)[:, :2]

    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    elem = gmsh.model.mesh.getElements(2)
    tri_nodes = np.array(elem[2][0], dtype=int).reshape(-1, 3)
    triangles = np.vectorize(tag_to_idx.get)(tri_nodes)

    t_mask = np.array([top_poly.boundary.distance(Point(p)) < 1e-7 for p in pts])
    b_mask = np.array([bot_poly.boundary.distance(Point(p)) < 1e-7 for p in pts])

    tol = max(1e-10, 1e-8 * max(abs(cfg.xmax - cfg.xmin), abs(cfg.ymax - cfg.ymin)))
    bd_mask = (
        (np.abs(pts[:, 0] - cfg.xmin) <= tol)
        | (np.abs(pts[:, 0] - cfg.xmax) <= tol)
        | (np.abs(pts[:, 1] - cfg.ymin) <= tol)
        | (np.abs(pts[:, 1] - cfg.ymax) <= tol)
    )

    gmsh.finalize()
    return pts, triangles, t_mask, b_mask, bd_mask


def _fem_solve_phi(pts: np.ndarray, tris: np.ndarray, t_m: np.ndarray, b_m: np.ndarray, bd_m: np.ndarray, cfg: SolveConfig) -> np.ndarray:
    n = len(pts)
    K = lil_matrix((n, n))

    p_tri = pts[tris]
    x1, y1 = p_tri[:, 0, 0], p_tri[:, 0, 1]
    x2, y2 = p_tri[:, 1, 0], p_tri[:, 1, 1]
    x3, y3 = p_tri[:, 2, 0], p_tri[:, 2, 1]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * np.abs(twoA)
    A = np.maximum(A, 1e-30)

    b_vec = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c_vec = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)

    for i in range(3):
        for j in range(3):
            vals = (b_vec[:, i] * b_vec[:, j] + c_vec[:, i] * c_vec[:, j]) / (4.0 * A)
            for k in range(len(vals)):
                K[tris[k, i], tris[k, j]] += float(vals[k])

    K = K.tocsr()

    dir_mask = t_m | b_m | bd_m
    dir_vals = np.zeros(n)
    dir_vals[t_m] = cfg.Vt
    dir_vals[b_m] = cfg.Vb

    free = np.where(~dir_mask)[0]
    fixed = np.where(dir_mask)[0]

    phi = np.zeros(n)
    phi[fixed] = dir_vals[fixed]
    phi[free] = spsolve(K[free][:, free], -K[free][:, fixed] @ phi[fixed])
    return phi


def _nodal_field_from_phi(pts: np.ndarray, tris: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_tri = pts[tris]
    x1, y1 = p_tri[:, 0, 0], p_tri[:, 0, 1]
    x2, y2 = p_tri[:, 1, 0], p_tri[:, 1, 1]
    x3, y3 = p_tri[:, 2, 0], p_tri[:, 2, 1]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * np.abs(twoA)
    A = np.maximum(A, 1e-30)

    b_vec = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c_vec = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)

    ph = phi[tris]
    ex_tri = -(ph[:, 0] * b_vec[:, 0] + ph[:, 1] * b_vec[:, 1] + ph[:, 2] * b_vec[:, 2]) / (2.0 * A)
    ey_tri = -(ph[:, 0] * c_vec[:, 0] + ph[:, 1] * c_vec[:, 1] + ph[:, 2] * c_vec[:, 2]) / (2.0 * A)

    n = len(pts)
    Exn = np.zeros(n)
    Eyn = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(3):
        np.add.at(Exn, tris[:, i], ex_tri)
        np.add.at(Eyn, tris[:, i], ey_tri)
        np.add.at(cnt, tris[:, i], 1.0)

    Exn /= np.maximum(cnt, 1.0)
    Eyn /= np.maximum(cnt, 1.0)
    return Exn, Eyn


def _grid_interpolate_from_mesh(
    pts: np.ndarray,
    tris: np.ndarray,
    phi: np.ndarray,
    Exn: np.ndarray,
    Eyn: np.ndarray,
    metal: Polygon,
    cfg: SolveConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(cfg.xmin, cfg.xmax, cfg.n_sample)
    yi = np.linspace(cfg.ymin, cfg.ymax, cfg.n_sample)

    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    Xg, Yg = np.meshgrid(xi, yi)
    xf = Xg.ravel()
    yf = Yg.ravel()

    t_id = tri_obj.get_trifinder()(xf, yf)
    air_mask_flat = (t_id >= 0) & (~shapely.contains_xy(metal, xf, yf))

    Ex_flat = np.zeros_like(xf)
    Ey_flat = np.zeros_like(xf)
    phi_flat = np.zeros_like(xf)

    if np.any(air_mask_flat):
        tri_idx = t_id[air_mask_flat]
        phi_flat[air_mask_flat] = phi[tris[tri_idx]].mean(axis=1)
        Ex_flat[air_mask_flat] = Exn[tris[tri_idx]].mean(axis=1)
        Ey_flat[air_mask_flat] = Eyn[tris[tri_idx]].mean(axis=1)

    Ex_grid = Ex_flat.reshape(cfg.n_sample, cfg.n_sample)
    Ey_grid = Ey_flat.reshape(cfg.n_sample, cfg.n_sample)
    phi_grid = phi_flat.reshape(cfg.n_sample, cfg.n_sample)
    air_mask_grid = air_mask_flat.reshape(cfg.n_sample, cfg.n_sample)

    return xi, yi, phi_grid, Ex_grid, Ey_grid, air_mask_grid


def _max_and_roi(
    xi: np.ndarray,
    yi: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    air_mask_grid: np.ndarray,
    cfg: SolveConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    Emag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    Emag_masked = np.where(air_mask_grid, Emag, 0.0)

    flat_idx = int(np.argmax(Emag_masked))
    iy, ix = np.unravel_index(flat_idx, Emag_masked.shape)
    max_result = np.array([xi[ix], yi[iy], Emag_masked[iy, ix]], dtype=float)

    X_grid, Y_grid = np.meshgrid(xi, yi)
    roi_mask = (np.abs(X_grid) <= cfg.roi) & (np.abs(Y_grid) <= cfg.roi)

    vals_small = np.column_stack([X_grid[roi_mask], Y_grid[roi_mask], Ex_grid[roi_mask], Ey_grid[roi_mask]])
    return max_result, vals_small


def uniformity_radius(cfg: SolveConfig, xi: np.ndarray, yi: np.ndarray, Ex_grid: np.ndarray, Ey_grid: np.ndarray, air_mask_grid: np.ndarray) -> float:
    dx = float(xi[1] - xi[0])
    dy = float(yi[1] - yi[0])
    dr = 0.5 * min(dx, dy)

    interp_Ex = RegularGridInterpolator((yi, xi), Ex_grid, bounds_error=False, fill_value=0.0)
    interp_Ey = RegularGridInterpolator((yi, xi), Ey_grid, bounds_error=False, fill_value=0.0)

    Ex0 = float(interp_Ex([[0.0, 0.0]])[0])
    Ey0 = float(interp_Ey([[0.0, 0.0]])[0])
    E0 = float(math.hypot(Ex0, Ey0))
    if not np.isfinite(E0) or E0 <= 0.0:
        return float("nan")

    X, Y = np.meshgrid(xi, yi)
    R = np.sqrt(X**2 + Y**2)
    Emag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    rel = np.abs(Emag - E0) / E0

    mask = air_mask_grid & (R <= cfg.roi)
    if not np.any(mask):
        return float("nan")

    nbins = int(np.floor(cfg.roi / dr))
    if nbins <= 1:
        return float("nan")

    for k in range(1, nbins + 1):
        r_lo = k * dr
        r_hi = (k + 1) * dr
        sel = mask & (R >= r_lo) & (R < r_hi)
        if not np.any(sel):
            continue
        d_k = float(np.max(rel[sel]))
        if d_k >= cfg.UNIFORMITY_TARGET:
            return float((k + 0.5) * dr)

    return float("nan")


def multipole_coeffs_on_circle(cfg: SolveConfig, xi: np.ndarray, yi: np.ndarray, Ex_grid: np.ndarray, Ey_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_ref = cfg.cutrad
    interp_Ex = RegularGridInterpolator((yi, xi), Ex_grid, bounds_error=False, fill_value=0.0)
    interp_Ey = RegularGridInterpolator((yi, xi), Ey_grid, bounds_error=False, fill_value=0.0)

    thetas = np.linspace(0.0, 2.0 * math.pi, cfg.M_theta, endpoint=False)
    xs = r_ref * np.cos(thetas)
    ys = r_ref * np.sin(thetas)
    pts_yx = np.column_stack([ys, xs])

    Ex_s = interp_Ex(pts_yx)
    Ey_s = interp_Ey(pts_yx)
    F = Ex_s - 1j * Ey_s

    n_arr = np.arange(1, cfg.NMAX + 1, dtype=int)
    C = np.zeros(cfg.NMAX, dtype=complex)

    for i, n in enumerate(n_arr):
        k = n - 1
        phase = np.exp(-1j * k * thetas)
        raw = np.mean(F * phase)
        denom = (r_ref ** k) if k > 0 else 1.0
        C[i] = raw / denom

    return n_arr, C


def objective_sum_high_orders(cfg: SolveConfig, n_arr: np.ndarray, C: np.ndarray) -> float:
    r_ref = cfg.cutrad
    mask = n_arr > 1
    k = (n_arr[mask] - 1).astype(int)
    return float(np.sum(np.abs(C[mask]) * (r_ref ** k)))


def write_points_yaml(path: Path, cfg: SolveConfig, y_points: np.ndarray) -> None:
    x_ctrl = x_control_points(cfg)
    pts = [{"x": float(x), "y": float(y)} for x, y in zip(x_ctrl, y_points)]

    try:
        import yaml  # type: ignore
        with path.open("w") as f:
            yaml.safe_dump({"Npoint": int(cfg.Npoint), "points": pts}, f, sort_keys=False)
    except Exception:
        with path.open("w") as f:
            f.write(f"Npoint: {int(cfg.Npoint)}\n")
            f.write("points:\n")
            for p in pts:
                f.write(f"  - x: {p['x']:.12e}\n")
                f.write(f"    y: {p['y']:.12e}\n")


def write_eval_out_txt(
    path: Path,
    cfg: SolveConfig,
    max_result: np.ndarray,
    n_arr: np.ndarray,
    C: np.ndarray,
    sum_high: float,
    uni_r: float,
) -> None:
    r_ref = cfg.cutrad

    lines: List[str] = []
    lines.append(f"Max |E| = {max_result[2]:.12e} at (x,y)=({max_result[0]:.12e}, {max_result[1]:.12e})")
    lines.append(
        f"Uniformity radius (first annulus where d(r) >= {cfg.UNIFORMITY_TARGET:.3g}) = "
        + (f"{uni_r:.12e}" if np.isfinite(uni_r) else "nan")
    )
    lines.append("")
    lines.append("n  name      Re(Cn)            Im(Cn)            |Cn|             phase(rad)     |Cn|*r_ref^(n-1)")

    for n, c in zip(n_arr, C):
        mag = abs(c)
        ph = math.atan2(c.imag, c.real)
        scaled = mag * (r_ref ** (n - 1))
        lines.append(f"{n:<2d} C{n:<2d}  {c.real: .12e}  {c.imag: .12e}  {mag: .12e}  {ph: .8f}  {scaled: .12e}")

    lines.append("")
    lines.append(f"Sum_{{n>1}} |Cn|*r_ref^(n-1) = {sum_high:.12e}  (V/m)")

    path.write_text("\n".join(lines) + "\n")


def write_best_points_txt(path: Path, cfg: SolveConfig, y_points: np.ndarray) -> None:
    x_ctrl = x_control_points(cfg)
    arr = np.column_stack([x_ctrl, y_points])
    np.savetxt(path, arr, header="x y")


def write_best_simulation_output_map(path: Path, max_result: np.ndarray, vals_small: np.ndarray) -> None:
    rows = np.asarray(vals_small, dtype=float).reshape(-1, 4)
    if rows.size:
        order = np.lexsort((rows[:, 1], rows[:, 0]))  # x asc, then y asc
        rows = rows[order]

    with path.open("w") as f:
        f.write(f"{max_result[0]:.12e} {max_result[1]:.12e} {max_result[2]:.12e}\n")
        f.write("x y Ex Ey\n")
        for row in rows:
            f.write(f"{row[0]:.12e} {row[1]:.12e} {row[2]:.12e} {row[3]:.12e}\n")


def plot_mesh_wireframe(out: Path, pts: np.ndarray, tris: np.ndarray, top: Polygon, bot: Polygon) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.triplot(pts[:, 0], pts[:, 1], tris, linewidth=0.15)
    for poly in (top, bot):
        x, y = poly.exterior.xy
        ax.plot(x, y, linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_title("Conformal Gmsh Mesh")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_phi_contour(out: Path, xi: np.ndarray, yi: np.ndarray, phi_grid: np.ndarray) -> None:
    Xg, Yg = np.meshgrid(xi, yi)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    cs = ax.contourf(Xg, Yg, phi_grid, levels=60)
    fig.colorbar(cs, ax=ax, label="Potential [V]")
    ax.set_title("Electric Potential (phi)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_emag_with_vectors(out: Path, xi: np.ndarray, yi: np.ndarray, Ex_grid: np.ndarray, Ey_grid: np.ndarray, air_mask: np.ndarray) -> None:
    Xg, Yg = np.meshgrid(xi, yi)
    Emag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    Exp = np.where(air_mask, Ex_grid, 0.0)
    Eyp = np.where(air_mask, Ey_grid, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(
        Emag,
        origin="lower",
        extent=[xi[0], xi[-1], yi[0], yi[-1]],
        aspect="equal",
        interpolation="bicubic",
        cmap="magma",
    )
    fig.colorbar(im, ax=ax, label="|E| [V/m]")

    stride = max(1, int(len(xi) / 60))
    xs = Xg[::stride, ::stride]
    ys = Yg[::stride, ::stride]
    us = Exp[::stride, ::stride]
    vs = Eyp[::stride, ::stride]
    ax.streamplot(xs, ys, us, vs, density=1.2, linewidth=0.6, arrowsize=0.8)

    ax.set_title("Field Magnitude |E| with E-direction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_convergence(out: Path, best_hist: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.plot(np.arange(1, len(best_hist) + 1), best_hist)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best objective so far")
    ax.set_title("CMA-ES convergence")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


@dataclass
class CMAState:
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    pc: np.ndarray
    ps: np.ndarray
    B: np.ndarray
    D: np.ndarray
    invsqrtC: np.ndarray
    chiN: float
    counteval: int = 0
    eigeneval: int = 0


def cma_init(dim: int, x0: np.ndarray, sigma0: float, seed: int) -> Tuple[CMAState, Dict]:
    rng = np.random.default_rng(seed)
    lam = 4 + int(3 * np.log(dim))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = (np.sum(weights) ** 2) / np.sum(weights**2)

    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

    mean = x0.astype(float).copy()
    sigma = float(sigma0)
    C = np.eye(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    invsqrtC = np.eye(dim)
    chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))

    st = CMAState(mean=mean, sigma=sigma, C=C, pc=pc, ps=ps, B=B, D=D, invsqrtC=invsqrtC, chiN=chiN)
    params = dict(rng=rng, lam=lam, mu=mu, weights=weights, mueff=mueff, cc=cc, cs=cs, c1=c1, cmu=cmu, damps=damps)
    return st, params


def cma_ask(st: CMAState, params: Dict) -> np.ndarray:
    rng = params["rng"]
    lam = params["lam"]
    dim = st.mean.size
    arz = rng.normal(size=(dim, lam))
    ary = (st.B @ (st.D[:, None] * arz))
    return st.mean[:, None] + st.sigma * ary


def cma_tell(st: CMAState, params: Dict, arx: np.ndarray, fitness: np.ndarray) -> None:
    lam = params["lam"]
    mu = params["mu"]
    weights = params["weights"]
    dim = st.mean.size

    idx = np.argsort(fitness)
    xsel = arx[:, idx[:mu]]
    old_mean = st.mean.copy()

    st.mean = xsel @ weights

    y = (st.mean - old_mean) / st.sigma
    st.ps = (1 - params["cs"]) * st.ps + math.sqrt(params["cs"] * (2 - params["cs"]) * params["mueff"]) * (st.invsqrtC @ y)

    hsig = float(
        (np.linalg.norm(st.ps) / math.sqrt(1 - (1 - params["cs"]) ** (2 * (st.counteval / lam + 1))) / st.chiN)
        < (1.4 + 2 / (dim + 1))
    )

    st.pc = (1 - params["cc"]) * st.pc + hsig * math.sqrt(params["cc"] * (2 - params["cc"]) * params["mueff"]) * y

    artmp = (xsel - old_mean[:, None]) / st.sigma
    Cmu = artmp @ np.diag(weights) @ artmp.T

    st.C = (
        (1 - params["c1"] - params["cmu"]) * st.C
        + params["c1"] * (np.outer(st.pc, st.pc) + (1 - hsig) * params["cc"] * (2 - params["cc"]) * st.C)
        + params["cmu"] * Cmu
    )

    st.sigma *= math.exp((params["cs"] / params["damps"]) * (np.linalg.norm(st.ps) / st.chiN - 1))

    st.counteval += lam

    if st.counteval - st.eigeneval > lam / (params["c1"] + params["cmu"]) / dim / 10:
        st.eigeneval = st.counteval
        st.C = np.triu(st.C) + np.triu(st.C, 1).T
        D2, B = np.linalg.eigh(st.C)
        D2 = np.maximum(D2, 1e-30)
        st.D = np.sqrt(D2)
        st.B = B
        st.invsqrtC = st.B @ np.diag(1.0 / st.D) @ st.B.T


def evaluate_candidate(cfg: SolveConfig, y_points: np.ndarray) -> Tuple[float, Dict]:
    y_points = clamp_y_points(cfg, y_points)

    top = build_top_electrode_polygon(cfg, y_points)
    bot = shp_scale(top, 1.0, -1.0, origin=(0, 0))
    metal = unary_union([top, bot])

    pts, tris, t_m, b_m, bd_m = _build_mesh_gmsh(top, bot, cfg)
    phi = _fem_solve_phi(pts, tris, t_m, b_m, bd_m, cfg)
    Exn, Eyn = _nodal_field_from_phi(pts, tris, phi)

    xi, yi, phi_grid, Ex_grid, Ey_grid, air_mask_grid = _grid_interpolate_from_mesh(pts, tris, phi, Exn, Eyn, metal, cfg)
    max_result, vals_small = _max_and_roi(xi, yi, Ex_grid, Ey_grid, air_mask_grid, cfg)

    uni_r = uniformity_radius(cfg, xi, yi, Ex_grid, Ey_grid, air_mask_grid)

    n_arr, C = multipole_coeffs_on_circle(cfg, xi, yi, Ex_grid, Ey_grid)
    obj = objective_sum_high_orders(cfg, n_arr, C)

    diag = dict(
        y_points=y_points,
        top=top,
        bot=bot,
        metal=metal,
        pts=pts,
        tris=tris,
        xi=xi,
        yi=yi,
        phi_grid=phi_grid,
        Ex_grid=Ex_grid,
        Ey_grid=Ey_grid,
        air_mask_grid=air_mask_grid,
        max_result=max_result,
        vals_small=vals_small,
        uniformity_radius=uni_r,
        n_arr=n_arr,
        C=C,
        obj=obj,
    )
    return obj, diag


def run_optimization(cfg: SolveConfig) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = Path(cfg.outdir_opt)
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / f"optruns_{stamp}"
    runs_dir = run_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Initial CMA mean (y-control points). Endpoints are always pinned to st via clamp_y_points().
    # If mean_range>0, start from a slightly perturbed mean inside [st-mean_range, st+mean_range].
    y0 = np.full(cfg.Npoint, cfg.st, dtype=float)
    if cfg.mean_range and float(cfg.mean_range) > 0.0 and cfg.Npoint > 2:
        rng0 = np.random.default_rng(cfg.seed + 1337)
        y0[1:-1] = cfg.st + rng0.uniform(-float(cfg.mean_range), float(cfg.mean_range), size=cfg.Npoint - 2)
        # Keep initial mean inside the hard optimization bounds.
        lo = cfg.st - cfg.y_delta_max
        hi = cfg.st + cfg.y_delta_max
        y0[1:-1] = np.clip(y0[1:-1], lo, hi)
    y0[0] = cfg.st
    y0[-1] = cfg.st

    st_cma, params = cma_init(cfg.Npoint, x0=y0, sigma0=cfg.y_delta_max / 3.0, seed=cfg.seed)

    best_obj = float("inf")
    best_pack: Optional[Dict] = None
    best_tag: str = ""

    best_hist: List[float] = []

    pool = None
    if cfg.nproc and cfg.nproc > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=cfg.nproc)

    eval_counter = 0
    gen = 0
    t0 = time.time()

    print(
        f"[start] outdir={run_dir} budget={cfg.budget} nproc={cfg.nproc} Npoint={cfg.Npoint} "
        f"split_point={cfg.split_point:.3f} mean_range={cfg.mean_range:.3g}"
    )

    try:
        while eval_counter < cfg.budget:
            arx = cma_ask(st_cma, params)
            lam = arx.shape[1]

            remaining = cfg.budget - eval_counter
            if lam > remaining:
                arx = arx[:, :remaining]
                lam = remaining

            cand_list = [arx[:, k].copy() for k in range(lam)]

            if pool is None:
                results = [evaluate_candidate(cfg, yk) for yk in cand_list]
            else:
                results = pool.starmap(evaluate_candidate, [(cfg, yk) for yk in cand_list])

            fitness = np.array([r[0] for r in results], dtype=float)

            for k, (obj, diag) in enumerate(results):
                eval_id = eval_counter + k
                base = f"gen{gen:04d}_eval{eval_id:06d}"

                yaml_path = runs_dir / f"{base}_.yaml"
                out_path = runs_dir / f"{base}_out.txt"

                write_points_yaml(yaml_path, cfg, diag["y_points"])
                write_eval_out_txt(
                    out_path,
                    cfg,
                    diag["max_result"],
                    diag["n_arr"],
                    diag["C"],
                    diag["obj"],
                    diag["uniformity_radius"],
                )

                if obj < best_obj:
                    best_obj = obj
                    best_pack = diag
                    best_tag = base
                    print(f"[new best] {best_tag}  best_sum={best_obj:.6e}")

            eval_counter += lam
            cma_tell(st_cma, params, arx, fitness)

            best_hist.append(best_obj)
            plot_convergence(run_dir / "convergence.png", best_hist)

            pct = 100.0 * eval_counter / float(cfg.budget)
            elapsed = time.time() - t0
            print(
                f"[gen {gen:04d}] eval {eval_counter:04d}/{cfg.budget} ({pct:5.1f}%) "
                f"best={best_obj:.6e} sigma={st_cma.sigma:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
            gen += 1

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if best_pack is None:
        raise RuntimeError("No valid evaluation produced a best solution.")

    write_best_points_txt(run_dir / "best.points.txt", cfg, best_pack["y_points"])
    write_eval_out_txt(
        run_dir / "best.out.txt",
        cfg,
        best_pack["max_result"],
        best_pack["n_arr"],
        best_pack["C"],
        best_pack["obj"],
        best_pack["uniformity_radius"],
    )
    write_best_simulation_output_map(run_dir / "best_simulation_output_map.txt", best_pack["max_result"], best_pack["vals_small"])

    plot_mesh_wireframe(run_dir / "mesh_wireframe.png", best_pack["pts"], best_pack["tris"], best_pack["top"], best_pack["bot"])
    plot_phi_contour(run_dir / "phi_map.png", best_pack["xi"], best_pack["yi"], best_pack["phi_grid"])
    plot_emag_with_vectors(run_dir / "emag_density.png", best_pack["xi"], best_pack["yi"], best_pack["Ex_grid"], best_pack["Ey_grid"], best_pack["air_mask_grid"])

    if best_tag:
        src_yaml = runs_dir / f"{best_tag}_.yaml"
        src_out = runs_dir / f"{best_tag}_out.txt"
        if src_yaml.exists():
            shutil.copy2(src_yaml, run_dir / f"{best_tag}_.yaml")
        if src_out.exists():
            shutil.copy2(src_out, run_dir / f"{best_tag}_out.txt")

    print(f"[done] best_tag={best_tag} best_sum={best_obj:.6e} outputs written in {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", type=str, default=None, help="YAML file with config overrides")
    p.add_argument("--budget", type=int, default=None, help="Total evaluations")
    p.add_argument("--nproc", type=int, default=None, help="Parallel processes")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SolveConfig()

    if args.yaml:
        cfg = apply_overrides(cfg, load_yaml_config(Path(args.yaml)))

    if args.budget is not None:
        cfg.budget = args.budget
    if args.nproc is not None:
        cfg.nproc = args.nproc
    if args.seed is not None:
        cfg.seed = args.seed

    run_dir = run_optimization(cfg)
    print("Run directory:", str(run_dir.resolve()))


if __name__ == "__main__":
    main()
