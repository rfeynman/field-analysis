"""electrodes_fast.py

Triangular FEM electrostatics solver (Mathematica ToElementMesh/TriangleElement style)

This script is a Python port of your Mathematica notebook logic:
  - airRegion = RegionDifference[outerBox, fullGeometry]
  - mesh = ToElementMesh[airRegion, MaxCellMeasure -> ...]  (triangles)
  - Solve Laplacian(phi)=0 in airRegion with Dirichlet on electrode boundaries
  - E = -grad(phi)
  - Sample E on a uniform grid in [-0.1,0.1]^2
  - Return:
      maxResult = [x_at_max, y_at_max, |E|_max]
      valsSmall = [[x,y,Ex,Ey], ...] for |x|,|y|<=0.03

Key guarantees (per your requirements):
  - Electrode metal is REMOVED from the PDE domain (no field solved inside metal).
  - Any evaluation point outside the air FEM region OR inside metal returns Ex=Ey=0.
  - Includes debug plotting:
      * full mesh wireframe (entire outer box)
      * |E| 2D density plot (Mathematica DensityPlot equivalent)

Meshing notes
-------------
If you have the Gmsh Python API installed, this script can use it to create a
true constrained triangular mesh with holes (best match to Mathematica).
If Gmsh is not available, it falls back to a fast Delaunay + refinement mesh
that still respects holes by masking triangles (plus optional adaptive refinement).

Usage
-----
  python electrodes_fast.py

Or import:
  from electrodes_fast import run_simulation, SolveConfig
  maxResult, valsSmall = run_simulation(SolveConfig())

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import scale as shp_scale
from shapely.ops import unary_union

from scipy.optimize import root
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    import gmsh  # type: ignore
    _HAS_GMSH = True
except Exception:
    gmsh = None
    _HAS_GMSH = False


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
        f1 = np.linalg.det(np.c_[t1, t2])
        f2 = np.linalg.det(np.c_[p2 - p1, t1])
        return np.array([f1, f2], dtype=float)

    sol = root(fun, np.array(guess, dtype=float), method="hybr")
    if not sol.success:
        return None

    th1, th2 = map(float, sol.x)
    p1 = _ellipse_point(c1, a1, b1, th1)
    p2 = _ellipse_point(c2, a2, b2, th2)
    return p1, p2, th1, th2


def _solve_point_tangent(pt: np.ndarray, e: Tuple[np.ndarray, float, float], guess: float):
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



def _find_best_ellipse_tangent(e1, e2, side: str = "right", select_mode: str = "outer"):
    """
    Finds the best tangent between two ellipses.
    - select_mode="outer": Picks the right-most points (convex hull style).
    - select_mode="inner_e2": Picks the left-most point on the first ellipse (for e2 indentation).
    - select_mode="inner_e3": Picks the left-most point on the second ellipse (for e3 transition).
    """
    guesses = np.linspace(-math.pi, math.pi, 9)
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
        raise RuntimeError(f"No tangent found between ellipses with mode {select_mode}")

    if select_mode == "inner_e2":
        # Force the tangent point on the first ellipse to be 'inward'
        sols.sort(key=lambda s: s[0][0])
    elif select_mode == "inner_e3":
        # Force the tangent point on the second ellipse to be 'inward'
        sols.sort(key=lambda s: s[1][0])
    else:
        # Standard convex tangent
        sols.sort(key=lambda s: (s[0][0] + s[1][0]), reverse=True)
    
    return sols[0]


def _find_best_point_tangent(pt, e, side: str = "right"):
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
        raise RuntimeError("No point-ellipse tangent found. Try changing ellipse params.")

    sols.sort(key=lambda s: s[0][0], reverse=True)
    return sols[0]


def _arc_points(e: Tuple[np.ndarray, float, float], th_start: float, th_end: float, n: int = 25, force_long: bool = False) -> np.ndarray:
    """
    Generates points along an elliptical arc.
    - Uses modular arithmetic to find the shortest angular path by default.
    - If force_long is True, it forces the path the 'long way' around (essential for the e2 indentation).
    """
    s = float(th_start)
    ee = float(th_end)
    
    # Calculate shortest angular difference in [-pi, pi]
    diff = (ee - s + np.pi) % (2 * np.pi) - np.pi
    
    if force_long:
        # Flip to the long path by adding/subtracting 2*pi
        diff = diff - (2 * np.pi) if diff > 0 else diff + (2 * np.pi)
            
    ts = np.linspace(s, s + diff, n + 1)
    c, a, b = e
    return np.column_stack([c[0] + a * np.cos(ts), c[1] + b * np.sin(ts)])

def generate_electrode(
    params: Sequence[Tuple[Tuple[float, float], float, float]],
    symmetry_x: float,
    st: float,
    ed: float,
    n_arc: int = 25,
) -> Polygon:
    """Corrected electrode with opposite tangents for e2-e3 and e3-e4."""
    e1, e2, e3, e4 = [(np.array(p[0], dtype=float), float(p[1]), float(p[2])) for p in params]

    top_pt = np.array([symmetry_x, ed], dtype=float)
    start_pt = np.array([symmetry_x, st], dtype=float)

    # 1. Solve for the specific tangent orientations required
    # e1 to e2: Standard outer tangent
    t12 = _find_best_ellipse_tangent(e1, e2, side="right", select_mode="outer")
    
    # e2 to e3: Opposite tangent (inward on e2)
    t23 = _find_best_ellipse_tangent(e2, e3, side="right", select_mode="inner_e2")
    
    # e3 to e4: Opposite tangent (inward on e3) - THIS FIXES THE E3-E4 ATTACHMENT
    t34 = _find_best_ellipse_tangent(e3, e4, side="right", select_mode="inner_e2")
    
    # Final point tangent
    p4E, th4E = _find_best_point_tangent(start_pt, e4, side="right")

    right = []
    right.append(top_pt)
    right.append(np.array([e1[0][0], ed], dtype=float))

    # All segments now use the standard short-arc logic as the correct points are selected
    right.extend(_arc_points(e1, math.pi / 2, t12[2], n_arc)[1:])
    right.extend(_arc_points(e2, t12[3], t23[2], n_arc)[1:])
    right.extend(_arc_points(e3, t23[3], t34[2], n_arc)[1:])
    right.extend(_arc_points(e4, t34[3], th4E, n_arc)[1:])
    right.append(start_pt)

    # Mirroring and Cleanup
    right = np.array(right, dtype=float)
    mirrored = right.copy()
    mirrored[:, 0] = 2 * symmetry_x - mirrored[:, 0]
    full = np.vstack([right, mirrored[::-1]])

    cleaned = [full[0]]
    for p in full[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-10:
            cleaned.append(p)

    poly = Polygon(cleaned)
    return poly if poly.is_valid else poly.buffer(0)

# -----------------------------
# Config
# -----------------------------

@dataclass
class SolveConfig:
    # Voltages
    Vt: float = 60000.0
    Vb: float = -60000.0

    # Outer box half size [m]
    Lx: float = 0.25
    Ly: float = 0.25

    # Electrode vertical span
    thick: float = 0.01
    st: float = 0.035

    # Meshing control: Mathematica MaxCellMeasure is a max triangle *area*.
    # Use 2e-5 to match the notebook.
    max_cell_area: float = 2e-5

    # Boundary sampling density relative to h0
    boundary_h_factor: float = 0.33   # electrode boundary spacing = h0*boundary_h_factor

    # Optional adaptive refinement after a solve (adds points where |E| is largest)
    adapt_iters: int = 1
    adapt_top_frac: float = 0.06      # refine top ~6% triangles by |E|

    # Sampling grid (Mathematica: xmin/xmax etc)
    xmin: float = -0.10
    xmax: float = 0.10
    ymin: float = -0.10
    ymax: float = 0.10
    n_sample: int = 160

    roi: float = 0.03

    seed: int = 123

    # If True and gmsh is available, prefer gmsh mesh.
    prefer_gmsh: bool = True

    # Plot controls
    debug_plots: bool = True
    outdir: str = "."

    @property
    def ed(self) -> float:
        return self.st + self.thick


def default_example_params(cfg: SolveConfig):
    # Matches notebook's exampleParams
    return [
        ((0.06, 0.042), 0.003, cfg.ed - 0.042),
        ((0.06, 0.025), 0.005, 0.005),
        ((0.052, 0.025), 0.005, 0.005),
        ((0.045, 0.032), 0.003, 0.003),
    ]


# -----------------------------
# Meshing utilities
# -----------------------------

def _h0_from_area(max_area: float) -> float:
    # Equilateral triangle area: A = sqrt(3)/4 * h^2  => h = sqrt(4A/sqrt(3))
    return math.sqrt(4.0 * max_area / math.sqrt(3.0))


def _resample_ring(coords: np.ndarray, ds: float) -> np.ndarray:
    """Resample a closed ring (Nx2, last may equal first) with arc-step ds."""
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 3:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])

    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt((seg**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = s[-1]
    if L <= 0:
        return pts[:-1]

    n = max(3, int(math.ceil(L / ds)))
    tq = np.linspace(0.0, L, n, endpoint=False)

    out = []
    j = 0
    for t in tq:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(s):
            j = len(s) - 2
        u = (t - s[j]) / max(1e-30, (s[j + 1] - s[j]))
        out.append(pts[j] * (1 - u) + pts[j + 1] * u)
    return np.array(out, dtype=float)


def _polygon_band(poly: Polygon, tol: float) -> Polygon:
    # band around boundary: buffer(+tol) - buffer(-tol)
    outer = poly.buffer(tol)
    inner = poly.buffer(-tol)
    if inner.is_empty:
        return outer
    band = outer.difference(inner)
    if not band.is_valid:
        band = band.buffer(0)
    return band


def _build_mesh_delaunay_refine(
    air: Polygon,
    top: Polygon,
    bottom: Polygon,
    cfg: SolveConfig,
    extra_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast unstructured triangle mesh via Delaunay + masking + max-area refinement."""

    rng = np.random.default_rng(cfg.seed)

    h0 = _h0_from_area(cfg.max_cell_area)
    hb = max(1e-6, cfg.boundary_h_factor * h0)

    # base jittered grid points over outer domain
    xs = np.arange(-cfg.Lx, cfg.Lx + h0, h0)
    ys = np.arange(-cfg.Ly, cfg.Ly + h0, h0)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    pts = pts + (rng.random(pts.shape) - 0.5) * (0.25 * h0)

    # filter to air region
    mask_air = shapely.contains_xy(air, pts[:, 0], pts[:, 1])
    pts = pts[mask_air]

    # outer boundary points
    ob = []
    xb = np.arange(-cfg.Lx, cfg.Lx + hb, hb)
    yb = np.arange(-cfg.Ly, cfg.Ly + hb, hb)
    ob.append(np.column_stack([xb, np.full_like(xb, -cfg.Ly)]))
    ob.append(np.column_stack([xb, np.full_like(xb, cfg.Ly)]))
    ob.append(np.column_stack([np.full_like(yb, -cfg.Lx), yb]))
    ob.append(np.column_stack([np.full_like(yb, cfg.Lx), yb]))
    ob = np.vstack(ob)

    # electrode boundary points (dense)
    top_b = _resample_ring(np.asarray(top.exterior.coords), ds=hb)
    bot_b = _resample_ring(np.asarray(bottom.exterior.coords), ds=hb)

    # optional extra points (for adaptive refinement)
    if extra_points is None:
        extra_points = np.empty((0, 2), dtype=float)

    all_pts = np.vstack([pts, ob, top_b, bot_b, extra_points])

    # Remove points that accidentally fall inside electrodes or outside air
    mask_air2 = shapely.contains_xy(air, all_pts[:, 0], all_pts[:, 1])
    all_pts = all_pts[mask_air2]

    # deduplicate (grid-scale)
    q = max(1e-12, hb * 0.25)
    key = np.round(all_pts / q).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    all_pts = all_pts[np.sort(idx)]

    # refinement loop to enforce max triangle area
    for _ in range(6):
        tri = Delaunay(all_pts)
        simplices = tri.simplices
        p = all_pts[simplices]

        cent0 = p.mean(axis=1)
        # keep triangles whose centroid AND edge midpoints are in air
        c_ok = shapely.contains_xy(air, cent0[:, 0], cent0[:, 1])
        mid01 = (p[:, 0, :] + p[:, 1, :]) * 0.5
        mid12 = (p[:, 1, :] + p[:, 2, :]) * 0.5
        mid20 = (p[:, 2, :] + p[:, 0, :]) * 0.5
        m_ok = (
            shapely.contains_xy(air, mid01[:, 0], mid01[:, 1])
            & shapely.contains_xy(air, mid12[:, 0], mid12[:, 1])
            & shapely.contains_xy(air, mid20[:, 0], mid20[:, 1])
        )
        keep = c_ok & m_ok
        simplices = simplices[keep]
        p = all_pts[simplices]
        cent = p.mean(axis=1)

        # areas
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        too_big = area > (cfg.max_cell_area * 1.05)
        if not np.any(too_big):
            # compress to used points and return
            used = np.unique(simplices.ravel())
            remap = -np.ones(len(all_pts), dtype=int)
            remap[used] = np.arange(len(used))
            pts2 = all_pts[used]
            tri2 = remap[simplices]
            return pts2, tri2

        # add centroids of too-large triangles (but only if they are well inside air)
        newp = cent[too_big]
        # small jitter
        newp = newp + (rng.random(newp.shape) - 0.5) * (0.02 * h0)
        ok = shapely.contains_xy(air, newp[:, 0], newp[:, 1])
        newp = newp[ok]
        if len(newp) == 0:
            break
        all_pts = np.vstack([all_pts, newp])

    # final attempt
    tri = Delaunay(all_pts)
    simplices = tri.simplices
    p = all_pts[simplices]
    cent = p.mean(axis=1)
    keep = shapely.contains_xy(air, cent[:, 0], cent[:, 1])
    simplices = simplices[keep]
    used = np.unique(simplices.ravel())
    remap = -np.ones(len(all_pts), dtype=int)
    remap[used] = np.arange(len(used))
    pts2 = all_pts[used]
    tri2 = remap[simplices]
    return pts2, tri2

def _build_mesh_gmsh(air_poly: Polygon, top_poly: Polygon, bot_poly: Polygon, cfg: SolveConfig):
    if not _HAS_GMSH:
        raise RuntimeError("gmsh is not available")

    h_fine = cfg.boundary_h_factor * _h0_from_area(cfg.max_cell_area)
    h_coarse = _h0_from_area(cfg.max_cell_area)

    gmsh.initialize()
    gmsh.model.add("electrode_simulation")
    # Use OpenCASCADE for reliable Boolean subtractions
    occ = gmsh.model.occ

    # 1. Create the outer box
    outer_tag = occ.addRectangle(-cfg.Lx, -cfg.Ly, 0, 2*cfg.Lx, 2*cfg.Ly)
    
    # 2. Helper to add electrode as a wire/surface
    def poly_to_occ(poly):
        coords = np.asarray(poly.exterior.coords, dtype=float)
        p_tags = [occ.addPoint(x, y, 0) for x, y in coords[:-1]]
        l_tags = []
        for i in range(len(p_tags)):
            l_tags.append(occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]))
        wire = occ.addCurveLoop(l_tags)
        return occ.addPlaneSurface([wire]), l_tags

    top_surf, top_lines = poly_to_occ(top_poly)
    bot_surf, bot_lines = poly_to_occ(bot_poly)

    # 3. BOOLEAN SUBTRACTION (Crucial for attachment)
    # This physically removes the metal from the domain
    air_surf_tags, _ = occ.cut([(2, outer_tag)], [(2, top_surf), (2, bot_surf)])
    occ.synchronize()

    # 4. REFINEMENT FIELDS (Mathematica Style)
    # Refine based on local curvature
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 40) # Higher = finer curves

    # Threshold field for high-density "black" regions near electrodes
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", top_lines + bot_lines)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)   # Target density at boundary
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse) # Default density far away
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.002)    # 2mm of max density
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.030)    # Fade out by 30mm

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # 5. Generate and Extract
    gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    _, _, elem_node_tags = gmsh.model.mesh.getElements(2)
    triangles = np.array(elem_node_tags[0], dtype=int).reshape(-1, 3)
    triangles = np.vectorize(tag_to_idx.get)(triangles)

    # Correct Dirichlet masking using proximity (since Booleans re-tag IDs)
    tree = shapely.STRtree([Point(p) for p in points])
    top_mask = np.zeros(len(points), dtype=bool)
    bot_mask = np.zeros(len(points), dtype=bool)
    
    # Nodes within 1e-5 of the original polygons are boundary nodes
    for i, p in enumerate(points):
        pt = Point(p)
        if top_poly.boundary.distance(pt) < 1e-6: top_mask[i] = True
        if bot_poly.boundary.distance(pt) < 1e-6: bot_mask[i] = True

    gmsh.finalize()
    return points, triangles, top_mask, bot_mask



# -----------------------------
# FEM assembly/solve
# -----------------------------

def _assemble_stiffness(points: np.ndarray, triangles: np.ndarray) -> csr_matrix:
    n = len(points)
    K = lil_matrix((n, n), dtype=float)

    p = points[triangles]  # (m,3,2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]

    # twice area with sign
    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * np.abs(twoA)

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    # local stiffness entries: (b_i b_j + c_i c_j)/(4A)
    denom = 4.0 * A
    denom = np.maximum(denom, 1e-30)

    bb = np.stack([b1, b2, b3], axis=1)
    cc = np.stack([c1, c2, c3], axis=1)

    for i in range(3):
        for j in range(3):
            val = (bb[:, i] * bb[:, j] + cc[:, i] * cc[:, j]) / denom
            I = triangles[:, i]
            J = triangles[:, j]
            for k in range(len(val)):
                K[I[k], J[k]] += val[k]

    return K.tocsr()


def _solve_dirichlet(K: csr_matrix, dir_mask: np.ndarray, dir_values: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    dir_idx = np.where(dir_mask)[0]
    free_idx = np.where(~dir_mask)[0]

    u = np.zeros(n, dtype=float)
    u[dir_idx] = dir_values[dir_idx]

    if len(free_idx) == 0:
        return u

    Kff = K[free_idx][:, free_idx]
    Kfc = K[free_idx][:, dir_idx]

    rhs = -Kfc @ u[dir_idx]

    uf = spsolve(Kff, rhs)
    u[free_idx] = uf
    return u


def _triangle_grad_phi(points: np.ndarray, triangles: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-triangle grad(phi) and |E| (piecewise constant)."""
    p = points[triangles]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * np.abs(twoA)
    A = np.maximum(A, 1e-30)

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    ph = phi[triangles]
    dphidx = (ph[:, 0] * b1 + ph[:, 1] * b2 + ph[:, 2] * b3) / (2.0 * A)
    dphidy = (ph[:, 0] * c1 + ph[:, 1] * c2 + ph[:, 2] * c3) / (2.0 * A)

    Ex = -dphidx
    Ey = -dphidy
    emag = np.sqrt(Ex**2 + Ey**2)
    return Ex, Ey, emag


def _nodal_field_from_triangles(points: np.ndarray, triangles: np.ndarray, tri_Ex: np.ndarray, tri_Ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Node-averaged E (smoother, closer to Mathematica's InterpolatingFunction derivatives)."""
    n = len(points)
    Exn = np.zeros(n, dtype=float)
    Eyn = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=float)

    for t, (i, j, k) in enumerate(triangles):
        Exn[i] += tri_Ex[t]; Eyn[i] += tri_Ey[t]; cnt[i] += 1
        Exn[j] += tri_Ex[t]; Eyn[j] += tri_Ey[t]; cnt[j] += 1
        Exn[k] += tri_Ex[t]; Eyn[k] += tri_Ey[t]; cnt[k] += 1

    cnt = np.maximum(cnt, 1.0)
    Exn /= cnt
    Eyn /= cnt
    return Exn, Eyn


# -----------------------------
# Sampling + plotting
# -----------------------------

def plot_mesh_wireframe(points: np.ndarray, triangles: np.ndarray, top: Polygon, bottom: Polygon, cfg: SolveConfig, savepath: str):
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.triplot(tri, linewidth=0.25)

    # overlay electrode outlines (white fill to mimic your screenshot)
    for poly in (top, bottom):
        x, y = poly.exterior.xy
        ax.fill(x, y, color="white", zorder=3)
        ax.plot(x, y, color="black", linewidth=0.6, zorder=4)

    ax.set_aspect("equal", "box")
    ax.set_xlim([-cfg.Lx, cfg.Lx])
    ax.set_ylim([-cfg.Ly, cfg.Ly])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Triangular mesh (air region)")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_emag_density(xi: np.ndarray, yi: np.ndarray, emag2d: np.ndarray, cfg: SolveConfig, savepath: str):
    fig = plt.figure(figsize=(7, 6), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        emag2d,
        origin="lower",
        extent=[xi[0], xi[-1], yi[0], yi[-1]],
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="|E| [V/m]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("|E| density (Mathematica DensityPlot equivalent)")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def _sample_on_grid(
    points: np.ndarray,
    triangles: np.ndarray,
    Exn: np.ndarray,
    Eyn: np.ndarray,
    air: Polygon,
    metal: Polygon,
    cfg: SolveConfig,
) -> Tuple[List[float], List[List[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample node-averaged E on uniform grid in [xmin,xmax]x[ymin,ymax]."""

    xi = np.linspace(cfg.xmin, cfg.xmax, cfg.n_sample)
    yi = np.linspace(cfg.ymin, cfg.ymax, cfg.n_sample)

    tri_obj = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    trifinder = tri_obj.get_trifinder()

    # Precompute for barycentric interpolation of nodal E
    tri_pts = points[triangles]  # (m,3,2)

    Ex2d = np.zeros((cfg.n_sample, cfg.n_sample), dtype=float)
    Ey2d = np.zeros((cfg.n_sample, cfg.n_sample), dtype=float)

    # vectorized grid
    Xg, Yg = np.meshgrid(xi, yi)
    xflat = Xg.ravel()
    yflat = Yg.ravel()

    # outside air or inside metal -> 0
    in_air = shapely.contains_xy(air, xflat, yflat)
    in_metal = shapely.contains_xy(metal, xflat, yflat)
    ok = in_air & (~in_metal)

    tri_id = np.full_like(xflat, -1, dtype=int)
    tri_id[ok] = trifinder(xflat[ok], yflat[ok]).astype(int)

    good = ok & (tri_id >= 0)
    if np.any(good):
        tid = tri_id[good]
        P = np.column_stack([xflat[good], yflat[good]])

        T = tri_pts[tid]
        # barycentric weights
        x1, y1 = T[:, 0, 0], T[:, 0, 1]
        x2, y2 = T[:, 1, 0], T[:, 1, 1]
        x3, y3 = T[:, 2, 0], T[:, 2, 1]

        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        detT = np.where(np.abs(detT) < 1e-30, 1e-30, detT)

        l1 = ((y2 - y3) * (P[:, 0] - x3) + (x3 - x2) * (P[:, 1] - y3)) / detT
        l2 = ((y3 - y1) * (P[:, 0] - x3) + (x1 - x3) * (P[:, 1] - y3)) / detT
        l3 = 1.0 - l1 - l2

        idx0 = triangles[tid, 0]
        idx1 = triangles[tid, 1]
        idx2 = triangles[tid, 2]

        ex = l1 * Exn[idx0] + l2 * Exn[idx1] + l3 * Exn[idx2]
        ey = l1 * Eyn[idx0] + l2 * Eyn[idx1] + l3 * Eyn[idx2]

        out_ex = np.zeros_like(xflat, dtype=float)
        out_ey = np.zeros_like(xflat, dtype=float)
        out_ex[good] = ex
        out_ey[good] = ey

        Ex2d = out_ex.reshape(cfg.n_sample, cfg.n_sample)
        Ey2d = out_ey.reshape(cfg.n_sample, cfg.n_sample)

    emag2d = np.sqrt(Ex2d**2 + Ey2d**2)

    # build bigVals and then match Mathematica's logic:
    # goodVals are already finite here; outside region is 0
    # Find max
    flat_emag = emag2d.ravel()
    imax = int(np.argmax(flat_emag))
    iy, ix = divmod(imax, cfg.n_sample)
    maxResult = [float(xi[ix]), float(yi[iy]), float(flat_emag[imax])]

    # valsSmall
    valsSmall: List[List[float]] = []
    for j, y in enumerate(yi):
        if abs(y) > cfg.roi:
            continue
        for i, x in enumerate(xi):
            if abs(x) > cfg.roi:
                continue
            valsSmall.append([float(x), float(y), float(Ex2d[j, i]), float(Ey2d[j, i])])

    return maxResult, valsSmall, (xi, yi, emag2d)


# -----------------------------
# Main simulation
# -----------------------------

def run_simulation(cfg: SolveConfig = SolveConfig()):
    os.makedirs(cfg.outdir, exist_ok=True)

    # Geometry
    params = default_example_params(cfg)
    top = generate_electrode(params, symmetry_x=0.0, st=cfg.st, ed=cfg.ed, n_arc=35)
    bottom = shp_scale(top, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))

    metal = unary_union([top, bottom])
    outer = Polygon([(-cfg.Lx, -cfg.Ly), (cfg.Lx, -cfg.Ly), (cfg.Lx, cfg.Ly), (-cfg.Lx, cfg.Ly)])
    air = outer.difference(metal)
    if not air.is_valid:
        air = air.buffer(0)

    # Build mesh
    if cfg.prefer_gmsh and _HAS_GMSH:
        points, triangles, top_mask, bot_mask = _build_mesh_gmsh(air, top, bottom, cfg)
        # Dirichlet masks from gmsh
        dir_mask = top_mask | bot_mask
        dir_values = np.zeros(len(points), dtype=float)
        dir_values[top_mask] = cfg.Vt
        dir_values[bot_mask] = cfg.Vb
    else:
        points, triangles = _build_mesh_delaunay_refine(air, top, bottom, cfg)

        # Dirichlet masks (band around boundary)
        tol = _h0_from_area(cfg.max_cell_area) * 0.35
        top_band = _polygon_band(top, tol)
        bot_band = _polygon_band(bottom, tol)

        top_mask = shapely.contains_xy(top_band, points[:, 0], points[:, 1])
        bot_mask = shapely.contains_xy(bot_band, points[:, 0], points[:, 1])

        # ensure we don't accidentally tag outer boundary as electrode
        dir_mask = top_mask | bot_mask
        dir_values = np.zeros(len(points), dtype=float)
        dir_values[top_mask] = cfg.Vt
        dir_values[bot_mask] = cfg.Vb

    # Optional debug: mesh wireframe
    if cfg.debug_plots:
        mesh_path = os.path.join(cfg.outdir, "mesh_wireframe.png")
        plot_mesh_wireframe(points, triangles, top, bottom, cfg, savepath=mesh_path)

    # Adaptive loop
    extra_pts = np.empty((0, 2), dtype=float)
    phi = None

    for it in range(max(1, cfg.adapt_iters)):
        if (it > 0) and (not (cfg.prefer_gmsh and _HAS_GMSH)):
            # rebuild mesh with extra points (delaunay path only)
            points, triangles = _build_mesh_delaunay_refine(air, top, bottom, cfg, extra_points=extra_pts)
            tol = _h0_from_area(cfg.max_cell_area) * 0.35
            top_band = _polygon_band(top, tol)
            bot_band = _polygon_band(bottom, tol)
            top_mask = shapely.contains_xy(top_band, points[:, 0], points[:, 1])
            bot_mask = shapely.contains_xy(bot_band, points[:, 0], points[:, 1])
            dir_mask = top_mask | bot_mask
            dir_values = np.zeros(len(points), dtype=float)
            dir_values[top_mask] = cfg.Vt
            dir_values[bot_mask] = cfg.Vb

        K = _assemble_stiffness(points, triangles)
        phi = _solve_dirichlet(K, dir_mask, dir_values)

        # refine where |E| is largest (skip last iter)
        if it < cfg.adapt_iters - 1 and not (cfg.prefer_gmsh and _HAS_GMSH):
            tri_Ex, tri_Ey, tri_emag = _triangle_grad_phi(points, triangles, phi)
            m = len(tri_emag)
            k = max(50, int(cfg.adapt_top_frac * m))
            idx = np.argpartition(tri_emag, -k)[-k:]
            cent = points[triangles[idx]].mean(axis=1)
            # keep inside air
            ok = shapely.contains_xy(air, cent[:, 0], cent[:, 1])
            cent = cent[ok]
            extra_pts = np.vstack([extra_pts, cent])

    assert phi is not None

    tri_Ex, tri_Ey, _ = _triangle_grad_phi(points, triangles, phi)
    Exn, Eyn = _nodal_field_from_triangles(points, triangles, tri_Ex, tri_Ey)

    maxResult, valsSmall, (xi, yi, emag2d) = _sample_on_grid(points, triangles, Exn, Eyn, air, metal, cfg)

    if cfg.debug_plots:
        dens_path = os.path.join(cfg.outdir, "emag_density.png")
        plot_emag_density(xi, yi, emag2d, cfg, savepath=dens_path)

    return maxResult, valsSmall


if __name__ == "__main__":
    cfg = SolveConfig(debug_plots=True, outdir=".")
    maxResult, valsSmall = run_simulation(cfg)
    print("maxResult =", maxResult)
    print("valsSmall length =", len(valsSmall))
    print("Wrote debug plots: mesh_wireframe.png and emag_density.png")