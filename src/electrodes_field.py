"""
electrodes_field.py — 2D Electrostatic FEM Solver for Wien Filter Electrodes

Author: Erdong Wang

This script performs a 2D finite-element electrostatic simulation of a
two-electrode system defined by a smooth tangent/ellipse geometry.

It solves Laplace’s equation:

    ∇²φ = 0

with Dirichlet boundary conditions:
    - Top electrode:  φ = +Vt
    - Bottom electrode: φ = Vb
    - Outer rectangular boundary: φ = 0 (grounded metal box)

Electric field is computed from:

    E = -∇φ

The script generates:
    - FEM mesh (via Gmsh)
    - Potential distribution φ(x,y)
    - Electric field components Ex, Ey
    - Field magnitude |E|
    - Region-of-interest (ROI) data for multipole analysis

------------------------------------------------------------------------------
INPUT
------------------------------------------------------------------------------

Geometry can be provided in two ways:

1) Default built-in geometry (no arguments)

    python electrodes_field.py

2) YAML geometry file (recommended)

    python electrodes_field.py --geom-yaml geometry.yaml

The YAML file must define ellipses e1–e4:

    electrode_geometry:
      e1: {x0: ..., y0: ..., a: ...}
      e2: {x0: ..., y0: ..., a: ..., b: ...}
      e3: {x0: ..., y0: ..., a: ..., b: ...}
      e4: {x0: ..., y0: ..., a: ..., b: ...}

Note:
    e1_b is automatically computed as:
        e1_b = cfg.ed - e1_y0

Simulation parameters are controlled in the SolveConfig dataclass:
    - Vt, Vb                electrode voltages
    - thick, st             geometric offsets
    - xmin,xmax,ymin,ymax   outer metal boundary (grounded)
    - max_cell_area         mesh density
    - n_sample              grid resolution
    - roi                   region-of-interest half-width
    - debug_plots           enable/disable PNG output
    - writefile             enable/disable text output

------------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------------

Outputs are written to:

    <cfg.outdir>/

Generated files may include:

    mesh_wireframe.png
    phi_map.png
    emag_density.png

If writefile=True:
    simulation_output.txt

This file contains:
    Line 1: x_max, y_max, |E|max
    Line 2: header
    Remaining lines:
        x   y   Ex   Ey
    for all points within |x|,|y| ≤ roi.

The function run_simulation(...) returns:

    max_result   = [x_max, y_max, |E|max]
    vals_small   = array of [x, y, Ex, Ey] within ROI

These outputs are used by:
    electrodes_field_analysis.py
    CMA-ES optimization driver

------------------------------------------------------------------------------
HOW TO USE AS A MODULE
------------------------------------------------------------------------------

You can import and run directly:

    from electrodes_field import SolveConfig, run_simulation

    cfg = SolveConfig()
    max_result, vals_small = run_simulation(cfg, geom_yaml="geometry.yaml")

------------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------------

- Mesh generation uses Gmsh.
- Geometry construction uses Shapely.
- Linear system solved using SciPy sparse solver.
- Outer boundary is treated as grounded conductor (φ = 0).
- Designed for 2D transverse electrostatic field analysis.

------------------------------------------------------------------------------
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import scale as shp_scale
from shapely.ops import unary_union

from scipy.optimize import root
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh

# YAML support is optional and only used when you pass `--geom-yaml` (or
# `geom_yaml=...` into `run_simulation`).

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class SolveConfig:
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
    debug_plots: bool = True
    writefile: bool = True
    filename: str = "simulation_output.txt"
    outdir: str = "."

    @property
    def ed(self) -> float:
        return self.st + self.thick

# -----------------------------
# Geometry Logic (Tangents & Arcs)
# -----------------------------

# The electrode boundary on the RIGHT side is constructed from:
#   (0, ed) -> top of e1 -> short arc on e1 -> tangent segment e1-e2 ->
#   short arc on e2 -> tangent segment e2-e3 -> short arc on e3 ->
#   tangent segment e3-e4 -> short arc on e4 -> tangent segment e4-(0,st) -> (0, st)
# Then mirrored about the y-axis.
#
# Robustness rules implemented here (per project discussion):
#   - e1,e2,e3 are INSIDE the electrode polygon.
#   - ellipse4's ENTIRE INTERIOR is OUTSIDE the electrode polygon.
#   - Always use the SHORT arc between tangent points (defined by arc-length).
#   - Tangent family selection via center-line crossing:
#       * e1-e2 and e2-e3: tangent line must NOT cross the segment between centers.
#       * e3-e4: tangent line MUST cross the segment between centers.
#   - e3-e4: overlap OR touching is an error.
#   - e1-e2 and e2-e3: overlap/touch is allowed, BUT if one ellipse totally contains the other -> error.


def _ellipse_point(c, a, b, th):
    return np.array([c[0] + a * math.cos(th), c[1] + b * math.sin(th)], dtype=float)


def _ellipse_tangent(a, b, th):
    return np.array([-a * math.sin(th), b * math.cos(th)], dtype=float)


def _wrap_angle(th: float) -> float:
    return (float(th) + math.pi) % (2.0 * math.pi) - math.pi


def _polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _ellipse_polygon(e, n: int = 1200) -> Polygon:
    """Polygonal approximation of an ellipse (for robust topology tests)."""
    c, a, b = e
    ts = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.column_stack([c[0] + a * np.cos(ts), c[1] + b * np.sin(ts)])
    return Polygon(pts)


def _check_containment_only(ei, ej, *, name: str) -> None:
    """Allow overlap/touch, but forbid total containment (including boundary-touch containment)."""
    pi = _ellipse_polygon(ei)
    pj = _ellipse_polygon(ej)
    if not pi.is_valid or not pj.is_valid:
        raise RuntimeError(f"Invalid ellipse polygon approximation for {name}")

    # If one fully contains/covers the other (even if just tangent internally), tangents become
    # degenerate/ambiguous for our boundary logic.
    if pi.covers(pj) or pj.covers(pi):
        raise RuntimeError(f"Ellipse pair {name}: one ellipse contains the other (error)")


def _check_no_touch_or_overlap(ei, ej, *, name: str) -> None:
    """Forbid overlap or touching (any intersection at all)."""
    pi = _ellipse_polygon(ei)
    pj = _ellipse_polygon(ej)
    if not pi.is_valid or not pj.is_valid:
        raise RuntimeError(f"Invalid ellipse polygon approximation for {name}")
    if pi.intersects(pj):
        raise RuntimeError(f"Ellipse pair {name} overlaps or touches (error)")


def _line_signed_distance(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    v = p1 - p0
    w = q - p0
    return float(v[0] * w[1] - v[1] * w[0])


def _line_crosses_center_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    eps: float = 1e-12,
) -> Optional[bool]:
    """Return True if infinite line through p0,p1 crosses the segment c0-c1.

    Returns None for near-degenerate cases (line nearly passes through a center).
    """
    s0 = _line_signed_distance(p0, p1, c0)
    s1 = _line_signed_distance(p0, p1, c1)
    if abs(s0) <= eps or abs(s1) <= eps:
        return None
    return (s0 > 0) != (s1 > 0)


def _solve_ellipse_tangent(e1, e2, guess):
    c1, a1, b1 = e1
    c2, a2, b2 = e2

    def fun(v):
        th1, th2 = float(v[0]), float(v[1])
        p1 = _ellipse_point(c1, a1, b1, th1)
        p2 = _ellipse_point(c2, a2, b2, th2)
        t1 = _ellipse_tangent(a1, b1, th1)
        t2 = _ellipse_tangent(a2, b2, th2)
        # Conditions:
        # 1) tangents parallel  -> det([t1,t2]) = 0
        # 2) line through p1,p2 aligned with tangent direction -> det([p2-p1, t1]) = 0
        return np.array([
            float(np.linalg.det(np.c_[t1, t2])),
            float(np.linalg.det(np.c_[p2 - p1, t1])),
        ], dtype=float)

    sol = root(fun, np.array(guess, dtype=float), method="hybr")
    if not sol.success:
        return None

    th1, th2 = _wrap_angle(sol.x[0]), _wrap_angle(sol.x[1])
    p1 = _ellipse_point(c1, a1, b1, th1)
    p2 = _ellipse_point(c2, a2, b2, th2)
    t1 = _ellipse_tangent(a1, b1, th1)
    t2 = _ellipse_tangent(a2, b2, th2)

    r1 = abs(float(np.linalg.det(np.c_[t1, t2])))
    r2 = abs(float(np.linalg.det(np.c_[p2 - p1, t1])))
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return None
    if max(r1, r2) > 1e-9:
        return None
    return (p1, p2, th1, th2)


def _dedup_tangents(sols, tol: float = 1e-6):
    out = []
    for s in sols:
        p1, p2, th1, th2 = s
        keep = True
        for o in out:
            if np.linalg.norm(p1 - o[0]) <= tol and np.linalg.norm(p2 - o[1]) <= tol:
                keep = False
                break
        if keep:
            out.append(s)
    return out


def _all_ellipse_tangents(e1, e2, n_guess: int = 17):
    guesses = np.linspace(-math.pi, math.pi, n_guess)
    sols = []
    for g1 in guesses:
        for g2 in guesses:
            out = _solve_ellipse_tangent(e1, e2, (g1, g2))
            if out is not None:
                sols.append(out)
    return _dedup_tangents(sols)


def _solve_point_tangent(e, P: np.ndarray, guess_th: float):
    c, a, b = e

    def fun(v):
        th = float(v[0])
        p = _ellipse_point(c, a, b, th)
        t = _ellipse_tangent(a, b, th)
        return np.array([float(np.linalg.det(np.c_[p - P, t]))], dtype=float)

    sol = root(fun, np.array([guess_th], dtype=float), method="hybr")
    if not sol.success:
        return None

    th = _wrap_angle(float(sol.x[0]))
    p = _ellipse_point(c, a, b, th)
    t = _ellipse_tangent(a, b, th)
    r = abs(float(np.linalg.det(np.c_[p - P, t])))
    if not np.isfinite(r) or r > 1e-9:
        return None
    return (p, th)


def _all_point_tangents(e, P: np.ndarray, n_guess: int = 25):
    guesses = np.linspace(-math.pi, math.pi, n_guess)
    sols = []
    for g in guesses:
        out = _solve_point_tangent(e, P, g)
        if out is not None:
            sols.append(out)
    # dedup by point
    uniq = []
    for p, th in sols:
        keep = True
        for q, _ in uniq:
            if np.linalg.norm(p - q) <= 1e-6:
                keep = False
                break
        if keep:
            uniq.append((p, th))
    return uniq


def _arc_points_short(e, th_start: float, th_end: float, n: int = 90) -> np.ndarray:
    """Points along the *short* ellipse arc between two parameter angles.

    Shortness is decided by approximate arc-length (polyline length), not |Δθ|.
    """
    c, a, b = e
    t0 = float(th_start)
    t1 = float(th_end)

    d = _wrap_angle(t1 - t0)  # in (-pi, pi]
    d2 = d - 2.0 * math.pi if d > 0 else d + 2.0 * math.pi

    ts1 = np.linspace(t0, t0 + d, n + 1)
    pts1 = np.column_stack([c[0] + a * np.cos(ts1), c[1] + b * np.sin(ts1)])

    ts2 = np.linspace(t0, t0 + d2, n + 1)
    pts2 = np.column_stack([c[0] + a * np.cos(ts2), c[1] + b * np.sin(ts2)])

    return pts1 if _polyline_length(pts1) <= _polyline_length(pts2) else pts2


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    dot = float(np.dot(u, v) / (nu * nv))
    dot = max(-1.0, min(1.0, dot))
    return float(math.acos(dot))


def _max_corner_angle(pts: np.ndarray) -> float:
    """Maximum turning angle between consecutive segments along a polyline."""
    if len(pts) < 3:
        return 0.0
    worst = 0.0
    for i in range(1, len(pts) - 1):
        v_in = pts[i] - pts[i - 1]
        v_out = pts[i + 1] - pts[i]
        worst = max(worst, _angle_between(v_in, v_out))
    return worst


def _clean_consecutive_duplicates(pts: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > tol:
            out.append(p)
    return np.asarray(out)


def _filter_tangents(
    tangents,
    *,
    crosses_required: bool,
    c0: np.ndarray,
    c1: np.ndarray,
    right_side_eps: float = 1e-12,
):
    keep = []
    for p0, p1, th0, th1 in tangents:
        if p0[0] < -right_side_eps or p1[0] < -right_side_eps:
            continue
        cr = _line_crosses_center_segment(p0, p1, c0, c1)
        if cr is None:
            continue
        if bool(cr) != bool(crosses_required):
            continue
        keep.append((p0, p1, th0, th1))
    return keep


def _polygon_ok_for_rules(poly: Polygon, ellipses, *, eps: float = 1e-9) -> bool:
    """Validate inside/outside rules using shrunken-ellipse interiors."""
    if not poly.is_valid:
        return False

    # e1,e2,e3 interior must be inside
    for idx in (0, 1, 2):
        c, a, b = ellipses[idx]
        epi = _ellipse_polygon((c, a, b))
        shrink = max(1e-12, min(a, b) * 1e-3)
        inner = epi.buffer(-shrink)
        if inner.is_empty:
            inner = epi
        if not poly.buffer(eps).covers(inner):
            return False

    # ellipse4 interior must be outside
    c, a, b = ellipses[3]
    ep4 = _ellipse_polygon((c, a, b))
    shrink = max(1e-12, min(a, b) * 1e-3)
    inner4 = ep4.buffer(-shrink)
    if inner4.is_empty:
        inner4 = ep4
    if poly.intersects(inner4):
        return False

    return True


def _build_top_candidate(
    e1, e2, e3, e4,
    st: float,
    ed: float,
    t12,
    t23,
    t34,
    t4p,
    *,
    n_arc: int = 90,
):
    """Build top electrode polygon points for a particular tangent choice."""
    # Unpack tangents
    p1_12, p2_12, th1_12, th2_12 = t12
    p2_23, p3_23, th2_23, th3_23 = t23
    p3_34, p4_34, th3_34, th4_34 = t34
    p4_p, th4_p = t4p

    top_point_e1 = _ellipse_point(e1[0], e1[1], e1[2], math.pi / 2.0)
    Ptop0 = np.array([0.0, float(ed)], dtype=float)
    Pbot0 = np.array([0.0, float(st)], dtype=float)

    pts = [Ptop0, top_point_e1]

    # e1: top -> tangent with e2
    arc1 = _arc_points_short(e1, math.pi / 2.0, th1_12, n=n_arc)
    pts.extend(list(arc1[1:]))

    # tangent segment e1->e2
    pts.append(p2_12)

    # e2: tangent with e1 -> tangent with e3
    arc2 = _arc_points_short(e2, th2_12, th2_23, n=n_arc)
    pts.extend(list(arc2[1:]))

    # tangent segment e2->e3
    pts.append(p3_23)

    # e3: tangent with e2 -> tangent with e4
    arc3 = _arc_points_short(e3, th3_23, th3_34, n=n_arc)
    pts.extend(list(arc3[1:]))

    # tangent segment e3->e4
    pts.append(p4_34)

    # e4: tangent with e3 -> tangent with (0,st)
    arc4 = _arc_points_short(e4, th4_34, th4_p, n=n_arc)
    pts.extend(list(arc4[1:]))

    # tangent segment e4->(0,st)
    pts.append(Pbot0)

    right = _clean_consecutive_duplicates(np.asarray(pts, dtype=float))

    # Mirror to form full top electrode polygon
    left = np.column_stack([-right[::-1, 0], right[::-1, 1]])
    full = np.vstack([right, left[1:]])
    full = _clean_consecutive_duplicates(full)
    return full, len(right)


def generate_electrode(params, st, ed):
    """Generate one electrode polygon (top)."""
    e1, e2, e3, e4 = [(np.array(p[0], dtype=float), float(p[1]), float(p[2])) for p in params]

    # Topology constraints
    _check_containment_only((e1[0], e1[1], e1[2]), (e2[0], e2[1], e2[2]), name="e1-e2")
    _check_containment_only((e2[0], e2[1], e2[2]), (e3[0], e3[1], e3[2]), name="e2-e3")
    _check_no_touch_or_overlap((e3[0], e3[1], e3[2]), (e4[0], e4[1], e4[2]), name="e3-e4")

    # Enumerate tangents
    t12_all = _all_ellipse_tangents((e1[0], e1[1], e1[2]), (e2[0], e2[1], e2[2]))
    t23_all = _all_ellipse_tangents((e2[0], e2[1], e2[2]), (e3[0], e3[1], e3[2]))
    t34_all = _all_ellipse_tangents((e3[0], e3[1], e3[2]), (e4[0], e4[1], e4[2]))

    # Apply crossing rule families
    t12 = _filter_tangents(t12_all, crosses_required=False, c0=e1[0], c1=e2[0])
    t23 = _filter_tangents(t23_all, crosses_required=False, c0=e2[0], c1=e3[0])
    t34 = _filter_tangents(t34_all, crosses_required=True, c0=e3[0], c1=e4[0])

    if not t12:
        raise RuntimeError("No valid e1-e2 tangent after filtering")
    if not t23:
        raise RuntimeError("No valid e2-e3 tangent after filtering")
    if not t34:
        raise RuntimeError("No valid e3-e4 tangent after filtering")

    # Point tangents from e4 to bottom center point
    Pbot0 = np.array([0.0, float(st)], dtype=float)
    t4p_all = _all_point_tangents((e4[0], e4[1], e4[2]), Pbot0)
    # keep only right-side tangency points
    t4p = [(p, th) for (p, th) in t4p_all if p[0] >= -1e-12]
    if not t4p:
        raise RuntimeError("No valid point tangent from e4 to (0,st)")

    # Evaluate combinations globally and choose the most smooth, rule-satisfying polygon.
    ellipses = [
        (e1[0], e1[1], e1[2]),
        (e2[0], e2[1], e2[2]),
        (e3[0], e3[1], e3[2]),
        (e4[0], e4[1], e4[2]),
    ]

    best_poly = None
    best_score = None

    for cand12 in t12:
        for cand23 in t23:
            for cand34 in t34:
                for cand4p in t4p:
                    pts, n_right = _build_top_candidate(
                        (e1[0], e1[1], e1[2]),
                        (e2[0], e2[1], e2[2]),
                        (e3[0], e3[1], e3[2]),
                        (e4[0], e4[1], e4[2]),
                        st, ed,
                        cand12, cand23, cand34, cand4p,
                    )
                    poly = Polygon(pts)
                    if not _polygon_ok_for_rules(poly, ellipses):
                        continue
                    # Scoring: (1) smoothness (no sharp corners), (2) monotone-down right boundary, (3) larger area tie-break.
                    smooth = _max_corner_angle(pts)
                    right = pts[:n_right]
                    dy = np.diff(right[:, 1])
                    up_penalty = float(np.sum(np.clip(dy, 0.0, None)))
                    score = (smooth, up_penalty, -float(poly.area))

                    if best_score is None or score < best_score:
                        best_score = score
                        best_poly = poly

    if best_poly is None:
        raise RuntimeError("Could not find a tangent/arc combination satisfying all geometry rules")

    return best_poly

# -----------------------------
# Meshing & Utility
# -----------------------------

def _build_mesh_gmsh(top_poly, bot_poly, cfg):
    h0 = math.sqrt(4.0 * cfg.max_cell_area / math.sqrt(3.0))
    h_fine = cfg.boundary_h_factor * h0

    gmsh.initialize()
    gmsh.model.add("conformal")
    occ = gmsh.model.occ

    # Outer simulation boundary (metal box at phi=0)
    # NOTE: this boundary is treated as a grounded conductor via Dirichlet BC.
    width = float(cfg.xmax - cfg.xmin)
    height = float(cfg.ymax - cfg.ymin)
    if width <= 0 or height <= 0:
        raise ValueError('Invalid bounds: require xmax>xmin and ymax>ymin')
    outer = occ.addRectangle(cfg.xmin, cfg.ymin, 0, width, height)
    
    def add_occ_poly(poly):
        c = np.asarray(poly.exterior.coords)
        p_tags = [occ.addPoint(x, y, 0) for x, y in c[:-1]]
        l_tags = [occ.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]) for i in range(len(p_tags))]
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
    triangles = np.vectorize(tag_to_idx.get)(np.array(gmsh.model.mesh.getElements(2)[2][0], dtype=int).reshape(-1, 3))

    t_mask = np.array([top_poly.boundary.distance(Point(p)) < 1e-7 for p in pts])
    b_mask = np.array([bot_poly.boundary.distance(Point(p)) < 1e-7 for p in pts])
    
    # Grounded outer boundary node mask (xmin/xmax/ymin/ymax)
    # Tolerance chosen to be robust to meshing floating point noise.
    tol = max(1e-10, 1e-8 * max(abs(cfg.xmax - cfg.xmin), abs(cfg.ymax - cfg.ymin)))
    bd_mask = (
        (np.abs(pts[:, 0] - cfg.xmin) <= tol) | (np.abs(pts[:, 0] - cfg.xmax) <= tol) |
        (np.abs(pts[:, 1] - cfg.ymin) <= tol) | (np.abs(pts[:, 1] - cfg.ymax) <= tol)
    )

    gmsh.finalize()
    return pts, triangles, t_mask, b_mask, bd_mask

# -----------------------------
# Visualization
# -----------------------------

def plot_mesh_wireframe(pts, tris, top, bot, cfg, savepath):
    fig, ax = plt.subplots(figsize=(16, 16), dpi=800)
    ax.triplot(pts[:, 0], pts[:, 1], tris, linewidth=0.15, color='royalblue')
    for p, c in [(top, 'red'), (bot, 'green')]:
        x, y = p.exterior.xy
        ax.plot(x, y, color=c, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title("Conformal Gmsh Mesh")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

def plot_field(xi, yi, data, title, label, savepath):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(data, origin="lower", extent=[xi[0], xi[-1], yi[0], yi[-1]], 
                   aspect="equal", interpolation="bicubic", cmap="magma")
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

def plot_phi_contour(xi, yi, phi_grid, title, label, savepath, n_levels=60):
    """Contour plot for potential (phi).

    Uses filled contours so equipotential structure is visible and consistent with MATLAB/Mathematica-style plots.
    """
    Xg, Yg = np.meshgrid(xi, yi)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    cs = ax.contourf(Xg, Yg, phi_grid, levels=n_levels)
    fig.colorbar(cs, ax=ax, label=label)
    # Optional thin contour lines for readability
    ax.contour(Xg, Yg, phi_grid, levels=max(10, n_levels//6), linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_emag_with_vectors(xi, yi, Ex_grid, Ey_grid, air_mask_grid, title, label, savepath, stride=None):
    """Field magnitude density map with E-field direction overlay (VectorDensityPlot-like).

    - Background: |E|
    - Overlay: streamlines from (Ex,Ey)
    """
    Xg, Yg = np.meshgrid(xi, yi)
    Emag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    # Avoid stream artifacts in masked regions
    Exp = np.where(air_mask_grid, Ex_grid, 0.0)
    Eyp = np.where(air_mask_grid, Ey_grid, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(Emag, origin='lower', extent=[xi[0], xi[-1], yi[0], yi[-1]],
                   aspect='equal', interpolation='bicubic', cmap='magma')
    fig.colorbar(im, ax=ax, label=label)

    # Downsample for streamline density control
    if stride is None:
        # Keep roughly <= 60 vectors across the plot in each direction
        stride = max(1, int(len(xi) / 60))

    xs = Xg[::stride, ::stride]
    ys = Yg[::stride, ::stride]
    us = Exp[::stride, ::stride]
    vs = Eyp[::stride, ::stride]

    # Streamplot gives a VectorDensityPlot-like look (direction + density).
    # Use a small linewidth so it doesn't overpower the |E| heatmap.
    ax.streamplot(xs, ys, us, vs, density=1.2, linewidth=0.6, arrowsize=0.8)

    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

# -----------------------------
# Sampling & Outputs
# -----------------------------

def sample_and_outputs(
    xs: np.ndarray,
    ys: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    air_mask: np.ndarray,
    cfg: SolveConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process grid data to find max field and extract a region of interest.
    If cfg.writefile is True, it saves results to a text file.
    
    Returns:
        max_result: [x_pos, y_pos, |E|_max]
        valsSmall: array of [x, y, Ex, Ey] for |x|, |y| <= roi
    """
    Emag_grid = np.sqrt(Ex_grid**2 + Ey_grid**2)
    
    # Apply air mask to ignore points inside metal or outside simulation bounds
    Emag_masked = np.where(air_mask, Emag_grid, 0.0)
    
    # 1. Find Maximum Result
    flat_idx = np.argmax(Emag_masked)
    iy, ix = np.unravel_index(flat_idx, Emag_masked.shape)
    max_result = np.array([xs[ix], ys[iy], Emag_masked[iy, ix]])
    
    # 2. Extract Small Region (ROI)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    roi_mask = (np.abs(X_grid) <= cfg.roi) & (np.abs(Y_grid) <= cfg.roi)
    
    x_roi = X_grid[roi_mask]
    y_roi = Y_grid[roi_mask]
    ex_roi = Ex_grid[roi_mask]
    ey_roi = Ey_grid[roi_mask]
    
    valsSmall = np.column_stack([x_roi, y_roi, ex_roi, ey_roi])

    # 3. File Output Logic
    if cfg.writefile:
        file_path = os.path.join(cfg.outdir, cfg.filename)
        with open(file_path, "w") as f:
            # First line: max_result
            f.write(f"{max_result[0]}, {max_result[1]}, {max_result[2]}\n")
            # Second line: titleline
            f.write("x  y   Ex  Ey\n")
            # Subsequent lines: valsSmall
            np.savetxt(f, valsSmall, delimiter="    ")
    
    return max_result, valsSmall

# -----------------------------
# Main Simulation
# -----------------------------

def _default_geometry_params(cfg: SolveConfig):
    """Built-in legacy geometry defaults.

    Returns:
        params: list of 4 ellipses: [((x0,y0), a, b), ...]
    """
    e1_y0 = 0.042
    e1_x0, e1_a, e1_b = 0.06, 0.003, cfg.ed - e1_y0
    e2_x0, e2_y0, e2_a, e2_b = 0.06, 0.025, 0.005, 0.005
    e3_x0, e3_y0, e3_a, e3_b = 0.042, 0.025, 0.005, 0.005
    e4_x0, e4_y0, e4_a, e4_b = 0.045, 0.032, 0.003, 0.003
    return [
        ((e1_x0, e1_y0), e1_a, e1_b),
        ((e2_x0, e2_y0), e2_a, e2_b),
        ((e3_x0, e3_y0), e3_a, e3_b),
        ((e4_x0, e4_y0), e4_a, e4_b),
    ]


def _load_geometry_params_from_yaml(path: str, cfg: SolveConfig):
    """Load ellipse geometry (e1..e4) from YAML.

    YAML schema (either top-level keys or under `electrode_geometry`):

        electrode_geometry:
          e1: {x0: 0.06, y0: 0.042, a: 0.003}
          e2: {x0: 0.06, y0: 0.028, a: 0.005, b: 0.005}
          e3: {x0: 0.042, y0: 0.028, a: 0.005, b: 0.005}
          e4: {x0: 0.045, y0: 0.032, a: 0.003, b: 0.003}

    Note:
      - `e1_b` is computed, not read: e1_b = cfg.ed - e1_y0
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError(
            "PyYAML is required to load geometry from YAML. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got: {type(data)}")

    g = data.get("electrode_geometry", data)
    if not isinstance(g, dict):
        raise ValueError("`electrode_geometry` must be a mapping/dict")

    def req(node: dict, key: str, name: str) -> float:
        if key not in node:
            raise KeyError(f"Missing key '{key}' for {name} in YAML geometry")
        return float(node[key])

    def node(name: str) -> dict:
        v = g.get(name)
        if not isinstance(v, dict):
            raise KeyError(f"Missing or invalid '{name}' mapping in YAML geometry")
        return v

    e1 = node("e1")
    e2 = node("e2")
    e3 = node("e3")
    e4 = node("e4")

    e1_x0 = req(e1, "x0", "e1")
    e1_y0 = req(e1, "y0", "e1")
    e1_a = req(e1, "a", "e1")
    e1_b = float(cfg.ed - e1_y0)
    if not np.isfinite(e1_b) or e1_b <= 0:
        raise ValueError(
            f"Computed e1_b = cfg.ed - e1_y0 is not positive/finite: {e1_b} (cfg.ed={cfg.ed}, e1_y0={e1_y0})"
        )

    e2_x0 = req(e2, "x0", "e2")
    e2_y0 = req(e2, "y0", "e2")
    e2_a = req(e2, "a", "e2")
    e2_b = float(e2.get("b", e2_a))

    e3_x0 = req(e3, "x0", "e3")
    e3_y0 = req(e3, "y0", "e3")
    e3_a = req(e3, "a", "e3")
    e3_b = float(e3.get("b", e3_a))

    e4_x0 = req(e4, "x0", "e4")
    e4_y0 = req(e4, "y0", "e4")
    e4_a = req(e4, "a", "e4")
    e4_b = float(e4.get("b", e4_a))

    return [
        ((e1_x0, e1_y0), e1_a, e1_b),
        ((e2_x0, e2_y0), e2_a, e2_b),
        ((e3_x0, e3_y0), e3_a, e3_b),
        ((e4_x0, e4_y0), e4_a, e4_b),
    ]

def run_simulation(cfg: SolveConfig = SolveConfig(), geom_yaml: Optional[str] = None, params=None):
    os.makedirs(cfg.outdir, exist_ok=True)
    
    # 1. Geometry
    if params is None:
        if geom_yaml:
            params = _load_geometry_params_from_yaml(str(geom_yaml), cfg)
        else:
            params = _default_geometry_params(cfg)
    
    top = generate_electrode(params, cfg.st, cfg.ed)
    bot = shp_scale(top, 1.0, -1.0, origin=(0, 0))
    metal = unary_union([top, bot])
    
    # 2. Meshing
    pts, tris, t_m, b_m, bd_m = _build_mesh_gmsh(top, bot, cfg)
    
    if cfg.debug_plots:
        plot_mesh_wireframe(pts, tris, top, bot, cfg, os.path.join(cfg.outdir, "mesh_wireframe.png"))

    # 3. FEM Assembly
    n = len(pts)
    K = lil_matrix((n, n))
    p_tri = pts[tris]
    x1, y1 = p_tri[:, 0, 0], p_tri[:, 0, 1]
    x2, y2 = p_tri[:, 1, 0], p_tri[:, 1, 1]
    x3, y3 = p_tri[:, 2, 0], p_tri[:, 2, 1]
    
    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * np.abs(twoA)
    A = np.maximum(A, 1e-30)
    
    b_vec = np.stack([y2-y3, y3-y1, y1-y2], axis=1)
    c_vec = np.stack([x3-x2, x1-x3, x2-x1], axis=1)
    
    for i in range(3):
        for j in range(3):
            vals = (b_vec[:, i]*b_vec[:, j] + c_vec[:, i]*c_vec[:, j]) / (4.0 * A)
            for k in range(len(vals)):
                K[tris[k, i], tris[k, j]] += vals[k]
    K = K.tocsr()

    # 4. Dirichlet Solve
    # Electrode surfaces are fixed at Vt/Vb, and the outer rectangle boundary is grounded (phi=0).
    dir_mask = t_m | b_m | bd_m
    dir_vals = np.zeros(n)  # default: grounded
    # Apply electrode potentials (override if any node happens to coincide with the boundary).
    dir_vals[t_m] = cfg.Vt
    dir_vals[b_m] = cfg.Vb
    
    free, fixed = np.where(~dir_mask)[0], np.where(dir_mask)[0]
    phi = np.zeros(n)
    phi[fixed] = dir_vals[fixed]
    phi[free] = spsolve(K[free][:, free], -K[free][:, fixed] @ phi[fixed])

    # 5. Field Gradients (Nodal)
    ph = phi[tris]
    ex_tri = -(ph[:, 0]*b_vec[:, 0] + ph[:, 1]*b_vec[:, 1] + ph[:, 2]*b_vec[:, 2]) / (2.0 * A)
    ey_tri = -(ph[:, 0]*c_vec[:, 0] + ph[:, 1]*c_vec[:, 1] + ph[:, 2]*c_vec[:, 2]) / (2.0 * A)
    
    Exn, Eyn, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(3):
        np.add.at(Exn, tris[:, i], ex_tri)
        np.add.at(Eyn, tris[:, i], ey_tri)
        np.add.at(cnt, tris[:, i], 1.0)
    Exn, Eyn = Exn/np.maximum(cnt, 1), Eyn/np.maximum(cnt, 1)

    # 6. Grid Interpolation
    xi, yi = np.linspace(cfg.xmin, cfg.xmax, cfg.n_sample), np.linspace(cfg.ymin, cfg.ymax, cfg.n_sample)
    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    Xg, Yg = np.meshgrid(xi, yi)
    xf, yf = Xg.ravel(), Yg.ravel()
    
    t_id = tri_obj.get_trifinder()(xf, yf)
    air_mask_flat = (t_id >= 0) & (~shapely.contains_xy(metal, xf, yf))
    
    Ex_flat, Ey_flat, phi_flat = np.zeros_like(xf), np.zeros_like(xf), np.zeros_like(xf)
    if np.any(air_mask_flat):
        phi_flat[air_mask_flat] = phi[tris[t_id[air_mask_flat]]].mean(axis=1)
        Ex_flat[air_mask_flat] = Exn[tris[t_id[air_mask_flat]]].mean(axis=1)
        Ey_flat[air_mask_flat] = Eyn[tris[t_id[air_mask_flat]]].mean(axis=1)

    # 7. Outputs & Processing
    Ex_grid = Ex_flat.reshape(cfg.n_sample, cfg.n_sample)
    Ey_grid = Ey_flat.reshape(cfg.n_sample, cfg.n_sample)
    phi_grid = phi_flat.reshape(cfg.n_sample, cfg.n_sample)
    air_mask_grid = air_mask_flat.reshape(cfg.n_sample, cfg.n_sample)

    max_result, vals_small = sample_and_outputs(xi, yi, Ex_grid, Ey_grid, air_mask_grid, cfg)

    if cfg.debug_plots:
        plot_phi_contour(xi, yi, phi_grid, "Electric Potential (phi)", "Potential [V]", os.path.join(cfg.outdir, "phi_map.png"))
        plot_emag_with_vectors(xi, yi, Ex_grid, Ey_grid, air_mask_grid, "Field Magnitude |E| with E-direction", "|E| [V/m]", os.path.join(cfg.outdir, "emag_density.png"))

    print(f"Simulation Finished. Max |E| = {max_result[2]:.2e} V/m at {max_result[0:2]}")
    return max_result, vals_small

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run FEM field simulation for electrode geometry")
    ap.add_argument(
        "--geom-yaml",
        "-g",
        default=None,
        help="YAML file providing e1..e4 geometry. e1_b is computed as cfg.ed - e1_y0.",
    )
    args = ap.parse_args()

    res, small = run_simulation(SolveConfig(), geom_yaml=args.geom_yaml)