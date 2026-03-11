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
    roi: float = 0.03
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

def _ellipse_point(c, a, b, th):
    return np.array([c[0] + a * math.cos(th), c[1] + b * math.sin(th)], dtype=float)

def _ellipse_tangent(a, b, th):
    return np.array([-a * math.sin(th), b * math.cos(th)], dtype=float)

def _solve_ellipse_tangent(e1, e2, guess):
    c1, a1, b1 = e1
    c2, a2, b2 = e2
    def fun(v):
        p1, p2 = _ellipse_point(c1, a1, b1, v[0]), _ellipse_point(c2, a2, b2, v[1])
        t1, t2 = _ellipse_tangent(a1, b1, v[0]), _ellipse_tangent(a2, b2, v[1])
        return np.array([np.linalg.det(np.c_[t1, t2]), np.linalg.det(np.c_[p2 - p1, t1])], dtype=float)
    sol = root(fun, np.array(guess, dtype=float), method="hybr")
    if not sol.success: return None
    return (_ellipse_point(c1, a1, b1, sol.x[0]), _ellipse_point(c2, a2, b2, sol.x[1]), sol.x[0], sol.x[1])

def _find_best_ellipse_tangent(e1, e2, select_mode="outer"):
    guesses = np.linspace(-math.pi, math.pi, 9)
    sols = []
    for g1 in guesses:
        for g2 in guesses:
            out = _solve_ellipse_tangent(e1, e2, (g1, g2))
            if out and out[0][0] > 0: sols.append(out)
    if not sols: raise RuntimeError(f"Tangent not found: {select_mode}")
    sols.sort(key=lambda s: s[0][0] if select_mode == "inner_e2" else (s[0][0] + s[1][0]), reverse=(select_mode != "inner_e2"))
    return sols[0]

def _arc_points(e, th_start, th_end, n=40):
    s, ee = float(th_start), float(th_end)
    diff = (ee - s + np.pi) % (2 * np.pi) - np.pi
    ts = np.linspace(s, s + diff, n + 1)
    return np.column_stack([e[0][0] + e[1] * np.cos(ts), e[0][1] + e[2] * np.sin(ts)])

def generate_electrode(params, st, ed):
    e1, e2, e3, e4 = [(np.array(p[0]), p[1], p[2]) for p in params]
    t12 = _find_best_ellipse_tangent(e1, e2, "outer")
    t23 = _find_best_ellipse_tangent(e2, e3, "inner_e2") 
    t34 = _find_best_ellipse_tangent(e3, e4, "inner_e2") 
    
    right_list = [np.array([0, ed]), np.array([e1[0][0], ed])]
    right_list.extend(_arc_points(e1, math.pi/2, t12[2])[1:])
    right_list.extend(_arc_points(e2, t12[3], t23[2])[1:])
    right_list.extend(_arc_points(e3, t23[3], t34[2])[1:])
    
    p4E_sol = root(lambda v: np.linalg.det(np.c_[_ellipse_point(e4[0], e4[1], e4[2], v[0]) - [0, st], _ellipse_tangent(e4[1], e4[2], v[0])]), [0])
    right_list.extend(_arc_points(e4, t34[3], p4E_sol.x[0])[1:])
    right_list.append(np.array([0, st]))
    
    right = np.array(right_list)
    mirror_x = -right[::-1, 0]
    mirror_y = right[::-1, 1]
    left = np.column_stack([mirror_x, mirror_y])
    
    full_pts = np.vstack([right, left[1:]])
    cleaned = [full_pts[0]]
    for p in full_pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-9:
            cleaned.append(p)
    return Polygon(cleaned)

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
            f.write("x,y,Ex,Ey\n")
            # Subsequent lines: valsSmall
            np.savetxt(f, valsSmall, delimiter=",")
    
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
    e1_x0, e1_a, e1_b = 0.06, 0.007, cfg.ed - e1_y0
    e2_x0, e2_y0, e2_a, e2_b = 0.06, 0.027, 0.005, 0.005
    e3_x0, e3_y0, e3_a, e3_b = 0.042, 0.025, 0.005, 0.005
    e4_x0, e4_y0, e4_a, e4_b = 0.038, 0.032, 0.003, 0.003
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