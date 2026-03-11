#!/usr/bin/env python3
"""
CMA-ES driver (Rejection Sampling version)

Improvements:
1) Results saved to:
   PROJECT_ROOT/outputs/<outdir>_<date_time>/

2) After run, best yaml + out copied to:
   genXXXX_evalYYYYYY_best.yaml
   genXXXX_evalYYYYYY_best.out.txt

3) Summary file:
   summary_<date_time>.dat
   (tab-separated, one line per evaluation)
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import shutil
import yaml

try:
    import cma
except ImportError as e:
    raise SystemExit("Missing dependency 'cma'. Install with: pip install cma\n" + str(e))


PROJECT_ROOT = Path("/Users/wange/Coding/Python/fieldanalysis")


VAR_ORDER = [
    ("e1", "x0"), ("e1", "y0"), ("e1", "a"),
    ("e2", "x0"), ("e2", "y0"), ("e2", "a"), ("e2", "b"),
    ("e3", "x0"), ("e3", "y0"), ("e3", "a"), ("e3", "b"),
    ("e4", "x0"), ("e4", "y0"), ("e4", "a"), ("e4", "b"),
]


@dataclass
class Bounds:
    lo: float
    hi: float


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _get_geom_root(doc: Dict[str, Any]) -> Dict[str, Any]:
    if "electrode_geometry" in doc and isinstance(doc["electrode_geometry"], dict):
        return doc["electrode_geometry"]
    return doc


def extract_x0(template_doc: Dict[str, Any]) -> List[float]:
    geom = _get_geom_root(template_doc)
    return [float(geom[e][k]) for e, k in VAR_ORDER]


def apply_vector_to_doc(template_doc: Dict[str, Any], x: List[float]) -> Dict[str, Any]:
    doc = yaml.safe_load(yaml.safe_dump(template_doc))
    geom = _get_geom_root(doc)
    for (e, k), val in zip(VAR_ORDER, x):
        geom[e][k] = float(val)
    return doc


REL_CONSTRAINTS = [
    ("e2.y0 < e1.y0", lambda p: p("e2", "y0") < p("e1", "y0")),
    ("e3.x0 < e2.x0", lambda p: p("e3", "x0") < p("e2", "x0")),
    ("e1.x0 + e1.a < 0.072", lambda p: (p("e1", "x0") + p("e1", "a")) < 0.072),
    ("e2.x0 + e2.a < 0.072", lambda p: (p("e2", "x0") + p("e2", "a")) < 0.072),
    ("e2.y0 - e2.b > 0.02", lambda p: (p("e2", "y0") - p("e2", "b")) > 0.02),
    ("e3.y0 - e3.b > 0.02", lambda p: (p("e3", "y0") - p("e3", "b")) > 0.02),
]


def check_constraints(x: List[float]) -> bool:
    pmap = {(e, k): float(v) for (e, k), v in zip(VAR_ORDER, x)}
    def p(e, k): return pmap[(e, k)]
    for _, fn in REL_CONSTRAINTS:
        if not fn(p):
            return False
    return True


def default_bounds_from_template(x0: List[float]) -> List[Bounds]:
    bnds = []
    for (e, k), v in zip(VAR_ORDER, x0):
        if k in ("x0", "y0"):
            lo = v - 0.012
            hi = v + 0.012
        else:
            lo = max(0.0005, v / 3)
            hi = min(0.02, v * 3)
        bnds.append(Bounds(lo, hi))
    return bnds


def clip_to_bounds(x: List[float], bnds: List[Bounds]) -> List[float]:
    return [min(max(xi, b.lo), b.hi) for xi, b in zip(x, bnds)]


_SUM_RE = re.compile(r"Sum_\{n>2\}.*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_RAD_RE = re.compile(r"Uniformity radius.*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def run_analysis_subprocess(analysis_py: Path, geom_yaml: Path, python_exe: str):
    cmd = [python_exe, str(analysis_py), str(geom_yaml)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    msum = _SUM_RE.search(out)
    if not msum:
        raise RuntimeError("Could not parse sum from output.\n" + out)

    sum_high = float(msum.group(1))
    mrad = _RAD_RE.search(out)
    radius = float(mrad.group(1)) if mrad else None

    return sum_high, radius, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="electrode_geometry.yaml")
    ap.add_argument("--analysis", default="electrodes_field_analysis.py")
    ap.add_argument("--outdir", default="opt_runs")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PROJECT_ROOT / "outputs" / f"{args.outdir}_{timestamp}"
    runs_dir = run_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / f"summary_{timestamp}.dat"

    template_doc = _load_yaml(Path(args.template))
    x0 = extract_x0(template_doc)
    bnds = default_bounds_from_template(x0)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.1,
        {"seed": args.seed,
         "bounds": [[b.lo for b in bnds], [b.hi for b in bnds]],
         "verbose": -9}
    )

    eval_id = 0
    gen = 0
    best_sum = float("inf")
    best_yaml = None
    best_out = None

    header = ["gen","eval","sum","radius","yaml"] + [f"{e}.{k}" for e,k in VAR_ORDER]
    summary_path.write_text("\t".join(header) + "\n")

    while not es.stop() and eval_id < args.budget:
        gen += 1
        feasible_solutions = []
        values = []

        while len(feasible_solutions) < es.popsize and eval_id < args.budget:
            cand = es.ask(1)[0]
            cand = clip_to_bounds(list(map(float, cand)), bnds)
            if not check_constraints(cand):
                continue

            feasible_solutions.append(cand)
            eval_id += 1

            yaml_name = f"gen{gen:04d}_eval{eval_id:06d}.yaml"
            yaml_path = runs_dir / yaml_name
            out_path = runs_dir / yaml_name.replace(".yaml",".out.txt")

            try:
                doc = apply_vector_to_doc(template_doc, cand)
                _dump_yaml(doc, yaml_path)
                sum_high, radius, full_out = run_analysis_subprocess(
                    Path(args.analysis), yaml_path, args.python_exe)
                out_path.write_text(full_out)

                if sum_high < best_sum:
                    best_sum = sum_high
                    best_yaml = yaml_path
                    best_out = out_path

            except Exception as ex:
                sum_high = 1e9
                radius = None
                out_path.write_text(str(ex))

            values.append(sum_high)

            row = [str(gen), str(eval_id), f"{sum_high:.12e}",
                   "" if radius is None else f"{radius:.12e}",
                   yaml_name] + [f"{v:.12e}" for v in cand]

            with summary_path.open("a") as f:
                f.write("\t".join(row) + "\n")

            print(f"gen={gen} eval={eval_id} sum={sum_high:.4e}")

        if feasible_solutions:
            es.tell(feasible_solutions, values)

    if best_yaml is not None:
        shutil.copy(best_yaml, runs_dir / (best_yaml.stem + "_best.yaml"))
        shutil.copy(best_out, runs_dir / (best_out.stem + "_best.out.txt"))

    print("Done.")
    print("Results folder:", run_root)


if __name__ == "__main__":
    main()
