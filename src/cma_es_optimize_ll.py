#!/usr/bin/env python3
"""
CMA-ES driver (Rejection Sampling + Multi-core + Convergence Plot)

======================================================================
HOW TO RUN
----------------------------------------------------------------------

python cma_es_optimize.py \
    --template electrode_geometry.yaml \
    --analysis electrodes_field_analysis.py \
    --outdir opt_runs \
    --budget 500 \
    --nproc 8

ALL ARGUMENTS ARE OPTIONAL.

If omitted, the following defaults are used:

--template   : electrode_geometry.yaml
--analysis   : electrodes_field_analysis.py
--outdir     : opt_runs
--budget     : 200          (total FEM evaluations)
--seed       : 0
--python     : current Python interpreter
--nproc      : 1            (serial mode)

======================================================================
OUTPUT STRUCTURE
----------------------------------------------------------------------

PROJECT_ROOT/outputs/<outdir>_<YYYYMMDD_HHMMSS>/

Inside that folder:

runs/                               -> all evaluated yaml/out files
convergence.png                     -> best objective vs evaluation
genXXXX_evalYYYYYY_best.yaml        -> best geometry (copied to root)
genXXXX_evalYYYYYY_best.out.txt     -> best output  (copied to root)

======================================================================
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import shutil
import yaml
import multiprocessing as mp
import matplotlib.pyplot as plt
import cma


# ==============================
# USER CONFIG
# ==============================

PROJECT_ROOT = Path("/Users/wange/Coding/Python/fieldanalysis")


# ==============================
# Parameter Definition
# ==============================

VAR_ORDER = [
    ("e1","x0"),("e1","y0"),("e1","a"),
    ("e2","x0"),("e2","y0"),("e2","a"),("e2","b"),
    ("e3","x0"),("e3","y0"),("e3","a"),("e3","b"),
    ("e4","x0"),("e4","y0"),("e4","a"),("e4","b"),
]

@dataclass
class Bounds:
    lo: float
    hi: float


# ==============================
# YAML Helpers
# ==============================

def _load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)

def _dump_yaml(obj, path: Path):
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def _get_geom_root(doc):
    if "electrode_geometry" in doc:
        return doc["electrode_geometry"]
    return doc

def extract_x0(template_doc):
    geom = _get_geom_root(template_doc)
    return [float(geom[e][k]) for e,k in VAR_ORDER]

def apply_vector_to_doc(template_doc, x):
    doc = yaml.safe_load(yaml.safe_dump(template_doc))
    geom = _get_geom_root(doc)
    for (e,k),v in zip(VAR_ORDER,x):
        geom[e][k] = float(v)
    return doc


# ==============================
# Constraints
# ==============================

REL_CONSTRAINTS = [
    ("e2.y0 < e1.y0", lambda p: p("e2","y0") < p("e1","y0")),
    ("e3.x0 < e2.x0", lambda p: p("e3","x0") < p("e2","x0")),
    ("e1.x0 + e1.a < 0.072", lambda p: (p("e1","x0")+p("e1","a")) < 0.072),
    ("e2.x0 + e2.a < 0.072", lambda p: (p("e2","x0")+p("e2","a")) < 0.072),
    ("e2.y0 - e2.b > 0.02", lambda p: (p("e2","y0")-p("e2","b")) > 0.02),
    ("e3.y0 - e3.b > 0.02", lambda p: (p("e3","y0")-p("e3","b")) > 0.02),
]

def check_constraints(x):
    pmap = {(e,k):float(v) for (e,k),v in zip(VAR_ORDER,x)}
    def p(e,k): return pmap[(e,k)]
    for _,fn in REL_CONSTRAINTS:
        if not fn(p):
            return False
    return True


# ==============================
# Bounds
# ==============================

def default_bounds_from_template(x0):
    bnds=[]
    for (e,k),v in zip(VAR_ORDER,x0):
        if k in ("x0","y0"):
            lo=v-0.012; hi=v+0.012
        else:
            lo=max(0.0005,v/4); hi=min(0.02,v*2)
        bnds.append(Bounds(lo,hi))
    return bnds

def clip_to_bounds(x,bnds):
    return [min(max(xi,b.lo),b.hi) for xi,b in zip(x,bnds)]


# ==============================
# Output Parsing
# ==============================

_SUM_RE=re.compile(r"Sum_\{n>2\}.*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_RAD_RE=re.compile(r"Uniformity radius.*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def run_one(args):
    cand,template_doc,analysis_path,python_exe,yaml_path,out_path=args
    try:
        doc=apply_vector_to_doc(template_doc,cand)
        _dump_yaml(doc,yaml_path)

        cmd=[python_exe,str(analysis_path),str(yaml_path)]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        out=proc.stdout

        m=_SUM_RE.search(out)
        if not m:
            raise RuntimeError("Parse error")

        sum_high=float(m.group(1))
        r=_RAD_RE.search(out)
        radius=float(r.group(1)) if r else None

        out_path.write_text(out)
        return sum_high,radius

    except Exception as ex:
        out_path.write_text(str(ex))
        return 1e9,None


# ==============================
# Main
# ==============================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--template",default="electrode_geometry.yaml")
    ap.add_argument("--analysis",default="electrodes_field_analysis.py")
    ap.add_argument("--outdir",default="opt_runs")
    ap.add_argument("--budget",type=int,default=200)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--python",dest="python_exe",default=sys.executable)
    ap.add_argument("--nproc",type=int,default=1)
    args=ap.parse_args()

    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root=PROJECT_ROOT/"outputs"/f"{args.outdir}_{timestamp}"
    runs_dir=run_root/"runs"
    runs_dir.mkdir(parents=True,exist_ok=True)

    template_doc=_load_yaml(Path(args.template))
    x0=extract_x0(template_doc)
    bnds=default_bounds_from_template(x0)

    es=cma.CMAEvolutionStrategy(
        x0,0.1,
        {"seed":args.seed,
         "bounds":[[b.lo for b in bnds],[b.hi for b in bnds]],
         "verbose":-9}
    )

    eval_id=0; gen=0
    best_sum=float("inf")
    best_yaml=None; best_out=None
    best_history=[]

    pool=mp.Pool(args.nproc) if args.nproc>1 else None

    while not es.stop() and eval_id<args.budget:
        gen+=1
        feasible=[]

        while len(feasible)<es.popsize and eval_id<args.budget:
            cand=es.ask(1)[0]
            cand=clip_to_bounds(list(map(float,cand)),bnds)
            if check_constraints(cand):
                feasible.append(cand)

        tasks=[]
        for cand in feasible:
            eval_id+=1
            yaml_name=f"gen{gen:04d}_eval{eval_id:06d}.yaml"
            yaml_path=runs_dir/yaml_name
            out_path=runs_dir/yaml_name.replace(".yaml",".out.txt")
            tasks.append((cand,template_doc,Path(args.analysis),
                          args.python_exe,yaml_path,out_path))

        results=pool.map(run_one,tasks) if pool else [run_one(t) for t in tasks]

        values=[]
        for (sum_high,_),task in zip(results,tasks):
            values.append(sum_high)
            if sum_high<best_sum:
                best_sum=sum_high
                best_yaml=task[4]
                best_out=task[5]
            best_history.append(best_sum)

        es.tell(feasible,values)
        print(f"Generation {gen} done. Best so far: {best_sum:.4e}")

    if pool:
        pool.close(); pool.join()

    if best_yaml:
        shutil.copy(best_yaml, run_root/(best_yaml.stem+"_best.yaml"))
        shutil.copy(best_out, run_root/(best_out.stem+"_best.out.txt"))

    if best_history:
        plt.figure()
        plt.plot(best_history)
        plt.xlabel("Evaluation")
        plt.ylabel("Best Objective")
        plt.title("CMA-ES Convergence")
        plt.tight_layout()
        plt.savefig(run_root/"convergence.png")
        plt.close()

    print("Done.")
    print("Results folder:",run_root)


if __name__=="__main__":
    main()