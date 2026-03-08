from electrodes_fast import SolveConfig, run_and_plot_emag

cfg = SolveConfig(
    max_cell_measure=1e-5,  # same as your Mathematica MaxCellMeasure -> 2*10^-5
    n_sample=160,           # same ROI sampling density
    xmin=-0.10, xmax=0.10,
    ymin=-0.10, ymax=0.10,
    prefer_gmsh=True
)

maxResult, valsSmall, (xi, yi, emag2d) = run_and_plot_emag(
    cfg=cfg,
    savepath="emag_tri.png",
    show=True,
)

print("maxResult =", maxResult)
print("len(valsSmall) =", len(valsSmall))