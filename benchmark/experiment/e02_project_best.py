"""e02 (simulated Franka+SoftHand) post-processing: a-posteriori feasibility check + per-link LMI projection of the best run of every case
(as solve_dynamics in the pipeline), re-evaluation of the projected solution, and the LaTeX table for the paper.

Reads  results/e02_multistart_softhand/results.json
Writes results/e02_multistart_softhand/projected.json, table_multistart_softhand.tex
"""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.setup_env import RESULTS
from bench.data import load_config, load_trajectory
from bench.pipeline import load_robot
from bench.ident import build_regressor, wls, rmse_stats, dyn2reg_np, feasibility_margins
from utils.identification_utils import LMI_projection

OUT = os.path.join(RESULTS, os.environ.get("E12_OUT", "e02_multistart_softhand"))
FAMS = ("Fourier", "Chirp", "Sinusoidal")
BAGS = {f: f"./rosbag/softhand_lean_{f.lower()}_CT" for f in FAMS}
TOL = -1e-6                                                         # same tolerance as e02_multistart_softhand

cfg = load_config("src/examples/franka_sim_trajectories/config/fourier_config.yaml")
r = load_robot("franka", "src/thunder/franka_generatedFiles/param/franka_par.yaml")
n = r.numJoints
beta = np.array(r.get_beta()); nb = beta.shape[0]
gt = np.array(load_robot("franka", "src/thunder/franka_generatedFiles/param/franka_SH_lean_par.yaml")
              .get_par_DYN()).ravel()
gt_bp = beta @ dyn2reg_np(gt)
data = {f: build_regressor(r, load_trajectory(cfg, f"e12p_{f}", interval=[5.0, 19.7], bag=BAGS[f])) for f in FAMS}
fric = {f: wls(Y, T, n)["pi"].ravel()[nb:] for f, (Y, T) in data.items()}   # WLS friction, as in e02_multistart_softhand


def evaluate(dyn, fam):
    pim = np.concatenate([beta @ dyn2reg_np(dyn), fric[fam]])
    ho = float(np.mean([rmse_stats(T.ravel(), Y @ pim, n)["rmse"] for f, (Y, T) in data.items() if f != fam]))
    d, g = dyn[60:70], gt[60:70]
    return dict(heldout_rmse=ho, link7_mass=float(d[0]), link7_dm=float(abs(d[0] - g[0])),
                link7_dc=float(np.linalg.norm(d[1:4] - g[1:4])),
                bp_err=float(np.linalg.norm(beta @ dyn2reg_np(dyn) - gt_bp) / np.linalg.norm(gt_bp)))


res = json.load(open(os.path.join(OUT, "results.json")))
proj = json.load(open(os.path.join(OUT, "projected.json"))) if "--table-only" in sys.argv else {}
for cname, c in ([] if proj else res.items()):
    ok = [x for x in c["runs"] if x["success"]]
    best = np.array(min(ok, key=lambda x: x["cost"])["dyn"], float)
    bad = [i for i, f in enumerate(feasibility_margins(best)) if not (f["mass"] > 0 and f["min_eig_sigma"] > TOL)]
    p = np.asarray(LMI_projection(r, par_DYN=best, links=bad), float).ravel() if bad else best.copy()
    fm = feasibility_margins(p)
    proj[cname] = dict(projected_links=bad, min_margin_before=min(f["min_eig_sigma"] for f in feasibility_margins(best)),
                       min_margin_after=min(f["min_eig_sigma"] for f in fm),
                       max_rel_change=float(np.max(np.abs(p - best) / np.maximum(np.abs(best), 1e-9))),
                       before=evaluate(best, c["family"]), after=evaluate(p, c["family"]))
    a, b = proj[cname]["after"], proj[cname]["before"]
    print(f"{cname:26s} links {bad}  margin {proj[cname]['min_margin_before']:.1e} -> {proj[cname]['min_margin_after']:.1e}"
          f"  held-out {b['heldout_rmse']:.4f} -> {a['heldout_rmse']:.4f}  m7 {b['link7_mass']:.3f} -> {a['link7_mass']:.3f}"
          f"  BP {b['bp_err']:.4f} -> {a['bp_err']:.4f}", flush=True)
if "--table-only" not in sys.argv:
    json.dump(proj, open(os.path.join(OUT, "projected.json"), "w"), indent=1)


# LaTeX table: one row per prior (scale k, mean), multistart spread pooled over the three trajectories,
# accuracy of the projected best solution per trajectory (Fourier / Chirp / Sinusoidal).
def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(abs(m), 1) >= 10:
        m /= 10; e += 1
    return f"${m:.1f}\\cdot10^{{{e}}}$"


rows = []
for mean, mlabel in (("none", "CAD"), ("declared", "Pert.")):
    for k in (0.1, 1, 10):
        cs = [res[f"{f}_x{k:g}_{mean}"] for f in FAMS]
        ps = [proj[f"{f}_x{k:g}_{mean}"]["after"] for f in FAMS]
        conv = sum(c["summary"]["n_success"] for c in cs); tot = sum(c["summary"]["n_starts"] for c in cs)
        gap = max(c["summary"]["rel_cost_quantiles"][2] for c in cs)
        hspread = max(c["summary"]["heldout_rmse_min_median_max"][2] - c["summary"]["heldout_rmse_min_median_max"][0]
                      for c in cs)
        pa = cs[0]["prior_alone"]
        m7 = [p["link7_mass"] for p in ps]
        rows.append((f"\\multirow{{3}}{{*}}{{{mlabel}}}" if k == 0.1 else "") + f" & {k:g} & {sci(gap)} & {sci(hspread)} & "
                    f"{pa['heldout_rmse']:.2f} / {pa['bp_err']:.2f} & "
                    + " / ".join(f"{p['heldout_rmse']:.2f}" for p in ps) + " & "
                    + " / ".join(f"{p['bp_err']:.3f}" for p in ps) + " & "
                    + " / ".join(f"{x:.2f}" for x in m7) + " \\\\")
    rows.append("\\midrule" if mean == "none" else "")
tex = r"""\begin{table*}[t]
\caption{Multi-start MLE on the three simulated Franka+SoftHand recordings (F/C/S: Fourier, chirp, sinusoidal) for six priors: Case~A std.\ dev.\ scaled by $k$, mean equal to the CAD values or perturbed (Pert.) by one draw $\sigma_k\xi$. 101 starts per problem, all converged. Cost gap: max.\ relative gap to the best run; RMSE range: max.\ spread of the held-out RMSE over the starts (both over the three trajectories). Held-out RMSE [N\,m] on the two other recordings; BP err.: $\|\hat{\pi}_b-\pi_b\|/\|\pi_b\|$; $m_7$ [kg]: last-link mass (ground truth 1.52).}
\label{tab:multistart_softhand}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
Mean & $k$ & Cost gap & RMSE range & Prior only: RMSE / BP err. & Held-out RMSE (F/C/S) & BP err.\ (F/C/S) & $m_7$ (F/C/S)\\
\midrule
""" + "\n".join(x for x in rows if x) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(OUT, "table_multistart_softhand.tex"), "w").write(tex)
print(tex)
