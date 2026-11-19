"""
E02 table - multi-start robustness of the proposed MAP step (IPOPT), compacted in one LaTeX table.

Reads results/e02_multistart/results.json (written by e02_multistart.py, which runs this script automatically). For every case and every converged run k
(IPOPT success with the exact constraint, or with the approximate one of the fallback: a run fails only if both
fail) it reports:
  * converged runs / starts, and in parentheses how many of them converged with the fallback
  * relative cost gap     g_k = (J_k - J_best) / J_best
  * relative parameter deviation from the best MAP solution, per link j and parameter group, as computed by
    e02_multistart.rel_deviation (summary["rel_dev"] in results.json):
        mass     |m_k,j - m_best,j| / m_best,j
        CoM      ||c_k,j - c_best,j|| / ||c_best,j||
        inertia  ||I_k,j - I_best,j||_F / ||I_best,j||_F
    (group norms, not single components: CoM coordinates and products of inertia can be ~0)
and reports mean +- std. dev. over runs and links. J_best, theta_best: run with the lowest MAP cost.
Output: results/e02_multistart/table_multistart.tex (table only, to \\input) and table_multistart.pdf (standalone).
"""
import json, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e02_multistart")
ROOT = os.path.join(HERE, "..", "..")
res = json.load(open(os.path.join(OUT, "results.json")))

# Case key in results.json and its label in the table
CASES = [("real_full", "Real, full window"), ("real_foldA", "Real, fold A"), ("real_foldB", "Real, fold B")]
GROUPS = ["mass", "com", "inertia"]   # keys of summary["rel_dev"] in results.json


def sci(x):
    """Number in LaTeX scientific notation, e.g. 3.1\\cdot10^{-11}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if abs(round(x / 10 ** e, 1)) >= 10:   # mantissa rounds up to 10.0 -> next exponent
        e += 1
    return f"{x / 10 ** e:.1f}\\cdot10^{{{e}}}" if abs(e) > 2 else f"{x:.3g}"


def pm(v):
    """mean +- std. dev. of an array, in math mode."""
    return f"${sci(np.mean(v))} \\pm {sci(np.std(v))}$"


rows = []
for key, label in CASES:
    runs = res[key]["runs"]                                     # all run for dataset 'key'
    ok = [r for r in runs if r["success"]]                      # converged runs (exact or fallback constraint)
    costs = np.array([r["cost"] for r in ok])                   # all optim. costs of the converged runs
    best = int(np.argmin(costs))                                # best run idx, i.e. the one with smallest optim. cost
    gap = (costs - costs[best]) / costs[best]                   # normalized cost gap across all converged runs

    n_fb = sum(r["fallback"] for r in ok)                       # of which converged with the fallback

    # Create table entry
    #       dataset     num of converged (fallback)       cost gap
    cells = [label, f"{len(ok)}/{len(runs)} ({n_fb})", pm(gap)]

    # Relative deviation from the best MAP solution, per run r and link l (computed in e02_multistart.rel_deviation):
    #   mass |m_rl - m_l,best| / m_l,best,  CoM ||c_rl - c_l,best||_2 / ||c_l,best||_2,
    #   inertia ||I_rl - I_l,best||_F / ||I_l,best||_F;  mean +- std. dev. over (r, l)
    for g in GROUPS:
        d = res[key]["summary"]["rel_dev"][g]
        cells.append(f"${sci(d['mean'])} \\pm {sci(d['std'])}$")
    h = [r["heldout_rmse"] for r in ok if "heldout_rmse" in r]
    cells.append(f"${np.mean(h):.3f} \\pm {sci(np.std(h))}$" if h else "--")
    rows.append(" & ".join(cells) + r" \\")

table = r"""\begin{table}[t]
\caption{Multi-start robustness of the proposed MLE step (IPOPT, 101 starts per case: prior, 50 random physically
consistent bodies, 50 Gaussian perturbations of the prior, $\mathcal{N}(0,\sigma^2)$ with $\sigma$ the prior std.\ dev.). Converged: IPOPT success with the exact eigenvalue constraint
or, where it fails, with the fallback on 2000 sampled directions (in parentheses, number of fallback runs). Mean $\pm$ std.\ dev.\ over the converged runs of the relative
cost gap to the best run and of the parameter deviation relative to the best MLE solution (per link: mass,
CoM norm, Frobenius norm of the inertia), over all links.}
\label{tab:multistart}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Case & Conv. & Cost gap (J)& $|\Delta m|/m^\star$ & $\|\Delta c\|/\|c^\star\|$ & $\|\Delta I\|_F/\|I^\star\|_F$ & Held-out RMSE [N\,m]\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(OUT, "table_multistart.tex"), "w").write(table)

# Standalone document (single column) -> PDF
doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{booktabs}
\pagestyle{empty}
\begin{document}
\input{table_multistart.tex}
\end{document}
"""
open(os.path.join(OUT, "table_multistart_standalone.tex"), "w").write(doc)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "table_multistart_standalone.tex"], cwd=OUT,
               stdout=subprocess.DEVNULL, check=True)
os.replace(os.path.join(OUT, "table_multistart_standalone.pdf"), os.path.join(OUT, "table_multistart.pdf"))
for ext in ("aux", "log"):
    os.remove(os.path.join(OUT, f"table_multistart_standalone.{ext}"))
print(table)
