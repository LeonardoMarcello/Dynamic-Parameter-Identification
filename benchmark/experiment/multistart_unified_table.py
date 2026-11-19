"""Unified multi-start table: real Franka (e02) + simulated Franka+SoftHand (e02_multistart_softhand) in one LaTeX table (+ PNG).

Reads  results/e02_multistart/results.json, results/e02_multistart_softhand/{results,projected}.json
Writes results/e02_multistart/table_multistart_unified.{tex,png}
Spread statistics are means over the converged runs (sim: pooled over the three trajectories); RMSE range is the
largest max-min of the held-out RMSE over the starts; held-out RMSE and BP error refer to the best (projected) run.
"""
import json
import os
import subprocess
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "e02_multistart")
FAMS = ("Fourier", "Chirp", "Sinusoidal")

real = json.load(open(os.path.join(RES, "e02_multistart", "results.json")))
sim = json.load(open(os.path.join(RES, "e02_multistart_softhand", "results.json")))
proj = json.load(open(os.path.join(RES, "e02_multistart_softhand", "projected.json")))


def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(abs(m), 1) >= 10:
        m /= 10; e += 1
    return f"${m:.1f}\\cdot10^{{{e}}}$"


def spread(summaries):
    """Mean over the pooled converged runs of cost gap and parameter deviations (equal run counts per summary)."""
    rc = lambda s: np.mean([(x["cost"] - s["best_cost"]) / s["best_cost"] for x in s["_ok"]])
    return dict(gap=np.mean([rc(s) for s in summaries]),
                **{g: np.mean([s["rel_dev"][g]["mean"] for s in summaries]) for g in ("mass", "com", "inertia")})


def with_ok(case):
    s = dict(case["summary"]); s["_ok"] = [x for x in case["runs"] if x["success"]]
    return s


rows = []
# Real Franka
for key, label in (("real_full", "Full window"), ("real_foldA", "Fold A"), ("real_foldB", "Fold B")):
    s = with_ok(real[key]); sp = spread([s])
    ok = s["_ok"]; best = min(ok, key=lambda x: x["cost"])
    if "heldout_rmse" in best:
        h = np.array([x["heldout_rmse"] for x in ok]); rng, ho = sci(h.max() - h.min()), f"{best['heldout_rmse']:.2f}"
    else:
        rng, ho = "--", "--"
    rows.append(f" & {label} & {len(ok)}/{s['n_starts']} ({s['n_fallback']}) & {sci(sp['gap'])} & {sci(sp['mass'])} & "
                f"{sci(sp['com'])} & {sci(sp['inertia'])} & {rng} & {ho} & -- \\\\")
rows[0] = "\\multirow{3}{*}{\\shortstack[l]{Real\\\\Franka}}" + rows[0]
rows.append("\\midrule")
# Simulated Franka + SoftHand
for mean, mlabel in (("none", "CAD"), ("declared", "Pert.")):
    for i, k in enumerate((0.1, 1, 10)):
        cs = [sim[f"{f}_x{k:g}_{mean}"] for f in FAMS]
        ss = [with_ok(c) for c in cs]; sp = spread(ss)
        ps = [proj[f"{f}_x{k:g}_{mean}"]["after"] for f in FAMS]
        conv = sum(s["n_success"] for s in ss); tot = sum(s["n_starts"] for s in ss)
        fb = sum(s["n_fallback"] for s in ss)
        hr = max(s["heldout_rmse_min_median_max"][2] - s["heldout_rmse_min_median_max"][0] for s in ss)
        first = f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{Sim.\\\\{mlabel}}}}}" if i == 0 else ""
        rows.append(f"{first} & $k={k:g}$ & {conv}/{tot} ({fb}) & {sci(sp['gap'])} & {sci(sp['mass'])} & "
                    f"{sci(sp['com'])} & {sci(sp['inertia'])} & {sci(hr)} & "
                    + "/".join(f"{p['heldout_rmse']:.2f}" for p in ps) + " & "
                    + "/".join(f"{p['bp_err']:.3f}" for p in ps) + " \\\\")
    if mean == "none":
        rows.append("\\midrule")

caption = (r"Multi-start robustness of the MLE (101 starts per problem). Conv.: converged runs (with the sampled "
           r"constraint). Spread statistics: means over runs and links of the relative deviation from the best run; "
           r"RMSE range: largest spread of the held-out RMSE [N\,m]. Held-out RMSE [N\,m] and BIP error "
           r"$\|\hat{\pi}_b-\pi_b\|/\|\pi_b\|$ of the best run (Sim.: Fourier/chirp/sinusoidal).")
body = (r"""\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llcccccccc}
\toprule
\multicolumn{2}{l}{Case} & Conv. & Cost gap & $|\Delta m|/m^\star$ & $\|\Delta c\|/\|c^\star\|$ & $\|\Delta I\|_F/\|I^\star\|_F$ & \shortstack{RMSE\\range} & \shortstack{Held-out\\RMSE} & BIP err.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
tex = ("\\begin{table*}[t]\n\\caption{" + caption + "}\n\\label{tab:multistart}\n" + body + "\n\\end{table*}\n")
open(os.path.join(OUT, "table_multistart_unified.tex"), "w").write(tex)
print(tex)

# PNG (standalone build)
doc = (r"""\documentclass[border=6pt]{standalone}
\usepackage{times}\usepackage{amsmath}\usepackage{booktabs}\usepackage{multirow}
\begin{document}\begin{minipage}{7.16in}\centering\footnotesize
\textsc{Table}: """ + caption + r"""\par\vspace{6pt}
""" + body + r"""
\end{minipage}\end{document}""")
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "t.tex"), "w").write(doc)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=d, stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", os.path.join(d, "t.pdf"),
                    os.path.join(OUT, "table_multistart_unified")], check=True)
