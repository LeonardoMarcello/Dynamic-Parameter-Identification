"""
Utils: figures of e01_franka_bench.py (which runs this script automatically at the end).
1) Real Franka: torque RMSE of results/e01_franka_bench/results_core.json -> rmse_core.pdf|png
2) Sim Franka + SoftHand, from results_sim.json and soft_hand_torque_train.npz:
     measured vs CAD prior / FIGAROH / proposed MLE torques -> soft_hand_torque_train.png
     link-7 (arm flange + SoftHand) m, com, I vs ground truth          -> soft_hand_comparison_results.png
     ... including the CAD prior and the FIGAROH baselines              -> soft_hand_comparison_all.png
   and prints the sim RMSE and the link-7 estimates.
3) Real Franka: training / held-out windows of each setting over the whole recording -> folds.pdf|png
4) Real Franka: LaTeX table of the MLE parameters (full window) vs Gaz et al. (CAD prior) -> table_real_parameters.tex
"""
import json, os, sys
import numpy as np
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.setup_env import RESULTS
from bench.data import read_bag
from bench.plots import torque_plot_paper, link_param_plot, PAPER_STYLES
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(RESULTS, "e01_franka_bench")


# ======================================================================================================================
# 1) Real Franka: torque RMSE per method on training and held-out windows
# ======================================================================================================================
res = json.load(open(os.path.join(OUT, "results_core.json")))

# Same colours as the paper figures (e99); one solid colour per method (hatch column unused here)
METHODS = [  # (key in results, label, colour, hatch)
    ("Prior (CAD)", "Prior (CAD)", "#a3a29c", ""),
    ("BP-OLS (= project WLS)", "LS base param. (OLS)", "#1baf7a", ".."),
    ("BP-WLS (fixed)", "LS base param. (WLS)", "#eb6834", "//"),
    ("MAP-WLS (IPOPT)", "Proposed MLE (IPOPT)", "#2a78d6", ""),
    ("FIGAROH physical fit", "FIGAROH: physical fit (LMI)", "#eda100", "xx"),
    ("FIGAROH per-link projection", "FIGAROH: per-link projection", "#4a3aa7", "\\\\"),
]
# (setting, window, x label): training windows first, then held-out windows
GROUPS = [("full", "train", "full\ntrain"), ("foldA", "train", "fold A\ntrain"),
          ("foldB", "train", "fold B\ntrain"), ("foldA", "test", "fold A\nheld-out"),
          ("foldB", "test", "fold B\nheld-out")]

# Paper style only for this figure (the sim figures below keep the bench.plots style)
with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "savefig.bbox": "tight", "pdf.fonttype": 42}):
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    x = np.arange(len(GROUPS))
    bw = 0.8 / len(METHODS)
    for k, (key, label, col, hatch) in enumerate(METHODS):
        vals = [res[s]["rmse"][w][key] for s, w, _ in GROUPS]
        xb = x - 0.4 + (k + 0.5) * bw
        ax.bar(xb, vals, bw * 0.9, color=col, edgecolor="white", linewidth=0.4, label=label, zorder=3)
        for xi, v in zip(xb, vals):  # RMSE value on every bar
            ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90, color="#0b0b0b")
    ax.axvline(2.5, color="#52514e", lw=0.6, ls=":")  # separates training from held-out windows
    ax.set_xticks(x, [g[2] for g in GROUPS])
    ax.set_ylim(0, 4)   # headroom for the rotated value labels
    ax.set_ylabel("torque RMSE [N m]")
    ax.set_title("Real Franka: torque RMSE on training and held-out windows", fontsize=8)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"rmse_core.{ext}"), dpi=220)
    plt.close(fig)
print("saved", os.path.join(OUT, "rmse_core.pdf"))


# ======================================================================================================================
# 2) Simulated Franka + SoftHand
# ======================================================================================================================
sim = json.load(open(os.path.join(OUT, "results_sim.json")))
full_dyn = {k: np.array(v) for k, v in sim["full_dyn"].items()}   # full standard parameters of each method
tz = np.load(os.path.join(OUT, "soft_hand_torque_train.npz"))
w0, w1 = sim["window"]

# Plot (paper style, total RMSE in the title): measured vs CAD prior (no hand) vs FIGAROH
# reconstructions vs proposed MLE model (drawn last, on top)
sh_preds = dict(zip(tz["names"].tolist(), tz["preds"]))
sh_preds.pop("Ground truth")                                   # ground truth not shown in the paper figure
sh_preds["MAP-WLS (IPOPT)"] = sh_preds.pop("MAP-WLS (IPOPT)")
torque_plot_paper(tz["t"], tz["T"], sh_preds, len(full_dyn["Ground truth"]) // 10,
                  os.path.join(OUT, "soft_hand_torque_train.png"), PAPER_STYLES, sim["rmse"]["train"], cond=sim.get("cond"))
# Plot: estimated link-7 (index 6) m, com, I vs ground truth, as src/examples/franka_sim_softhand/results/soft_hand_comparison_results.png
link_param_plot({"Model": full_dyn["Ground truth"], "Estimation": full_dyn["MAP-WLS (IPOPT)"]}, 6,
                os.path.join(OUT, "soft_hand_comparison_results.png"))
# ... and the same comparison including the CAD prior and the FIGAROH baselines
first = ["Ground truth", "MAP-WLS (IPOPT)", "Prior (CAD)"]
link_param_plot({k: full_dyn[k] for k in first + [k for k in full_dyn if k not in first]},
                6, os.path.join(OUT, "soft_hand_comparison_all.png"))
# Print: RMSE [N m] and link-7 estimates vs ground truth
print("sim train", {k: round(v, 3) for k, v in sim["rmse"]["train"].items()})
for k, v in sim["link7"].items():
    print(f"  link7 {k:40s} m={v['mass']:.3f} com={np.round(v['com'], 4)}",
          "" if k == "Ground truth" else {e: round(x, 4) for e, x in sim["link7_error"][k].items()})


print("saved", os.path.join(OUT, "soft_hand_comparison_*.png"))
# ======================================================================================================================
# 3) Real Franka: how the recording is split into training / held-out windows for each setting
# ======================================================================================================================
config_traj = yaml.safe_load(open('./src/examples/franka_real/config/identif_config.yaml'))['trajectory']
t_bag = read_bag(config_traj['bag'], config_traj['topic'], config_traj['joints'])['t'].to_numpy()
T_END = t_bag[-1] - t_bag[0]                     # windows are relative to the first sample of the bag [s]
ROWS = [("full", "full"), ("foldA", "fold A"), ("foldB", "fold B")]   # (setting, y label), top to bottom
COL = {"train": "#2a78d6", "test": "#eb6834", "unused": "#e4e3df"}

with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.grid": False, "savefig.bbox": "tight", "pdf.fonttype": 42}):
    fig, ax = plt.subplots(figsize=(7.1, 1.4))
    for y, (key, _) in enumerate(ROWS):
        win = res[key]["windows"]
        ax.barh(y, T_END, left=0, height=0.6, color=COL["unused"], zorder=1)          # whole recording
        segments = [("train", win["train"])] + [("test", iv) for iv in win["tests"].values()]
        for kind, (a, b) in segments:
            b = min(b, T_END)                                                         # windows past the end are truncated
            ax.barh(y, b - a, left=a, height=0.6, color=COL[kind], edgecolor="white", linewidth=0.6, zorder=2)
            ax.text((a + b) / 2, y, f"{a:.3g}-{b:.3g} s", ha="center", va="center", fontsize=6, color="white", zorder=3)
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, T_END)
    ax.set_xlabel("time from start of recording [s]")
    ax.legend(handles=[Patch(color=COL[k], label=l) for k, l in
                       (("train", "training window"), ("test", "held-out window"), ("unused", "not used"))],
              ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"folds.{ext}"), dpi=220)
    plt.close(fig)
print("saved", os.path.join(OUT, "folds.pdf"))


# ======================================================================================================================
# 4) Real Franka: MLE parameters identified on the full window vs Gaz et al. (the CAD prior) -> LaTeX table
# ======================================================================================================================
def sci(v):
    """Number as $a.bcd \\times 10^{e}$ (4 significant digits), as in the paper tables."""
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    m = round(v / 10 ** e, 3)
    if abs(m) >= 10:                                     # rounding carried to the next power of ten
        m, e = m / 10, e + 1
    return f"${m:.3f} \\times 10^{{{e}}}$"


mle = res["full"]["mle_params"]
dyn_mle = np.array(mle["par_DYN"]).reshape(-1, 10)
dl_mle = np.array(mle["par_Dl"]).reshape(len(dyn_mle), -1)        # per joint: [fs, fv]
dyn_gaz = np.array(res["full"]["prior_par_DYN"]).reshape(-1, 10)
n_links = len(dyn_mle)


def block(source, dyn, dl):
    rows = [f"\\multirow{{{n_links}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{source}}}}}}} "]
    for i in range(n_links):
        fr = [sci(v) for v in dl[i]] if dl is not None else ["$-$", "$-$"]
        rows.append(f" & {i} & " + " & ".join([sci(v) for v in dyn[i]] + fr) + " \\\\")
    return "\n".join(rows)


table = "\n".join([
    "\\begin{table*}[htpb]",
    "\\centering",
    "\\caption{Comparison of Inertial and Friction Parameters}",
    "\\label{tab:real_parameter_comparison}",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{cc|rrrrrrrrrrrr}",
    "\\hline",
    "Source & Link & $m$ & $cx$ & $cy$ & $cz$ & $ixx$ & $ixy$ & $ixz$ & $iyy$ & $iyz$ & $izz$ & $fs$ & $fv$ \\\\ \\hline",
    block("MLE", dyn_mle, dl_mle),
    "\\hline",
    block("Gaz et al. \\cite{gaz2019franka}", dyn_gaz, None),
    "\\hline",
    "\\end{tabular}%",
    "}",
    "\\end{table*}",
    ""])
open(os.path.join(OUT, "table_real_parameters.tex"), "w").write(table)
print("saved", os.path.join(OUT, "table_real_parameters.tex"))
