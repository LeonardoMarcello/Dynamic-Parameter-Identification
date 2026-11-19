"""
    Excitation diagnostics vs identification accuracy, with the MLE started from a wrong (perturbed) prior (paper Fig. 5).

Many identification windows (duration 3-14.7 s, different offsets) are cut from the three simulated
recordings with the lean SoftHand mounting (Fourier, chirp, sinusoidal; 19.7 s each, same payload,
all consistent with franka_SH_lean_par.yaml). For every window the project pipeline (Identifier) is run:
  * base parameters with the project WLS (utils.identification_utils.solve_WLS),
  * full dynamics with the project MLE (utils.identification_utils.solve_dynamics) whose prior and
    initial guess is the CAD model perturbed by 30 %.
We collect the diagnostics reported by the framework (condition number, trace = T-optimality,
log-det = D-optimality, max relative std of the base inertial parameters, BIP) and two accuracy scores:
  * held-out torque RMSE on the two OTHER recordings (full 5-19.7 s, never used for identification),
  * BIP error w.r.t. the lean ground truth.
Spearman rank correlations quantify how well each criterion predicts accuracy (3 families x 15 windows = 45 windows).

Run from the repository root, with src/ on the path (ROS 2 environment, as the other examples):
    PYTHONPATH=src python src/examples/franka_sim_trajectories/franka_identification.py

Output (src/examples/franka_sim_trajectories/results/excitation/):
    1) results.json            all windows, Spearman table, perturbed-prior reference
    2) windows.csv             one row per window (diagnostics + accuracy)
    3) table_spearman.tex      LaTeX table of the Spearman correlations
    4) excitation_diagnostic.png   criterion vs accuracy for every window (paper Fig. 5)
"""
import csv
import os
import json
import yaml
from yaml import SafeLoader
import numpy as np
from scipy.stats import spearmanr
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder
import matplotlib.pyplot as plt

OUT = './src/examples/franka_sim_trajectories/results/excitation'; os.makedirs(OUT, exist_ok=True)
json_filename = os.path.join(OUT, "results.json")
csv_filename = os.path.join(OUT, "windows.csv")
tex_filename = os.path.join(OUT, "table_spearman.tex")
png_filename = os.path.join(OUT, "excitation_diagnostic.png")

# Load Ground-Truth values (simulator, lean SoftHand)
franka_ground_truth = thunder.thunder_['franka']()
thunder.load_params(franka_ground_truth, "./src/thunder/franka_generatedFiles/param/franka_SH_lean_par.yaml")
franka_ground_truth.set_par_REG(franka_ground_truth.get_dyn2reg())
franka_ground_truth.set_par_REG_red(franka_ground_truth.get_reg2red())
gt_bp = franka_ground_truth.get_par_REG_red().copy()
nb = len(gt_bp)                                                     # number of inertial BP (the rest is friction)
n = franka_ground_truth.numJoints

# Base Parameter error: RMS over the n_b inertial BP of the relative error (denominator floored at 1e-3)
#   bp_err = sqrt( 1/n_b * sum_i ( (pi_hat_i - pi_gt_i) / max(|pi_gt_i|, 1e-3) )^2 )
scale = np.maximum(np.abs(gt_bp), 1e-3)
bp_err = lambda pi: float(np.linalg.norm((np.ravel(pi)[:nb] - gt_bp) / scale) / np.sqrt(nb))
# Torque RMSE (paper definition): sqrt( 1/M sum_k ||tau_k - tau_hat_k||^2 )
rmse = lambda TTau, TTau_hat: float(np.sqrt(np.mean(np.sum((TTau.reshape(-1, n) - TTau_hat.reshape(-1, n))**2, axis=1))))


# Configuration: trajectory/processing from the config of each family,
# identification (WLS + prior weights, link-7 std inflated: the prior ignores the hand) from the SoftHand config, solved with IPOPT
trajectories = ['Fourier', 'Chirp', 'Sinusoidal']
config_files = {
    'Fourier':    './src/examples/franka_sim_trajectories/config/fourier_config.yaml',
    'Chirp':      './src/examples/franka_sim_trajectories/config/chirp_config.yaml',
    'Sinusoidal': './src/examples/franka_sim_trajectories/config/sinusoidal_config.yaml'
}
configs = {}
for traj in trajectories:
    with open(config_files[traj], 'r') as f:
        configs[traj] = yaml.load(f, Loader=SafeLoader)
with open('./src/examples/franka_sim_softhand/config/softhand_config.yaml', 'r') as f:
    config_ident = yaml.load(f, Loader=SafeLoader)['identification']
config_ident.update(method='WLS', optimization='ipopt', positive_threshold=1e-10, inertia_scale_factor=1)
ipopt_opts = {"ipopt": {"tol": 1e-4, "constr_viol_tol": 1e-6, "acceptable_tol": 1e-5, "dual_inf_tol": 1e-6,
                        "compl_inf_tol": 1e-6, "max_iter": 50000, "print_level": 0}, "print_time": False}
T0, T1 = 5.0, 19.7                                                  # usable interval [s]

# Identification windows (15 per family):
#   * durations 3, 5, 7.5, 10 s, offsets every 2.5 s
#   * plus the full interval (14.7 s)
windows = [(dur, float(off)) for dur in (3, 5, 7.5, 10) for off in np.arange(T0, T1 - dur + 1e-6, 2.5)]
windows.append((round(T1 - T0, 1), T0))


# Raw bag of each family, read once (as Identifier.init) and cut into windows by load_window
raw_df = {traj: TrajectoryManagerROS2.read_bag(configs[traj]['trajectory']['bag'], configs[traj]['trajectory']['topic'],
                                               configs[traj]['trajectory']['joints'])
          for traj in trajectories}


def load_window(traj, dur, off):
    """Processed trajectory of one identification window, as Identifier.init() with interval [off, off + dur]."""
    trajectory = TrajectoryManager(config_files[traj])
    trajectory.config['trajectory']['interval'] = [float(off), float(off + dur)]
    trajectory.df = raw_df[traj]
    if trajectory.config['processing'].get('pipeline', False):
        trajectory.process_pipeline()
    else:
        trajectory.process()
    return trajectory


# -------- Perturbed prior: every CAD DYN entry scaled by (1 + DELTA * N(0,1)), masses floored at 0.05 kg, --------
# then projected onto the physically consistent set (LMI_projection: m > 0, pseudo-inertia P > 0)
DELTA, SEED = 0.3, 7
franka_prior = thunder.thunder_['franka']()
thunder.load_params(franka_prior, configs['Fourier']['robot']['path'])
rng = np.random.default_rng(SEED)
par_DYN_prior = np.array(franka_prior.get_par_DYN()).ravel() * (1 + DELTA * rng.standard_normal(10 * n))
par_DYN_prior[0::10] = np.maximum(par_DYN_prior[0::10], 0.05)
franka_prior.set_par_DYN(par_DYN_prior)
par_DYN_prior = np.ravel(LMI_projection(franka_prior))
franka_prior.set_par_DYN(par_DYN_prior)
print(f"[INFO] perturbed prior feasible: {check_par_DYN_feasibility(par_DYN_prior, n, verbose=True)}")
franka_prior.set_par_REG(franka_prior.get_dyn2reg())
franka_prior.set_par_REG_red(franka_prior.get_reg2red())
prior_model = np.concatenate([franka_prior.get_par_REG_red(), np.zeros(n * franka_prior.Dl_order)])  # zero friction

# -------- Held-out test sets: full interval of every recording --------
# A window cut from family k is evaluated only on the OTHER two families -> train/test never overlap
tests = {}
for traj in trajectories:
    YY, TTau, _ = get_big_Y_Tau(franka_prior, load_window(traj, T1 - T0, T0))
    tests[traj] = (YY, TTau)

# Reference: perturbed prior alone (no data)
prior_ref = dict(delta=DELTA, seed=SEED, bp_err=bp_err(prior_model),
                 rmse=float(np.mean([rmse(TTau, YY @ prior_model) for YY, TTau in tests.values()])))

# -------- Run Identification for each trajectory and window --------
rows = []
for traj in trajectories:
    for dur, off in windows:
        franka = thunder.thunder_['franka']()
        thunder.load_params(franka, configs[traj]['robot']['path'])

        identifier = Identifier(franka, config_path=config_files[traj])
        identifier.config['identification'] = config_ident      # SoftHand prior weights, IPOPT
        identifier.trajectory = load_window(traj, dur, off)     # = identifier.init() on the window
        identifier.solve_base_parameter()                       # project WLS
        hat_pi_ls = np.concatenate([franka.get_par_REG_red(), franka.get_par_Dl()])   # LS inertial BP + friction
        metrics = identifier.metrics

        # Proposed MLE with the perturbed prior: solve_dynamics takes prior and initial guess from robot.get_par_DYN()
        franka.set_par_DYN(par_DYN_prior)
        identifier.solve_full_dynamics(opts=ipopt_opts)
        hat_pi_map = np.concatenate([franka.get_par_REG_red(), franka.get_par_Dl()])  # MLE inertial BP + WLS friction

        # Diagnostics reported by the framework, computed on this window only:
        #   cond          kappa(Y) = s_max / s_min
        #   trace         tr(Y^T Y)          (T-optimality: larger = more excitation on average)
        #   logdet        log det(Y^T Y)     (D-optimality: larger = smaller confidence-ellipsoid volume)
        #   max_rel_std   max relative std. dev. of the inertial BP, 100 * sigma_pi / |pi| [%]
        YY, _, _ = get_big_Y_Tau(franka, identifier.trajectory)
        row = dict(family=traj, duration=float(dur), offset=off,
                   cond=float(metrics['conditioning number']),
                   trace=float(metrics['trace']),
                   logdet=float(np.linalg.slogdet(YY.T @ YY)[1]),
                   max_rel_std=float(np.max(metrics['parameters relative standard deviation'][:nb])))

        # Accuracy (a): Base Parameter error w.r.t. the ground truth
        row['bp_err_ls'] = bp_err(hat_pi_ls)
        row['bp_err_map'] = bp_err(hat_pi_map)
        # Accuracy (b): held-out torque RMSE, mean over the two other recordings
        row['rmse_ls_heldout_mean'] = float(np.mean([rmse(TTau, YY_t @ hat_pi_ls) for k, (YY_t, TTau) in tests.items() if k != traj]))
        row['rmse_map_heldout_mean'] = float(np.mean([rmse(TTau, YY_t @ hat_pi_map) for k, (YY_t, TTau) in tests.items() if k != traj]))
        rows.append(row)
        print(traj, dur, off, "cond %.0f  LS %.3f  MLE %.3f  bipLS %.2f  bipMLE %.2f" % (
            row['cond'], row['rmse_ls_heldout_mean'], row['rmse_map_heldout_mean'], row['bp_err_ls'], row['bp_err_map']))

# -------- Spearman rank correlation over the 45 windows --------
# (rank-based: robust to the very different scales of the criteria; tests a monotone relation)
# Expected sign if the criterion is useful: kappa, max rel. std -> positive; trace, log det -> negative
crit = {'cond': r"condition number $\kappa$",
        'trace': r"trace$(W^TW)$ (T-opt.)",
        'logdet': r"log det$(W^TW)$ (D-opt.)",
        'max_rel_std': r"max $\sigma_{\pi,\%}$"}
SIGN = {'cond': 1, 'trace': -1, 'logdet': -1, 'max_rel_std': 1}
targets = {'rmse_ls_heldout_mean': "RMSE LS",
           'rmse_map_heldout_mean': "RMSE MLE",
           'bp_err_ls': "BIP err. LS",
           'bp_err_map': "BIP err. MLE"}
corr = {}
for k in crit:
    corr[k] = {}
    for tgt in targets:
        rho, p = spearmanr([rw[k] for rw in rows], [rw[tgt] for rw in rows])   # rho in [-1, 1], p-value of rho = 0
        corr[k][tgt] = dict(spearman=float(rho), p=float(p))
works = lambda k, tgt: np.sign(corr[k][tgt]['spearman']) == SIGN[k] and corr[k][tgt]['p'] < 0.05

json.dump(dict(rows=rows, spearman=corr, perturbed_prior=prior_ref), open(json_filename, "w"), indent=1)

# -------- LaTeX Table Automation Generation --------
print("\n" + "="*20 + " GENERATING LATEX TABLE " + "="*20 + "\n")

latex_str = []
latex_str.append(r"\begin{table}[htpb]")
latex_str.append(r"\centering")
latex_str.append(r"\resizebox{\columnwidth}{!}{%")
latex_str.append(r"\begin{tabular}{|l|" + "c|" * len(targets) + "}")
latex_str.append(r"\hline")
latex_str.append(r"\textbf{Criterion} & " + " & ".join(rf"\textbf{{{v}}}" for v in targets.values()) + r" \\")
latex_str.append(r"\hline")
for k, label in crit.items():
    cells = []
    for tgt in targets:
        s = f"{corr[k][tgt]['spearman']:+.2f}"
        cells.append((rf"$\mathbf{{{s}}}$" if works(k, tgt) else f"${s}$") + f" ({corr[k][tgt]['p']:.1e})")
    latex_str.append(f"{label} & " + " & ".join(cells) + r" \\")
latex_str.append(r"\hline")
latex_str.append(r"\end{tabular}")
latex_str.append(r"}")
latex_str.append(rf"\caption{{Spearman rank correlation ($p$-value) between excitation criteria and accuracy over "
                 rf"{len(rows)} identification windows; MLE with a {DELTA:.0%} perturbed prior. "
                 r"Bold: expected sign and $p<0.05$.}")
latex_str.append(r"\label{tab:excitation_spearman}")
latex_str.append(r"\end{table}")

print("\n".join(latex_str))
with open(tex_filename, "w") as file:
    file.write("\n".join(latex_str) + "\n")

with open(csv_filename, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n[INFO] CSV data successfully saved to: {csv_filename}\n")

# -------- Printing Max/Min Metrics with Windows --------
print("\n" + "="*20 + " ADDITIONAL METRICS METADATA " + "="*20 + "\n")
print(f"perturbed prior alone: BIP error {prior_ref['bp_err']:.3f}, held-out RMSE {prior_ref['rmse']:.3f} N m")
print("-" * 30)
for traj in trajectories:
    fam = [rw for rw in rows if rw['family'] == traj]
    for key, label in (('cond', "condition number"), ('bp_err_map', "MLE BIP error"), ('rmse_map_heldout_mean', "MLE held-out RMSE")):
        lo, hi = min(fam, key=lambda rw: rw[key]), max(fam, key=lambda rw: rw[key])
        print(f"{traj} {label}: min {lo[key]:.4f} (window {lo['duration']} s @ {lo['offset']} s), "
              f"max {hi[key]:.4f} (window {hi['duration']} s @ {hi['offset']} s)")
    print("-" * 30)

# -------- Plot (paper Fig. 5): criterion (x) vs accuracy (y) for every window --------
#   row 1: held-out torque RMSE      row 2: base inertial parameter (BIP) error w.r.t. the ground truth
#   colour = trajectory family, marker = method (LS: hollow square, MLE: filled triangle)
#   grey dashed line = perturbed prior alone (no data)
#   Spearman in bold only when it confirms the correlation: expected sign, p < 0.05 and strong (|rho| >= 0.7)
FAMS = (("Fourier", "#0072B2"), ("Chirp", "#D55E00"), ("Sinusoidal", "#009E73"))
RHO_STRONG = 0.7


def rho_txt(k, t):
    s = f"{corr[k][t]['spearman']:+.2f}"
    return rf"$\mathbf{{{s}}}$" if works(k, t) and abs(corr[k][t]['spearman']) >= RHO_STRONG else rf"${s}$"


ACC = (("rmse_ls_heldout_mean", "rmse_map_heldout_mean", "rmse", "held-out RMSE [N m]"),
       ("bp_err_ls", "bp_err_map", "bp_err", "BIP error (RMS of relative error) [-]"))

# Serif text sized for the paper (STIX maths, so bold values match the serif font)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 16, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13})
fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
for i, (tls, tmap, pkey, ylab) in enumerate(ACC):
    for ax, k in zip(axs[i], ["cond", "logdet", "trace", "max_rel_std"]):
        ax.axhline(prior_ref[pkey], color="#7F7F7F", ls="--", lw=1, label=f"prior alone ({DELTA:.0%} pert.)")
        for fam, col in FAMS:
            x = [rw[k] for rw in rows if rw['family'] == fam]
            ax.scatter(x, [rw[tls] for rw in rows if rw['family'] == fam], marker="s", facecolors="none",
                       edgecolors=col, linewidths=1.1, s=26, label=f"LS ({fam})")
            ax.scatter(x, [rw[tmap] for rw in rows if rw['family'] == fam], marker="^", c=col, s=24,
                       label=f"MLE, {DELTA:.0%} prior ({fam})")
        if k in ("cond", "trace", "max_rel_std"):
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(crit[k])
        ax.set_title(f"Spearman LS {rho_txt(k, tls)} | MLE {rho_txt(k, tmap)}")
    axs[i, 0].set_ylabel(ylab)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.1, 1, 1]); fig.savefig(png_filename, dpi=170)
print(f"[INFO] figure saved to: {png_filename}")
