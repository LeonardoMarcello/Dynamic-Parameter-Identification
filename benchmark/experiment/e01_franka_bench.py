"""
Benchmark of the proposed method (project pipeline: Identifier WLS + MAP) against the FIGAROH baseline.
Usage:
    > benchmark/run.sh benchmark/experiment/e01_franka_bench.py

1) Real Franka Panda: torque RMSE of every method,
   (1.a) on the full window (2.5-15 s),
   (1.b) on held-out windows, with a 2-fold split of the Fourier period,
   with a prior relative to each link (mass 5 %, CoM 1 cm, inertia 25 %; results/e01_franka_bench/prior_relative.yaml).
2) Simulated Franka + lean SoftHand: estimate of the unknown end-effector dynamics lumped into link 7.

Run from benchmark/ with ./run.sh benchmark/experiment/e01_franka_bench.py (FIGAROH and IPOPT live in benchmark/.env).
Utils: e01_franka_bench_plot_core.py (figure generator, run automatically at the end).

Output (results/e01_franka_bench/):
    1) results_core.json           RMSE [N m] per setting, window and method,
       torque_<setting>_<window>.png  measured vs CAD prior vs proposed MAP torques
    2) results_sim.json            RMSE, link-7 estimates and errors, full parameters of each method,
       soft_hand_torque_train.npz  torques for the figures of e01_franka_bench_plot_core.py
"""
import json
import os
import subprocess
import sys
import yaml
from yaml import SafeLoader
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # make benchmark/bench importable
from bench.setup_env import RESULTS                                 # stubs ROS, puts src/ on the path, chdir to the repo root
from bench.data import read_bag                                     # rosbag reader without ROS2 (same output as TrajectoryManagerROS2.read_bag)
from bench import figaroh_baseline as fb                            # FIGAROH 1.0.2 baseline
from bench.plots import torque_plot_arrays, torque_plot_paper, PAPER_STYLES
from bench.ident import relative_prior_sigma, write_prior_yaml   # relative prior uncertainty (real Franka)
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder

OUT = os.path.join(RESULTS, "e01_franka_bench"); os.makedirs(OUT, exist_ok=True)

# Torque RMSE (paper definition): sqrt( 1/M sum_k ||tau_k - tau_hat_k||^2 )
rmse = lambda TTau, TTau_hat, n: float(np.sqrt(np.mean(np.sum((np.reshape(TTau, (-1, n)) - np.reshape(TTau_hat, (-1, n)))**2, axis=1))))


def load_window(config_path, interval):
    """Processed trajectory of the bag in config_path on interval [s], as Identifier.init()."""
    trajectory = TrajectoryManager(config_path)
    config_traj = trajectory.config['trajectory']
    config_traj['interval'] = [float(x) for x in interval]
    trajectory.df = read_bag(config_traj['bag'], config_traj['topic'], config_traj['joints'])
    if trajectory.config['processing'].get('pipeline', False):
        trajectory.process_pipeline()
    else:
        trajectory.process()
    return trajectory


def run_proposed(robot, config_path, trajectory, config_ident=None):
    """Proposed pipeline on one trajectory: OLS and WLS base parameters, then MAP (IPOPT) from the WLS estimate
    with the robot CAD values as prior. Returns {method: [inertial BP | friction]}, the WLS metrics and the MAP costs."""
    identifier = Identifier(robot, config_path=config_path)
    if config_ident is not None:
        identifier.config['identification'] = config_ident
    identifier.trajectory = trajectory                                      # = identifier.init() on the window
    estimates = {}
    for method in ('OLS', 'WLS'):                                           # WLS last: its covariance weights the MAP
        identifier.config['identification']['method'] = method
        identifier.solve_base_parameter()
        estimates[f"BP-{method}"] = np.concatenate([robot.get_par_REG_red(), robot.get_par_Dl()])
    metrics = identifier.metrics
    map_cost = identifier.solve_full_dynamics(opts=ipopt_opts)
    estimates["MAP-WLS"] = np.concatenate([robot.get_par_REG_red(), robot.get_par_Dl()])  # MAP inertial BP + WLS friction
    return estimates, metrics, {k: float(v) for k, v in map_cost.items()}


def cad_model(robot):
    """CAD prior of a thunder robot as [inertial BP | friction]."""
    robot.set_par_REG(robot.get_dyn2reg())
    return np.concatenate([robot.get_reg2red(), robot.get_par_Dl()])


ipopt_opts = {"ipopt": {"tol": 1e-4, "constr_viol_tol": 1e-6, "acceptable_tol": 1e-5, "dual_inf_tol": 1e-6,
                        "compl_inf_tol": 1e-6, "max_iter": 50000, "print_level": 0}, "print_time": False}


# ======================================================================================================================
# 1) Real Franka Estimation: reconstructed torque and rmse
# ======================================================================================================================
print("1) Benchmarking against figaroh")
config_path = './src/examples/franka_real/config/identif_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)

# Prior uncertainty relative to each link (bench.ident.relative_prior_sigma): mass 5 %, CoM 1 cm,
# inertia 25 % of the link inertia norm. Passed to solve_dynamics as identification.prior_yaml
# (per-parameter sigma, no link-weight normalisation), replacing the absolute weights of identif_config.yaml.
franka = thunder.thunder_['franka']()
thunder.load_params(franka, config['robot']['path'])
prior_dyn = np.array(franka.get_par_DYN()).ravel()
prior_sigma = relative_prior_sigma(prior_dyn)
config_ident_real = dict(config['identification'],
                         prior_yaml=write_prior_yaml(prior_dyn, prior_sigma, os.path.join(OUT, "prior_relative.yaml")))

# Training and held-out test windows [s] of each setting
settings = {
    # Full window of the recording
    "full": dict(train=[2.5, 15], tests={}),
    # Held-out 1) Train on early phases, test on later phases of the Fourier period (13.3 s)
    "foldA": dict(train=[2.5, 10.7], tests={"test": [10.7, 15.3]}),
    # Held-out 2) Train on late phases, test on early phases not seen in training
    "foldB": dict(train=[6.0, 15.3], tests={"test": [2.3, 6.0]}),
}

# -------- Run Identification for each setting --------
summary = {}
for name, s in settings.items():
    trajectories = {"train": load_window(config_path, s["train"]),
                    **{k: load_window(config_path, iv) for k, iv in s["tests"].items()}}

    franka = thunder.thunder_['franka']()
    thunder.load_params(franka, config['robot']['path'])                    # CAD prior
    n = franka.numJoints
    models = {"Prior (CAD)": cad_model(franka)}
    cad_reg = franka.get_dyn2reg().copy()

    # Proposed pipeline on the training window
    estimates, metrics, map_cost = run_proposed(franka, config_path, trajectories["train"], dict(config_ident_real))
    models.update({"BP-OLS (= project WLS)": estimates["BP-OLS"],
                   "BP-WLS (fixed)": estimates["BP-WLS"],
                   "MAP-WLS (IPOPT)": estimates["MAP-WLS"]})
    # Prior consistency: shift of the MAP solution from the prior in prior std. devs (|z| <~ 3 if the prior is plausible)
    z = np.abs((np.array(franka.get_par_DYN()).ravel() - prior_dyn) / prior_sigma).reshape(n, 10)
    prior_z = dict(mass=z[:, 0].tolist(), com=z[:, 1:4].max(1).tolist(), inertia=z[:, 4:].max(1).tolist())
    # Full MLE estimate (standard parameters + friction [fs, fv] per joint) for the parameter table of the paper
    mle_params = dict(par_DYN=np.array(franka.get_par_DYN()).ravel().tolist(), par_Dl=np.array(franka.get_par_Dl()).ravel().tolist())
    print(f"{name:9s} max |z| per link: mass {np.round(prior_z['mass'], 2)}  CoM {np.round(prior_z['com'], 2)}  "
          f"inertia {np.round(prior_z['inertia'], 2)}")

    # FIGAROH 1.0.2 on the same processed data: physically consistent full-parameter estimates (physical fit,
    # per-link projection, exact reconstruction when feasible), all anchored to the same CAD prior
    rob = fb.load_model()
    fig_models, fig_report = fb.estimates(rob, trajectories["train"], cad_reg)

    # Torque RMSE of every model on the training window and each held-out test window
    rmse_tab = {}
    for wname, trajectory in trajectories.items():
        YY, TTau, _ = get_big_Y_Tau(franka, trajectory)
        rmse_tab[wname] = {k: rmse(TTau, YY @ v, n) for k, v in models.items()}
        fig_preds = {k: fb.predict(rob, trajectory, th) for k, th in fig_models.items()}
        rmse_tab[wname].update({k: rmse(TTau, p, n) for k, p in fig_preds.items()})
        # Plot: measured torque vs CAD prior vs proposed MAP model (+ FIGAROH reconstructions on the full training window)
        window = s["train"] if wname == "train" else s["tests"][wname]
        plot_preds = {k: YY @ models[k] for k in ["Prior (CAD)", "MAP-WLS (IPOPT)"]}
        if name == "full" and wname == "train":
            # Paper figure, same style as src/examples/franka_real/results/ne_comparison.png, with FIGAROH curves
            plot_preds.update(fig_preds)
            plot_preds["MAP-WLS (IPOPT)"] = plot_preds.pop("MAP-WLS (IPOPT)")   # MLE on top of the FIGAROH curves
            torque_plot_paper(trajectory.t, TTau.ravel(), plot_preds, n, os.path.join(OUT, f"torque_{name}_{wname}.png"),
                              PAPER_STYLES, rmse_tab[wname], cond=metrics.get('conditioning number'))
        else:
            torque_plot_arrays(trajectory.t, TTau.ravel(), plot_preds, n,
                               os.path.join(OUT, f"torque_{name}_{wname}.png"), title=f"Real Franka - {name} - {wname} window {window} s")
        print(f"{name:9s} {wname:5s}", {k: round(v, 3) for k, v in rmse_tab[wname].items()})
    summary[name] = dict(windows=s, rmse=rmse_tab, map_cost=map_cost, prior_z=prior_z, figaroh=fig_report,
                         mle_params=mle_params, prior_par_DYN=prior_dyn.tolist())

json.dump(summary, open(os.path.join(OUT, "results_core.json"), "w"), indent=1)


# ======================================================================================================================
# 2) Simulated Franka + SoftHand: estimate the unknown end-effector (SoftHand) dynamics lumped into link 7
# ======================================================================================================================
print("2) Softhamd estimation")
# SoftHand config (lean SoftHand mounting, Fourier trajectory), prior weights with link-7 std x10, solved with IPOPT
config_path = './src/examples/franka_sim_softhand/config/softhand_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
config_ident = config['identification']
config_ident.update(optimization='ipopt', positive_threshold=1e-10, inertia_scale_factor=1)
WINDOW = [5, 19.7]

# Prior = Franka without SoftHand (config robot path); ground truth = simulator parameters with the lean SoftHand
franka = thunder.thunder_['franka']()
thunder.load_params(franka, config['robot']['path'])
franka_ground_truth = thunder.thunder_['franka']()
thunder.load_params(franka_ground_truth, "./src/thunder/franka_generatedFiles/param/franka_SH_lean_par.yaml")
n = franka.numJoints
prior_dyn = np.array(franka.get_par_DYN()).ravel()
gt_dyn = np.array(franka_ground_truth.get_par_DYN()).ravel()
models = {"Prior (CAD)": cad_model(franka), "Ground truth": cad_model(franka_ground_truth)}
cad_reg = franka.get_dyn2reg().copy()

# -------- Run Identification --------
trajectory = load_window(config_path, WINDOW)
estimates, metrics, map_cost = run_proposed(franka, config_path, trajectory, config_ident)
models.update({"BP-OLS (= project WLS)": estimates["BP-OLS"],
               "BP-WLS (fixed)": estimates["BP-WLS"],
               "MAP-WLS (IPOPT)": estimates["MAP-WLS"]})
map_dyn = np.array(franka.get_par_DYN()).ravel()

# FIGAROH 1.0.2 on the same data (full parameters in FIGAROH order, converted to thunder standard parameters)
rob = fb.load_model()
fig_models, fig_report = fb.estimates(rob, trajectory, cad_reg)
converter = thunder.thunder_['franka']()


def figaroh_to_dyn(theta):
    converter.set_par_REG(np.concatenate([fb.p10_to_thunder_reg(theta[10 * k:10 * k + 10]) for k in range(n)]))
    return np.array(converter.get_reg2dyn()).ravel()


# Full (physically consistent) standard parameters of each method -> estimated link-7 (arm flange + SoftHand) m, com, I
full_dyn = {"Prior (CAD)": prior_dyn, "Ground truth": gt_dyn, "MAP-WLS (IPOPT)": map_dyn,
            **{k: figaroh_to_dyn(th) for k, th in fig_models.items()}}
L7 = slice(60, 70)
link7 = {k: dict(mass=float(d[L7][0]), com=d[L7][1:4].tolist(), inertia=d[L7][4:10].tolist()) for k, d in full_dyn.items()}
link7_err = {k: dict(mass_abs=abs(v["mass"] - link7["Ground truth"]["mass"]),
                     com_norm=float(np.linalg.norm(np.subtract(v["com"], link7["Ground truth"]["com"]))),
                     inertia_fro=float(np.linalg.norm(np.subtract(v["inertia"], link7["Ground truth"]["inertia"]))))
             for k, v in link7.items() if k != "Ground truth"}

# Torque RMSE on the identification window
YY, TTau, _ = get_big_Y_Tau(franka, trajectory)
preds = {k: YY @ v for k, v in models.items()}
preds.update({k: fb.predict(rob, trajectory, th) for k, th in fig_models.items()})   # FIGAROH torque reconstruction
rmse_tab = {k: rmse(TTau, p, n) for k, p in preds.items()}

json.dump(dict(window=WINDOW, prior=config['robot']['path'], ground_truth="./src/thunder/franka_generatedFiles/param/franka_SH_lean_par.yaml",
               map_cost=map_cost, figaroh=fig_report, rmse={"train": rmse_tab}, link7=link7, link7_error=link7_err,
               cond=float(metrics.get('conditioning number', np.nan)),
               full_dyn={k: np.asarray(v).tolist() for k, v in full_dyn.items()}),
          open(os.path.join(OUT, "results_sim.json"), "w"), indent=1)
# Torques for the training-window plot: measured, CAD prior, ground truth, proposed MAP, then FIGAROH reconstructions
plot_preds = {k: preds[k] for k in ["Prior (CAD)", "Ground truth", "MAP-WLS (IPOPT)", *fig_models]}
np.savez(os.path.join(OUT, "soft_hand_torque_train.npz"), t=trajectory.t, T=TTau.ravel(), names=np.array(list(plot_preds)),
         preds=np.array([np.ravel(v) for v in plot_preds.values()]))

# All figures of parts 1 and 2 (rmse_core, sim torque_train, soft_hand_comparison_*) and the sim prints
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e01_franka_bench_plot_core.py")],
               check=True)
