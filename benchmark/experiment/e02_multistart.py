"""
This scripts evaluate the stability of the optimizer when considering different initial guesses of the IPOPT solver.
the following initial guesses are considered for the  IPOPT solver:
    * 1 run with the prior as initial guess,
    * 50 runs with initial gueeses extracted with gaussian perturbation N(0, sigma^2) of the prior with sigma the prior std. dev,
    * 50 runs with random physically-consistent bodies (mass U[0.2,5] kg, CoM U[-0.15,0.15] m, random principal moments satisfying the triangle inequalities, random orientation)

Rerun logic of every start (bench.ident.map_solve, diagram in multistart_rerun_logic.png):
    1. IPOPT with the exact constraint eig_symbolic(Sigma_C) >= 0
    2. if it fails (NaN of eig_symbolic at a degenerate Sigma_C with two equal eigenvalues), re-solve from the same
       start with the approximate constraint v^T Sigma_C v >= 0 on 2000 sphere directions (fallback)
    A run fails only if both IPOPT solves fail. The exact feasibility of the solution (min eig of Sigma_C, the
    fallback can violate it by ~1e-5) is stored per run as a diagnostic, not as a convergence criterion.

Usage:
   > benchmark/run.sh benchmark/experiment/e02_multistart.py
   > benchmark/run.sh benchmark/experiment/e02_multistart.py --resummarize
     (no solves: recompute the summaries of the stored runs, real and simulated, then the tables)

For each case we report success rate, spread of the final cost, relative deviation of mass / CoM / inertia from
the best solution (mean, std. dev., max over runs and links) and the held-out RMSE spread.

Paper Table IV: at the end, this script runs automatically (about 2 h in total)
    e02_multistart_table.py      real-Franka table (table_multistart.tex|pdf)
    e02_multistart_softhand.py   multi-start on the simulated Franka+SoftHand (results/e02_multistart_softhand/)
    e02_project_best.py          feasibility check and per-link LMI projection of the simulated best runs
    multistart_unified_table.py  unified table, real + simulated (results/e02_multistart/table_multistart_unified.tex|png)

Output: results/e02_multistart/results.json, table_multistart.tex|pdf, table_multistart_unified.tex|png
"""
import json
import os
import subprocess
import sys
import yaml
from yaml import SafeLoader
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # make benchmark/bench importable
from bench.setup_env import RESULTS                                 # stubs ROS, puts src/ on the path, chdir to the repo root
from bench.data import read_bag                                     # rosbag reader without ROS2 (same output as TrajectoryManagerROS2.read_bag)
# MAP problem of solve_dynamics with an explicit initial guess (solve_dynamics always starts from the prior)
from bench.ident import map_solve, relative_prior_sigma, feasibility_margins
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder

OUT = os.path.join(RESULTS, "e02_multistart"); os.makedirs(OUT, exist_ok=True)
N_RAND, N_GAUSS = 50, 50   # number of random-body starts and of Gaussian prior perturbations


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


# Utils for randomization of init guess
def random_body(rng):
    """Random physically consistent rigid body, in DYN coordinates [m, c(3), I_c(6)] (I_c barycentric)."""
    m = rng.uniform(0.2, 5.0); c = rng.uniform(-0.15, 0.15, 3)    # mass [kg] and CoM [m]
    # Principal moments: resample until the triangle inequality a1 + a2 > a3 holds (sorted a1 <= a2 <= a3),
    # i.e. the inertia is realisable by a physical mass distribution
    while True:
        a = np.sort(rng.uniform(1e-4, 0.05, 3))
        if a[0] + a[1] > a[2]:
            break
    # Random orientation of the principal axes: I = R diag(a) R^T
    R = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
    I = R @ np.diag(a) @ R.T
    return np.array([m, *c, I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]])


# Utils for the spread of the solutions (also read by e02_multistart_table.py)
def frob(v):
    """Frobenius norm of symmetric 3x3 matrices stored as [xx, xy, xz, yy, yz, zz] (off-diagonals count twice)."""
    return np.sqrt(np.sum(v[..., [0, 3, 5]] ** 2, -1) + 2 * np.sum(v[..., [1, 2, 4]] ** 2, -1))


def rel_deviation(par_DYN, par_DYN_best):
    """Relative deviation from the best MAP solution, per run r and link l (group norms, since single CoM
    coordinates and products of inertia can be ~0), summarised by mean, std. dev. and max over (r, l):
        mass     |m_rl - m_l,best| / m_l,best
        CoM      ||c_rl - c_l,best||_2 / ||c_l,best||_2
        inertia  ||I_rl - I_l,best||_F / ||I_l,best||_F
    par_DYN: runs x 10n solutions, par_DYN_best: 10n best solution (DYN coordinates)."""
    L, Lb = par_DYN.reshape(len(par_DYN), -1, 10), par_DYN_best.reshape(-1, 10)       # runs x links x 10, links x 10
    dev = dict(mass=np.abs(L[..., 0] - Lb[:, 0]) / np.abs(Lb[:, 0]),
               com=np.linalg.norm(L[..., 1:4] - Lb[:, 1:4], axis=-1) / np.linalg.norm(Lb[:, 1:4], axis=-1),
               inertia=frob(L[..., 4:10] - Lb[:, 4:10]) / frob(Lb[:, 4:10]))
    return {g: dict(mean=float(v.mean()), std=float(v.std()), max=float(v.max())) for g, v in dev.items()}

# ---------------------------------------------------------------------------------------------------------------
# 3 test cases: 
#   - real robot (full window).
#   - real robot (foldA and foldB)
# ---------------------------------------------------------------------------------------------------------------
config_path = './src/examples/franka_real/config/identif_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
cases = {
    "real_full": dict(train=[2.5, 15], test=None),
    "real_foldB": dict(train=[6.0, 15.3], test=[2.3, 6.0]),
    "real_foldA": dict(train=[2.5, 10.7], test=[10.7, 15.3]),
}
# Thunder robot used only to convert DYN -> REG coordinates (IPOPT's decision variable)
converter = thunder.thunder_['franka']()
def dyn2reg(par_DYN):
    converter.set_par_DYN(par_DYN)
    return np.array(converter.get_dyn2reg()).ravel()


def summarize(runs):
    """Spread of the solutions over the converged runs (IPOPT success, exact or fallback constraint)."""
    ok = [x for x in runs if x["success"]]                          # converged: fails only if both solves fail
    best = min(x["cost"] for x in ok)                               # best MAP cost found (used as ref. for std computation)
    par_DYN = np.array([x["dyn"] for x in ok])                            # all solutions, DYN parameters
    par_DYN_best = np.array(min(ok, key=lambda x: x["cost"])["dyn"])       # DYN parameters for best, i.e. best MAP cost found (used as ref. for std computation)
    rel_cost = np.array([(x["cost"] - best) / best for x in ok])    # relative cost gap to the best run
    within = rel_cost < 1e-3                                        # runs within 0.1 % of the best cost

    # create solution table
    s = dict(n_starts=len(runs), n_success=len(ok), best_cost=best,
             n_fallback=int(sum(x["fallback"] for x in ok)),    # converged with the approximate (fallback) constraint
             n_feasible=int(sum(x["feasible"] for x in ok)),    # exact check min eig(Sigma_C) > -1e-6
             n_strictly_feasible=int(sum(x["strictly_feasible"] for x in ok)),
             min_margin_best=float(min(ok, key=lambda x: x["cost"])["min_margin"]),
             min_margin_worst=float(min(x["min_margin"] for x in ok)),
             frac_within_0p1pct_of_best=float(np.mean(within)),
             rel_cost_quantiles=np.quantile(rel_cost, [0.5, 0.9, 1.0]).tolist(),
             # Relative deviation from the best solution over all converged runs and links: {mass, com, inertia}
             # -> {mean, std, max}; same metric as the table (e02_multistart_table.py)
             rel_dev=rel_deviation(par_DYN, par_DYN_best),
             mean_runtime_s= float(np.mean([x["runtime"] for x in runs])),
             )
    # if held out exist
    if "heldout_rmse" in runs[0]:
        # Spread of the held-out RMSE over the runs (min, median, max)
        h = np.array([x["heldout_rmse"] for x in ok])
        s["heldout_rmse_min_median_max"] = [float(h.min()), float(np.median(h)), float(h.max())]
    return s


# Iterate on cases:
summary = {}
if "--resummarize" in sys.argv:
    # No solves: recompute the summaries of the stored runs (e.g. after changing the convergence criterion)
    summary = json.load(open(os.path.join(OUT, "results.json")))
    for cname in summary:
        summary[cname]["summary"] = summarize(summary[cname]["runs"])
        print(cname, json.dumps(summary[cname]["summary"], indent=1))
    cases = {}
for cname, c in cases.items():
    # ---------------- 1. Data, prior and base-parameter estimate (fixed for all starts)
    franka = thunder.thunder_['franka']()
    thunder.load_params(franka, config['robot']['path'])    # CAD prior
    n = franka.numJoints
    beta = np.array(franka.get_beta()); nb = beta.shape[0]  # standard -> base map, number of inertial BP
    prior = np.array(franka.get_par_DYN()).ravel()          # prior in DYN coordinates (pi_0)
    # Diagonal prior precision C_0^-1 = 1/sigma^2, sigma relative to each link (mass 5 %, CoM 1 cm, inertia 25 %),
    # the same prior as e01_franka_bench.py (bench.ident.relative_prior_sigma)
    prec = 1 / relative_prior_sigma(prior) ** 2
    # Project WLS on the training window: BP estimate + covariance
    identifier = Identifier(franka, config_path=config_path)
    identifier.trajectory = load_window(config_path, c["train"])        # = identifier.init() on the window
    identifier.solve_base_parameter()
    pi_wls = np.concatenate([franka.get_par_REG_red(), franka.get_par_Dl()])
    C_wls = identifier.metrics['parameters covariance matrix']
    # Arguments of the MAP problem: beta, BP estimate pi_b*, its covariance C_pipi, prior pi_0, prior precision
    args = (beta, pi_wls[:nb], C_wls[:nb, :nb], prior, prec)
    # Held-out regressor/torques (folds A and B have a test window)
    test = get_big_Y_Tau(franka, load_window(config_path, c["test"]))[:2] if c["test"] else None

    # ---------------- 2. Initial guesses, all converted to regressor coordinates (IPOPT's decision variable)
    rng = np.random.default_rng(1)     # fixed seed: reproducible starts
    starts = [("prior", dyn2reg(prior))]   # the default start used by the pipeline
    # 50 starts with every link replaced by a random consistent body (far from the prior)
    starts += [("random", dyn2reg(np.concatenate([random_body(rng) for _ in range(n)]))) for _ in range(N_RAND)]
    # 50 starts perturbing every prior parameter by N(0, sigma^2), sigma = 1/sqrt(precision);
    # these can be physically inconsistent
    starts += [("gauss", dyn2reg(prior + 1 / np.sqrt(prec) * rng.standard_normal(prior.shape)))
            for _ in range(N_GAUSS)]


    # ---------------- 3. Solve the MAP problem (IPOPT) from every start
    runs = []
    for kind, x0 in starts:
        # get MAP solution (symbolic eigenvalues, numerical fallback if IPOPT fails)
        sol = map_solve(*args, x0_reg=x0)
        # Check per-link feasibility: mass and min eigenvalue of Sigma_C = tr(I_C)/2 - I_C (> 0 <=> triangle inequalities)
        fm = feasibility_margins(sol["dyn"])
        feas = all(f["mass"] > 0 and f["min_eig_sigma"] > -1e-6 for f in fm)  # within IPOPT constr_viol_tol

        # create runs table
        row = dict(kind=kind, cost=sol["cost"], success=sol["success"], status=sol["status"], feasible=feas,
                   strictly_feasible = all(f["feasible"] for f in fm), min_margin = min(f["min_eig_sigma"] for f in fm),
                   runtime=sol["runtime"], iters=sol["iters"], fallback=sol["fallback"], dyn=sol["dyn"].tolist())
        if test is not None:
            # Held-out torque RMSE of this solution: identified inertial BP (beta @ reg) + friction from WLS
            pim = np.concatenate([beta @ sol["reg"], pi_wls[nb:]])
            row["heldout_rmse"] = rmse(test[1], test[0] @ pim, n)
        runs.append(row)

    summary[cname] = dict(summary=summarize(runs), runs=runs)
    print(cname, json.dumps(summary[cname]["summary"], indent=1))

# Save all runs and summaries
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=1)
# Paper Table IV: real-Franka table, then the simulated Franka+SoftHand multi-start (same flag), the projection of
# its best runs and the unified table (results/e02_multistart/table_multistart_unified.tex|png)
HERE = os.path.dirname(os.path.abspath(__file__))
for script, args in (("e02_multistart_table.py", []),
                     ("e02_multistart_softhand.py", ["--resummarize"] if "--resummarize" in sys.argv else []),
                     ("e02_project_best.py", []),
                     ("multistart_unified_table.py", [])):
    print(f"=== {script}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True)
