"""
Multi-start robustness of the MLE step on the simulated Franka + SoftHand, for every exciting trajectory and for
scaled prior uncertainties.

Questions:
  (1) Is the optimum consistent across exciting trajectories? The three simulated recordings (Fourier, chirp,
      sinusoidal; same lean SoftHand payload, ground truth franka_SH_lean_par.yaml) are identified separately and
      the best solutions are compared with each other and with the ground truth.
  (2) Is it robust to the prior uncertainty and accuracy? The prior std. devs. are the uniform relative prior of
      the real-robot experiment (Case A, bench.ident.relative_prior_sigma: mass 5 %, CoM 1 cm, inertia 25 % of the
      link inertia norm, same on every link: no information on where the payload is), scaled by SIGMA_SCALES
      (std. dev. x k, covariance x k^2), and
      the prior mean is either the CAD values ("none") or drawn from the declared uncertainty ("declared"):
          pi_0,k = pi_0 + sigma_k * xi,   xi ~ N(0, I) (one fixed draw, same for every trajectory and scale),
      i.e. a prior that is wrong exactly as much as it declares. Links of the drawn prior that are not physically
      consistent (m <= 0 or Sigma_C not positive definite) are projected onto the feasible set with the per-link
      LMI projection of the pipeline (utils.identification_utils.LMI_projection).

For every (trajectory, prior) case the MLE (bench.ident.map_solve, same rerun logic as e02_multistart.py) is solved
from 101 initial guesses, all with the same data and the same prior of that case:
    * 1 run from the case prior,
    * 50 runs with every link replaced by a random physically consistent body,
    * 50 runs with Gaussian perturbations N(0, sigma_k^2) of the case prior.
The multistart spread is computed within each case only.
Per case: convergence, spread of cost and parameters over the runs (as e02), held-out torque RMSE on the two OTHER
recordings (never used for identification), link-7 (arm flange + SoftHand) mass / CoM / inertia errors and the
normalised BP error ||pi_b - pi_b,gt|| / ||pi_b,gt|| w.r.t. the ground truth.
Across trajectories (same prior): spread of the best solutions (link-7 parameters, BP). The accuracy of the prior
alone (no data) is reported for reference.

Usage:
   > benchmark/run.sh benchmark/experiment/e02_multistart_softhand.py
   > benchmark/run.sh benchmark/experiment/e02_multistart_softhand.py --resummarize   (no solves)

Output: results/e02_multistart_softhand/results.json (all runs + summaries, saved after every case)
        results/e02_multistart_softhand/summary.md   (tables per case and across trajectories)
"""
import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # make benchmark/bench importable
from bench.setup_env import RESULTS                                 # stubs ROS, puts src/ on the path, chdir to the repo root
from bench.data import load_config, load_trajectory
from bench.pipeline import load_robot
from bench.ident import (build_regressor, wls, map_solve, relative_prior_sigma, rmse_stats, dyn2reg_np,
                         feasibility_margins)
from utils.identification_utils import LMI_projection             # per-link projection onto the feasible set

# E12_N / E12_OUT: fewer starts and a separate output folder for a quick test run
OUT = os.path.join(RESULTS, os.environ.get("E12_OUT", "e02_multistart_softhand")); os.makedirs(OUT, exist_ok=True)
N_RAND = N_GAUSS = int(os.environ.get("E12_N", 50))   # random-body starts and Gaussian prior perturbations (+1 prior start)
SIGMA_SCALES = (0.1, 1.0, 10.0)     # prior std. dev. multipliers
PRIOR_MEANS = ("none", "declared")  # CAD prior mean, or CAD + sigma_k * xi
SEED, SEED_PRIOR = 1, 7             # starts, prior draw

# Simulated recordings (same payload, 19.7 s each) and usable interval, as the excitation study (src/examples/franka_sim_trajectories)
cfg = load_config("src/examples/franka_sim_trajectories/config/fourier_config.yaml")
BAGS = {"Fourier": "./rosbag/softhand_lean_fourier_CT",
        "Chirp": "./rosbag/softhand_lean_chirp_CT",
        "Sinusoidal": "./rosbag/softhand_lean_sinusoidal_CT"}
T0, T1 = 5.0, 19.7


# Utils (same definitions as experiment/e02_multistart.py)
def random_body(rng):
    """Random physically consistent rigid body [m, c(3), I_c(6)] (I_c barycentric)."""
    m = rng.uniform(0.2, 5.0); c = rng.uniform(-0.15, 0.15, 3)
    while True:   # principal moments satisfying the triangle inequality
        a = np.sort(rng.uniform(1e-4, 0.05, 3))
        if a[0] + a[1] > a[2]:
            break
    R = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
    I = R @ np.diag(a) @ R.T
    return np.array([m, *c, I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]])


def frob(v):
    """Frobenius norm of symmetric 3x3 matrices stored as [xx, xy, xz, yy, yz, zz]."""
    return np.sqrt(np.sum(v[..., [0, 3, 5]] ** 2, -1) + 2 * np.sum(v[..., [1, 2, 4]] ** 2, -1))


def rel_deviation(par_DYN, par_DYN_best):
    """Relative deviation from a reference solution per run and link (mass, CoM norm, inertia Frobenius)."""
    L, Lb = par_DYN.reshape(len(par_DYN), -1, 10), par_DYN_best.reshape(-1, 10)
    dev = dict(mass=np.abs(L[..., 0] - Lb[:, 0]) / np.abs(Lb[:, 0]),
               com=np.linalg.norm(L[..., 1:4] - Lb[:, 1:4], axis=-1) / np.linalg.norm(Lb[:, 1:4], axis=-1),
               inertia=frob(L[..., 4:10] - Lb[:, 4:10]) / frob(Lb[:, 4:10]))
    return {g: dict(mean=float(v.mean()), std=float(v.std()), max=float(v.max())) for g, v in dev.items()}


def link7_errors(dyn, gt):
    """Link-7 (index 6) errors w.r.t. the ground truth: |dm| [kg], ||dc|| [m], ||dI||_F / ||I_gt||_F."""
    d, g = np.asarray(dyn)[60:70], gt[60:70]
    return dict(mass=float(abs(d[0] - g[0])), com=float(np.linalg.norm(d[1:4] - g[1:4])),
                inertia_rel=float(frob(d[4:10] - g[4:10]) / frob(g[4:10])))


# ---------------------------------------------------------------------------------------------------------------
# Fixed data: robot, prior, prior std. devs., ground truth, regressors of every recording
# ---------------------------------------------------------------------------------------------------------------
r = load_robot("franka", "src/thunder/franka_generatedFiles/param/franka_par.yaml")    # CAD prior (bare arm)
n = r.numJoints
beta = np.array(r.get_beta()); nb = beta.shape[0]                   # standard -> base map, number of inertial BP
prior = np.array(r.get_par_DYN()).ravel()                           # prior pi_0 (DYN coordinates)
prec_nominal = 1 / relative_prior_sigma(prior) ** 2               # C_0^-1, uniform relative prior (Case A)
gt = np.array(load_robot("franka", "src/thunder/franka_generatedFiles/param/franka_SH_lean_par.yaml")
              .get_par_DYN()).ravel()
gt_bp = beta @ dyn2reg_np(gt)

xi = np.random.default_rng(SEED_PRIOR).standard_normal(prior.shape)   # fixed draw for the "declared" prior


def case_prior(k, mean):
    """Prior mean of a case and the links projected to restore physical consistency."""
    sig = 1 / np.sqrt(prec_nominal / k ** 2)
    if mean == "none":
        return prior.copy(), []
    p = prior + sig * xi
    bad = [i for i, f in enumerate(feasibility_margins(p)) if not f["feasible"]]
    if bad:
        p = np.asarray(LMI_projection(r, par_DYN=p, links=bad), float).ravel()
    return p, bad


def prior_alone(p):
    """Accuracy of a prior with no data: held-out RMSE (zero friction, all recordings), link-7 and BP errors."""
    pim = np.concatenate([beta @ dyn2reg_np(p), np.zeros(2 * n)])
    return dict(heldout_rmse=float(np.mean([rmse_stats(T.ravel(), Y @ pim, n)["rmse"] for Y, T in data.values()])),
                link7=link7_errors(p, gt),
                bp_err=float(np.linalg.norm(beta @ dyn2reg_np(p) - gt_bp) / np.linalg.norm(gt_bp)))


data = {}
for fam, bag in BAGS.items():
    tm = load_trajectory(cfg, f"e12_{fam}", interval=[T0, T1], bag=bag)
    data[fam] = build_regressor(r, tm)                               # (Y, T) on the full usable interval


def summarize(runs, fam):
    """Spread over the converged runs + accuracy of the best run (as e02, plus ground-truth errors)."""
    ok = [x for x in runs if x["success"]]
    if not ok:
        return dict(n_starts=len(runs), n_success=0)
    best_run = min(ok, key=lambda x: x["cost"])
    best = best_run["cost"]
    par = np.array([x["dyn"] for x in ok])
    rel_cost = np.array([(x["cost"] - best) / best for x in ok])
    h = np.array([x["heldout_rmse"] for x in ok])
    l7 = {k: np.array([x["link7"][k] for x in ok]) for k in ("mass", "com", "inertia_rel")}
    return dict(n_starts=len(runs), n_success=len(ok),
                n_fallback=int(sum(x["fallback"] for x in ok)),
                n_feasible=int(sum(x["feasible"] for x in ok)),
                best_cost=best, rel_cost_quantiles=np.quantile(rel_cost, [0.5, 0.9, 1.0]).tolist(),
                frac_within_0p1pct_of_best=float(np.mean(rel_cost < 1e-3)),
                rel_dev=rel_deviation(par, np.array(best_run["dyn"])),
                heldout_rmse_min_median_max=[float(h.min()), float(np.median(h)), float(h.max())],
                best=dict(heldout_rmse=best_run["heldout_rmse"], link7=best_run["link7"],
                          bp_err=best_run["bp_err"], dyn=best_run["dyn"]),
                link7_spread={k: [float(v.min()), float(np.median(v)), float(v.max())] for k, v in l7.items()},
                mean_runtime_s=float(np.mean([x["runtime"] for x in runs])))


# ---------------------------------------------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------------------------------------------
path = os.path.join(OUT, "results.json")
results = json.load(open(path)) if os.path.exists(path) else {}
if "--resummarize" in sys.argv:
    for cname, c in results.items():
        c["summary"] = summarize(c["runs"], c["family"])
else:
    for mean in PRIOR_MEANS:
      for k in SIGMA_SCALES:
        prec = prec_nominal / k ** 2                                 # std. dev. x k  ->  precision / k^2
        sig = 1 / np.sqrt(prec)
        p0, projected = case_prior(k, mean)                          # same prior for every trajectory and start
        for fam, (Y, T) in data.items():
            cname = f"{fam}_x{k:g}_{mean}"
            if cname in results:                                     # resume: case already solved
                continue
            # BP estimate (per-joint WLS) and covariance on this trajectory
            e = wls(Y, T, n)
            pi_wls = e["pi"].ravel()
            args = (beta, pi_wls[:nb], e["C"][:nb, :nb], p0, prec)
            tests = {t: v for t, v in data.items() if t != fam}      # held-out: the two other recordings

            rng = np.random.default_rng(SEED)
            starts = [("prior", dyn2reg_np(p0))]
            starts += [("random", dyn2reg_np(np.concatenate([random_body(rng) for _ in range(n)])))
                       for _ in range(N_RAND)]
            starts += [("gauss", dyn2reg_np(p0 + sig * rng.standard_normal(p0.shape))) for _ in range(N_GAUSS)]

            runs = []
            for kind, x0 in starts:
                sol = map_solve(*args, x0_reg=x0)
                fm = feasibility_margins(sol["dyn"])
                pim = np.concatenate([beta @ sol["reg"], pi_wls[nb:]])           # MLE inertial BP + WLS friction
                ho = float(np.mean([rmse_stats(Tt.ravel(), Yt @ pim, n)["rmse"] for Yt, Tt in tests.values()]))
                runs.append(dict(kind=kind, cost=sol["cost"], success=sol["success"], status=sol["status"],
                                 fallback=sol["fallback"], runtime=sol["runtime"], iters=sol["iters"],
                                 feasible=all(f["mass"] > 0 and f["min_eig_sigma"] > -1e-6 for f in fm),
                                 min_margin=min(f["min_eig_sigma"] for f in fm), heldout_rmse=ho,
                                 link7=link7_errors(sol["dyn"], gt),
                                 bp_err=float(np.linalg.norm(beta @ sol["reg"] - gt_bp) / np.linalg.norm(gt_bp)),
                                 dyn=sol["dyn"].tolist()))
            results[cname] = dict(family=fam, sigma_scale=k, prior_mean=mean, prior_projected_links=projected,
                                  prior=p0.tolist(), prior_alone=prior_alone(p0),
                                  summary=summarize(runs, fam), runs=runs)
            json.dump(results, open(path, "w"), indent=1)            # checkpoint after every case
            s = results[cname]["summary"]
            print(f"{cname:16s} conv {s['n_success']}/{s['n_starts']} (fallback {s['n_fallback']})  "
                  f"held-out {s['best']['heldout_rmse']:.3f} N m  link7 dm {s['best']['link7']['mass']:.3f} kg  "
                  f"dc {1e3 * s['best']['link7']['com']:.1f} mm  BP err {s['best']['bp_err']:.3f}", flush=True)
json.dump(results, open(path, "w"), indent=1)

# ---------------------------------------------------------------------------------------------------------------
# Summary tables: per case, and consistency of the best solutions across trajectories (same scale)
# ---------------------------------------------------------------------------------------------------------------
lines = ["# e02 — multistart on the simulated Franka + SoftHand\n",
         f"Starts per case: 1 prior + {N_RAND} random bodies + {N_GAUSS} Gaussian, all with the case prior; "
         "prior std. dev. scaled by k; prior mean CAD ('none') or CAD + sigma_k xi ('declared').\n",
         "## Per case (best run, and spread over converged runs)\n",
         "| case | conv. (fallback) | cost gap median / max | mass dev. mean | inertia dev. mean | held-out RMSE best [N m] (min–max) | link-7 dm [kg] | link-7 dc [mm] | link-7 dI rel | BP err |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for cname, c in results.items():
    s = c["summary"]
    if not s.get("n_success"):
        lines.append(f"| {cname} | 0/{s['n_starts']} | | | | | | | | |"); continue
    b, ho = s["best"], s["heldout_rmse_min_median_max"]
    lines.append(f"| {cname} | {s['n_success']}/{s['n_starts']} ({s['n_fallback']}) | "
                 f"{s['rel_cost_quantiles'][0]:.1e} / {s['rel_cost_quantiles'][2]:.1e} | "
                 f"{s['rel_dev']['mass']['mean']:.1e} | {s['rel_dev']['inertia']['mean']:.1e} | "
                 f"{b['heldout_rmse']:.3f} ({ho[0]:.3f}–{ho[2]:.3f}) | {b['link7']['mass']:.3f} | "
                 f"{1e3 * b['link7']['com']:.1f} | {b['link7']['inertia_rel']:.2f} | {b['bp_err']:.3f} |")
lines += ["\n## Across trajectories (best solution of each trajectory, same prior)\n",
          "| k | prior mean | projected prior links | prior alone: held-out RMSE / link-7 dm [kg] / BP err | link-7 mass [kg] (F / C / S) | spread [kg] | link-7 CoM spread [mm] | mass dev. between trajectories (mean over links) |",
          "|---|---|---|---|---|---|---|---|"]
for mean in PRIOR_MEANS:
    for k in SIGMA_SCALES:
        cs = [results.get(f"{fam}_x{k:g}_{mean}") for fam in BAGS]
        if not all(c and c["summary"].get("n_success") for c in cs):
            continue
        D = np.array([c["summary"]["best"]["dyn"] for c in cs])
        m7, c7 = D[:, 60], D[:, 61:64]
        dev = rel_deviation(D, D.mean(0))["mass"]["mean"]
        pa = cs[0]["prior_alone"]
        lines.append(f"| {k:g} | {mean} | {[i + 1 for i in cs[0]['prior_projected_links']]} | "
                     f"{pa['heldout_rmse']:.2f} / {pa['link7']['mass']:.3f} / {pa['bp_err']:.3f} | "
                     f"{' / '.join(f'{x:.3f}' for x in m7)} | {m7.max() - m7.min():.3f} | "
                     f"{1e3 * np.max(np.linalg.norm(c7 - c7.mean(0), axis=1)):.1f} | {dev:.2e} |")
lines.append(f"\nGround truth link-7 mass: {gt[60]:.3f} kg; prior (no hand): {prior[60]:.3f} kg.")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
