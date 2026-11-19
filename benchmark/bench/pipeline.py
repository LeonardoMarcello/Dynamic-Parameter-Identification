"""Run all compared methods on one train set and evaluate them on held-out sets."""
import time

import numpy as np

from bench import figaroh_baseline as fb
from bench.ident import (build_regressor, dyn2reg_np, feasibility_margins, map_convex, map_solve,
                         ols, prior_precision, reg2dyn_np, rmse_stats, wls)
from bench.setup_env import thunder


def load_robot(name, par_path):
    r = thunder.thunder_[name]()
    thunder.load_params(r, par_path)
    return r


def full_param_errors(dyn, gt_dyn):
    rows = []
    for k in range(len(gt_dyn) // 10):
        a, b = dyn[10 * k:10 * k + 10], gt_dyn[10 * k:10 * k + 10]
        rows.append(dict(link=k + 1, mass_abs=float(abs(a[0] - b[0])),
                         com_norm=float(np.linalg.norm(a[1:4] - b[1:4])),
                         inertia_fro=float(np.linalg.norm(a[4:10] - b[4:10]))))
    return rows


def run_methods(robot_name, prior_path, weights, tm_train, tests, gt_path=None, with_figaroh=True,
                positive_threshold=1e-10, link_w=None):
    r = load_robot(robot_name, prior_path)
    n = r.numJoints
    beta = np.array(r.get_beta())
    nb = beta.shape[0]
    prior_dyn = np.array(r.get_par_DYN()).ravel()
    prior_dl = np.array(r.get_par_Dl()).ravel()
    prec = prior_precision(n, weights["mass"], weights["CoM"], weights["inertia"], link_w)

    Y, T = build_regressor(r, tm_train)
    est = {"OLS": ols(Y, T), "WLS": wls(Y, T, n)}
    models = {}  # name -> parameter vector multiplying [Yr_red | reg_dl]
    models["Prior (CAD)"] = np.concatenate([beta @ dyn2reg_np(prior_dyn), prior_dl])
    models["BP-OLS (= project WLS)"] = est["OLS"]["pi"].ravel()
    models["BP-WLS (fixed)"] = est["WLS"]["pi"].ravel()

    out = dict(n_train_samples=int(tm_train.t.shape[0]), excitation={k: v for k, v in est["WLS"].items()
               if k in ("cond", "sigma_max", "sigma_min", "trace", "logdet", "cond_colnorm")},
               sigma_joint_wls=est["WLS"]["sigma_joint"].tolist(),
               bp_std_ols=est["OLS"]["std"].ravel().tolist(), bp_std_wls=est["WLS"]["std"].ravel().tolist(),
               bp_ols=est["OLS"]["pi"].ravel().tolist(), bp_wls=est["WLS"]["pi"].ravel().tolist())
    full = {}
    for tag, e in [("MAP-WLS (IPOPT)", est["WLS"]), ("MAP-OLS (IPOPT, as published)", est["OLS"])]:
        sol = map_solve(beta, e["pi"].ravel()[:nb], e["C"][:nb, :nb], prior_dyn, prec,
                        positive_threshold=positive_threshold)
        full[tag] = sol
        models[tag] = np.concatenate([beta @ sol["reg"], e["pi"].ravel()[nb:]])
    cv = map_convex(beta, est["WLS"]["pi"].ravel()[:nb], est["WLS"]["C"][:nb, :nb], prior_dyn, prec)
    full["MAP-convex (SDP)"] = cv
    models["MAP-convex (SDP)"] = np.concatenate([beta @ cv["reg"], est["WLS"]["pi"].ravel()[nb:]])

    if gt_path is not None:  # reference model: exact simulator parameters
        g = load_robot(robot_name, gt_path)
        models["Ground truth"] = np.concatenate([beta @ dyn2reg_np(np.array(g.get_par_DYN()).ravel()),
                                                 np.array(g.get_par_Dl()).ravel()])

    fig = None
    fig_models = {}
    if with_figaroh:
        rob = fb.load_model()
        cad_reg = dyn2reg_np(prior_dyn)
        t0 = time.time()
        fig = fb.identify(rob, tm_train, method="WLS", project=True, cad_reg=cad_reg)
        fig["runtime_total"] = time.time() - t0
        ns = fb.reconstruct(fig, cad_reg, method="nullspace")
        st = fb.physical_stages(rob, tm_train, cad_reg, row_weight=fig["row_weight"])
        out["figaroh"] = dict(version="1.0.2", rank=fig["rank"], cond=fig["cond"], runtime_ls=fig["runtime_ls"],
                              feasible_links_ls=fig["feasible_links_ls"],
                              nullspace_feasible_links=ns["feasible_links"],
                              stages={k: {f: v for f, v in d.items() if f != "theta_full"} for k, d in st.items()},
                              naive_projection=dict(status=fig["projection_status"],
                                                    feasible_links=fig["feasible_links_projected"]))
        fig_models = {"FIGAROH BP (WLS)": fig["theta_full"],
                      "FIGAROH reconstruction (nullspace, no consistency)": ns["theta_full"],
                      "FIGAROH per-link projection": st["per_link_projection"]["theta_full"],
                      "FIGAROH physical fit": st["physical_fit"]["theta_full"],
                      "naive min-norm + LMI proj. (not a FIGAROH stage)": fig["theta_full_projected"]}
        fig_models = {k: v for k, v in fig_models.items() if v is not None}   # failed solves give no estimate
        to_reg = lambda th: np.concatenate([fb.p10_to_thunder_reg(th[10 * k:10 * k + 10]) for k in range(n)])
        for k, th in fig_models.items():
            if k != "FIGAROH BP (WLS)":
                rg = to_reg(th)
                full[k] = dict(reg=rg, dyn=reg2dyn_np(rg))

    out["full"] = {k: dict(status=v.get("status"), runtime=v.get("runtime"), cost=v.get("cost"),
                           f_loss=v.get("f_loss"), f_prior=v.get("f_prior"),
                           max_constr_viol=v.get("max_constr_viol"), dyn=v["dyn"].tolist(),
                           feasibility=feasibility_margins(v["dyn"])) for k, v in full.items()}

    # ---------------- evaluation
    out["rmse"] = {}
    sets = {"train": tm_train, **tests}
    for sname, tm in sets.items():
        Yt, Tt = (Y, T) if sname == "train" else build_regressor(r, tm)
        res = {k: rmse_stats(Tt.ravel(), Yt @ v, n) for k, v in models.items()}
        if fig is not None:
            rob = fb.load_model()
            for k, th in fig_models.items():
                res[k] = rmse_stats(Tt.ravel(), fb.predict(rob, tm, th), n)
        out["rmse"][sname] = res

    if gt_path is not None:
        g = load_robot(robot_name, gt_path)
        gt_dyn = np.array(g.get_par_DYN()).ravel()
        gt_bp = beta @ dyn2reg_np(gt_dyn)
        scale = np.maximum(np.abs(gt_bp), 1e-3)
        out["gt_bp"] = gt_bp.tolist()
        out["bp_error"] = {k: dict(abs=np.abs(v[:nb] - gt_bp).tolist(),
                                   norm_rel=float(np.linalg.norm((v[:nb] - gt_bp) / scale) / np.sqrt(nb)))
                           for k, v in models.items()}
        for k in fig_models:
            if k in full:
                bp = beta @ full[k]["reg"]
                out["bp_error"][k] = dict(abs=np.abs(bp - gt_bp).tolist(),
                                          norm_rel=float(np.linalg.norm((bp - gt_bp) / scale) / np.sqrt(nb)))
        out["full_error"] = {k: full_param_errors(v["dyn"], gt_dyn) for k, v in full.items()}
        out["full_error"]["Prior (CAD)"] = full_param_errors(prior_dyn, gt_dyn)
        out["gt_dyn"] = gt_dyn.tolist()
    out["prior_dyn"] = prior_dyn.tolist()
    return out, dict(robot=r, beta=beta, est=est, full=full, models=models, prec=prec, prior_dyn=prior_dyn,
                     Y=Y, T=T, figaroh=fig, figaroh_models=fig_models)
