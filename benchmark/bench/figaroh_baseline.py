"""
FIGAROH (v1.0.2, PyPI) baseline on the *same processed data* as the proposed pipeline.

Base-parameter fit, as BaseIdentification.solve(wls=True):
  Pinocchio joint-torque regressor (+ viscous/Coulomb friction columns)
  -> elimination of non-excited columns -> double-QR base-parameter selection
  -> OLS -> WLS with inverse per-joint residual variance (BaseIdentification._apply_weighted_least_squares).
Full (standard) parameters, the estimates FIGAROH's select_stage can return (figaroh.identification):
  - reconstruction (select_stage: reconstruction): closest-to-CAD theta with M theta = phi_base,
    'nullspace' (FIGAROH default, no consistency) or 'sdp' (+ per-link pseudo-inertia LMI)   -> reconstruct()
  - physical fit (select_stage: physical_fit): WLS effort fit with per-link LMIs and a weak CAD prior
    (physical_fit.solve_direct_effort_fit)                                                    -> physical_stages()
  - per-link projection: CAD-anchored null-space representative, then independent per-link LMI projection
    (physical_fit.solve_per_link_projection)                                                  -> physical_stages()
identify(project=True) also keeps the min-norm redistribution + per-link LMI projection used by the earlier
version of this baseline. It is NOT a FIGAROH stage (FIGAROH projects the nominal model, not the fit) and is
only kept as a diagnostic.
"""
import time
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
from figaroh.identification.physical_consistency import project_p10_lmi, check_p10_feasibility
from figaroh.tools.qrdecomposition import QRDecomposer, redistribute_min_norm
from figaroh.tools.regressor import RegressorBuilder, RegressorConfig

URDF = "src/data/franka/urdf/frankaEmikaPanda.urdf"
ARM = [f"panda_joint{i}" for i in range(1, 8)]
P10_NAMES = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


def load_model(urdf=URDF, joints=ARM, thunder_frames=True):
    """Pinocchio model of the arm. With thunder_frames=True the last joint frame is moved by
    d7 = 0.107 m along its axis, so that link frames coincide with Thunder's DH frames
    (Thunder's link-7 frame is the flange frame, the URDF one is at joint 7)."""
    full = pin.buildModelFromUrdf(urdf)
    lock = [full.getJointId(n) for n in full.names[1:] if n not in joints]
    model = pin.buildReducedModel(full, lock, pin.neutral(full)) if lock else full
    if thunder_frames:
        jid = model.njoints - 1
        model.jointPlacements[jid] = model.jointPlacements[jid] * pin.SE3(np.eye(3), np.array([0, 0, 0.107]))
    model.gravity.linear = np.array([0, 0, -9.81])
    return SimpleNamespace(model=model, data=model.createData())


def thunder_reg_to_p10(reg10):
    """Thunder REG [m, mc, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] (origin) -> Pinocchio p10."""
    m, h, I = reg10[0], reg10[1:4], reg10[4:10]
    return np.array([m, *h, I[0], I[1], I[3], I[2], I[4], I[5]])


def p10_to_thunder_reg(p10):
    m, h = p10[0], p10[1:4]
    return np.array([m, *h, p10[4], p10[5], p10[7], p10[6], p10[8], p10[9]])


def set_model_inertias(robot, reg):
    for k in range(robot.model.njoints - 1):
        robot.model.inertias[k + 1] = pin.Inertia.FromDynamicParameters(thunder_reg_to_p10(reg[10 * k:10 * k + 10]))
    robot.data = robot.model.createData()


def regressor(robot, tm):
    n = robot.model.nv
    builder = RegressorBuilder(robot, RegressorConfig(has_friction=True))
    W = builder.build_basic_regressor(tm.q, tm.qd, tm.qdd, {"act_idxv": list(range(n))})
    tau = tm.tau.T.reshape(-1)  # FIGAROH ordering: joint-major
    return W, tau


def param_names(n):
    names = [f"{p}_{k + 1}" for k in range(n) for p in P10_NAMES]
    names += [f"fv_{k + 1}" for k in range(n)] + [f"fs_{k + 1}" for k in range(n)]
    return names


def identify(robot, tm, method="WLS", project=True, cad_reg=None):
    n = robot.model.nv
    t0 = time.time()
    W, tau = regressor(robot, tm)
    names = param_names(n)
    col = np.diag(W.T @ W)
    keep = [i for i in range(W.shape[1]) if col[i] >= 1e-6]
    W_e = W[:, keep]
    names_r = [names[i] for i in keep]
    qr = QRDecomposer().decompose(W_e, names_r, tau, method="double")
    W_b = W_e[:, qr.base_indices] if qr.W_b is None else qr.W_b
    W_b = W_e @ np.linalg.pinv(qr.M)  # base regressor consistent with M (phi_b = M theta)
    phi = np.linalg.lstsq(W_b, tau, rcond=None)[0]
    row_weight = np.ones(tau.size)
    if method == "WLS":  # FIGAROH 1.0.2 BaseIdentification._apply_weighted_least_squares (Gautier 1997)
        var = np.mean((tau - W_b @ phi).reshape(n, -1) ** 2, axis=1)   # per-joint residual variance (joint-major)
        row_weight = np.repeat(np.sqrt(1.0 / (var + 1e-12)), tau.size // n)
        phi = np.linalg.lstsq(row_weight[:, None] * W_b, row_weight * tau, rcond=None)[0]
    theta = redistribute_min_norm(qr.M, phi)
    full = np.zeros(len(names)); full[keep] = theta
    t_ls = time.time() - t0
    res = dict(rank=int(qr.rank), keep=keep, M=qr.M, phi=phi, theta_full=full, W_b=W_b, row_weight=row_weight,
               cond=float(np.linalg.cond(W_b)), runtime_ls=t_ls)
    # standard (full) parameters of the redistributed solution, before/after LMI projection
    reg_ls = np.concatenate([p10_to_thunder_reg(full[10 * k:10 * k + 10]) for k in range(n)])
    res["reg_ls"] = reg_ls
    res["feasible_links_ls"] = [check_p10_feasibility(full[10 * k:10 * k + 10]).status == "feasible" for k in range(n)]
    if project:
        t1 = time.time()
        full_p, status = project_full(full, n, cad_reg)
        res["runtime_projection"] = time.time() - t1
        res["projection_status"] = status
        res["theta_full_projected"] = full_p
        res["reg_projected"] = np.concatenate([p10_to_thunder_reg(full_p[10 * k:10 * k + 10]) for k in range(n)])
        res["feasible_links_projected"] = [check_p10_feasibility(full_p[10 * k:10 * k + 10]).status == "feasible" for k in range(n)]
    return res


def project_full(full, n, cad_reg=None):
    """Per-link FIGAROH LMI projection of a full parameter vector (friction untouched)."""
    proj, status = [], []
    for k in range(n):
        p10 = full[10 * k:10 * k + 10]
        p, rep = project_p10_lmi(p10)  # FIGAROH defaults (auto weights)
        st = rep.status
        if rep.status == "error":  # e.g. zero mass after min-norm redistribution
            p, rep = project_p10_lmi(p10, weights=np.ones(10))
            st = "error->uniform weights:" + rep.status
        if rep.status == "error" and cad_reg is not None:
            p = thunder_reg_to_p10(cad_reg[10 * k:10 * k + 10])
            st = "error->CAD value kept"
        proj.append(np.asarray(p, float))
        status.append(st)
    full_p = full.copy()
    full_p[:10 * n] = np.concatenate(proj)
    return full_p, status


def predict(robot, tm, theta_full):
    W, _ = regressor(robot, tm)
    return (W @ theta_full).reshape(robot.model.nv, -1).T.reshape(-1)  # back to sample-major


def reconstruct(res, cad_reg, weights_reg=None, method="sdp"):
    """FIGAROH prior-anchored reconstruction (figaroh.identification.reconstruction):
        min ||diag(w)(theta - theta0)||^2  s.t.  M theta = phi_base  (+ per-link LMI for method='sdp').
    theta0 = CAD prior (friction prior 0). weights_reg: per-parameter weights in Thunder REG order
    (None -> FIGAROH default, all ones)."""
    from figaroh.identification.reconstruction import reconstruct_full_parameters

    n = len(cad_reg) // 10
    names = param_names(n)
    keep = res["keep"]
    cad_p10 = np.concatenate([thunder_reg_to_p10(cad_reg[10 * k:10 * k + 10]) for k in range(n)])
    theta0_all = np.concatenate([cad_p10, np.zeros(2 * n)])
    w_all = np.ones(len(names))
    if weights_reg is not None:
        w_p10 = np.concatenate([thunder_reg_to_p10(weights_reg[10 * k:10 * k + 10]) for k in range(n)])
        w_all = np.concatenate([w_p10, 1e-3 * np.ones(2 * n)])
    params_r = [names[i] for i in keep]
    t0 = time.time()
    rr = reconstruct_full_parameters((res["M"], res["phi"], params_r), method=method,
                                     theta0=theta0_all[keep], weights=w_all[keep],
                                     joint_names=[str(k + 1) for k in range(n)])
    full = theta0_all.copy()
    full[keep] = rr.theta_r
    reg = np.concatenate([p10_to_thunder_reg(full[10 * k:10 * k + 10]) for k in range(n)])
    return dict(theta_full=full, reg=reg, status=rr.status, runtime=time.time() - t0,
                base_residual_norm=float(rr.base_residual_norm),
                feasible_links=[check_p10_feasibility(full[10 * k:10 * k + 10]).status == "feasible" for k in range(n)])


def physical_stages(robot, tm, cad_reg, row_weight=None, solver="cvxopt", second_solver="qics"):
    """FIGAROH 1.0.2 physical_fit problem on the same data: direct LMI effort fit (select_stage: physical_fit,
    default PhysicalPolicy) and per-link projection. cad_reg = Thunder REG prior (friction prior 0).
    row_weight: WLS row weights from identify() (FIGAROH uses them for the physical fit after solve(wls=True)).
    second_solver: retried when the first solve is not optimal (FIGAROH physical_fit.second_solver option);
    cvxopt can fail numerically ('math domain error') on long recordings."""
    from figaroh.identification import physical_fit as pf

    n = robot.model.nv
    W, tau = regressor(robot, tm)
    prior = np.concatenate([np.concatenate([thunder_reg_to_p10(cad_reg[10 * k:10 * k + 10]) for k in range(n)]),
                            np.zeros(2 * n)])
    prob = pf.build_problem(W, tau, param_names(n), [str(k + 1) for k in range(n)], prior=prior,
                            row_weight=row_weight)
    out = {}
    for key, fn in (("physical_fit", pf.solve_direct_effort_fit), ("per_link_projection", pf.solve_per_link_projection),
                    ("exact_reconstruction", pf.solve_exact_reconstruction)):
        rec, used = fn(prob, solver=solver), [solver]
        if rec.solver_status != "optimal" and second_solver and key != "per_link_projection":
            rec = fn(prob, solver=second_solver)
            used.append(second_solver)
        out[key] = dict(theta_full=rec.theta, status=rec.solver_status, accepted=rec.accepted, solvers=used,
                        runtime=rec.runtime_s, base_residual_rel=rec.base_residual["rel"],
                        feasible_links=[v["ok"] for v in rec.feasibility.values()])
    # Exact reconstruction (M theta = phi with per-link LMIs) fails when no consistent theta reproduces phi:
    # FIGAROH's phase-I problem (diagnose_exact, stage D1) certifies it with s* < 0
    if not out["exact_reconstruction"]["accepted"]:
        ph1 = pf._solve_core(prob, "phase1", solver=solver)
        if ph1["status"] != "optimal" and second_solver:
            ph1 = pf._solve_core(prob, "phase1", solver=second_solver)
        out["exact_reconstruction"]["phase1_s_star"] = ph1.get("s_star")
    return out


def estimates(robot, tm, cad_reg):
    """FIGAROH 1.0.2 full-parameter estimates that are physically consistent by construction, on the same
    data: {name: theta_full (FIGAROH order)}, plus the stage report (status, feasibility, exact-reconstruction
    infeasibility certificate). Failed solves are reported, not replaced."""
    fig = identify(robot, tm, method="WLS", project=False)
    st = physical_stages(robot, tm, cad_reg, row_weight=fig["row_weight"])
    models = {"FIGAROH physical fit": st["physical_fit"]["theta_full"],
              "FIGAROH per-link projection": st["per_link_projection"]["theta_full"],
              "FIGAROH exact reconstruction (SDP)": st["exact_reconstruction"]["theta_full"]}
    report = {k: {f: v for f, v in d.items() if f != "theta_full"} for k, d in st.items()}
    return {k: v for k, v in models.items() if v is not None}, report
