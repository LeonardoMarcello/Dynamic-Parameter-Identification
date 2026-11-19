"""
Identification core used by the benchmark.

* build_regressor  -> stacked [Yr_red | reg_dl] exactly as utils.identification_utils.get_big_Y_Tau
* ols / wls        -> base-parameter LS; `wls` applies the per-joint weights of Eq. (13) of the
                      paper (the project's solve_WLS applies 1/sigma_0 to every joint, see E0)
* map_solve        -> Eq. (27) solved with CasADi/IPOPT, same constraints/weights as
                      utils.identification_utils.solve_dynamics, but with explicit x0 and prior
* map_convex       -> convex SDP variant (prior expressed in regressor coordinates, pseudo-inertia LMI)
* qcp_scalar       -> scalar-weight QCP baseline: ||T - W pi||^2 + lam ||pi_REG - pi0_REG||^2 s.t. LMI
"""
import time

import casadi
import numpy as np

from utils.casadi_utils import reg2dyn as ca_reg2dyn
from utils.identification_utils import _generate_fibonacci_sphere

# ----------------------------------------------------------------------------------- regressor


def build_regressor(robot, tm):
    Ys = []
    for i in range(tm.t.shape[0]):
        robot.set_q(tm.q[i]); robot.set_dq(tm.qd[i]); robot.set_ddq(tm.qdd[i])
        robot.set_dqr(tm.qd[i]); robot.set_ddqr(tm.qdd[i])
        Ys.append(np.hstack([robot.get_Yr_red(), robot.get_reg_dl()]))
    Y = np.vstack(Ys)
    T = tm.tau.reshape(-1, 1)  # sample-major, joint-minor (same as get_big_Y_Tau)
    return Y, T


def excitation_metrics(Y):
    s = np.linalg.svd(Y, compute_uv=False)
    YtY = Y.T @ Y
    sign, logdet = np.linalg.slogdet(YtY)
    # column-normalised condition number: insensitive to the physical units of each column
    Yn = Y / np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
    return dict(cond=float(s[0] / s[-1]), sigma_max=float(s[0]), sigma_min=float(s[-1]),
                trace=float(np.trace(YtY)), logdet=float(logdet),
                cond_colnorm=float(np.linalg.cond(Yn)))


def _ls_stats(Yw, Tw, pi, n_rows):
    p = Yw.shape[1]
    r = Tw - Yw @ pi
    s2 = float(r.T @ r) / (n_rows - p)
    C = s2 * np.linalg.pinv(Yw.T @ Yw)
    std = np.sqrt(np.clip(np.diag(C), 0, None)).reshape(-1, 1)
    return C, std, s2


def ols(Y, T):
    pi = np.linalg.lstsq(Y, T, rcond=None)[0]
    C, std, s2 = _ls_stats(Y, T, pi, Y.shape[0])
    return dict(pi=pi, C=C, std=std, rel_std=100 * std / (np.abs(pi) + 1e-10), s2=s2,
                sigma_joint=None, **excitation_metrics(Y))


def _tsvd_solve(A, b, ratio):
    """Truncated-SVD LS (Eq. 19): drop singular values with s_max/s_i > ratio. Returns (x, V1 S1^-2 V1^T, r)."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(S[0] / S <= ratio)) if ratio else len(S)
    x = Vt[:r].T @ ((U[:, :r].T @ b) / S[:r, None])
    return x, Vt[:r].T @ np.diag(1 / S[:r] ** 2) @ Vt[:r], r


def wls(Y, T, n, conditioning_ratio=None):
    """Per-joint weighting Z_jj = 1/sigma_j (Eq. 13), sigma_j from OLS residuals of joint j.
    With conditioning_ratio the truncated-SVD solution / covariance of Eqs. (19)-(23) is used."""
    pi0 = _tsvd_solve(Y, T, conditioning_ratio)[0]
    r = T - Y @ pi0
    M = Y.shape[0] // n
    p = Y.shape[1]
    sig = np.array([np.linalg.norm(r[j::n]) / np.sqrt(M - p) for j in range(n)])
    w = np.tile(1.0 / sig, M).reshape(-1, 1)
    Yw, Tw = Y * w, T * w
    pi, P, rank = _tsvd_solve(Yw, Tw, conditioning_ratio)
    C, std, s2 = _ls_stats(Yw, Tw, pi, Y.shape[0])
    if conditioning_ratio:
        s2 = float(((Tw - Yw @ pi).T @ (Tw - Yw @ pi))) / (Y.shape[0] - rank)
        C = s2 * P
        std = np.sqrt(np.clip(np.diag(C), 0, None)).reshape(-1, 1)
    return dict(pi=pi, C=C, std=std, rel_std=100 * std / (np.abs(pi) + 1e-10), s2=s2,
                sigma_joint=sig, **excitation_metrics(Y))


def predict(Y, pi):
    return (Y @ pi).ravel()


def rmse_stats(tau, tau_hat, n):
    e = (tau.reshape(-1, n) - tau_hat.reshape(-1, n))
    tau = tau.reshape(-1, n)
    per_joint = np.sqrt(np.mean(e ** 2, axis=0))
    nrmse = per_joint / np.maximum(np.std(tau, axis=0), 1e-12)
    return dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),  # paper definition
                rmse_joint=per_joint.tolist(), nrmse_joint=nrmse.tolist(),
                nrmse_mean=float(np.mean(nrmse)))


# ------------------------------------------------------------------------- parameter maps


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def dyn2reg_np(dyn):
    """Correct parallel-axis map (I_O = I_C + m S(c)^T S(c)); matches thunder get_dyn2reg."""
    out = []
    for k in range(len(dyn) // 10):
        m, c, I = dyn[10 * k], np.asarray(dyn[10 * k + 1:10 * k + 4]), dyn[10 * k + 4:10 * k + 10]
        S = skew(c)
        A = m * S.T @ S
        out += [m, *(m * c), I[0] + A[0, 0], I[1] + A[0, 1], I[2] + A[0, 2],
                I[3] + A[1, 1], I[4] + A[1, 2], I[5] + A[2, 2]]
    return np.array(out)


def reg2dyn_np(reg):
    out = []
    for k in range(len(reg) // 10):
        m, h, I = reg[10 * k], np.asarray(reg[10 * k + 1:10 * k + 4]), reg[10 * k + 4:10 * k + 10]
        c = h / m
        S = skew(c)
        A = m * S.T @ S
        out += [m, *c, I[0] - A[0, 0], I[1] - A[0, 1], I[2] - A[0, 2],
                I[3] - A[1, 1], I[4] - A[1, 2], I[5] - A[2, 2]]
    return np.array(out)


def inertia3(v6):
    return np.array([[v6[0], v6[1], v6[2]], [v6[1], v6[3], v6[4]], [v6[2], v6[4], v6[5]]])


def feasibility_margins(dyn):
    """Per link: mass and min eigenvalue of Sigma_C = tr(I_C)/2 - I_C (>0 <=> triangle inequalities)."""
    rows = []
    for k in range(len(dyn) // 10):
        I = inertia3(dyn[10 * k + 4:10 * k + 10])
        Sig = 0.5 * np.trace(I) * np.eye(3) - I
        if not np.all(np.isfinite(Sig)):
            rows.append(dict(link=k + 1, mass=float(dyn[10 * k]), min_eig_sigma=float("nan"),
                             min_eig_inertia=float("nan"), rel_margin=float("nan"), feasible=False))
            continue
        ev = np.linalg.eigvalsh(Sig)
        rows.append(dict(link=k + 1, mass=float(dyn[10 * k]), min_eig_sigma=float(ev.min()),
                         min_eig_inertia=float(np.linalg.eigvalsh(I).min()),
                         rel_margin=float(ev.min() / max(abs(np.trace(I)), 1e-16)),
                         feasible=bool(dyn[10 * k] > 0 and ev.min() > 0)))
    return rows


# ------------------------------------------------------------------------------ MAP (Eq. 27)


def prior_precision(n, sig_mass, sig_com, sig_inertia, link_w=None, normalize_links=True):
    """Diagonal precision of the DYN prior, replicating solve_dynamics' weight branch.

    The project normalises link weights by their sum (w_link / sum(w_link)); this is reproduced
    with normalize_links=True. Returns a vector of length 10 n.
    """
    lw = np.ones(n) if link_w is None else np.asarray(link_w, float)
    if normalize_links:
        lw = lw / lw.sum()
    per = np.array([1 / sig_mass ** 2] + [1 / sig_com ** 2] * 3 + [1 / sig_inertia ** 2] * 6)
    return np.concatenate([lw[k] * per for k in range(n)])


DYN_NAMES = ["mass", "CoM_x", "CoM_y", "CoM_z", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]   # par_DYN order per link


def relative_prior_sigma(prior_dyn, rel_mass=0.05, sig_com=0.01, rel_inertia=0.25):
    """Per-parameter prior std. dev. [DYN order] scaled to each link:
        mass     rel_mass * m_l                      (5 %: manufacturing, cables, covers)
        CoM      sig_com [m] per coordinate          (1 cm)
        inertia  rel_inertia * ||I_l|| per component (25 % of the link inertia norm, barycentric)"""
    d = np.asarray(prior_dyn, float).reshape(-1, 10)
    sig = np.zeros_like(d)
    sig[:, 0] = rel_mass * np.abs(d[:, 0])
    sig[:, 1:4] = sig_com
    sig[:, 4:10] = rel_inertia * np.linalg.norm(d[:, 4:10], axis=1, keepdims=True)
    return sig.ravel()


def write_prior_yaml(prior_dyn, sigma, path):
    """Prior file for the project's solve_dynamics (identification.prior_yaml, src/templates/prior_template.yaml)."""
    import yaml
    d, s = np.asarray(prior_dyn, float).reshape(-1, 10), np.asarray(sigma, float).reshape(-1, 10)
    links = {f"link{l + 1}": {k: dict(mu=float(d[l, j]), sigma=float(s[l, j])) for j, k in enumerate(DYN_NAMES)}
             for l in range(len(d))}
    with open(path, "w") as f:
        yaml.safe_dump({"robot": {"prior": links}}, f, sort_keys=False)
    return path


def base_weight(C_b, tol=1e-9):
    U, S, Vh = np.linalg.svd(C_b)
    keep = S > tol
    return np.diag(np.sqrt(1.0 / S[keep])) @ Vh[keep, :]


def map_solve(beta, pib_star, C_b, prior_dyn, prec_dyn, x0_reg=None, positive_threshold=1e-10,
              inertia_scale=1.0, mass_bounds=(1e-3, 15.0), w_loss=1.0, ipopt_opts=None,
              numerical_eigen="fallback", n_sphere=2000):
    """Eq. (27): min (pi*-beta pi)^T C^+ (pi*-beta pi) + (pi0-pi_dyn)^T C0^-1 (pi0-pi_dyn)  s.t. feasibility.

    Decision variable: pi_REG (10 n), as in the project. Returns dict with solution and diagnostics.
    numerical_eigen selects the Sigma > 0 constraint (as identification.numerical_eigen in the project):
      False       eig_symbolic(Sigma) >= thr (exact, but NaN when two eigenvalues of Sigma coincide)
      True        v^T Sigma v >= thr on n_sphere Fibonacci-sphere directions (NaN-free, approximate)
      "fallback"  symbolic first; if IPOPT does not succeed, re-solve from the same x0 with the numerical one
    """
    if numerical_eigen == "fallback":
        kw = dict(x0_reg=x0_reg, positive_threshold=positive_threshold, inertia_scale=inertia_scale,
                  mass_bounds=mass_bounds, w_loss=w_loss, ipopt_opts=ipopt_opts, n_sphere=n_sphere)
        sol = map_solve(beta, pib_star, C_b, prior_dyn, prec_dyn, numerical_eigen=False, **kw)
        if sol["success"]:
            return sol
        num = map_solve(beta, pib_star, C_b, prior_dyn, prec_dyn, numerical_eigen=True, **kw)
        num.update(fallback=True, status_symbolic=sol["status"], runtime=sol["runtime"] + num["runtime"])
        return num
    n = len(prior_dyn) // 10
    x = casadi.SX.sym("x", 10 * n)
    dyn = ca_reg2dyn(x)
    L = casadi.DM(base_weight(C_b))
    e = casadi.DM(pib_star.reshape(-1, 1)) - casadi.mtimes(casadi.DM(beta), x)
    f_loss = w_loss * casadi.sumsqr(casadi.mtimes(L, e))
    d = dyn - casadi.DM(prior_dyn)
    f_prior = casadi.sum1(casadi.DM(prec_dyn) * d * d)
    g, lb, ub = [], [], []
    for k in range(n):
        m = dyn[10 * k]
        g.append(m); lb.append(mass_bounds[0]); ub.append(mass_bounds[1])
        v = inertia_scale * dyn[10 * k + 4:10 * k + 10]
        I = casadi.vertcat(casadi.horzcat(v[0], v[1], v[2]), casadi.horzcat(v[1], v[3], v[4]),
                           casadi.horzcat(v[2], v[4], v[5]))
        Sig = 0.5 * casadi.trace(I) * casadi.SX.eye(3) - I
        if numerical_eigen:
            V = casadi.DM(_generate_fibonacci_sphere(n_sphere))
            qf = casadi.sum2(casadi.mtimes(V, Sig) * V)   # v_j^T Sigma v_j, one row per direction
            g.append(qf); lb += [positive_threshold] * n_sphere; ub += [casadi.inf] * n_sphere
        else:
            ev = casadi.eig_symbolic(Sig)
            for j in range(3):
                g.append(ev[j]); lb.append(positive_threshold); ub.append(casadi.inf)
    opts = {"ipopt": {"tol": 1e-4, "constr_viol_tol": 1e-6, "acceptable_tol": 1e-5,
                      "dual_inf_tol": 1e-6, "compl_inf_tol": 1e-6, "max_iter": 50000,
                      "print_level": 0}, "print_time": False}
    if ipopt_opts:
        opts["ipopt"].update(ipopt_opts)
    solver = casadi.nlpsol("map", "ipopt", {"x": x, "f": f_loss + f_prior, "g": casadi.vertcat(*g)}, opts)
    if x0_reg is None:
        x0_reg = dyn2reg_np(prior_dyn)
    t0 = time.time()
    sol = solver(x0=x0_reg, lbg=lb, ubg=ub)
    rt = time.time() - t0
    xs = np.array(sol["x"]).ravel()
    fl = casadi.Function("fl", [x], [f_loss])
    fp = casadi.Function("fp", [x], [f_prior])
    gv = np.array(sol["g"]).ravel()
    viol = np.maximum(np.array(lb) - gv, 0).max()
    return dict(reg=xs, dyn=reg2dyn_np(xs), cost=float(sol["f"]), f_loss=float(fl(xs)),
                f_prior=float(fp(xs)), status=solver.stats()["return_status"],
                success=bool(solver.stats()["success"]), runtime=rt, max_constr_viol=float(viol),
                iters=int(solver.stats()["iter_count"]), eig_method="numerical" if numerical_eigen else "symbolic",
                fallback=False)


def map_objective(beta, pib_star, C_b, prior_dyn, prec_dyn, reg, w_loss=1.0):
    L = base_weight(C_b)
    e = pib_star.ravel() - beta @ reg
    d = reg2dyn_np(reg) - prior_dyn
    return float(w_loss * np.sum((L @ e) ** 2) + np.sum(prec_dyn * d * d))


# ---------------------------------------------------------------------------- convex variants


def dyn2reg_jacobian(prior_dyn):
    xs = casadi.SX.sym("d", len(prior_dyn))
    n = len(prior_dyn) // 10
    out = []
    for k in range(n):
        m = xs[10 * k]; c = xs[10 * k + 1:10 * k + 4]; I = xs[10 * k + 4:10 * k + 10]
        S = casadi.vertcat(casadi.horzcat(0, -c[2], c[1]), casadi.horzcat(c[2], 0, -c[0]),
                           casadi.horzcat(-c[1], c[0], 0))
        A = m * casadi.mtimes(S.T, S)
        out += [m, m * c[0], m * c[1], m * c[2], I[0] + A[0, 0], I[1] + A[0, 1], I[2] + A[0, 2],
                I[3] + A[1, 1], I[4] + A[1, 2], I[5] + A[2, 2]]
    J = casadi.Function("J", [xs], [casadi.jacobian(casadi.vertcat(*out), xs)])
    return np.array(J(prior_dyn))


def _pseudo_inertia_cp(cp, x, k):
    m = x[10 * k]; h = x[10 * k + 1:10 * k + 4]; I = x[10 * k + 4:10 * k + 10]
    Io = cp.bmat([[I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]]])
    Sig = 0.5 * cp.trace(Io) * np.eye(3) - Io
    hcol = cp.reshape(h, (3, 1), order="C")
    return cp.bmat([[Sig, hcol], [hcol.T, cp.reshape(m, (1, 1), order="C")]])


def map_convex(beta, pib_star, C_b, prior_dyn, prec_dyn, eps=1e-10, solver="CLARABEL"):
    """Convex surrogate of Eq. (27): prior mapped to regressor coordinates by linearisation
    (C0_REG = J C0 J^T), feasibility via pseudo-inertia LMI J_k >= eps I (equivalent to m>0 and
    triangle inequalities). Globally optimal for this surrogate."""
    import cvxpy as cp

    n = len(prior_dyn) // 10
    J = dyn2reg_jacobian(prior_dyn)
    C0 = np.diag(1.0 / prec_dyn)
    C0r = J @ C0 @ J.T
    Lp = np.linalg.cholesky(np.linalg.inv(C0r + 1e-30 * np.eye(len(C0r))))
    reg0 = dyn2reg_np(prior_dyn)
    s = np.sqrt(np.diag(C0r))  # variable scaling
    z = cp.Variable(10 * n)
    x = cp.multiply(s, z) + reg0
    L = base_weight(C_b)
    obj = cp.sum_squares(L @ (pib_star.ravel() - beta @ x)) + cp.sum_squares(Lp.T @ (x - reg0))
    cons = [_pseudo_inertia_cp(cp, x, k) >> eps * np.eye(4) for k in range(n)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.time()
    prob.solve(solver=solver)
    reg = np.asarray(x.value).ravel()
    return dict(reg=reg, dyn=reg2dyn_np(reg), status=prob.status, runtime=time.time() - t0,
                cost_surrogate=float(prob.value))


def qcp_scalar(Yb, T, beta, nfric, prior_dyn, lam, eps=1e-10, solver="CLARABEL"):
    """Scalar-weight QCP baseline (torque residual + lam * ||pi_REG - pi0_REG||^2, LMI constraints).
    Yb = [Yr_red | reg_dl]; the friction parameters are free."""
    import cvxpy as cp

    n = len(prior_dyn) // 10
    reg0 = dyn2reg_np(prior_dyn)
    G = Yb.T @ Yb
    R = np.linalg.cholesky(G + 1e-12 * np.trace(G) / G.shape[0] * np.eye(G.shape[0])).T
    b = np.linalg.solve(R.T, Yb.T @ T).ravel()
    x = cp.Variable(10 * n)
    f = cp.Variable(nfric)
    pib = cp.hstack([beta @ x, f])
    obj = cp.sum_squares(R @ pib - b) + lam * cp.sum_squares(x - reg0)
    cons = [_pseudo_inertia_cp(cp, x, k) >> eps * np.eye(4) for k in range(n)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver)
    reg = np.asarray(x.value).ravel()
    return dict(reg=reg, dyn=reg2dyn_np(reg), fric=np.asarray(f.value).ravel(), status=prob.status)
