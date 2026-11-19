"""
utilities function for computation of the idientification process.
"""

import os
import datetime
import numpy as np
import yaml

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

import casadi
from utils.casadi_utils import *
from utils.loader_utils import *
from typing import Any, Optional

def print_base_inertial_parameters(robot:Any) -> dict:
    """
    Display the Base inertial Parameters of a robot model by name.

    Args:
        robot (Any): robot model imported by user build with thunder dynamic tool
                     (e.g., thunder_franka).

    Returns:
        dict: A dictionary mapping row indices to lists of formatted parameter strings.
    """
    param_names = ['M','MX','MY','MZ','XX','XY','XZ','YY','YZ','ZZ','Ia']

    beta = np.round(robot.get_beta(), 10)
    n = robot.numJoints
    parPerLink = robot.STD_PAR_LINK
    description = ""
    description_dict = {}

    for i in range(beta.shape[0]):
        line_terms = []
        for j in range(beta.shape[1]):
            coeff = beta[i, j]
            if coeff == 0:
                continue
            # get param name
            if j >= n * parPerLink:
                p_name = f"{param_names[10]}_{j - n * parPerLink}"
            else:
                p_name = f"{param_names[j % parPerLink]}_{j // parPerLink}"

            # Formatting the term
            if abs(coeff) == 1.0:
                formatted_term = p_name
            else:
                formatted_term = f"{abs(coeff):.2e} * {p_name}"
            if not line_terms:
                prefix = "-" if coeff < 0 else ""
                line_terms.append(f"{prefix}{formatted_term}")
            else:
                op = " - " if coeff < 0 else " + "
                line_terms.append(f"{op}{formatted_term}")

        description += "".join(line_terms) + "\n"
        description_dict[f"{i}"] = line_terms

    print(description)
    return description_dict

def get_big_Y_Tau(robot:Any, traject:TrajectoryManager) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """
    Compute the stacked regressor matrix and measurement vector given an exciting trajectory
    such that: TTau = YY * pi

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        trajec (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.

    Returns:
        tuple[np.ndarray, np.ndarray, list[bool]]: A tuple containing:
            - YY (np.ndarray): The stacked regressor matrix of shape (N * numJoints, numParams).
            - TTau (np.ndarray): The stacked torque measurement vector of shape (N * numJoints, 1).
            - mask (list[bool]): A boolean index list indicating which base inertial parameters
              are sufficiently excited (observable) across the given trajectory.
    """
    # YY is oredred for multiplication with par =  [par_reg_red', friction_par']'
    Y_log = []
    for i in range(traject.t.shape[0]):
        q = traject.q[i,:]
        dq = traject.qd[i,:]
        ddq = traject.qdd[i,:]

        robot.set_q(q)
        robot.set_dq(dq)
        robot.set_ddq(ddq)
        robot.set_dqr(dq)
        robot.set_ddqr(ddq)

        # Full inverse-dynamics regressor (per-joint rows)
        Y_red = robot.get_Yr_red()
        Y_dl = robot.get_reg_dl()
        Y = np.hstack([Y_red, Y_dl])
        Y_log.append(Y)

    # Stack regressors and torques into the big LS problem
    M = len(traject.t)
    n = robot.numJoints
    YY = np.concatenate([np.asarray(Y) for Y in Y_log], axis=0)  # (M*n) x p
    TTau = np.concatenate([np.asarray(t).reshape(n, 1) for t in traject.tau], axis=0)  # (M*n) x 1

    observable_parameters_mask = [1 for _ in range(YY.shape[1])]

    if np.linalg.matrix_rank(YY) < YY.shape[1]:
        print(f"[WARNING] Matrix is not full colum rank {np.linalg.matrix_rank(YY)}/{YY.shape[1]} - the solution is not unique" )
        for i in range(YY.shape[1]):
            if np.linalg.norm(YY[:,i]) < 1e-10:
                # Check for zero-column
                print(f"parameters {i} of the base is non-observable")
                observable_parameters_mask[i] = 0

    return YY, TTau, observable_parameters_mask

def solve_OLS(robot: Any, traject: TrajectoryManager, conditioning_ratio: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """
    Compute the estimation of base inertial parameters via Ordinary Least Squares (OLS)
    using the stacked regressor matrix and torque measurements:
        pi_OLS = pinv(Y) @ tau = (Y.T @ Y)^-1 @ Y.T @ tau

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.
        conditioning_ratio (float, optional): Threshold ratio for SVD truncation during the 
                                              pseudo-inverse calculation. Singular values smaller 
                                              than this ratio multiplied by the largest singular 
                                              value will be discarded to mitigate ill-conditioning. 
                                              Defaults to None (standard pseudo-inverse).

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - hat_pi (np.ndarray): OLS estimate vector of the identifiable base inertial parameters.
            - metrics (dict): A dictionary containing quality metrics of the estimation 
                              (e.g., condition number, standard deviation, residual error).
    """
    # > Compute pi_OLS = (Y.t @ Y)^-1 @ Y.t @ tau
    YY, TTau, observable_mask = get_big_Y_Tau(robot, traject)
    p = YY.shape[1]     # all values
    r = p
    if conditioning_ratio is not None:
        Sinv_trunc = np.zeros((p,p))
        U,S,Ut = np.linalg.svd(YY.T @ YY,  compute_uv = True)
        sigma_max = S[0]
        for i in range(1,YY.shape[1]):
            sigma_p = S[i]
            cond = np.sqrt(sigma_max/sigma_p)
            if cond > conditioning_ratio:
                print(f"OLS: From now on (index {i}), high ill-conditioning")
                r = i
                break


        Sinv_trunc[:r,:r] = np.diag(1.0 / S[:r])
        Sinv_trunc[r:,r:] = 0
        pseudo_inv = U @ Sinv_trunc @ Ut
    else:
        pseudo_inv =  np.linalg.pinv(YY.T @ YY)

    hat_pi = pseudo_inv @ YY.T @ TTau
    # > Compute Solution metrics
    metrics = {}
    metrics['observable base parameters mask'] = observable_mask
    metrics['conditioning number'] = np.linalg.cond(YY)
    metrics['sigma max'] = np.sqrt(np.max(np.linalg.eigvals(YY.T@YY)))
    metrics['trace'] = np.trace(YY.T@YY)

    residual = TTau - YY @ hat_pi
    metrics['residual'] = np.linalg.norm(residual)

    # std.dev. of residual error
    n = YY.shape[0]
    sigma_w_2 = np.linalg.norm((TTau - YY@hat_pi),ord=2)**2 / (
        n - r
    )
    metrics['error standard deviation'] = sigma_w_2

    # covariance matrix of estimated parameters
    C_w = np.round(sigma_w_2 * pseudo_inv, decimals=20)
    metrics['parameters covariance matrix'] = C_w

    # relative std.dev. of estimated parameters
    sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i] + 1e-10) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    metrics['parameters standard deviation'] = sigma_pi
    metrics['parameters relative standard deviation'] = sigma_pi_perc


    return hat_pi, metrics

def solve_OLS_with_prior(robot:Any, traject:TrajectoryManager, conditioning_ratio:Optional[float] = None) -> tuple[np.ndarray, dict]:
    """
    Compute estimation of base inertial parameters via Ordinary Least Squares and using prior 
        pi_OLSprior = pi_prior + (Y.t @ Y)^-1 @ Y.t @ (tau - Y pi_prior)
    
    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.
        conditioning_ratio (float, optional): Threshold ratio for SVD truncation during the 
                                              pseudo-inverse calculation. Singular values smaller 
                                              than this ratio multiplied by the largest singular 
                                              value will be discarded to mitigate ill-conditioning. 
                                              Defaults to None (standard pseudo-inverse).

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - hat_pi (np.ndarray): OLS estimate vector of the identifiable base inertial parameters.
            - metrics (dict): A dictionary containing quality metrics of the estimation 
                              (e.g., condition number, standard deviation, residual error).
    """
    hat_pi_ref = np.hstack([robot.get_par_REG_red(), robot.get_par_Dl()]).reshape(-1,1)
    YY, TTau, observable_mask = get_big_Y_Tau(robot, traject)
    p = YY.shape[1]     # all values
    r = p
    if conditioning_ratio is not None:
        Sinv_trunc = np.zeros((p,p))
        U,S,Ut = np.linalg.svd(YY.T @ YY,  compute_uv = True)
        sigma_max = S[0]
        for i in range(1,YY.shape[1]):
            sigma_p = S[i]
            cond = np.sqrt(sigma_max/sigma_p)
            if cond > conditioning_ratio:
                print(f"OLS: From now on (index {i}), high ill-conditioning")
                r = i
                break
        Sinv_trunc[:r,:r] = np.diag(1.0 / S[:r])
        Sinv_trunc[r:,r:] = 0
        pseudo_inv = U @ Sinv_trunc @ Ut
    else:
        pseudo_inv =  np.linalg.pinv(YY.T @ YY)

    hat_pi = hat_pi_ref + pseudo_inv @ YY.T @ (TTau - YY @ hat_pi_ref)

    # > Compute metrics
    metrics = {}
    metrics['observable base parameters mask'] = observable_mask
    metrics['conditioning number'] = np.linalg.cond(YY)
    metrics['sigma max'] = np.sqrt(np.max(np.linalg.eigvals(YY.T@YY)))
    metrics['trace'] = np.trace(YY.T@YY)


    residual = TTau - YY @ hat_pi
    metrics['residual'] = np.linalg.norm(residual)

    # std.dev. of residual error
    n = YY.shape[0]
    sigma_w_2 = np.linalg.norm((TTau - YY@hat_pi),ord=2)**2 / (
        n - r
    )
    metrics['error standard deviation'] = sigma_w_2

    # covariance matrix of estimated parameters
    C_w = np.round(sigma_w_2 * pseudo_inv, decimals=20)
    metrics['parameters covariance matrix'] = C_w

    # relative std.dev. of estimated parameters
    sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i] + 1e-10) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    metrics['parameters standard deviation'] = sigma_pi
    metrics['parameters relative standard deviation'] = sigma_pi_perc

    return hat_pi, metrics

def solve_WLS(robot:Any, traject:TrajectoryManager, conditioning_ratio:Optional[float] = None) -> tuple[np.ndarray, dict]:
    """
    Compute estimation of base inertial parameters via weighted Least Squares
    Each joint is weighted differently with 
        Z = blkdiag( diag(rho1, ..., rhon), diag(rho1, ..., rhon), ..., diag(rho1, ..., rhon)) in NbxNb

        pi_WLS = (Y.t @ Z.t @ Z @ Y)^-1 @ Y.t @ Z.t @ tau
    
    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.
        conditioning_ratio (float, optional): Threshold ratio for SVD truncation during the 
                                              pseudo-inverse calculation. Singular values smaller 
                                              than this ratio multiplied by the largest singular 
                                              value will be discarded to mitigate ill-conditioning. 
                                              Defaults to None (standard pseudo-inverse).

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - hat_pi (np.ndarray): OLS estimate vector of the identifiable base inertial parameters.
            - metrics (dict): A dictionary containing quality metrics of the estimation 
                              (e.g., condition number, standard deviation, residual error).
    """
    # OLS solution for residuals (weight computation)
    YY, TTau, observable_mask = get_big_Y_Tau(robot, traject)
    hat_pi, _ = solve_OLS(robot, traject, conditioning_ratio=conditioning_ratio)
    # computation of weighting matrix
    sigma_w_2 = np.zeros(robot.numJoints)   # effort measurements noise std.dev. at joint level
    nb_samples = YY.shape[0]            # num of samples
    p = YY.shape[1]                     # num of parameters
    # W_diag = np.zeros(nb_samples*robot.numJoints)

    # pre-compute all effort residual
    residual = TTau - YY @ hat_pi

    # std.dev for joint i-th measurements using OLS residuals
    for i in range(robot.numJoints):
        diff = residual[i::robot.numJoints]
        denom = nb_samples - p
        sigma_w_2[i] = np.sqrt(np.linalg.norm(diff)**2 / denom)

        # # compute weight (inverse of std.dev) of joint i-th
        # for j in range(nb_samples):
        #     W_diag[j + i*robot.numJoints] = 1 / sigma_w_2[i]

    # compute weights multiplication for each joint
    YY_weighted = np.zeros_like(YY)
    TTau_weighted = np.zeros_like(TTau)
    for i in range(robot.numJoints):
        # Previous  BUG (previous implementation): W_diag[i] equals 1/sigma_w_2[0] for every i, so all joints got the
        # same weight and WLS returned exactly the OLS estimate and covariance.
        # YY_weighted[i::robot.numJoints,:] = W_diag[i]*YY[i::robot.numJoints, :] # apply weight for i-th raw
        # TTau_weighted[i::robot.numJoints] = W_diag[i]*TTau[i::robot.numJoints]  # apply weight for i-th value
        w_i = 1 / sigma_w_2[i]          # weight (inverse of std.dev) of joint i-th
        YY_weighted[i::robot.numJoints,:] = w_i*YY[i::robot.numJoints, :]   # apply weight for i-th raw
        TTau_weighted[i::robot.numJoints] = w_i*TTau[i::robot.numJoints]    # apply weight for i-th value

    # Compute inverted matrix and solution
    r = p                # index of last singular value
    if conditioning_ratio is not None:
        Sinv_trunc = np.zeros((p,p))
        U,S,Ut = np.linalg.svd(YY_weighted.T @ YY_weighted,  compute_uv = True)
        sigma_max = S[0]
        for i in range(1,YY_weighted.shape[1]):
            sigma_p = S[i]
            cond = np.sqrt(sigma_max/sigma_p)
            if cond > conditioning_ratio:
                print(f"WLS: From now on (index {i}), high ill-conditioning")
                r = i
                break

        Sinv_trunc[:r,:r] = np.diag(1.0 / S[:r])
        Sinv_trunc[r:,r:] = 0.0
        pseudo_inv = U @ Sinv_trunc @ Ut
    else:
        pseudo_inv = np.linalg.pinv(YY_weighted.T @ YY_weighted)

    hat_pi = pseudo_inv @ YY_weighted.T @ TTau_weighted

    # > Compute metrics
    metrics = {}
    metrics['observable base parameters mask'] = observable_mask
    metrics['conditioning number'] = np.linalg.cond(YY)
    metrics['sigma max'] = np.sqrt(np.max(np.linalg.eigvals(YY.T@YY)))
    metrics['trace'] = np.trace(YY.T@YY)


    residual = TTau_weighted - YY_weighted @ hat_pi
    metrics['residual'] = np.linalg.norm(residual)

    # std.dev. of residual error
    n = YY.shape[0]
    sigma_w_2 = np.linalg.norm((TTau_weighted - YY_weighted@hat_pi),ord=2)**2 / (
        n - r
    )
    metrics['error standard deviation'] = sigma_w_2

    C_w = np.round(sigma_w_2 * pseudo_inv, decimals=20)
    metrics['parameters covariance matrix'] = C_w

    # relative std.dev. of estimated parameters
    sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i] + 1e-10) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    metrics['parameters standard deviation'] = sigma_pi
    metrics['parameters relative standard deviation'] = sigma_pi_perc

    return hat_pi, metrics

def solve_WLS_with_prior(robot:Any, traject:TrajectoryManager, conditioning_ratio:Optional[float] = None) -> tuple[np.ndarray, dict]:
    """
    Compute estimation of base inertial parameters via weighted Least Squares and using prior
    Each joint is weighted differently with 
        Z = blkdiag( diag(rho1, ..., rhon), diag(rho1, ..., rhon), ..., diag(rho1, ..., rhon)) in NbxNb

        pi_WLSprior = pi_prior + (Y.t @ Z.t @ Z @ Y)^-1 @ Y.t @ Z.t @ (tau - Y pi_prior)
    
    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.
        conditioning_ratio (float, optional): Threshold ratio for SVD truncation during the 
                                              pseudo-inverse calculation. Singular values smaller 
                                              than this ratio multiplied by the largest singular 
                                              value will be discarded to mitigate ill-conditioning. 
                                              Defaults to None (standard pseudo-inverse).

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - hat_pi (np.ndarray): OLS estimate vector of the identifiable base inertial parameters.
            - metrics (dict): A dictionary containing quality metrics of the estimation 
                              (e.g., condition number, standard deviation, residual error).
    """

    # OLS solution for residuals  (weight computation)
    hat_pi_ref = np.hstack([robot.get_par_REG_red(),robot.get_par_Dl()]).reshape(-1,1)
    YY, TTau, observable_mask = get_big_Y_Tau(robot, traject)
    hat_pi, _ = solve_OLS_with_prior(robot, traject, conditioning_ratio=conditioning_ratio)


    # computation of weighting matrix
    sigma_w_2 = np.zeros(robot.numJoints)   # effort measurements noise std.dev. at joint level
    nb_samples = YY.shape[0]            # num of samples
    p = YY.shape[1]                     # num of parameters
    # W_diag = np.zeros(nb_samples*robot.numJoints)

    residual = TTau - YY @ hat_pi
    for i in range(robot.numJoints):
        # std.dev for joint i-th measuerements using OLS residuals
        diff = residual[i::robot.numJoints]
        denom = nb_samples - p
        sigma_w_2[i] = np.sqrt(np.linalg.norm(diff)**2 / denom)

        # # assign inverse for all samples of joint i-th
        # for j in range(nb_samples):
        #     W_diag[j + i*robot.numJoints] = 1 / sigma_w_2[i]

    # compute weights multiplication for each joint
    YY_weighted = np.zeros_like(YY)
    TTau_weighted = np.zeros_like(TTau)
    for i in range(robot.numJoints):
        # BUG (previous implementation): W_diag[i] equals 1/sigma_w_2[0] for every i, so all joints got the
        # same weight and WLS returned exactly the OLS estimate and covariance.
        # YY_weighted[i::robot.numJoints,:] = W_diag[i]*YY[i::robot.numJoints, :] # apply weight for i-th raw
        # TTau_weighted[i::robot.numJoints] = W_diag[i]*TTau[i::robot.numJoints]  # apply weight for i-th value
        w_i = 1 / sigma_w_2[i]          # weight (inverse of std.dev) of joint i-th
        YY_weighted[i::robot.numJoints,:] = w_i*YY[i::robot.numJoints, :]   # apply weight for i-th raw
        TTau_weighted[i::robot.numJoints] = w_i*TTau[i::robot.numJoints]    # apply weight for i-th value

    # Compute inverted matrix and solution
    r = p                 # num of parameters
    if conditioning_ratio is not None:
        Sinv_trunc = np.zeros((p,p))
        U,S,Ut = np.linalg.svd(YY_weighted.T @ YY_weighted,  compute_uv = True)
        sigma_max = S[0]
        print(sigma_max)
        for i in range(YY_weighted.shape[1]):
            sigma_p = S[i]
            cond = np.sqrt(sigma_max/sigma_p)
            if cond > conditioning_ratio:
                print(f"WLS: From now on (index {i}), high ill-conditioning")
                r = i
                break
        Sinv_trunc[:r,:r] = np.diag(1.0 / S[:r])
        Sinv_trunc[r:,r:] = 0
        pseudo_inv = U @ Sinv_trunc @ Ut
    else:
        pseudo_inv = np.linalg.pinv(YY_weighted.T @ YY_weighted)
    hat_pi = hat_pi_ref + pseudo_inv @ YY_weighted.T @ (TTau_weighted - YY_weighted @ hat_pi_ref)

    # > Compute metrics
    metrics = {}
    metrics['observable base parameters mask'] = observable_mask
    metrics['conditioning number'] = np.linalg.cond(YY)
    metrics['sigma max'] = np.sqrt(np.max(np.linalg.eigvals(YY.T@YY)))
    metrics['trace'] = np.trace(YY.T@YY)


    residual = TTau_weighted - YY_weighted @ hat_pi
    metrics['residual'] = np.linalg.norm(residual)

    # std.dev. of residual error
    n = YY.shape[0]
    sigma_w_2 = np.linalg.norm((TTau_weighted - YY_weighted@hat_pi),ord=2)**2 / (
        n - r
    )
    metrics['error standard deviation'] = sigma_w_2

    # covariance matrix of estimated parameters
    C_w = np.round(sigma_w_2 * pseudo_inv, decimals=20)
    metrics['parameters covariance matrix'] = C_w

    # relative std.dev. of estimated parameters
    sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i] + 1e-10) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    metrics['parameters standard deviation'] = sigma_pi
    metrics['parameters relative standard deviation'] = sigma_pi_perc

    return hat_pi, metrics


def compute_essential(robot:Any, traject:TrajectoryManager, ratio_essential:float = 30) -> tuple[np.ndarray, list[int]]:
    """
    Compute the essential base inertial parameters iteratively. 

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka, thunder_ahand_finger).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                    velocities (dq), accelerations (ddq), and measured
                                    torques (tau) over time.
        ratio_essential (float, optional): Percentile or energy threshold ratio used during the 
                                           iterative pruning process to classify a parameter 
                                           as "essential". Defaults to 30.0.

    Returns:
        tuple[np.ndarray, list[int]]: A tuple containing:
            - hat_pi_essential (np.ndarray): Estimated value vector of the remaining essential parameters.
            - indices_essential (list[int]): A list containing the indices of base parameters that are also essential.
    """

    YY, TTau = get_big_Y_Tau(robot, traject)
    mask_essential = np.ones((YY.shape[1]), dtype=bool) # mask of essential parameters
    indeces_essential = [i for i in range(YY.shape[1]) if mask_essential[i]]

    #
    print(f"Computing essential parameters with ratio {ratio_essential} between max and min relative std.dev.")
    hat_pi = np.linalg.pinv(YY) @ TTau
    n = YY.shape[0]
    p = hat_pi.shape[0]
    sigma_w_2 = np.linalg.norm((TTau - np.dot(YY, hat_pi)),ord=2)**2 / (
        n - p
    )
    C_w = sigma_w_2 * np.linalg.inv(np.dot(YY.T, YY))
    sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
    s_max, idx_max = np.max(sigma_pi_perc), np.argmax(sigma_pi_perc)
    s_min, idx_min = np.min(sigma_pi_perc), np.argmin(sigma_pi_perc)

    while s_max/s_min > ratio_essential:
        index_to_remove = indeces_essential[idx_max]
        mask_essential[index_to_remove] = False
        indeces_essential = [i for i in range(YY.shape[1]) if mask_essential[i]]

        YY_e = YY[:, indeces_essential]

        hat_pi = np.linalg.pinv(YY_e) @ TTau
        n = YY_e.shape[0]
        p = hat_pi.shape[0]
        sigma_w_2 = np.linalg.norm((TTau - np.dot(YY_e, hat_pi)),ord=2)**2 / (
            n - p
        )
        C_w = sigma_w_2 * np.linalg.inv(np.dot(YY_e.T, YY_e))
        sigma_pi = np.array([np.sqrt(C_w[i,i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
        sigma_pi_perc = np.array([100*sigma_pi[i]/np.abs(hat_pi[i]) for i in range(hat_pi.shape[0])]).reshape(hat_pi.shape)
        s_max, idx_max = np.max(sigma_pi_perc), np.argmax(sigma_pi_perc)
        s_min, idx_min = np.min(sigma_pi_perc), np.argmin(sigma_pi_perc)

    hat_pi_essential = hat_pi

    return hat_pi_essential, indeces_essential

def solve_dynamics(robot:Any, config:dict, sigma_pi: Optional[np.ndarray] = None, opts:Optional[dict] = None, export_file:bool = False, path:Optional[str] = None) -> tuple:
    """
    Estimate the full robot dynamic via MLE approach

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka, thunder_ahand_finger).
        config (dict): MLE optimization configuration dictionary containing weights and loss parameters.
        sigma_pi (np.ndarray, optional): Covariance matrix of the base inertial parameters. Defaults to None, e.g. identity weighting.
        opts (dict, optional): CasADi solver configuration and option tuning parameters. Defaults to None.
        export_file (bool, optional): Whether the evaluation and optimization results should be exported to disk. Defaults to False.
        path (str, optional): Absolute or relative system path where exported results files will be saved. Defaults to None.

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - hat_par_REG_optim (np.ndarray): Estimated parameters via MLE.
            - results (dict): A dictionary containing optimization loss value.

    Note:
        The target robot model instance must pre-specify a valid base parameter initialization. 
        Specifically, `robot.set_par_REG_red(...)` must be executed prior to calling this optimization function 
        to bind baseline values estimated via OLS. All newly optimized MLE results are updated inside the `robot` instance automatically.

    """

    # Load solution parameters
    POSITIVE_THRESH = float(config['identification'].get('positive_threshold',1e-16))
    INERTIA_SCALE_FACTOR = float(config['identification'].get('inertia_scale_factor', 1))
    NUMERICAL_EIGEN = float(config['identification'].get('numerical_eigen', False))

    # Load dynamic parameters prior from robot
    par_per_link = robot.STD_PAR_LINK

    # Standard Parameters
    FULL_DYN_PARAM_INITIAL_GUESS = {}
    FULL_DYN_PARAM_INITIAL_GUESS['mass']  = robot.get_par_DYN()[0::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['CoM_x'] = robot.get_par_DYN()[1::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['CoM_y'] = robot.get_par_DYN()[2::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['CoM_z'] = robot.get_par_DYN()[3::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Ixx']   = robot.get_par_DYN()[4::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Ixy']   = robot.get_par_DYN()[5::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Ixz']   = robot.get_par_DYN()[6::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Iyy']   = robot.get_par_DYN()[7::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Iyz']   = robot.get_par_DYN()[8::10].copy()
    FULL_DYN_PARAM_INITIAL_GUESS['Izz']   = robot.get_par_DYN()[9::10].copy()
    # Motor inwertia
    if hasattr(robot,"get_par_Ia"):
        par_per_link += 1
        FULL_DYN_PARAM_INITIAL_GUESS['Ia']   = robot.get_par_Ia().copy()
    else:
        print(f"[INFO]: the model used does not include motor inertia" )


    # init guess on urdf values for inertial values. Convert from dynamic to regressor
    n = robot.numJoints
    hat_par_REG_red_star = robot.get_par_REG_red().copy() # Extract reduced estimated base inertial parameters via LS 

    # store prior
    hat_par_DYN_0 = np.zeros(n*par_per_link)    # Set initial guess for full regressor parameters 
    hat_par_REG_0 = np.zeros(n*par_per_link)
    for i in range(n):
        hat_par_DYN_0[i*10]   = FULL_DYN_PARAM_INITIAL_GUESS['mass'][i]
        hat_par_DYN_0[i*10+1] = FULL_DYN_PARAM_INITIAL_GUESS['CoM_x'][i]
        hat_par_DYN_0[i*10+2] = FULL_DYN_PARAM_INITIAL_GUESS['CoM_y'][i]
        hat_par_DYN_0[i*10+3] = FULL_DYN_PARAM_INITIAL_GUESS['CoM_z'][i]
        hat_par_DYN_0[i*10+4] = FULL_DYN_PARAM_INITIAL_GUESS['Ixx'][i]
        hat_par_DYN_0[i*10+5] = FULL_DYN_PARAM_INITIAL_GUESS['Ixy'][i]
        hat_par_DYN_0[i*10+6] = FULL_DYN_PARAM_INITIAL_GUESS['Ixz'][i]
        hat_par_DYN_0[i*10+7] = FULL_DYN_PARAM_INITIAL_GUESS['Iyy'][i]
        hat_par_DYN_0[i*10+8] = FULL_DYN_PARAM_INITIAL_GUESS['Iyz'][i]
        hat_par_DYN_0[i*10+9] = FULL_DYN_PARAM_INITIAL_GUESS['Izz'][i]
        if par_per_link == 11:
            hat_par_DYN_0[n*10+i] = FULL_DYN_PARAM_INITIAL_GUESS['Ia'][i]


    # convert CAD parameters into regressor and base parameters
    robot.set_par_DYN(hat_par_DYN_0[:n*10])
    hat_par_REG_0[:len(robot.get_dyn2reg())] = robot.get_dyn2reg()                          # regressor
    if (par_per_link == 11): hat_par_REG_0[len(robot.get_dyn2reg()):] = robot.get_par_Ia()  # motor inertia

    # Define optimization problem CasADi variables
    hat_par_REG_sym = casadi.SX.sym('Pi', n*par_per_link)    # Parameters to estimate (dyn + motor inertia)
    hat_par_DYN_sym = reg2dyn(hat_par_REG_sym)               # CasADi conversion


    # - Feasability Costraint
    penalty_costraint = 0
    g = []          # Constraints
    ub = []         # Upper bound
    lb = []         # Lower bound
    d = casadi.SX(n,5)           # determinant
    tr = casadi.SX(n,1)          # trace
    l_max = casadi.SX(n,1)       # max_eigen
    eig = casadi.SX(n,3)       # all_eigen
    for i in range(n):
        m = hat_par_DYN_sym[i*10]                     # mass

        # Mass must be positive
        g += [m]             # [Kg]
        lb += [1e-3]
        ub += [15]

        # Center of mass
        c = casadi.SX(3,1)
        c[0,0] = hat_par_DYN_sym[i*10 + 1]
        c[1,0] = hat_par_DYN_sym[i*10 + 2]
        c[2,0] = hat_par_DYN_sym[i*10 + 3]

        # positive definite inertia matrix
        Ib = casadi.SX(3,3)
        Ib[0,0] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 4]
        Ib[0,1] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 5]
        Ib[0,2] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 6]
        Ib[1,0] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 5]
        Ib[1,1] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 7]
        Ib[1,2] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 8]
        Ib[2,0] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 6]
        Ib[2,1] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 8]
        Ib[2,2] = INERTIA_SCALE_FACTOR*hat_par_DYN_sym[i*10 + 9]


        # Positive Pseudo-Definite Matrix constraint
        Sigma = 0.5*casadi.trace(Ib)*np.eye(3) - Ib
        P = casadi.vertcat(
                 casadi.horzcat(m,                  np.zeros((1, 3))),
                 casadi.horzcat(np.zeros((3, 1)),               Sigma)
            )


        if NUMERICAL_EIGEN:
            # Use numerical approximetion to asses def. pos. of P
            v_array = _generate_fibonacci_sphere(2000)
            for j in range(2000):
                v = v_array[j,:].reshape(-1,1)
                g += [casadi.mtimes(casadi.mtimes(v.T, P[1:,1:]), v)]
                lb += [POSITIVE_THRESH]
                ub += [casadi.inf]
        else:
            # Use symbolic computation of eigenvalues to asses def. pos. of P
            eig[i,:] = casadi.eig_symbolic(P[1:,1:])
            g +=  [eig[i,0], eig[i,1], eig[i,2]]
            lb += [POSITIVE_THRESH, POSITIVE_THRESH, POSITIVE_THRESH]
            ub += [casadi.inf, casadi.inf, casadi.inf]


        # build penalty cost function to use in SA method
        tr[i] = casadi.trace(Ib)
        l_max[i] = -minimum(-casadi.eig_symbolic(Ib))
        penalty_costraint += -minimum([0, (tr[i]/2 - l_max[i] - POSITIVE_THRESH) ], n=2)

        # Positive motor inertias
        if par_per_link > 10:
            Ia = hat_par_REG_sym[n*10 + i]
            g +=  [Ia]
            lb += [POSITIVE_THRESH]
            ub += [1.0]

    # - Define Optimization Cost
    # -- Residual in the base parameters
    w_loss = 1
    sigma_pi_inv = 0
    beta = np.round(robot.get_beta(), 16)

    if sigma_pi is not None and sigma_pi.shape[0] == sigma_pi.shape[1]:
        # Use covariance matrix of base parameters
        C_w = sigma_pi[:-n*2, :-n*2]
        U, S, Vh = np.linalg.svd(C_w)
        tol = 1e-9 # tolerance for truncated invertion
        valid_modes = S > tol

        # Compute residual as (Le)'(Le)
        S_inv_sqrt = np.sqrt(1.0 / S[valid_modes])
        L = casadi.DM(np.diag(S_inv_sqrt) @ Vh[valid_modes, :])
        e = hat_par_REG_red_star - casadi.mtimes(beta, hat_par_REG_sym)
        weighted_residual = casadi.mtimes(L, e)
        f_loss = casadi.dot(weighted_residual, weighted_residual)

    elif sigma_pi is not None and sigma_pi.shape[0] != sigma_pi.shape[1]:
        # Weight with inverse of relative std.dev. of base parameters
        print("[WARN] Provided covariance matrix is not squared. Treating as relative std.deviation of base parameters")
        sigma_pi_inv = np.diag(1/sigma_pi[:-n*2].flatten())
        W_sigma_inv = sigma_pi_inv/np.trace(sigma_pi_inv) # sigma_pi is parameters relative std.dev
        f_loss = (hat_par_REG_red_star - beta@hat_par_REG_sym).T @ W_sigma_inv @ (hat_par_REG_red_star - beta@hat_par_REG_sym)
    else:
        # Do not use weights of base parameters
        print("[WARN] No Base parameters covariance matrix is provided. Proceeding with Identity weighting")
        f_loss = (hat_par_REG_red_star - beta@hat_par_REG_sym).T @ (hat_par_REG_red_star - beta@hat_par_REG_sym)

    # -- Residual in the dynamic parameters
    f_mass = 0
    f_com = 0
    f_inertia = 0
    f_additional = 0

    w_mass    = 1
    w_com     = 1
    w_inertia = 1

    config_ident = config['identification']
    if not config_ident.get('prior_yaml',False):
        # Reduced definition of uncertainties
        w_loss    = float(config_ident['weight']['loss'])                # Pull toward base parameters
        w_mass    = float(config_ident['weight']['mass'])                # Pull toward URDF mass
        w_com     = float(config_ident['weight']['CoM'])                 # Pull toward URDF CoM
        w_inertia = float(config_ident['weight']['inertia'])             # Pull toward URDF inertia
        w_link = np.array(config_ident['weight']['link'], dtype=float)   # Weight link differently
        if len(w_link)!=n:
            print("[WARN] The lenght of link weight is ill-setted. proceding with equal weighting for each link")
            w_link = [1 for _ in range(n)]
        w_link = w_link/np.sum(w_link)
        # Computing Sigma inverse
        w_mass    = 1/(w_mass*w_mass)
        w_com     = 1/(w_com*w_com)
        w_inertia = 1/(w_inertia*w_inertia)

        for i in range(n):
            # DYN version
            par_DYN_link_i = hat_par_DYN_0[10*i:10*(i+1)]

            f_mass      +=  (w_link[i]*w_mass) @ casadi.sumsqr(hat_par_DYN_sym[10*i] - par_DYN_link_i[0]) # M
            f_com       +=  (w_link[i]*w_com) @ (casadi.sumsqr(hat_par_DYN_sym[10*i+1] - par_DYN_link_i[1]) + # CX
                                                 casadi.sumsqr(hat_par_DYN_sym[10*i+2] - par_DYN_link_i[2]) + # CY
                                                 casadi.sumsqr(hat_par_DYN_sym[10*i+3] - par_DYN_link_i[3]))  # CZ
            f_inertia   +=  (w_link[i]*w_inertia) @ (casadi.sumsqr(hat_par_DYN_sym[10*i+4] - par_DYN_link_i[4]) +   # Ixx
                                                     casadi.sumsqr(hat_par_DYN_sym[10*i+5] - par_DYN_link_i[5]) +   # Ixy
                                                     casadi.sumsqr(hat_par_DYN_sym[10*i+6] - par_DYN_link_i[6]) +   # Ixz
                                                     casadi.sumsqr(hat_par_DYN_sym[10*i+7] - par_DYN_link_i[7]) +   # Iyy
                                                     casadi.sumsqr(hat_par_DYN_sym[10*i+8] - par_DYN_link_i[8]) +   # Iyz
                                                     casadi.sumsqr(hat_par_DYN_sym[10*i+9] - par_DYN_link_i[9]))    # Izz
    else:
        # Explicit definition of prior uncrtainities
        prior = load_prior(config_ident['prior_yaml'])
        for i in range(n):
            # Load mean and std deviation of i-th link
            link_i = prior['robot']['prior'][f"link{i+1}"]
            mu_link_i = [link_i['mass']['mu'],
                         link_i['CoM_x']['mu'],
                         link_i['CoM_y']['mu'],
                         link_i['CoM_z']['mu'],
                         link_i['Ixx']['mu'],
                         link_i['Ixy']['mu'],
                         link_i['Ixz']['mu'],
                         link_i['Iyy']['mu'],
                         link_i['Iyz']['mu'],
                         link_i['Izz']['mu']]
            sdt2_link_i = [link_i['mass']['sigma']* link_i['mass']['sigma'],
                          link_i['CoM_x']['sigma']* link_i['CoM_x']['sigma'],
                          link_i['CoM_y']['sigma']* link_i['CoM_y']['sigma'],
                          link_i['CoM_z']['sigma']* link_i['CoM_z']['sigma'],
                          link_i['Ixx']['sigma']* link_i['Ixx']['sigma'],
                          link_i['Ixy']['sigma']* link_i['Ixy']['sigma'],
                          link_i['Ixz']['sigma']* link_i['Ixz']['sigma'],
                          link_i['Iyy']['sigma']* link_i['Iyy']['sigma'],
                          link_i['Iyz']['sigma']* link_i['Iyz']['sigma'],
                          link_i['Izz']['sigma']* link_i['Izz']['sigma']]
            f_mass      +=   1/sdt2_link_i[0] @ casadi.sumsqr(hat_par_DYN_sym[10*i]   - mu_link_i[0]) # M
            f_com       +=  (1/sdt2_link_i[1] @ casadi.sumsqr(hat_par_DYN_sym[10*i+1] - mu_link_i[1]) + # CX
                             1/sdt2_link_i[2] @ casadi.sumsqr(hat_par_DYN_sym[10*i+2] - mu_link_i[2]) + # CY
                             1/sdt2_link_i[3] @ casadi.sumsqr(hat_par_DYN_sym[10*i+3] - mu_link_i[3]))  # CZ
            f_inertia   += (1/sdt2_link_i[4] @ casadi.sumsqr(hat_par_DYN_sym[10*i+4]  - mu_link_i[4]) +   # Ixx
                            1/sdt2_link_i[5] @ casadi.sumsqr(hat_par_DYN_sym[10*i+5]  - mu_link_i[5]) +   # Ixy
                            1/sdt2_link_i[6] @ casadi.sumsqr(hat_par_DYN_sym[10*i+6]  - mu_link_i[6]) +   # Ixz
                            1/sdt2_link_i[7] @ casadi.sumsqr(hat_par_DYN_sym[10*i+7]  - mu_link_i[7]) +   # Iyy
                            1/sdt2_link_i[8] @ casadi.sumsqr(hat_par_DYN_sym[10*i+8]  - mu_link_i[8]) +   # Iyz
                            1/sdt2_link_i[9] @ casadi.sumsqr(hat_par_DYN_sym[10*i+9]  - mu_link_i[9]))    # Izz
            # additional
            if par_per_link == 11 and not link_i.get('motor',False):
                # motor inertia
                mu_Ia_i = prior['robot']['prior'][f"link{i+1}"]['motor']['sigma']
                std_Ia_i = prior['robot']['prior'][f"link{i+1}"]['motor']['sigma']
                f_additional += 1/std_Ia_i @ casadi.sumsqr(hat_par_DYN_sym[10*n + i]   - mu_Ia_i) # Ia

    # - Define Problem
    prob = {}
    prob['x'] = hat_par_REG_sym
    prob['f'] = w_loss* f_loss + (
                                    f_mass +
                                    f_com +
                                    f_inertia +
                                    f_additional
                                 )
    prob['g'] = casadi.vertcat(*g)


    # convert cost terms into CasADi functions
    f_loss_func = casadi.Function('f_loss_func',
                             [hat_par_REG_sym],
                             [w_loss*f_loss],
                             ['params'], ['cost'])
    f_mass_func = casadi.Function('f_mass_func',
                             [hat_par_REG_sym],
                             [f_mass],
                             ['params'], ['cost'])
    f_com_func = casadi.Function('f_com_func',
                             [hat_par_REG_sym],
                             [f_com],
                             ['params'], ['cost'])
    f_inertia_func = casadi.Function('f_inertia_func',
                             [hat_par_REG_sym],
                             [f_inertia],
                             ['params'], ['cost'])

    f_additional_func = casadi.Function('f_additional_func',
                             [hat_par_REG_sym],
                             [f_additional],
                             ['params'], ['cost'])

    f_penalty_costraint_func = casadi.Function('f_penalty_costraint',
                             [hat_par_REG_sym],
                             [penalty_costraint],
                             ['params'], ['cost'])

    # Solver Options
    if opts is None:
        opts = {
            "ipopt": {
                "tol": 1e-4,
                "constr_viol_tol": 1e-6,
                "acceptable_tol": 1e-5,
                "dual_inf_tol": 1e-6,
                "compl_inf_tol": 1e-6,
                "max_iter": 50000,
            }
        }

    # ---------------------------SIMULATED ANNEALING--------------------------------------------------------
    if config_ident.get('optimization','ipopt')=='SA':
        try:
            from scipy.optimize import dual_annealing
        except:
            raise ImportError
        # Define penalty based cost function
        def J_SA(par_DYN, penalty_weight = 1):
            par_DYN = par_DYN.reshape(-1, 1)
            cost  = float(f_loss_func(par_DYN).full()[0,0])
            cost += float(f_mass_func(par_DYN).full()[0,0])
            cost += float(f_com_func(par_DYN).full()[0,0])
            cost += float(f_inertia_func(par_DYN).full()[0,0])
            # Add the constraint penalty scaled by the current iteration's weight
            penalty = float(f_penalty_costraint_func(par_DYN).full()[0,0])

            return cost + (penalty_weight * penalty)

        N_k = config_ident.get('k', 10)
        Q = config_ident.get('Q', 100)
        # 2. Define the Search Space Bounds for SA
        bounds = []
        for i in range(n):
            bounds.append((1e-3, 50.0)) # M  [kg]
            bounds.append((-1.0, 1.0))  # CX [m]
            bounds.append((-1.0, 1.0))  # CY [m]
            bounds.append((-1.0, 1.0))  # CZ [m]
            bounds.append((1e-6, 5.0))  # XX [kg*m^2]
            bounds.append((-1.0, 1.0))  # XY [kg*m^2]
            bounds.append((-1.0, 1.0))  # XZ [kg*m^2]
            bounds.append((1e-6, 5.0))  # YY [kg*m^2]
            bounds.append((-1.0, 1.0))  # YZ [kg*m^2]
            bounds.append((1e-6, 5.0))  # ZZ [kg*m^2]

        if par_per_link == 11:
            for i in range(n):
                bounds.append((1e-6, 1.0)) # Ia


        current_guess = hat_par_REG_0.flatten()

        for k in range(N_k):
            # Increase penalty per outer iteration
            penalty_weight = 10**k 
            print(f"Outer Iteration {k+1}/{N_k} | Penalty Weight: {penalty_weight:.0e}")
            # Run the Simulated Annealing global optimization
            result = dual_annealing(
                func=J_SA, 
                bounds=bounds, 
                args=(penalty_weight,), # Pass the penalty weight to J_SA
                x0=current_guess,       # Give it the current best guess to start
                maxiter=Q,             # Max internal SA iterations (jumping/cooling) # Q
                initial_temp=5230.0,   # Default SA starting temperature
                seed=41                # set deterministic
            )
            print(f"Current best cost: {result.fun:.4f}")
            # Update our guess with the best state found in this SA run
            current_guess = result.x 
            if check_par_DYN_feasibility(reg2dyn(current_guess),n, verbose=False).all(): break # If the current guess is feasible, we can stop early

        hat_par_REG_optim = current_guess
        print("\n--- Simulated Annealing Complete ---")

    # ---------------------------CasADi IPOPT--------------------------------------------------------
    elif config_ident.get('optimization','ipopt')=='ipopt':# Solver
        solver = casadi.nlpsol('sol', 'ipopt', prob, opts)
        sol = solver(x0=hat_par_REG_0, lbg=lb,ubg=ub)
        hat_par_REG_optim = sol['x'].full()


    # Print residuals
    results = {}
    results['loss'] = f_loss_func(hat_par_REG_optim)
    results['mass'] = f_mass_func(hat_par_REG_optim)
    results['com'] = f_com_func(hat_par_REG_optim)
    results['inertia'] = f_inertia_func(hat_par_REG_optim)
    results['additional'] = f_additional_func(hat_par_REG_optim)
    print(f"Loss: {results['loss']}, Mass: {results['mass']}, Center of Mass: {results['com']}, Tensor inertia: {results['inertia']}, Additional: {results['additional']}")
    print(f"Penalty check: {f_penalty_costraint_func(hat_par_REG_optim)}")

    feas_links = check_par_DYN_feasibility(reg2dyn(hat_par_REG_optim),n)
    if not feas_links.all():
        print(f"[WARN] Estimated parameters do not pass feasibility check (links: {np.flatnonzero(~feas_links).tolist()}), they lie on feasibility bounds. Using LMI projection tring to force feasiblity")
        hat_par_REG_optim = np.asarray(hat_par_REG_optim, dtype=float)
        hat_par_DYN_optim = np.array(reg2dyn(hat_par_REG_optim[:10*n])).flatten()
        hat_par_DYN_optim = LMI_projection(robot, par_DYN=hat_par_DYN_optim, links=np.flatnonzero(~feas_links).tolist())
        hat_par_REG_optim[:10*n] = np.array(dyn2reg(casadi.DM(hat_par_DYN_optim))).reshape(hat_par_REG_optim[:10*n].shape)  # motor inertia untouched
        if not check_par_DYN_feasibility(reg2dyn(hat_par_REG_optim),n).all():
            print("[WARN] Remapped parameters still do not pass feasibility check")
        else:
            print("[INFO] Feasibility solved")

    if export_file:
        # 1. Define and create the results directory
        if path is None:
            date_str = datetime.datetime.now().strftime("%d_%m_%Y")
            results_dir = f"results/data_{date_str}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
        else:
            results_dir = path
        # 2. Create identification_par.yaml
        config_path = os.path.join(results_dir, "identification_par.yaml")

        with open(config_path, "w") as f:
            # Add other parameters
            f.write(f"d3q: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"d4q: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"ddq: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"ddqr: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"dq: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"dqr: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"q: {[0 for _ in range(robot.numJoints)]}\n")
            f.write(f"w: {[0 for _ in range(6)]}\n")
            try:
                f.write(f"par_Ln2EE: {robot.get_par_Ln2EE().tolist()}\n")
            except AttributeError:
                pass
            try:
                f.write(f"par_world2L0: {robot.get_world2L0().tolist()}\n")
            except AttributeError:
                pass
            try:
                f.write(f"par_gravity: {robot.get_par_gravity().tolist()}\n")
            except AttributeError:
                pass

            # Add identified parameters
            f.write(f"par_DYN: {robot.get_par_DYN().tolist()}\n")
            f.write(f"par_REG: {robot.get_par_REG().tolist()}\n")
            f.write(f"par_Dl: {robot.get_par_Dl().tolist()}\n")
            f.write(f"par_REG_red: {robot.get_par_REG_red().tolist()}\n")
            try:
                # Check for actuator inertia or additional parameters
                f.write(f"par_Ia: {robot.get_par_Ia().tolist()}\n")
            except AttributeError:
                pass

        print(f"Estimated Dynamic Parameters written in '{config_path}'")

    if par_per_link == 10:
        return (hat_par_REG_optim, results)
    elif par_per_link == 11:
        hat_par_REG = hat_par_REG_optim[:n*10]
        hat_par_Ia = hat_par_REG_optim[n*10:n*11]
        return (hat_par_REG, hat_par_Ia, results)


def LMI_projection(robot:Any, weights:Optional[np.ndarray] = None, psd_tol:float = 1e-5,
                   par_DYN:Optional[np.ndarray] = None, links:Optional[list[int]] = None) -> np.ndarray :
    """
    Projection of the estimated parameters into the nearest pheasible set of parameter via an Linear Matrix Inequality problem.
    min || par_DYN_proj - par_DYN ||_w ^2
    s.t.
        m > 0
        P > 0
    defined by the LMI constraints for positive definiteness of inertia matrix.

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka, thunder_ahand_finger).
        weights (np.ndarray, optional): weights for the projection cost function. Defaults to None.
        threshold (float, optional): threshold for positive definiteness. Defaults to 1e-5.
        par_DYN (np.ndarray, optional): standard dynamic parameters to project. Defaults to robot.get_par_DYN().
        links (list[int], optional): indices of the links to project, the others are left unchanged. Defaults to all links.
    Returns:
        np.ndarray: Feasible set of parameters
    """

    import cvxpy as cp

    par_DYN = robot.get_par_DYN().copy() if par_DYN is None else np.asarray(par_DYN, dtype=float).flatten().copy()
    n = robot.numJoints
    for i in (range(n) if links is None else links):
        # Dynamic parameters o i-th link
        par_DYN_proj = cp.Variable(10)

        if weights is None:
            m_max = np.max(par_DYN[10*i])  # Extract mass values
            com_max = np.linalg.norm(par_DYN[i*10 + 1:i*10 + 4])
            I_max = np.linalg.norm(par_DYN[i*10 + 4:i*10 + 10])  # Extract inertia values

            # Diagonal weights (previous BUG: W[1:4, 1:4] = ... filled the whole block, penalising only the
            # sum of the CoM / inertia deviations)
            W = np.diag([1/m_max] + [1/com_max]*3 + [1/I_max]*6)
        else:
            W = cp.diag(weights)

        residual = W @ (par_DYN_proj - par_DYN[i*10:(i+1)*10])
        J = cp.sum_squares(residual)
        constraints = []
        
        m = cp.reshape(par_DYN_proj[0], (1, 1), order='C') # mass
        I_bar = cp.bmat([[par_DYN_proj[4], par_DYN_proj[5], par_DYN_proj[6]],  # Barycentric Inertia
                         [par_DYN_proj[5], par_DYN_proj[7], par_DYN_proj[8]],
                         [par_DYN_proj[6], par_DYN_proj[8], par_DYN_proj[9]]])

        # Positive Pseudo-Definite Matrix constraint
        Sigma = 0.5*cp.trace(I_bar)*np.eye(3) - I_bar

        P = cp.bmat([
                    [m,                np.zeros((1, 3))],
                    [np.zeros((3, 1)), Sigma           ]
                ])
        constraints += [P >> psd_tol * np.eye(4)]
        constraints += [m >= 1e-5]
        prob = cp.Problem(cp.Minimize(J), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        par_DYN[i*10:(i+1)*10] = par_DYN_proj.value.flatten().copy()

    return par_DYN




def plot_base_identification(robot, traject, metrics, block = True,
                             title_size = 20, lable_size = 16, tick_size = 14, legend_size = 13):
    """
    Plot of the reconstructed dynamic via NE using base inertial parameters

    Args:
        robot (_type_): _description_
        traject (_type_): _description_
        metrics (_type_): _description_
        block (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    delta_tau = []
    par = np.hstack([robot.get_par_REG_red(),robot.get_par_Dl()]).reshape(-1,1)
    for i in range(traject.t.shape[0]):
        q = traject.q[i,:]
        dq = traject.qd[i,:]
        ddq = traject.qdd[i,:]

        robot.set_q(q)
        robot.set_dq(dq)
        robot.set_ddq(ddq)
        robot.set_dqr(dq)
        robot.set_ddqr(ddq)

        # Full inverse-dynamics regressor (per-joint rows)
        Y_red = robot.get_Yr_red()
        Y_dl = robot.get_reg_dl()
        Y = np.hstack([Y_red, Y_dl])


        tau_est = Y @ par
        delta_tau.append(traject.tau[i,:] - tau_est.flatten())

    delta_tau = np.array(delta_tau)

    rmse = np.sqrt(np.mean(np.sum(delta_tau**2, axis=1))) # RMSE = Sqrt{ 1/M Sum{||e||_2}  }
    cond_num = metrics.get('conditioning number', None)
    sigma_max = metrics.get('sigma max', None)

    cond_str = f"{cond_num:.3f}" if cond_num is not None else "N/A"
    sigma_str = f"{sigma_max:.3f}" if sigma_max is not None else "N/A"
    # --- 1. Plot tau ---
    colors = {
        'measured': '#D62728',   # Strong Red
        'estimate': '#1F77B4',   # Strong Blue
        'mass':     '#2CA02C',   # Forest Green
        'coriolis': '#FF7F0E',   # Safety Orange
        'gravity':  '#9467BD',   # Royal Purple
        'friction': '#7F7F7F',   # Medium Gray
        'motor': '#7F0F7F'       # Medium Gray
    }
    n = robot.numJoints
    cols = int(np.sqrt(n)) 
    rows = (n + cols - 1) // cols  # Ceiling division to handle odd numbers
    JOINT_NAMES = traject.config['trajectory']['joints']

    fig = plt.figure(figsize=(12,8))
    try:
        # Removed the internal :.3f since cond_str is already a string
        plt.suptitle(f'Torque Estimation Results - Method {metrics["method"]}\nRMSE: {rmse:.4f} Nm | Conditioning Number (κ): {cond_str} | Sigma max: {sigma_str}', 
                    fontsize=16, fontweight='bold')
    except Exception as e:
        # Fixed here as well
        plt.suptitle(f'Torque Estimation Results\nRMSE: {rmse:.4f} Nm | Conditioning Number (κ): {cond_str} | Sigma max: {sigma_str}', 
                    fontsize=16, fontweight='bold')
    
    for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            ax.plot(traject.t, traject.tau[:, i],
                    color=colors['measured'], linewidth=2, label='tau_filtered')
            ax.plot(traject.t, traject.tau[:, i] - delta_tau[:, i],
                    color=colors['estimate'], linewidth=1.5, label='hat_tau')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Torque [Nm]')
            ax.set_title(f'Torque {JOINT_NAMES[i]}')
            ax.grid(True)
            if i == 0:
                ax.legend()
            ax.title.set_fontsize(title_size)
            ax.xaxis.label.set_fontsize(lable_size)
            ax.yaxis.label.set_fontsize(lable_size)
            ax.tick_params(axis='both', labelsize=tick_size)
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(legend_size)


    plt.tight_layout()
    plt.show(block=block)

    return fig


def plot_identification(robot: Any, traject: TrajectoryManager, metrics: dict, title: Optional[str] = None, block: bool = True,
                        title_size: int = 20, suptitle_size: int = 24, label_size: int = 16, tick_size: int = 14, legend_size: int = 13) -> plt.Figure:
    """
    Plot the reconstructed joint torques computed via Newton-Euler (NE) inverse dynamics 
    against the actual measured trajectory torques to visually validate the identification quality.

    Args:
        robot (Any): Robot model instance built with the thunder dynamic tool
                     (e.g., thunder_franka).
        traject (TrajectoryManager): Managed tracking object containing joint positions (q),
                                     velocities (dq), accelerations (ddq), and measured
                                     torques (tau) over time.
        metrics (dict): Estimation quality statistics generated by the OLS or MLE solver 
                        (e.g., residual errors, standard deviations).
        title (str, optional): Main text title displayed at the top window of the figure space. Defaults to None.
        block (bool, optional): If True, blocks script execution until the user manually closes the window 
                                (matplotlib execution control). Defaults to True.
        title_size (int, optional): Font sizing parameter applied to individual subplot titles. Defaults to 20.
        suptitle_size (int, optional): Font sizing parameter applied to the absolute main figure heading. Defaults to 24.
        label_size (int, optional): Font sizing parameter applied to X and Y axis text titles. Defaults to 16.
        tick_size (int, optional): Font sizing parameter applied to sequential numerical tick scales along the axes. Defaults to 14.
        legend_size (int, optional): Font sizing parameter applied to the internal descriptive label keys. Defaults to 13.

    Returns:
        plt.Figure: The Matplotlib figure instance containing the generated plot.
    """
    tau_M = []
    tau_C = []
    tau_G = []
    tau_dl = []
    tau_Ia = []
    delta_tau = []
    for i in range(traject.t.shape[0]):
        q = traject.q[i,:]
        dq = traject.qd[i,:]
        ddq = traject.qdd[i,:]

        robot.set_q(q)
        robot.set_dq(dq)
        robot.set_ddq(ddq)
        robot.set_dqr(dq)
        robot.set_ddqr(ddq)

        # Full inverse-dynamics regressor (per-joint rows)

        tau_est_M = robot.get_M()@ddq
        tau_est_C = robot.get_C()@dq
        tau_est_G = robot.get_G()
        tau_est_dl = robot.get_dl()
        try:
            tau_est_Ia = robot.get_Ia()
            tau_est = tau_est_M + tau_est_C + tau_est_G + tau_est_dl + tau_est_Ia
            tau_Ia.append(tau_est_Ia)
        except:
            tau_est = tau_est_M + tau_est_C + tau_est_G + tau_est_dl

        delta_tau.append(traject.tau[i,:] - tau_est.flatten())
        tau_M.append(tau_est_M)
        tau_C.append(tau_est_C)
        tau_G.append(tau_est_G)
        tau_dl.append(tau_est_dl)

    delta_tau = np.array(delta_tau)
    tau_M = np.array(tau_M)
    tau_C = np.array(tau_C)
    tau_G = np.array(tau_G)
    tau_dl = np.array(tau_dl)
    tau_Ia = np.array(tau_Ia)


    rmse = np.sqrt(np.mean(np.sum(delta_tau**2, axis=1))) # RMSE = Sqrt{ 1/M Sum{||e||_2}  }
    cond_num = metrics.get('conditioning number', None)
    sigma_max = metrics.get('sigma max', None)
    # --- 1. Plot tau ---
    colors = {
        'measured': '#D62728',   # Strong Red
        'estimate': '#1F77B4',   # Strong Blue
        'mass':     '#2CA02C',   # Forest Green
        'coriolis': '#FF7F0E',   # Safety Orange
        'gravity':  '#9467BD',   # Royal Purple
        'friction': '#7F7F7F',   # Medium Gray
        'motor': '#7F0F7F'       # Medium Gray
    }
    n = robot.numJoints
    cols = int(np.sqrt(n)) 
    rows = (n + cols - 1) // cols  # Ceiling division to handle odd numbers
    JOINT_NAMES = traject.config['trajectory']['joints']

    fig = plt.figure(figsize=(12,8))
    if title is not None:
        plt.suptitle(title + f"\nRMSE: {rmse:.4f} Nm", fontsize=16, fontweight='bold')
    else:
        sigma_max_title = r"$\sigma_{max}$" 
        try:
            plt.suptitle(f'Torque Estimation Results - Method {metrics["method"]} (RMSE: {rmse:.4f} Nm)\nConditioning Number (κ): {cond_num:.3f}| {sigma_max_title}: {sigma_max:.3f}', 
                        fontsize=16, fontweight='bold')
        except Exception as e:
            plt.suptitle(f'Torque Estimation Results (RMSE: {rmse:.4f} Nm)', 
                    fontsize=16, fontweight='bold')
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(traject.t, traject.tau[:, i],
                color=colors['measured'], linewidth=2, label='tau_filtered')
        ax.plot(traject.t, traject.tau[:, i] - delta_tau[:, i],
                color=colors['estimate'], linewidth=1.5, label='hat_tau')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [Nm]')
        ax.set_title(f'Torque {JOINT_NAMES[i]}')
        ax.grid(True)
        if i == 0:
            ax.legend()
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                    text.set_fontsize(legend_size)

    plt.tight_layout()
    plt.show(block=block)

    return fig

def plot_LS_solution(hat_pi: np.ndarray, metrics: dict, pi_gt: Optional[np.ndarray] = None, block: bool = True,
                    title_size: int = 20, label_size: int = 16, tick_size: int = 14, legend_size: int = 13) -> plt.Figure:
    """
    Generate an error bar plot representation for visualization of the estimated base parameters
    against optional ground truth reference values.

    Args:
        hat_pi (np.ndarray): Estimation vector of the identifiable base inertial parameters.
        metrics (dict): Quality metrics containing 'parameters standard deviation' key array.
        pi_gt (np.ndarray, optional): Ground truth parameter vector for verification. Defaults to None.
        block (bool, optional): If True, blocks script execution until the user manually closes the figure. Defaults to True.
        title_size (int, optional): Font sizing applied to the figure title. Defaults to 20.
        label_size (int, optional): Font sizing applied to the X and Y axes labels. Defaults to 16.
        tick_size (int, optional): Font sizing applied to the numerical axis tick markings. Defaults to 14.
        legend_size (int, optional): Font sizing applied to the text elements inside the legend box. Defaults to 13.

    Returns:
        plt.Figure: The Matplotlib figure instance containing the generated plot.
    """
    # Create an array of indices for the x-axis
    x = np.arange(len(hat_pi))

    fig = plt.figure(figsize=(10, 6))

    # 1. Plot the ground truth (using green squares for clear contrast)
    if pi_gt is not None:
        plt.plot(x, pi_gt.flatten(), 'gs', markersize=6, label='Ground Truth')

    # 2. Plot hat_pi with its standard deviation as error bars
    # fmt='bo' uses blue circles for the points. ecolor sets the error bar color.
    plt.errorbar(x, hat_pi.flatten(), yerr=metrics['parameters standard deviation'].flatten(), fmt='bo', ecolor='red', 
                capsize=5, elinewidth=1.5, label='Estimated hat_pi')

    ax = plt.gca()
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Estimation vs. Ground Truth')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    # Axis-Style (title_size = 20, lable_size = 16, tick_size = 14, legend_size = 13)
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)
    # suptitle not used here but keep title consistent
    ax.title.set_fontsize(title_size)


    plt.tight_layout()
    plt.show(block=block)
    return fig

def load_prior(prior_path:str) -> dict :
    """  
    Load the extended prior configuration of the robot.

    Args:
        prior_path (str): Path where the robot model full dynamic priors are stored.
    
    Returns:
        dict: The dictionary containing the dynamic prior values.
    """
    with open(prior_path, 'r') as f:
        try:
            prior = yaml.load(f, Loader=SafeLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    if 'robot' not in prior or 'prior' not in prior['robot']:
        raise KeyError(f"Prior file is not well-formatted")

    return prior


def check_par_DYN_feasibility(par_DYN:np.ndarray ,n: int, verbose: bool = False, tol: float = 0.0) -> np.ndarray:
    """ 
    Return whether a set of dynamic parameters is feasible.

    Args:
        par_DYN (np.ndarray): standar dynamic parameters of robot model.
        n (int): Number of robot link.
        verbose (bool, optional): Wheter run in verbose mode. Default is false.
        tol (float, optional): Accepted violation of the triangle inequalities, tr(I)/2 - l_max(I) > -tol [Kg m^2].
                               When the optimum lies on the feasibility boundary (degenerate inertia) the solver
                               meets the constraint only up to its constraint tolerance, so use that value
                               (IPOPT constr_viol_tol, divided by the inertia scale factor). Default is 0 (strict).

    Returns:
        np.ndarray: (n,) boolean array, feasibility of each link. """
    feas_flag = np.ones(n, dtype=bool)
    for i in range(n):
        # Mass check
        par_DYN_link_i = par_DYN[10*i:10*(i+1)]
        if par_DYN_link_i[0] <= 0: 
            if verbose: 
                print(f"mass of link {i} is non positive")
            feas_flag[i] = False

        # Triangular inequalities
        Ib = np.zeros((3,3))
        Ib[0,0] = par_DYN_link_i[4]
        Ib[0,1] = par_DYN_link_i[5]
        Ib[0,2] = par_DYN_link_i[6]
        Ib[1,0] = par_DYN_link_i[5]
        Ib[1,1] = par_DYN_link_i[7]
        Ib[1,2] = par_DYN_link_i[8]
        Ib[2,0] = par_DYN_link_i[6]
        Ib[2,1] = par_DYN_link_i[8]
        Ib[2,2] = par_DYN_link_i[9]

        eig,_ = np.linalg.eig(Ib)
        l_max = np.max(eig)
        if not (np.trace(Ib)/2 - l_max > -tol): 
            if verbose:  print(f"Inertia of link {i} do not pass matrix inequality test. (resid: {np.trace(Ib)/2 - l_max})")
            feas_flag[i] = False

    if verbose and feas_flag.all():  print(f"Test passed")

    return feas_flag


def _generate_fibonacci_sphere(n_samples: int = 2000) -> np.ndarray:
    """
    Generates n_samples points nearly uniformly distributed on a unit sphere
    using the Fibonacci spiral method.

    Returns:
        np.ndarray: An (n_samples, 3) array of Cartesian coordinates (x, y, z).
    """
    # Create an index array from 0 to n_samples - 1
    indices = np.arange(n_samples) + 0.5

    # Calculate the z coordinate (linearly from 1 to -1)
    phi = np.arccos(1 - 2 * indices / n_samples)

    # Calculate the golden angle increment
    # Golden ratio approximation: (1 + sqrt(5)) / 2
    golden_angle = np.pi * (1 + 5**0.5)

    # Calculate the azimuth angle (theta)
    theta = golden_angle * indices

    # Convert spherical coordinates to Cartesian
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Stack into an (N, 3) array
    return np.column_stack((x, y, z))