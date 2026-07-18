"""
utilities function for the evaluation of identification process 
against a ground truth robot model.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

from utils.identification_utils import *

def plot_eval_identification(robot, metrics, traject, robot_gt = None, block = True,
                    title_size = 20, suptitle_size = 24, lable_size = 16, tick_size = 14, legend_size = 13):
    """ Plot measurement and reconstructed effort """
    delta_tau = []
    delta_tau_GT = []

    pi_red = robot.get_par_REG_red()
    pi_dl = robot.get_par_Dl()
    pi_red_CAD = robot_gt.get_par_REG_red()
    pi_dl_CAD = robot_gt.get_par_Dl()


    for i in range(traject.t.shape[0]):
        q = traject.q[i,:]
        dq = traject.qd[i,:]
        ddq = traject.qdd[i,:]

        robot.set_q(q)
        robot.set_dq(dq)
        robot.set_ddq(ddq)
        robot.set_dqr(dq)
        robot.set_ddqr(ddq)

        Y_red = robot.get_Yr_red()
        Y_dl = robot.get_reg_dl()

        tau_est = (Y_red@pi_red + Y_dl@pi_dl).flatten()
        delta_tau.append(traject.tau[i,:] - tau_est)
        delta_tau_GT.append(traject.tau[i,:] - (Y_red @ pi_red_CAD + Y_dl @ pi_dl_CAD).flatten())
    delta_tau = np.array(delta_tau)
    delta_tau_GT = np.array(delta_tau_GT)


    rmse = np.sqrt(np.mean(np.sum(delta_tau**2, axis=1))) # RMSE = Sqrt{ 1/M Sum{||e||_2}  }
    print(f"RMSE: {rmse} [Nm]" )

    cond_num = metrics['conditioning number']
    # --- 1. Plot tau ---
    colors = {
        'estimate': '#D62728',   # Strong Red
        'measured': '#1F77B4',   # Strong Blue
        'ground-truth':     '#2CA02C',   # Forest Green
    }
    n = robot.numJoints
    cols = int(np.sqrt(n)) 
    rows = (n + cols - 1) // cols  # Ceiling division to handle odd numbers
    JOINT_NAMES = traject.config['trajectory']['joints']

    fig = plt.figure(figsize=(12,8))
    plt.suptitle(f'Torque Estimation Results\nRMSE: {rmse:.4f} Nm | Conditioning Number (κ): {cond_num}', 
                fontsize=16, fontweight='bold')
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(traject.t, traject.tau_raw[:, i],
                color=colors['measured'], linewidth=2, label=r'$\tau$')
        ax.plot(traject.t, traject.tau[:, i] - delta_tau[:, i],
                color=colors['estimate'], linewidth=1.5, label=r'$\hat{\tau}$')
        ax.plot(traject.t, traject.tau[:, i] - delta_tau_GT[:, i],
                color=colors['ground-truth'], linewidth=1.5, label=r'$\tau_{GT}$')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [Nm]')
        ax.set_title(f'Torque {JOINT_NAMES[i]}')
        ax.grid(True)
        if i == 0:
            ax.legend(loc='lower right')
        # Axis-Style (title_size = 20, lable_size = 16, tick_size = 14, legend_size = 13)
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontsize(lable_size)
        ax.yaxis.label.set_fontsize(lable_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                    text.set_fontsize(legend_size)
    ax.title.set_fontsize(suptitle_size)

    plt.tight_layout()
    plt.show(block=block)

    return fig




def plot_eval_LS_solution(hat_pi, metrics, robot_gt = None, block = True,
                    title_size = 20, suptitle_size = 24, lable_size = 16, tick_size = 14, legend_size = 13):
    """ Plot Base parameters estim as boxplot """
    # Create an array of indices for the x-axis
    x = np.arange(len(hat_pi))

    fig = plt.figure(figsize=(10, 6))

    # 1. Plot the ground truth (using green squares for clear contrast)
    if robot_gt is not None:
        plt.plot(x, np.hstack([robot_gt.get_par_REG_red(),robot_gt.get_par_Dl()]).flatten(), 'gs', markersize=6, label='Ground Truth')

    # 2. Plot hat_pi with its standard deviation as error bars
    # fmt='bo' uses blue circles for the points. ecolor sets the error bar color.
    plt.errorbar(x, hat_pi.flatten(), yerr=metrics['parameters standard deviation'].flatten(), fmt='bo', ecolor='red', 
                capsize=5, elinewidth=1.5, label='Estimated hat_pi')

    ax = plt.gca()
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Estimation vs. Ground Truth')
    ax.set_xticks(x)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    # Axis-Style (title_size = 20, lable_size = 16, tick_size = 14, legend_size = 13)
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontsize(lable_size)
    ax.yaxis.label.set_fontsize(lable_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
                text.set_fontsize(legend_size)
    ax.title.set_fontsize(suptitle_size)


    plt.tight_layout()
    plt.show(block = block)
    return fig


def plot_table(robot, robot_ground_truth, format = 'plain'):
    """ 
    Print Table with the full set of dynamic parameters
    """
    print(f" ------------------")
    print(f"| Evaluation Table |")
    print(f" ------------------")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    headers = ["m", "cx", "cy", "cz", "ixx", "ixy", "ixz", "iyy", "iyz", "izz",'Ia']

    if hasattr(robot, "get_par_Ia"):
        par_per_link = 11
        headers.append("Ia")
    else:
        par_per_link = 10
    n = robot.numJoints
    headers = headers[:par_per_link]

    if hasattr(robot, "get_par_Dl"):
        Dl_order = robot.Dl_order
        if Dl_order == 1:
            headers += ["fs"]
        else:
            headers += ["fs","fv"]


    if format.lower() == 'latex':
        # 1) Optimized params
        # Motor inertia
        if par_per_link == 11:
            params = np.hstack([robot.get_par_DYN().reshape(n,10), 
                                robot.get_par_Ia().reshape(n,1)]).reshape(n, 11)
        else:
            params = robot.get_par_DYN().reshape((n, 10))
        # Friction
        if hasattr(robot, "get_par_Dl"):
            params = np.hstack([params.reshape(n,-1),
                                robot.get_par_Dl().reshape(n,-1)]).reshape(n, -11)
        title = "Optimized parameters"
        col_format = "c|" + "r" * len(headers)
        print(r"\begin{table}[htpb]")
        print(r"\centering")
        print(rf"\caption{{{title}}}")
        # Requires \usepackage{graphicx} in your LaTeX preamble
        print(r"\resizebox{\textwidth}{!}{%")
        print(rf"\begin{{tabular}}{{{col_format}}}")
        print(r"\hline")

        # Print Headers
        header_str = "Link & " + " & ".join([f"${h}$" for h in headers]) + r" \\ \hline"
        print(header_str)

        # Print Rows
        for i, row in enumerate(params):
            row_strs = []
            for val in row:
                # Convert to LaTeX scientific notation (e.g. 1.23e-04 -> $1.23 \times 10^{-4}$)
                val_e = f"{val:.6e}"
                base, exp = val_e.split('e')
                exp_int = int(exp) # Removes leading zeros from exponent
                row_strs.append(f"${base} \\times 10^{{{exp_int}}}$")
            print(f"{i} & " + " & ".join(row_strs) + r" \\")
        print(r"\hline")
        print(r"\end{tabular}%")
        print(r"}")
        print(r"\end{table}")
        print() # Add spacing between tables

        if robot_ground_truth is not None:
            # ground-truth params
            # Motor inertia
            if par_per_link == 11:
                gt_params = np.hstack([robot_ground_truth.get_par_DYN().reshape(n,10), robot_ground_truth.get_par_Ia().reshape(n,1)]).reshape(n, 11)
            else:
                gt_params = robot_ground_truth.get_par_DYN().reshape((n, 10))
            # Friction
            if hasattr(robot, "get_par_Dl"):
                gt_params = np.hstack([gt_params.reshape(n,-1),
                                        robot_ground_truth.get_par_Dl().reshape(n,-1)]).reshape(n, -11)

            # compute error model
            e_DYN = np.abs(robot_ground_truth.get_par_DYN() - robot.get_par_DYN())
            e_model = e_DYN.reshape((n, 10))
            if par_per_link == 11:
                e_Ia = np.abs(robot_ground_truth.get_par_Ia() - robot.get_par_Ia())
                e_model = np.hstack([e_DYN.reshape(n,10), e_Ia.reshape(n,1)]).reshape(n, 11)
            # Friction
            if hasattr(robot, "get_par_Dl"):
                e_Dl = np.abs(robot_ground_truth.get_par_Dl() - robot.get_par_Dl())
                e_model = np.hstack([e_model.reshape(n,-1),
                                     e_Dl.reshape(n,-1)]).reshape(n, -1)

            # 2) Robot Ground-Truth ------------------------
            title = "Robot model ground-truth (DYNAMIC)"
            col_format = "c|" + "r" * len(headers)
            print(r"\begin{table}[htpb]")
            print(r"\centering")
            print(rf"\caption{{{title}}}")
            # Requires \usepackage{graphicx} in your LaTeX preamble
            print(r"\resizebox{\textwidth}{!}{%")
            print(rf"\begin{{tabular}}{{{col_format}}}")
            print(r"\hline")

            # Print Headers
            header_str = "Link & " + " & ".join([f"${h}$" for h in headers]) + r" \\ \hline"
            print(header_str)

            # Print Rows
            for i, row in enumerate(gt_params):
                row_strs = []
                for val in row:
                    # Convert to LaTeX scientific notation (e.g. 1.23e-04 -> $1.23 \times 10^{-4}$)
                    val_e = f"{val:.6e}"
                    base, exp = val_e.split('e')
                    exp_int = int(exp) # Removes leading zeros from exponent
                    row_strs.append(f"${base} \\times 10^{{{exp_int}}}$")

                print(f"{i} & " + " & ".join(row_strs) + r" \\")

            print(r"\hline")
            print(r"\end{tabular}%")
            print(r"}")
            print(r"\end{table}")
            print() # Add spacing between tables

            # 3) Error ------------------------
            title = "Variation from ground-truth (absolute error)"
            col_format = "c|" + "r" * len(headers)
            print(r"\begin{table}[htpb]")
            print(r"\centering")
            print(rf"\caption{{{title}}}")
            # Requires \usepackage{graphicx} in your LaTeX preamble
            print(r"\resizebox{\textwidth}{!}{%")
            print(rf"\begin{{tabular}}{{{col_format}}}")
            print(r"\hline")

            # Print Headers
            header_str = "Link & " + " & ".join([f"${h}$" for h in headers]) + r" \\ \hline"
            print(header_str)

            # Print Rows
            for i, row in enumerate(e_model):
                row_strs = []
                for val in row:
                    # Convert to LaTeX scientific notation (e.g. 1.23e-04 -> $1.23 \times 10^{-4}$)
                    val_e = f"{val:.6e}"
                    base, exp = val_e.split('e')
                    exp_int = int(exp) # Removes leading zeros from exponent
                    row_strs.append(f"${base} \\times 10^{{{exp_int}}}$")

                print(f"{i} & " + " & ".join(row_strs) + r" \\")

            print(r"\hline")
            print(r"\end{tabular}%")
            print(r"}")
            print(r"\end{table}")
            print() # Add spacing between tables
    else:
        # Motor inertia
        if par_per_link == 11:
            params = np.hstack([robot.get_par_DYN().reshape(n,10), 
                                robot.get_par_Ia().reshape(n,1)]).reshape(n, 11)
        else:
            params = robot.get_par_DYN().reshape((n, 10))
        # Friction
        if hasattr(robot, "get_par_Dl"):
            params = np.hstack([params.reshape(n,-1),
                                robot.get_par_Dl().reshape(n,-1)]).reshape(n, -11)
        # Print header
        header_str = f"{'Link':<6}" + "".join([f"{h:>14}" for h in headers])
        print(header_str)
        print("-" * len(header_str))
        # Print rows
        for i, row in enumerate(params):
            row_str = f"{i:<6}" + "".join([f"{val:>14.6e}" for val in row])
            print(row_str)

        if robot_ground_truth is not None:
            # ground-truth params
            # Motor inertia
            if par_per_link == 11:
                gt_params = np.hstack([robot_ground_truth.get_par_DYN().reshape(n,10), robot_ground_truth.get_par_Ia().reshape(n,1)]).reshape(n, 11)
            else:
                gt_params = robot_ground_truth.get_par_DYN().reshape((n, 10))
            # Friction
            if hasattr(robot, "get_par_Dl"):
                gt_params = np.hstack([gt_params.reshape(n,-1),
                                        robot_ground_truth.get_par_Dl().reshape(n,-1)]).reshape(n, -11)

            # compute error model
            e_DYN = np.abs(robot_ground_truth.get_par_DYN() - robot.get_par_DYN())
            e_model = e_DYN.reshape((n, 10))
            if par_per_link == 11:
                e_Ia = np.abs(robot_ground_truth.get_par_Ia() - robot.get_par_Ia())
                e_model = np.hstack([e_DYN.reshape(n,10), e_Ia.reshape(n,1)]).reshape(n, 11)
            # Friction
            if hasattr(robot, "get_par_Dl"):
                e_Dl = np.abs(robot_ground_truth.get_par_Dl() - robot.get_par_Dl())
                e_model = np.hstack([e_model.reshape(n,-1),
                                     e_Dl.reshape(n,-1)]).reshape(n, -1)

            # 2) Robot Ground-Truth ------------------------
            print(f"Robot model ground-truth (DYNAMIC)")
            if par_per_link == 11:
                params = np.hstack([robot.get_par_DYN().reshape(n,10), robot.get_par_Ia().reshape(n,1)]).reshape(n, 11)
            else:
                params = robot_ground_truth.get_par_DYN().reshape((n, 10))
            # Friction
            if hasattr(robot, "get_par_Dl"):
                params = np.hstack([params.reshape(n,-1),
                                    robot_ground_truth.get_par_Dl().reshape(n,-1)]).reshape(n, -11)
            # Print header
            header_str = f"{'Link':<6}" + "".join([f"{h:>14}" for h in headers])
            print(header_str)
            print("-" * len(header_str))
            # Print rows
            for i, row in enumerate(params):
                row_str = f"{i:<6}" + "".join([f"{val:>14.6e}" for val in row])
                print(row_str)

            # 3) Variation from  Ground-Truth ------------------------
            print(f"Variation from ground-truth (absolute error)")
            params = e_model
            # Print header
            header_str = f"{'Link':<6}" + "".join([f"{h:>14}" for h in headers])
            print(header_str)
            print("-" * len(header_str))
            # Print rows
            for i, row in enumerate(params):
                row_str = f"{i:<6}" + "".join([f"{val:>14.6e}" for val in row])
                print(row_str)


def plot_link_solution(robot, robot_gt, n, title = None, block = True,
                    title_size = 20, suptitle_size=24, lable_size = 16, tick_size = 14, legend_size = 13):
    """
    Plot estimated dynamic parameters of a single robot link against their ground-truth

    Args:
        robot (_type_): _description_
        robot_gt (_type_): _description_
        n (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
        block (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Ground truth
    m_gt = robot_gt.get_par_DYN()[n*10 + 0]
    com_gt = np.array([robot_gt.get_par_DYN()[n*10 + 1],
                       robot_gt.get_par_DYN()[n*10 + 2],
                       robot_gt.get_par_DYN()[n*10 + 3]])
    Ib_gt = np.array([ robot_gt.get_par_DYN()[n*10 + 4],
                       robot_gt.get_par_DYN()[n*10 + 5],
                       robot_gt.get_par_DYN()[n*10 + 6],
                       robot_gt.get_par_DYN()[n*10 + 7],
                       robot_gt.get_par_DYN()[n*10 + 8],
                       robot_gt.get_par_DYN()[n*10 + 9]])
    # Estimation
    m_estim  = robot.get_par_DYN()[n*10 + 0]
    com_estim  = np.array([robot.get_par_DYN()[n*10 + 1],
                           robot.get_par_DYN()[n*10 + 2],
                           robot.get_par_DYN()[n*10 + 3]])
    Ib_estim  = np.array([ robot.get_par_DYN()[n*10 + 4],
                           robot.get_par_DYN()[n*10 + 5],
                           robot.get_par_DYN()[n*10 + 6],
                           robot.get_par_DYN()[n*10 + 7],
                           robot.get_par_DYN()[n*10 + 8],
                           robot.get_par_DYN()[n*10 + 9]])

    # 3. Prepare data for plotting
    com_labels = ['x', 'y', 'z']
    inertia_labels = ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz']

    urdf_inertia = [Ib_gt[0], Ib_gt[3], Ib_gt[5], 
                    Ib_gt[1], Ib_gt[2], Ib_gt[4]]
    params_inertia = [Ib_estim[0], Ib_estim[3], Ib_estim[5], 
                    Ib_estim[1], Ib_estim[2], Ib_estim[4]]

    # 4. Create Subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sup = title if title is not None else f"Estimation of link {n}"
    fig.suptitle(sup, fontsize=suptitle_size, fontweight='bold')


    width = 0.35
    # Subplot A: Mass
    axes[0].bar(['Model', 'Estimation'], [m_gt, m_estim], color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_title('Mass (kg)', fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    # Subplot B: CoM
    x_com = np.arange(len(com_labels))
    axes[1].bar(x_com - width/2, com_gt, width, label='Model', color='#3498db')
    axes[1].bar(x_com + width/2, com_estim, width, label='Estimation', color='#e74c3c')
    axes[1].set_xticks(x_com)
    axes[1].set_xticklabels(com_labels)
    axes[1].set_title('Center of Mass (m)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    # Subplot C: Inertia Elements
    x_ine = np.arange(len(inertia_labels))
    axes[2].bar(x_ine - width/2, urdf_inertia, width, label='Model', color='#3498db')
    axes[2].bar(x_ine + width/2, params_inertia, width, label='Estimation', color='#e74c3c')
    axes[2].set_xticks(x_ine)
    axes[2].set_xticklabels(inertia_labels)
    axes[2].set_title('Inertia (kg·m²)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)
    for ax in axes:
        # Axis-Style (title_size = 20, lable_size = 16, tick_size = 14, legend_size = 13)
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontsize(lable_size)
        ax.yaxis.label.set_fontsize(lable_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                    text.set_fontsize(legend_size)

    ax.title.set_fontsize(suptitle_size)
    plt.tight_layout()
    plt.show(block = block)
    return fig


def check_feasibility(robot):
    par_DYN = robot.get_par_DYN()
    for i in range(robot.numJoints):
        par_DYN_link_i = par_DYN[10*i:10*(i+1)]
        if par_DYN_link_i[0] <= 0: print(f"mass of link {i} is non positive")
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

        # Triangular inequality
        eig,_ = np.linalg.eig(Ib)
        l_max = np.max(eig)
        casadi_eig = f_powerm(Ib)
        if not (np.trace(Ib)/2 - l_max > 0): 
            print(f"Inertia of link {i} do not pass matrix inequality test. (resid: {np.trace(Ib)/2 - l_max})")
            # print(f"CasADi - Inertia of link {i} do not pass matrix inequality test. (resid: {np.trace(Ib)/2 - casadi_eig[2]})")
            # print(f"num - eig")
            # print(eig)
            # print(f"CasADi - eig")
            # print(casadi_eig)

        # Determinant Test
        ds = [det(Ib[:3,:3]), det(Ib[:2,:2]), det(Ib[:1,:1])]
        for d in ds: 
            if d <= 0: print(f"Inertia of link {i} do not pass Sylvewster Criterion. (d: {d})")


def check_par_feasibility(par_DYN,n):
    for i in range(n):
        par_DYN_link_i = par_DYN[10*i:10*(i+1)]
        if par_DYN_link_i[0] <= 0: print(f"mass of link {i} is non positive")
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

        # Triangular inequality
        eig,_ = np.linalg.eig(Ib)
        l_max = np.max(eig)
        if not (np.trace(Ib)/2 - l_max > 0): 
            print(f"Inertia of link {i} do not pass matrix inequality test. (resid: {np.trace(Ib)/2 - l_max})")
