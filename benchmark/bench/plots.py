"""Shared plotting style and helpers for the benchmark figures."""
import matplotlib.pyplot as plt
import numpy as np

from bench.ident import build_regressor

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10, "axes.labelsize": 10,
                     "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})

COLORS = {"Measured": "#C0392B", "Prior (CAD)": "#7F7F7F", "BP-OLS (= project WLS)": "#E67E22",
          "BP-WLS (fixed)": "#2E86C1", "MAP-WLS (IPOPT)": "#1F3A93", "MAP-OLS (IPOPT, as published)": "#AF7AC5",
          "MAP-convex (SDP)": "#16A085", "FIGAROH BP (WLS)": "#D4AC0D", "FIGAROH (WLS + LMI proj.)": "#B7950B",
          "FIGAROH physical fit": "#D4AC0D", "FIGAROH per-link projection": "#16A085",
          "FIGAROH exact reconstruction (SDP)": "#AF7AC5", "Ground truth": "#3498db"}
SHORT = {"FIGAROH physical fit": "FIGAROH-PF", "FIGAROH per-link projection": "FIGAROH-PLP",
         "FIGAROH exact reconstruction (SDP)": "FIGAROH-exact", "Ground truth": "GT",
         "MAP-WLS (IPOPT)": "MLE-WLS"}  # RMSE tags where the first word is ambiguous
# Names shown in the figures (the results keep the historical "MAP" keys; the paper calls the method MLE,
# the prior being interpreted as virtual measurements)
DISPLAY = {"MAP-WLS (IPOPT)": "MLE-WLS (IPOPT)", "MAP-OLS (IPOPT, as published)": "MLE-OLS (IPOPT, as published)",
           "MAP-convex (SDP)": "MLE-convex (SDP)"}


def torque_plot(obj, tm, model_names, path, title="", preds=None):
    """preds: optional {name: stacked torque prediction (same layout as T.ravel())} from external models
    (e.g. FIGAROH), plotted after model_names."""
    preds = preds or {}
    r = obj["robot"]
    Y, T = build_regressor(r, tm)
    th = {m: Y @ obj["models"][m] for m in model_names if m not in preds}
    th.update(preds)
    torque_plot_arrays(tm.t, T, th, r.numJoints, path, title)


def torque_plot_arrays(t, T, preds, n, path, title=""):
    """Same figure as torque_plot from precomputed arrays (e.g. saved to disk): t (N,) time, T measured torques
    stacked as T.ravel() (N n), preds {name: stacked prediction (N n)} in plot order."""
    tau = np.asarray(T).reshape(-1, n)
    cols = 2 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.1 * rows), sharex=True)
    axs = axs.ravel()
    for j in range(n):
        ax = axs[j]
        ax.plot(t, tau[:, j], color=COLORS["Measured"], lw=1.6, label=r"$\tau$ measured (filtered)")
        names = list(preds)
        for m in names:
            th = np.asarray(preds[m]).reshape(-1, n)[:, j]
            rm = np.sqrt(np.mean((th - tau[:, j]) ** 2))
            ax.plot(t, th, color=COLORS.get(m), lw=1.1, ls="--" if "Prior" in m else "-",
                    label=DISPLAY.get(m, m))
            ax.text(0.01, 0.95 - 0.13 * names.index(m), f"{SHORT.get(m, m.split(' ')[0])} {rm:.3f}", transform=ax.transAxes,
                    fontsize=7, color=COLORS.get(m), va="top")
        ax.set_title(f"joint {j + 1}", fontsize=9)
        ax.set_ylabel("N m")
    for k in range(n, len(axs)):
        axs[k].axis("off")
    h, l = axs[0].get_legend_handles_labels()
    (axs[n] if n < len(axs) else axs[0]).legend(h, l, loc="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Labels, colours and line styles of the paper torque figure (measured torque is added in red by the plot)
PAPER_STYLES = {"Prior (CAD)": (r"$\hat{\tau}_{\mathrm{prior}}$", "black", "--"),
                "MAP-WLS (IPOPT)": (r"$\hat{\tau}_{\mathrm{MLE}}$", "tab:blue", "-", 2.5),   # drawn last, on top
                "FIGAROH physical fit": (r"$\hat{\tau}_{\mathrm{Fig,PF}}$", "tab:orange", "-"),
                "FIGAROH per-link projection": (r"$\hat{\tau}_{\mathrm{Fig,PLP}}$", "tab:green", "-"),
                "FIGAROH exact reconstruction (SDP)": (r"$\hat{\tau}_{\mathrm{Fig,exact}}$", "tab:purple", "-"),
                "Ground truth": (r"$\hat{\tau}_{\mathrm{GT}}$", "tab:gray", ":", 2.0)}


def torque_plot_paper(t, T, preds, n, path, styles, rmse, cond=None):
    """Paper-style torque figure, same layout as plot_unified_results in src/examples/franka_real (LaTeX fonts,
    4x2 grid, legend in the last panel, RMSE of each model in the title). t (N,), T measured torques stacked as
    T.ravel() (N n), preds {name: stacked prediction (N n)} in plot order, styles {name: (latex label, color, ls[, lw])},
    rmse {name: torque RMSE [N m]} shown in the title."""
    tau = np.asarray(T).reshape(-1, n)
    rc = {"text.usetex": True, "font.family": "serif", "axes.labelsize": 24, "font.size": 28, "legend.fontsize": 28,
          "xtick.labelsize": 28, "ytick.labelsize": 28, "figure.titlesize": 28, "axes.grid": False}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(4, 2, figsize=(16, 14))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        axs = axs.flatten()
        for i in range(n):
            ax = axs[i]
            for m, p in preds.items():
                label, color, ls, lw = (*styles[m], 1.5)[:4]
                ax.plot(t, np.asarray(p).reshape(-1, n)[:, i], label=label, color=color, linestyle=ls, linewidth=lw)
            ax.plot(t, tau[:, i], label=r'$\tau_{\mathrm{filtered}}$', color='tab:red', linewidth=1.5)
            ax.set_title(rf'\textbf{{Torque panda\_joint{i + 1}}}')
            ax.set_xlabel(r'Time [s]')
            ax.set_ylabel(r'Torque [Nm]')
            ax.grid(True, linestyle=':', alpha=0.7)
        for k in range(n, len(axs)):
            axs[k].axis('off')
        handles, labels = axs[0].get_legend_handles_labels()
        axs[-1].legend(handles, labels, loc='center', frameon=True, framealpha=1.0, edgecolor='black',
                       facecolor='white', shadow=True, fancybox=True)
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        title = "RMSE " + r" $\vert$ ".join(rf"{styles[m][0]}: {rmse[m]:.4f} Nm" for m in preds)
        if cond is not None:
            title += "\n" + rf"Conditioning Number ($\kappa$): {cond:.3f}"
        fig.suptitle(title, y=0.98)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def link_param_plot(dyns, n, path, title=None, colors=None):
    """Mass / CoM / inertia of link n (0-based, par_DYN order) for several models, styled as
    src/utils/evaluation_utils.plot_link_solution. dyns: {label: full par_DYN vector}."""
    colors = colors or {}
    default = ["#3498db", "#e74c3c", "#7F7F7F", "#16A085", "#D4AC0D", "#AF7AC5"]
    labels = list(dyns)
    cols = [colors.get(k, default[i % len(default)]) for i, k in enumerate(labels)]
    p = {k: np.asarray(d)[10 * n:10 * n + 10] for k, d in dyns.items()}
    width = 0.8 / len(labels)
    off = lambda i: (i - (len(labels) - 1) / 2) * width
    fig, axes = plt.subplots(1, 3, figsize=(16 if len(labels) <= 2 else 20, 5))
    fig.suptitle(title if title is not None else f"Estimation of link {n}", fontsize=24, fontweight="bold")
    # Mass
    many = len(labels) > 2   # several models: one shared legend below the figure instead of per-axis legends
    axes[0].bar(labels, [p[k][0] for k in labels], color=cols, alpha=0.8)
    if many:
        axes[0].set_xticks([])
    axes[0].set_title("Mass (kg)", fontweight="bold")
    # CoM and inertia (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    for ax, idx, ticks, name in [(axes[1], [1, 2, 3], ["x", "y", "z"], "Center of Mass (m)"),
                                 (axes[2], [4, 7, 9, 5, 6, 8], ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"],
                                  "Inertia (kg·m²)")]:
        x = np.arange(len(idx))
        for i, k in enumerate(labels):
            ax.bar(x + off(i), p[k][idx], width, label=DISPLAY.get(k, k), color=cols[i])
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_title(name, fontweight="bold")
        if not many:
            ax.legend(fontsize=24)
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.grid(axis="x", visible=False)
        ax.title.set_fontsize(28)
        ax.tick_params(axis="both", labelsize=24)
    fig.tight_layout()
    if many:
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), fontsize=20, frameon=False)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
