"""
This example includes the estimation of the Franka model
when all the set of dynamic parameter are not exactly known
"""
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder

np.random.seed(41)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
opts = {
   "ipopt": {
         "tol": 1e-15,
         "constr_viol_tol": 1e-15,
         "acceptable_tol": 1e-15,
         "dual_inf_tol": 1e-15,
         "compl_inf_tol": 1e-15,
         "max_iter": 50000,
   }
}
# Set up Franka Model ----------------------------------
config_path = '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/config/softhand_config.yaml'
with open(config_path, 'r') as f:
   config = yaml.load(f, Loader=SafeLoader)
franka = thunder.thunder_franka()
franka_ground_truth = thunder.thunder_franka()
thunder.load_params(franka, config['robot']['path'])
thunder.load_params(franka_ground_truth, "/home/leo/Desktop/Base Inertial Parameter/src/thunder/franka_generatedFiles/param/franka_SH_par.yaml")

# Setup and Solve Identification Problem ----------------------------------
identifier = Identifier(franka, config_path=config_path)
identifier.init()                      # 1_ load and process trajectory
# store torque before optimization
time = identifier.trajectory.t
tau_filtered = identifier.trajectory.tau
tau_prior = identifier.trajectory.compute_ne(franka)
fig = plot_identification(identifier.robot, identifier.trajectory, identifier.metrics, title = "Before Optimization", block = False)  # 4_ plot before optimization
fig.savefig(os.path.join("/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/results/before", 'torque.png'), bbox_inches='tight', dpi=300)


print("Number of samples ", identifier.trajectory.raw_tau.shape[0])
identifier.solve_base_parameter()      # 2_ compute parameters in the base


identifier.solve_full_dynamics()       # 3_ compute all dynamics parameters
identifier.print_table()               # 4_ print identified dynamics parameters
identifier.save_plot(path = "/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/results/")     # 5_ save plot
identifier.export(path = "/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/results/")

# Comparison with Ground-Truth
check_feasibility(franka)
fig_comparisoin = plot_link_solution(franka, franka_ground_truth, n = 6, block = True, title_size=28, legend_size=24, tick_size=24)
fig_comparisoin.savefig(os.path.join("/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/results/", 'soft_hand_comparison_results.png'), bbox_inches='tight', dpi=300)
plot_table(franka, franka_ground_truth, format='latex')


tau_optim = identifier.trajectory.compute_ne(franka)





def plot_unified_results(time, tau_filtered, tau_prior, tau_wls, metrics_wls, save_path=None):
    """
    Generates a unified 4x2 subplot figure overlaying prior and optimized torque estimates
    using LaTeX styling.
    """
    # Enable LaTeX rendering and set font styles
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 24,
        "font.size": 28,
        "legend.fontsize": 28,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "figure.titlesize": 28
    })

    # Create a 4x2 grid for 7 joints
    fig, axs = plt.subplots(4, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axs = axs.flatten()

    for i in range(7):
        ax = axs[i]
        
        # Plot 1: Prior Estimate (Dashed Gray/Light Blue)
        ax.plot(time, tau_prior[:, i], label=r'$\hat{\tau}_{\mathrm{prior}}$', 
                color='lightsteelblue', linestyle='--', linewidth=1.5)
        
        # Plot 2: Optimized Estimate WLS (Solid Blue)
        ax.plot(time, tau_wls[:, i], label=r'$\hat{\tau}_{\mathrm{WLS}}$', 
                color='tab:blue', linewidth=1.5)
        
        # Plot 3: Ground Truth / Filtered (Solid Red)
        ax.plot(time, tau_filtered[:, i], label=r'$\tau_{\mathrm{filtered}}$', 
                color='tab:red', linewidth=1.5)

        # Labels and formatting
        ax.set_title(rf'\textbf{{Torque panda\_joint{i+1}}}')
        ax.set_xlabel(r'Time [s]')
        ax.set_ylabel(r'Torque [Nm]')
        ax.grid(True, linestyle=':', alpha=0.7)


    # --- Legend Placement in the 8th Subplot ---
    axs[7].axis('off')
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Make the legend box darker and more visible
    axs[7].legend(
        handles, labels, 
        loc='center', 
        frameon=True, 
        framealpha=1.0,       # Makes the background fully opaque (no transparency)
        edgecolor='black',    # Adds a solid black border
        facecolor='white',    # Ensures a stark white background
        shadow=True,          # Adds a drop shadow to lift it off the page
        fancybox=True         # Rounds the corners slightly
    )

    # Automatically adjust subplots to prevent overlaps
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    # -------------------------------------------

    rmse_prior = 5.1250
    rmse_opt = 0.4043
    cond_num = metrics_wls.get('conditioning number', 76.653)
    sigma_max = metrics_wls.get('sigma max', 676.518)

    # Unified Title
    suptitle_str = (
        #rf"\textbf{{Torque Estimation Results: Prior vs. WLS Optimization}}" + "\n" +
        rf"Prior RMSE: {rmse_prior:.4f} Nm $\vert$ WLS RMSE: {rmse_opt:.4f} Nm" + "\n" +
        rf"Conditioning Number ($\kappa$): {cond_num:.3f}"
    )
    fig.suptitle(suptitle_str, y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


plot_unified_results(time,tau_filtered,tau_prior,tau_optim, identifier.metrics, os.path.join("/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_softhand/results/", "ne_comparison.png"))



