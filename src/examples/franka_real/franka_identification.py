"""Estimate reduced inertial parameters from ROS2 bag joint data.

This script reads joint states from a rosbag, filters the signals,
computes the dynamic regressor for a reduced parameter set, and
estimates parameters by least-squares. It also computes simple
friction terms (viscous + smooth static) and plots results.

The code relies on helper functions and classes from `utils`.
"""
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder

# Set up Franka Model ------
config_path = './src/examples/franka_real/config/identif_config.yaml'
with open(config_path, 'r') as f:
   config = yaml.load(f, Loader=SafeLoader)
franka = thunder.thunder_['franka']()
franka_ground_truth = thunder.thunder_['franka']()


thunder.load_params(franka, config['robot']['path'])
thunder.load_params(franka_ground_truth, "./src/thunder/franka_generatedFiles/param/franka_par.yaml")

# Setup Identifier Object and Solve Identification Problem ----
identifier = Identifier(franka, config_path=config_path)
identifier.init()                      # 1_ load and process trajectory
identifier.save_plot(path = "./src/examples/franka_real/results/before")     # 5_ save plot
# store torque before optimization
time = identifier.trajectory.t
tau_filtered = identifier.trajectory.tau
tau_prior = identifier.trajectory.compute_ne(franka)

identifier.solve_base_parameter()      # 2_ compute parameters in the base
fig_base = plot_base_identification(robot=franka, traject=identifier.trajectory, metrics=identifier.metrics)

identifier.solve_full_dynamics()       # 3_ compute all dynamics parameters
identifier.save_plot(path = "./src/examples/franka_real/results")     # 5_ save plot
check_feasibility(franka)
plot_table(franka, franka_ground_truth)
plt.show(block=True)
identifier.export(path = "./src/examples/franka_real/results")        # 6_ export in thunder config yaml file


tau_optim = identifier.trajectory.compute_ne(franka)



# Paper plot unifying before and after optimization

# def plot_unified_results(time, tau_filtered, tau_prior, tau_wls, metrics_wls, save_path=None):
#     """
#     Generates a unified 4x2 subplot figure overlaying prior and optimized torque estimates
#     using LaTeX styling. Places the legend in the empty 8th subplot.
#     """
#     # Enable LaTeX rendering and set font styles
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "serif",
#         "axes.labelsize": 24,
#         "font.size": 28,
#         "legend.fontsize": 28,
#         "xtick.labelsize": 28,
#         "ytick.labelsize": 28,
#         "figure.titlesize": 28
#     })

#     # Create a 4x2 grid for 7 joints
#     fig, axs = plt.subplots(4, 2, figsize=(16, 14))
#     fig.subplots_adjust(hspace=0.5, wspace=0.3)
#     axs = axs.flatten()

#     for i in range(7):
#         ax = axs[i]
        
#         # Plot 1: Prior Estimate (Dashed Gray/Light Blue)
#         ax.plot(time, tau_prior[:, i], label=r'$\hat{\tau}_{\mathrm{prior}}$', 
#                 color='black', linestyle='--', linewidth=1.5)
        
#         # Plot 2: Optimized Estimate WLS (Solid Blue)
#         ax.plot(time, tau_wls[:, i], label=r'$\hat{\tau}_{\mathrm{WLS}}$', 
#                 color='tab:blue', linewidth=1.5)
        
#         # Plot 3: Ground Truth / Filtered (Solid Red)
#         ax.plot(time, tau_filtered[:, i], label=r'$\tau_{\mathrm{filtered}}$', 
#                 color='tab:red', linewidth=1.5)

#         # Labels and formatting
#         ax.set_title(rf'\textbf{{Torque panda\_joint{i+1}}}')
#         ax.set_xlabel(r'Time [s]')
#         ax.set_ylabel(r'Torque [Nm]')
#         ax.grid(True, linestyle=':', alpha=0.7)
        
#         # Notice: The individual legend call has been removed from here!

#     # --- Legend Placement in the 8th Subplot ---
#     axs[7].axis('off')
#     handles, labels = axs[0].get_legend_handles_labels()
    
#     # Make the legend box darker and more visible
#     axs[7].legend(
#         handles, labels, 
#         loc='center', 
#         frameon=True, 
#         framealpha=1.0,       # Makes the background fully opaque (no transparency)
#         edgecolor='black',    # Adds a solid black border
#         facecolor='white',    # Ensures a stark white background
#         shadow=True,          # Adds a drop shadow to lift it off the page
#         fancybox=True         # Rounds the corners slightly
#     )

#     # Automatically adjust subplots to prevent overlaps
#     fig.tight_layout(rect=[0, 0, 1, 0.90])
#     # -------------------------------------------

#     rmse_prior = 3.1364  
#     rmse_opt = 0.6734
#     cond_num = metrics_wls.get('conditioning number', 76.653)
#     # sigma_max = metrics_wls.get('sigma max', 676.518) # Unused in title

#     # Unified Title
#     suptitle_str = (
#         rf"Prior RMSE: {rmse_prior:.4f} Nm $\vert$ WLS RMSE: {rmse_opt:.4f} Nm" + "\n" +
#         rf"Conditioning Number ($\kappa$): {cond_num:.3f}"
#     )
#     fig.suptitle(suptitle_str, y=0.98)

#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
#     return fig

# plot_unified_results(time,tau_filtered,tau_prior,tau_optim, identifier.metrics, "./src/examples/franka_real/results/ne_comparison.png")



