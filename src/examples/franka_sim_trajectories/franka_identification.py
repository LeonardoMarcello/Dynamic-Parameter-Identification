# """Comparison of Base PArameters estimation using different trajectories.
# """
# from utils.Identifier import *
# from utils.evaluation_utils import *
# import utils.import_thunder as thunder

# franka_ground_truth = thunder.thunder_franka()
# thunder.load_params(franka_ground_truth, "/home/leo/Desktop/Base Inertial Parameter/src/thunder/franka_generatedFiles/param/franka_par.yaml")

# # --------
# # 1) Fourier Decomposition

# # Set up Franka Model ------
# config_path = '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/fourier_config.yaml'
# with open(config_path, 'r') as f:
#    config = yaml.load(f, Loader=SafeLoader)
# franka = thunder.thunder_franka()
# thunder.load_params(franka, config['robot']['path'])




# # Setup Identifier Object and Solve Identification Problem ----
# identifier = Identifier(franka, config_path=config_path)
# identifier.init()                      # 1_ load and process trajectory
# identifier.solve_base_parameter()      # 2_ compute parameters in the base
# fourier_BP = franka.get_par_REG_red().copy() # Save base parameters for comparison
# print("Base Parameters estimated with Fourier trajectory:")
# print(fourier_BP)
# print(identifier.metrics['conditioning number'])
# print(identifier.metrics['parameters relative standard deviation'])


# # --------
# # 2) Chirp Decomposition

# # Set up Franka Model ------
# config_path = '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/chirp_config.yaml'
# with open(config_path, 'r') as f:
#    config = yaml.load(f, Loader=SafeLoader)
# franka = thunder.thunder_franka()
# thunder.load_params(franka, config['robot']['path'])




# # Setup Identifier Object and Solve Identification Problem ----
# identifier = Identifier(franka, config_path=config_path)
# identifier.init()                      # 1_ load and process trajectory
# identifier.solve_base_parameter()      # 2_ compute parameters in the base
# chirp_BP = franka.get_par_REG_red().copy() # Save base parameters for comparison
# print("Base Parameters estimated with Chirp trajectory:")
# print(chirp_BP)
# print(identifier.metrics['conditioning number'])
# print(identifier.metrics['parameters relative standard deviation'])


# # --------
# # 3) Sinusoidal Decomposition

# # Set up Franka Model ------
# config_path = '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/sinusoidal_config.yaml'
# with open(config_path, 'r') as f:
#    config = yaml.load(f, Loader=SafeLoader)
# franka = thunder.thunder_franka()
# thunder.load_params(franka, config['robot']['path'])




# # Setup Identifier Object and Solve Identification Problem ----
# identifier = Identifier(franka, config_path=config_path)
# identifier.init()                      # 1_ load and process trajectory
# identifier.solve_base_parameter()      # 2_ compute parameters in the base
# sinusoidal_BP = franka.get_par_REG_red().copy() # Save base parameters for comparison
# print("Base Parameters estimated with Sinusoidal trajectory:")
# print(sinusoidal_BP)
# print(identifier.metrics['conditioning number'])
# print(identifier.metrics['parameters relative standard deviation'])
"""Comparison of Base Parameters estimation using different trajectories and LaTeX table generation.
"""
import re
import yaml
from yaml import SafeLoader
import numpy as np
from utils.Identifier import *
from utils.evaluation_utils import *
import utils.import_thunder as thunder

franka_ground_truth = thunder.thunder_franka()
thunder.load_params(franka_ground_truth, "/home/leo/Desktop/Base Inertial Parameter/src/thunder/franka_generatedFiles/param/franka_SH_par.yaml")
franka_ground_truth.set_par_REG(franka_ground_truth.get_dyn2reg())
franka_ground_truth.set_par_REG_red(franka_ground_truth.get_reg2red())


csv_filename =  '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/bp_data.csv'

# Dictionaries to hold results for table automation
trajectories = ['Fourier', 'Chirp', 'Sinusoidal']
config_files = {
    'Fourier': '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/fourier_config.yaml',
    'Chirp': '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/chirp_config.yaml',
    'Sinusoidal': '/home/leo/Desktop/Base Inertial Parameter/src/examples/franka_sim_trajectories/config/sinusoidal_config.yaml'
}

cond_numbers = {}
sigmas_max = {}
traces = {}
estimated_vals = {}
std_deviations = {}

# Read your custom-defined parameters structure
base_param_names = list(print_base_inertial_parameters(franka_ground_truth).values())

# -------- Run Identification for each trajectory --------
for traj in trajectories:
    with open(config_files[traj], 'r') as f:
       config = yaml.load(f, Loader=SafeLoader)
    
    franka = thunder.thunder_franka()
    thunder.load_params(franka, config['robot']['path'])

    identifier = Identifier(franka, config_path=config_files[traj])
    identifier.init()                      
    identifier.solve_base_parameter()      
    
    # Store matrix estimation metrics
    cond_numbers[traj] = identifier.metrics['conditioning number']
    sigmas_max[traj] = identifier.metrics['sigma max']
    traces[traj] = identifier.metrics['trace']
    std_deviations[traj] = identifier.metrics['parameters standard deviation']
    estimated_vals[traj] = franka.get_par_REG_red().copy()

# Extract the true base inertial parameter values from your ground truth instance
# depending on whether it stores a dict structure or numeric sequence array
gt_source = franka_ground_truth.get_par_REG_red().copy()
print(gt_source)

# -------- LaTeX Table Automation Generation --------
print("\n" + "="*20 + " GENERATING LATEX TABLE " + "="*20 + "\n")

latex_str = []
latex_str.append(r"\begin{table*}[htpb]")
latex_str.append(r"\centering")
latex_str.append(r"\resizebox{\textwidth}{!}{%")
latex_str.append(r"\begin{tabular}{|l|c|c|c|c|}")  # Expanded to 5 columns
latex_str.append(r"\hline")

def clean_cond(val):
    if isinstance(val, (np.ndarray, list)):
        return float(val[0])
    return float(val)

# Updated Header Row with Ground Truth element header included
#header_row = (
#    rf"\textbf{{Base Parameters}} & "
#    rf"\textbf{{Ground Truth}} & "
#    rf"\textbf{{Fourier}} ($\kappa={clean_cond(cond_numbers['Fourier']):.2f}$, $\sigma_{{\max}}={clean_cond(sigmas_max['Fourier']):.2f}$) & "
#    rf"\textbf{{Chirp}} ($\kappa={clean_cond(cond_numbers['Chirp']):.2f}$, $\sigma_{{\max}}={clean_cond(sigmas_max['Chirp']):.2f}$) & "
#    rf"\textbf{{Sinusoidal}} ($\kappa={clean_cond(cond_numbers['Sinusoidal']):.2f}$, $\sigma_{{\max}}={clean_cond(sigmas_max['Sinusoidal']):.2f}$) \\"
#)
header_row = (
    rf"\textbf{{Base Parameters}} & "
    rf"\textbf{{Ground Truth}} & "
    rf"\textbf{{Fourier}} ($\kappa={clean_cond(cond_numbers['Fourier']):.2f}$, tr={clean_cond(traces['Fourier']):.2e}$) & "
    rf"\textbf{{Chirp}} ($\kappa={clean_cond(cond_numbers['Chirp']):.2f}$, tr={clean_cond(traces['Chirp']):.2e}$) & "
    rf"\textbf{{Sinusoidal}} ($\kappa={clean_cond(cond_numbers['Sinusoidal']):.2f}$, tr={clean_cond(traces['Sinusoidal']):.2e}$) \\"
)
csv_rows = ["parameter,gt,fourier_est,fourier_std,chirp_est,chirp_std,sine_est,sine_std"]

latex_str.append(header_row)
latex_str.append(r"\hline")




def get_float_val(data_source, idx):
    """Safely extracts metric data value as native python float."""
    if isinstance(data_source, dict):
        raw_val = data_source.get(str(idx), data_source.get(idx, 0.0))
        if isinstance(raw_val, dict) and 'value' in raw_val:
            raw_val = raw_val['value']
    else:
        raw_val = data_source[idx]

    if hasattr(raw_val, 'item'): 
        return raw_val.item()
    elif isinstance(raw_val, (list, np.ndarray)):
        return float(raw_val[0])
    return float(raw_val)

def format_latex_expression(item):
    """Processes either list-of-strings or strings into clean LaTeX equations."""
    full_str = "".join(item) if isinstance(item, list) else str(item)
    full_str = re.sub(r"([\d.]+)[eE]([-\+]\d+)", r"\1 \\times 10^{\2}", full_str)
    full_str = full_str.replace("*", r"\cdot ")
    return f"${full_str}$"

# Data rows loop matching sequentially index-by-index
for i, item in enumerate(base_param_names):
    param_latex = format_latex_expression(item)
    
    # Extract Ground Truth scalar value
    val_gt = get_float_val(gt_source, i)
    
    # Extract estimated value properties
    val_f = get_float_val(estimated_vals['Fourier'], i)
    std_f = get_float_val(std_deviations['Fourier'], i)
    
    val_c = get_float_val(estimated_vals['Chirp'], i)
    std_c = get_float_val(std_deviations['Chirp'], i)
    
    val_s = get_float_val(estimated_vals['Sinusoidal'], i)
    std_s = get_float_val(std_deviations['Sinusoidal'], i)
    
    row = (
        rf"{i+1}. {param_latex} & "
        rf"{val_gt:.2e} & "  # Added Ground Truth output
        rf"{val_f:.2e} ($\pm$ {std_f:.2e}) & "
        rf"{val_c:.2e} ($\pm$ {std_c:.2e}) & "
        rf"{val_s:.2e} ($\pm$ {std_s:.2e}) \\"
    )

    latex_str.append(row)

    csv_row = f"{i+1},{val_gt:.2e},{val_f:.2e},{std_f:.2e},{val_c:.2e},{std_c:.2e},{val_s:.2e},{std_s:.2e}"
    csv_rows.append(csv_row)


latex_str.append(r"\hline")
latex_str.append(r"\end{tabular}")
latex_str.append(r"}")
latex_str.append(r"\caption{Ground Truth comparison and Percentage Error ($\pm$ Estimated Covariance) for the reduced base inertial parameters across three datasets.}")
latex_str.append(r"\label{tab:reduced_base_params_comparison}")
latex_str.append(r"\end{table*}")




print("\n".join(latex_str))


with open(csv_filename, "w") as file:
    file.write("\n".join(csv_rows) + "\n")
print(f"\n[INFO] CSV data successfully saved to: {csv_filename}\n")



# -------- Printing Max/Min Metrics with Indices --------
print("\n" + "="*20 + " ADDITIONAL METRICS METADATA " + "="*20 + "\n")
# Convert estimations and ground truth cleanly to numpy arrays for vector math
gt_arr = np.array([get_float_val(gt_source, i) for i in range(len(base_param_names))])
fourier_arr = np.array([get_float_val(estimated_vals['Fourier'], i) for i in range(len(base_param_names))])
chirp_arr = np.array([get_float_val(estimated_vals['Chirp'], i) for i in range(len(base_param_names))])
sinusoidal_arr = np.array([get_float_val(estimated_vals['Sinusoidal'], i) for i in range(len(base_param_names))])

fourier_std = np.array([get_float_val(std_deviations['Fourier'], i) for i in range(len(base_param_names))])
chirp_std = np.array([get_float_val(std_deviations['Chirp'], i) for i in range(len(base_param_names))])
sinusoidal_std = np.array([get_float_val(std_deviations['Sinusoidal'], i) for i in range(len(base_param_names))])

# Calculate absolute parameter errors
fourier_err = np.abs(fourier_arr - gt_arr)
chirp_err = np.abs(chirp_arr - gt_arr)
sinusoidal_err = np.abs(sinusoidal_arr - gt_arr)

# 1) Max Standard Deviations
idx_max_std_f = np.argmax(fourier_std)
idx_max_std_c = np.argmax(chirp_std)
idx_max_std_s = np.argmax(sinusoidal_std)

print(f"max Fourier std.dev: {fourier_std[idx_max_std_f]:.4f} (at Index: {idx_max_std_f})")
print(f"max Chirp std.dev: {chirp_std[idx_max_std_c]:.4f} (at Index: {idx_max_std_c})")
print(f"max Sinusoidal std.dev: {sinusoidal_std[idx_max_std_s]:.4f} (at Index: {idx_max_std_s})")
print("-" * 30)

# 2) Max Estimation Error
idx_max_err_f = np.argmax(fourier_err)
idx_max_err_c = np.argmax(chirp_err)
idx_max_err_s = np.argmax(sinusoidal_err)

print(f"max Fourier est. error: {fourier_err[idx_max_err_f]:.4f} (at Index: {idx_max_err_f})")
print(f"max Chirp est. error: {chirp_err[idx_max_err_c]:.4f} (at Index: {idx_max_err_c})")
print(f"max Sinusoidal est. error: {sinusoidal_err[idx_max_err_s]:.4f} (at Index: {idx_max_err_s})")
print("-" * 30)

# 3) Min Estimation Error
idx_min_err_f = np.argmin(fourier_err)
idx_min_err_c = np.argmin(chirp_err)
idx_min_err_s = np.argmin(sinusoidal_err)

print(f"min Fourier est. error: {fourier_err[idx_min_err_f]:.4f} (at Index: {idx_min_err_f})")
print(f"min Chirp est. error: {chirp_err[idx_min_err_c]:.4f} (at Index: {idx_min_err_c})")
print(f"min Sinusoidal est. error: {sinusoidal_err[idx_min_err_s]:.4f} (at Index: {idx_min_err_s})")