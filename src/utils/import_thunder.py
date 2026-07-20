import glob
import os
import sys
import yaml
from typing import Any

# IMPORT THUNDER FILES

# 1) Add installation path (where thunderd generated file are located)
#  --------------------------------------------------------------------
rel_paths = glob.glob("./src/thunder/*/build")

for rel_path in rel_paths:
    if os.path.exists(rel_path):
        sys.path.append(os.path.abspath(rel_path))
    else:
        print(f"[Thunder Import] Warning: path '{rel_path}' does not exist.")

# 2) Import modules
#  --------------------------------------------------------------------

import importlib
thunder_ = {}

for path in sys.path:
    # Scan for any shared objects (.so) or dynamically built modules starting with thunder_
    for filename in os.listdir(path) if os.path.isdir(path) else []:
        if filename.startswith("thunder_") and (filename.endswith(".so") or filename.endswith(".pyd") or os.path.isdir(os.path.join(path, filename))):
            # Extract the base module name (i.e., the robot name)
            module_name = filename.split('.')[0]
            if not module_name.endswith("_py"):
                continue
            robot_name = module_name.replace("_py", "") 
            try:
                # Dynamically import the module at runtime
                module = importlib.import_module(module_name)
                robot_class = getattr(module, robot_name)
                clean_key = robot_name.replace("thunder_", "")
                thunder_[clean_key] = robot_class
                print(f"[Thunder Import] Successfully auto-imported: {clean_key} from {module_name}")
            except (ImportError, AttributeError) as e:
                print(f"[Thunder Import] Failed to auto-import module {module_name}: {e}")

# 3) Define some utilities
#  --------------------------------------------------------------------
def load_params(robot:Any, config_path:str) -> None:
    """
    Parse robot parameters from a yaml file into robot model

    Args:
        filename (str): The configuration file location where parameters are stored.
    """
    with open(config_path, 'r') as f:
        config_robot = yaml.safe_load(f)

    # Dynamic parameters
    robot.set_par_DYN(config_robot['par_DYN'])
    robot.set_par_Dl(config_robot['par_Dl'])
    robot.set_par_REG(config_robot['par_REG'])
    robot.set_par_REG_red(config_robot['par_REG_red'])
    if hasattr(robot,"set_par_Ia"):
        robot.set_par_Ia(config_robot['par_Ia'])
    # Other parameters (EE effector transformation)
    if hasattr(robot,"set_par_Ln2EE"):
        robot.set_par_Ln2EE(config_robot['par_Ln2EE'])
    # Other parameters (Gravity definition in world frame)
    if hasattr(robot,"set_par_gravity"):
        robot.set_par_gravity(config_robot['par_gravity'])