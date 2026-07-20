# Dynamic Parameter Identification

This directory contains the main Python modules and scripts for identifying the dynamics of a robotic serial manipulator.

## Project Overview

The project contains an Object-Oriented software implementing the identification pipeline:
1. Loading and processing trajectory data
2. Estimating base dynamic parameter sets via different LS solution
3. Estimating full dynamic parameter sets via costrained optimization problem
4. Results visualization and export


The **base parameter identification** is implemented via:
- **OLS** (Ordinary Least Squared error)
- **WLS** (Weighted Least Squared error)
- **LS|prior** (Least Squared error with prior)

The **full parameter identification** is implemented in CasADi using the solver:
- in CasADi using the solver **IPOTP** (Interior Point) or 
- in SciPy using the **simulated annealing** algorithm

The **Results** are stored in a folder containing the following:
- **Trajectory Plots**
- **Reconstructed trajectory and metrics Plot**
- **Thunder formatted configiguration file**

## Installation
### Option 1: Using Docker
```bash
git clone https://github.com/LeonardoMarcello/Dynamic-Parameter-Identification.git

cd base_inertial_parameters

docker build -t base-inertial-parameters:latest -f docker/Dockerfile .
```

### Option 2: Using Conda virtual environment
```bash
git clone https://github.com/LeonardoMarcello/Dynamic-Parameter-Identification.git

cd dynamic_parameters_identification

conda env create -f environment.yaml
conda activate dynamic_parameters_identification

pip install -e .
```



## Example

This project contains some example of the **dynamic parameter identification** on real hardware:
- **7DoF collaporative robot, the Franka Emika Panda** The identification of the dynamiks of the Franka Emika Panda Manipulator
- **Anthropomorphic robotic hand, the Wonkik Allegro Hand** The identification of theparameters for the fingers of the Wonik Allegro Hand V5
--
The following data were collected in Gazebo simualtive environment for accuracy evaluation
- **Franka Panda with payload** The Identification of the dynamics of a SoftHand robotic platform mounted on the end-effector of the Franka Emika Panda Manipulator
- **Franka Panda estimation with different trajectorries** The Identification of the dynamics of the Franka Emika Panda Manipulator is compared across 3 trajectories: Fourier, chirp, and Sinusoidal

The example files ar located
```
examples/
├── franka_real/
│    ├── results/
│    │   └── ...
│    ├── config/
│    │   └── ...
│    └── franka_identification.py
├── franka_sim_softhand/
├── franka_sim_trajectories/
└── allegro_hand/
     ├── config/
     ├── ahand_finger/
     └── ahand_thumb/
```

### Directories

#### `utils/`
Utility modules providing core functionality:

- **`Identifier.py`** - Main identification class
  - Loads and processes trajectory data
  - Solves base parameter identification problem
  - Computes full dynamics parameters
  - Handles result storage and export
  - Methods: `init()`, `solve_base_parameter()`, `solve_full_dynamics()`, `print_table()`, `save_plot()`, `export()`

- **`import_thunder.py`** - Thunder dynamics generated code interface
  - Wraps Thunder generated code import in the identification framework
  - Handles parameter loading

- **`loader_utils.py`** - TrajectoryManager class definition aiding data loading and processing
  - ROS2 bag file parsing
  - Joint state extraction
  - Signal filtering (low-pass filters)
  - Trajectory segmentation

- **`identification_utils.py`** - Identification algorithms
  - Dynamic regressor computation
  - Reduced parameter set generation
  - Least-squares estimation
  - Friction model fitting

- **`evaluation_utils.py`** - Results evaluation
  - Parameter error computation
  - Prediction error analysis
  - Plot generation (actual vs predicted)
  - Statistical metrics

- **`casadi_utils.py`** - CasADi function definition utilities
  - Optimization problem setup
  - iLQR solver integration
  - Constraint handling

- **`ros2_utils.py`** - ROS2 interface utilities
  - Topic subscription and data collection
  - Message parsing

#### `examples/`
Example configurations and workflows for different robots:
- `franka_softhand/` - Franka + soft hand examples
- `ahand_thumb/` - Allegro Hand thumb examples
- `ahand_finger/` - Allegro Hand finger examples

#### `thunder/`
Thunder generated file for robot modelling, all robot generated code must be placed here: 
(see [Thunder Dynamics](https://github.com/CentroEPiaggio/thunder_dynamics) for more details about how to generating robot model code)
- Auto-generated robot models and parameters
- Dynamic equations and regressor computation
- Parameter definitions and constraints

#### `templates/`
Templates to ease the environment set-up:
- Identification configuration file
- directory with CSV data format

## Usage

### Basic Workflow

1. **Prepare configuration file**:
   ```yaml
    # config/identification_config.yaml
    robot:
      path: 'path/to/thunder/config/file/par.yaml'

    trajectory:
      type: 'ros2' # 'csv' or 'ros2' loader
      bag: 'path/to/ros2/bag'     # use for 'ros2' loader
      topic: '/joint/topic/name'  # use for 'ros2' loader
      data: '/path/to/csv/dir'    # use for 'csv' loader

      interval: [0, 1000]         # trajectory time interval for data selection [s]
      joints: ['joint1_name', 'joint2_name', '...', 'jointN_name']  # names of robot joint


    processing:
      pipeline:   # Which pipeline apply for data processing ['filtering','differentiating','decimating'] 
        - 'filtering'
        - 'differentiating'
        - 'filtering'
        - 'differentiating'
        - 'filtering'
        - 'decimating'
      cut_off_frequency: 30   # lowpass filtering cutoff frequency [rad/s]
      butterworth_order: 5    # butterworth filtering order
      apply_decimation: true  #
      decimate_factor: 10     # reduce samples of this many times
      flip_torques: false     # wheter change joint torce sign (e.g. when using raw Franka Emika Panda)


    identification:
      method: 'WLS' # OLS, WLS, OLS|prior, WLS|prior: Base Inertial parameters Estimation Method
      conditioning_ratio: None # truncate SVD inverse by forcing this conditioning (Default: None)

      positive_threshold: 1e-16   # Positive threshold for constrained problem solver (Default: 1e-16)
      inertia_scale_factor: 1     # Scaling of inertia matrix to ease check on definitive positive costraint  (Default: 1)
      weight:
        loss: 1         # gain factor for base parameters optimization (Default: 1)
        mass: 1e-3      # mass std.dev [Kg]
        CoM: 1e-3       # CoM std.dev [m]
        inertia: 1e-6   # Inertia std.dev [Kg/m^2]
        link: [1, 1, 1, 1, 1, 1, 1] # Link for link prior selection, i.e. multiplier of 1/std.dev. Use 0 to ignore that link

        prior_yaml: "path/to/yaml/with/prior.yaml" # wheter use specific prior (Default: None)
   ```
2. **Set up the identification file**:
   ```python
   # main.py
   
   # import thunder libraries and 
   import utils.import_thunder as thunder
   from utils.Identifier import *

   # Set up Robot Model ------
   config_path = '/path/to/identification/config/identification_config.yaml'
   with open(config_path, 'r') as f:
      config = yaml.load(f, Loader=SafeLoader)

   robot = thunder.thunder_['robot']()
   thunder.load_params(robot, config['robot']['path']) # This function allows to load priors


   # Setup Identifier Object and Solve Identification Problem ----
   Identifier = Identifier(robot, config_path=config_path)

   Identifier.init()                     # 1_ load and process trajectory
   Identifier.solve_base_parameter()     # 2_ compute parameters in the base. 
                                         #    They are stored using robot.set_par_REG_red(value)
   Identifier.solve_full_dynamics()      # 3_ compute all dynamics parameters
                                         #    They are stored using robot.set_par_DYN(value), 
                                         #    robot.set_par_REG(value)
   Identifier.print_table()              # 4_ print identified dynamics parameters
   Identifier.save_plot(path = "/path/to/directoy/results")     # 5_ save plot
   Identifier.export(path = "/path/to/directoy/results")        # 6_ export thunder config yaml file
  
2. **Run identification**:
   ```bash
   python main.py
   ```

3. **Results**:
   - Identified parameters saved to `/path/to/directoy/results` (Default: `results/data_DD_MM_YYYY/`)
   - Plots generated automatically
   - YAML exports for integration with robot controllers

### Configuration Parameters

Key parameters in YAML configs:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `robot.path` | str | Path to the robot's parameter configuration file (`par.yaml`). |
| `trajectory.type` | str | Specifies the data loader type: `'ros2'` or `'csv'`. |
| `trajectory.bag` | str | Path to the ROS2 bag file (used if type is `'ros2'`). |
| `trajectory.topic` | str | Name of the ROS2 topic containing joint data (used if type is `'ros2'`). |
| `trajectory.data` | str | Path to the directory containing CSV data (used if type is `'csv'`). |
| `trajectory.interval` | list | Time interval `[t_start, t_end]` in seconds for selecting the trajectory window. |
| `trajectory.joints` | list | List of the names of the robot joints to be processed. |
| `processing.pipeline` | list | Ordered sequence of data processing steps (e.g., `'filtering'`, `'differentiating'`, `'decimating'`). |
| `processing.cut_off_frequency` | float | Low-pass filter cutoff frequency in rad/s. |
| `processing.butterworth_order` | int | Order of the Butterworth filter. |
| `processing.apply_decimation` | bool | Flag to enable or disable sample decimation. |
| `processing.decimate_factor` | int | Factor by which to reduce the number of samples during decimation. |
| `processing.flip_torques` | bool | Flag to invert joint torque signs (useful for robots like the raw Franka Emika Panda). |
| `identification.method` | str | Estimation method for Base Inertial Parameters (`'OLS'`, `'WLS'`, `'OLS\|prior'`, `'WLS\|prior'`). |
| `identification.conditioning_ratio` | float | Ratio to truncate the SVD inverse for handling ill-conditioning (Default: `None`). |
| `identification.positive_threshold` | float | Positive threshold tolerance for the constrained problem solver (Default: `1e-16`). |
| `identification.numerical_eigen` | bool | Wheter to costraint Positive Definitess of matrix with samples rather than  symbolic eigenvalues computation (Default: `false`). |
| `identification.inertia_scale_factor` | float | Scaling of inertia matrix to ease numerical check on definitive positive costraint (Default: `1`). |
| `identification.weight.loss` | float | Gain factor for the base parameters optimization loss function. |
| `identification.weight.mass` | float | Mass standard deviation prior [Kg]. |
| `identification.weight.CoM` | float | Center of Mass standard deviation prior [m]. |
| `identification.weight.inertia` | float | Inertia standard deviation prior [Kg/m^2]. |
| `identification.weight.link` | list | Multipliers (1/std.dev) for link prior selection. Use `0` to completely ignore a specific link. |
| `identification.weight.prior_yaml` | str | Path to a specific YAML file defining custom parameter priors. |

## Dependencies

### Core Dependencies
- **NumPy**: Numerical computations
- **SciPy**: Signal filtering, optimization
- **PyYAML**: Configuration parsing
- **Matplotlib**: Visualization
- **CasADi**: Symbolic optimization and AD
- **Thunder Dynamic**: Symbolic robot modeling (only when generating the robot model)

### ROS2 Integration
- **ROS2**: For rosbag data loading
- **Rosbag2**: Python interface for ROS2 bags

## Output Structure

Results are stored in the desired directory (default is `results/data_DD_MM_YYYY/`):
```
results/
    ├── identification_par.yaml
    ├── q.png
    ├── qd.png
    ├── qdd.png
    ├── tau.png
    └── identification_results.png
```

## Algorithm Overview
![System Architecture Pipeline](doc/pipeline.png)
### 1. **Data Processing**
   - Load joint positions, velocities, accelerations from raw data
   - Apply low-pass filtering to smooth signals
   - Compute numeric derivatives if needed
   - Decimate data

The data processing pipeline and its parametes can be defined in the config file

### 2. **Base Parameter Estimation**
Build the extended regressor equation:
$\boldsymbol{Y} = \boldsymbol{W}\boldsymbol{\pi}_b$
   - OLS: $\hat{\boldsymbol{\pi}}^{\text{ols}}_b = \boldsymbol{W}^\dagger \boldsymbol{Y}$
   - WLS: $\hat{\boldsymbol{\pi}}^{\text{wls}}_b = \tilde{\boldsymbol{W}}^\dagger \tilde{\boldsymbol{Y}}$
   - LS|prior: $`\hat{\boldsymbol{\pi}}^{\text{prior}}_{b} = \hat{\boldsymbol{\pi}}_{b,0} + \boldsymbol{W}^\dagger (\boldsymbol{Y} - \boldsymbol{W} \hat{\boldsymbol{\pi}}_{b,0})`$ 

### 3. **Least-Squares Estimation**
   - Solve: 
```math
\begin{aligned}
\min_{\boldsymbol{\pi}_{\text{DYN}}} \quad & (\boldsymbol{\pi}^* - \boldsymbol{\pi}_b(\boldsymbol{\pi}_{\text{DYN}}))^\top \mathbf{C}_{\boldsymbol{\pi} \boldsymbol{\pi}}^{-1} (\boldsymbol{\pi}^* - \boldsymbol{\pi}_b(\boldsymbol{\pi}_{\text{DYN}})) + \\
& (\boldsymbol{\pi}_0 - \boldsymbol{\pi}_{\text{DYN}})^\top \boldsymbol{\Sigma}_0^{-1} (\boldsymbol{\pi}_0 - \boldsymbol{\pi}_{\text{DYN}}) \\
\textrm{s.t.} \quad & m_i > 0, \quad \forall i \in \{1, \dots, N\} \\
& \mathbf{P}_i \succ 0, \quad \forall i \in \{1, \dots, N\}
\end{aligned}
```

### 4. **Robot Modeling**
The developed package relies on the robot model generated via the Thunder Dynamics software. In particular, it allows to define:
```math
\begin{aligned}
\boldsymbol{\tau} &= \boldsymbol{M}(\boldsymbol{q})\ddot{\boldsymbol{q}} + \boldsymbol{C}(\boldsymbol{q},\dot{\boldsymbol{q}})\dot{\boldsymbol{q}} +\boldsymbol{G}(\boldsymbol{q}) + \boldsymbol{\tau}_f + \boldsymbol{\tau}_{Ia} \\
\boldsymbol{\tau}_f &= \boldsymbol{F}\dot{\boldsymbol{q}} + \boldsymbol{D}\text{sign}(\dot{\boldsymbol{q}}) \\  
\boldsymbol{\tau}_{Ia} &= \boldsymbol{I}_a\ddot{\boldsymbol{q}}
\end{aligned}
```

   - Mass Matrix: $\boldsymbol{M}(\boldsymbol{q})$
   - Coriolis terms: $\boldsymbol{C}(\boldsymbol{q},\dot{\boldsymbol{q}})$
   - Gravity vector: $\boldsymbol{G}(\boldsymbol{q})$
   - Viscous friction: $\boldsymbol{f}_v = \boldsymbol{F} \cdot \dot{\boldsymbol{q}}$ (optional)
   - Static friction: $\boldsymbol{f}_s(\boldsymbol{q}) = \boldsymbol{D} \cdot \text{sign}(\dot{\boldsymbol{q}})$ (optional)
   - Rotor Inertia: $\boldsymbol{\tau}_{Ia} = \boldsymbol{I}_a\ddot{\boldsymbol{q}}$ (optional)

for more detail about the usage of [Thunder Dynamic](https://github.com/CentroEPiaggio/thunder_dynamics.git) software, please refer to its official repo.

## Related Documentation

- See `doc/` for detailed methodology papers

