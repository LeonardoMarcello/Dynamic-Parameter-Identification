# Benchmark experiments

Scripts that regenerate the benchmark figures and tables of the paper. The project code in `../src` is
**imported, not modified**; the only environment created is `benchmark/.env`.

## Environment

```bash
/usr/bin/python3.10 -m venv benchmark/.env           # Thunder .so files are cp310
benchmark/.env/bin/pip install numpy scipy casadi matplotlib pandas pyyaml rosbags cvxpy clarabel \
                               figaroh pin cvxopt picos
```

Tested with FIGAROH 1.0.2, Pinocchio 4.1.0, CasADi 3.8.1, cvxpy 1.7.5, PICOS 2.6.2, NumPy 2.2.6, SciPy 1.15.3.

`run.sh` runs a script inside `.env` from the repository root and clears `PYTHONPATH`, `LD_LIBRARY_PATH` and
`AMENT_PREFIX_PATH`, so that a sourced ROS 2 installation (whose Pinocchio is built against NumPy 1) does not
shadow the pip packages. ROS 2 bags are read with the pure-Python `rosbags` package; `rosbag2_py`, `rclpy` and
`sensor_msgs` are stubbed (`bench/setup_env.py`) only so that `utils.*` can be imported. The recordings are
expected in `<repo>/rosbag/`.

## Layout

```
benchmark/
├── run.sh          # runs a script inside .env with ROS paths cleared
├── bench/          # library
├── experiment/     # one script per paper figure/table
└── results/        # outputs of experiment/ (JSON, figures, LaTeX tables)
```
At run time, parsed bags are cached in `cache/` and per-run configs are written to `configs/generated/`
(both untracked, regenerated automatically).

## Library (`bench/`)

| file | content |
|---|---|
| `ident.py` | regressor, OLS, per-joint WLS, MLE (IPOPT), feasibility margins, RMSE |
| `figaroh_baseline.py` | FIGAROH 1.0.2 on the same processed data: base parameters (double QR, WLS) and its physically consistent estimators (physical fit, per-link projection, exact SDP reconstruction with phase-I check) |
| `pipeline.py` | runs all methods on a training set and evaluates them on held-out sets |
| `data.py`, `plots.py`, `setup_env.py` | bag loading and processing, torque plots, environment setup |

## Paper figures and tables

```bash
benchmark/run.sh benchmark/experiment/<script>.py
```

Approximate times on one core: `e01` a few minutes, `e02` about 3 h.

| paper item | script | output (`results/`) |
|---|---|---|
| Fig. 2: real Franka torques and RMSE (prior, LS, FIGAROH PF/PLP, MLE; folds A/B) | `e01_franka_bench.py` (calls `e01_franka_bench_plot_core.py`) | `e01_franka_bench/torque_full_train.png`, `rmse_core.png` |
| Fig. 4: last-link parameters, simulated Franka + SoftHand | same | `e01_franka_bench/soft_hand_comparison_all.png` |
| Table III: MLE parameters vs Gaz et al. | same | `e01_franka_bench/table_real_parameters.tex` |
| Table IV: multi-start robustness (real + simulated) | `e02_multistart.py` (then runs `e02_multistart_softhand.py`, `e02_project_best.py` and `multistart_unified_table.py` automatically) | `e02_multistart/table_multistart_unified.{tex,png}` |

Fig. 1 and Tables I–II are not generated from data. Fig. 3 (Allegro Hand) comes from the MuJoCo validation in
`../src/examples/allegro_hand`, and Fig. 5 (excitation metrics vs accuracy, 45 windows) from
`../src/examples/franka_sim_trajectories/franka_identification.py`, which uses only the project pipeline.

Useful flags: `e02_multistart.py --resummarize` recomputes the summaries of the stored runs (real and simulated)
and the tables without new solves; `e02_project_best.py --table-only` rebuilds its table from `projected.json`.
