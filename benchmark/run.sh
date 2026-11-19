#!/usr/bin/env bash
# Run a benchmark script inside benchmark/.env, isolated from ROS paths sourced by the shell profile.
#   script                               paper item                     approx. time (1 core)
#   e01_franka_bench.py                  Fig. 2, Fig. 4, Tab. III       a few min
#   e02_multistart.py                    Tab. IV                        ~3 h

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # repo root (scripts use paths relative to it)
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH MPLBACKEND=Agg "$HERE/.env/bin/python" "$@"
