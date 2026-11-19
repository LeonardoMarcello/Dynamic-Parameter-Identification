"""
Make the unmodified project code in ../src importable from benchmark/.env.

The project modules import ROS2 python bindings (rosbag2_py, rclpy, sensor_msgs)
at import time. The benchmark reads bags with the pure-python `rosbags` package
instead, so light stub modules are registered before importing `utils.*`.
Nothing in ../src is modified.
"""
import os
import sys
import types

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
RESULTS = os.path.join(BENCH, "results")
CACHE = os.path.join(BENCH, "cache")

os.chdir(ROOT)  # project code uses paths relative to the repository root
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


def _unavailable(*_a, **_k):
    raise RuntimeError("ROS2 bindings are stubbed in the benchmark; use bench.data.read_bag")


_stub("rosbag2_py", SequentialReader=_unavailable, StorageOptions=_unavailable, ConverterOptions=_unavailable)
_stub("rclpy")
_stub("rclpy.serialization", deserialize_message=_unavailable)
_stub("sensor_msgs")
_stub("sensor_msgs.msg", JointState=object)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import utils.import_thunder as thunder  # noqa: E402,F401

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
