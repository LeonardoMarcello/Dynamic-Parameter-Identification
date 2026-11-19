"""Trajectory loading (rosbags) and processing with the project's own TrajectoryManager."""
import copy
import hashlib
import os

import numpy as np
import pandas as pd
import yaml

from bench.setup_env import BENCH, CACHE, ROOT
from utils.loader_utils import TrajectoryManager


def read_bag(bag: str, topic: str, joints: list) -> pd.DataFrame:
    """Same output layout as utils.ros2_utils.TrajectoryManagerROS2.read_bag (bag receive time)."""
    key = hashlib.md5(f"{bag}|{topic}|{joints}".encode()).hexdigest()[:12]
    cache = os.path.join(CACHE, f"bag_{os.path.basename(bag.rstrip('/'))}_{key}.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows, has_vel = [], None
    with Reader(os.path.join(ROOT, bag)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, t, raw in reader.messages(connections=conns):
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            names = list(msg.name)
            idx = [names.index(j) for j in joints if j in names]
            if len(idx) != len(joints):
                continue
            if has_vel is None:
                has_vel = len(msg.velocity) == len(msg.position)
            row = [t / 1e9] + [msg.position[i] for i in idx] + [msg.effort[i] for i in idx]
            if has_vel:
                row += [msg.velocity[i] for i in idx]
            rows.append(row)
    cols = ["t"] + [f"pos_{j}" for j in joints] + [f"eff_{j}" for j in joints]
    if has_vel:
        cols += [f"vel_{j}" for j in joints]
    df = pd.DataFrame(rows, columns=cols)
    df.to_pickle(cache)
    return df


def write_config(cfg: dict, name: str) -> str:
    d = os.path.join(BENCH, "configs", "generated")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trajectory(cfg: dict, name: str, interval=None, bag=None) -> TrajectoryManager:
    """Load and process a trajectory exactly as Identifier.init() does (pipeline branch)."""
    cfg = copy.deepcopy(cfg)
    if interval is not None:
        cfg["trajectory"]["interval"] = [float(x) for x in interval]
    if bag is not None:
        cfg["trajectory"]["bag"] = bag
    path = write_config(cfg, name)
    tm = TrajectoryManager(path)
    tc = cfg["trajectory"]
    tm.df = read_bag(tc["bag"], tc["topic"], tc["joints"])
    tm.process_pipeline()
    return tm
