import math

import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .utils.actor_utils import Actor, ArticulationActor


DEFAULT_OBJECTS_LAYOUT_CONFIG = {
    "enabled": False,
    "o1_distractor_objects": True,
    "o1_distractor_count_range": [3, 15],
    "o2_target_pose": True,
    "o2_xy_noise_std": 0.02,
    "o2_yaw_deg": 15,
}


def merge_objects_layout_config(config):
    merged = dict(DEFAULT_OBJECTS_LAYOUT_CONFIG)
    merged.update(config or {})
    return merged


def sample_objects_layout(config, rng):
    config = merge_objects_layout_config(config)
    enabled = bool(config.get("enabled", False))
    summary = {
        "enabled": enabled,
        "o1_distractor_objects": False,
        "o1_distractor_count": None,
        "o2_target_pose": False,
        "o2_xy_noise_std": None,
        "o2_yaw_deg": None,
        "registered_actor_count": 0,
        "applied_actor_count": 0,
        "registered_actors": [],
        "applied_actors": [],
    }

    if not enabled:
        return config, summary

    if config.get("o1_distractor_objects", True):
        low, high = config["o1_distractor_count_range"]
        low = int(low)
        high = int(high)
        if high < low:
            low, high = high, low
        summary["o1_distractor_objects"] = True
        summary["o1_distractor_count"] = int(rng.integers(low, high + 1))

    if config.get("o2_target_pose", True):
        summary["o2_target_pose"] = True
        summary["o2_xy_noise_std"] = float(config["o2_xy_noise_std"])
        summary["o2_yaw_deg"] = float(config["o2_yaw_deg"])

    return config, summary


def _unwrap_actor(actor):
    if isinstance(actor, (Actor, ArticulationActor)):
        return actor.actor
    return actor


def _get_actor_name(actor):
    raw_actor = _unwrap_actor(actor)
    if raw_actor is None:
        return "unknown"
    try:
        name = raw_actor.get_name()
    except AttributeError:
        name = None
    return name or type(raw_actor).__name__


def _get_actor_pose(actor):
    if isinstance(actor, (Actor, ArticulationActor)):
        return actor.get_pose()
    return actor.get_pose()


def _set_actor_pose(actor, pose):
    raw_actor = _unwrap_actor(actor)
    if hasattr(raw_actor, "set_root_pose"):
        raw_actor.set_root_pose(pose)
        return
    if hasattr(raw_actor, "set_pose"):
        raw_actor.set_pose(pose)
        return
    raise AttributeError(f"Actor {_get_actor_name(actor)} does not support pose updates")


def _apply_yaw_delta(quat, yaw_delta_rad):
    yaw_quat = t3d.euler.euler2quat(0.0, 0.0, yaw_delta_rad)
    return t3d.quaternions.qmult(yaw_quat, quat)


def apply_target_pose_perturbation(task, summary, rng):
    if not summary.get("enabled", False) or not summary.get("o2_target_pose", False):
        summary["registered_actor_count"] = len(getattr(task, "objects_layout_targets", []))
        summary["registered_actors"] = [
            _get_actor_name(entry["actor"]) for entry in getattr(task, "objects_layout_targets", [])
        ]
        return summary

    entries = getattr(task, "objects_layout_targets", [])
    summary["registered_actor_count"] = len(entries)
    summary["registered_actors"] = [_get_actor_name(entry["actor"]) for entry in entries]

    xy_std = float(summary["o2_xy_noise_std"])
    yaw_limit_rad = math.radians(float(summary["o2_yaw_deg"]))
    applied_actors = []

    for entry in entries:
        if not entry.get("pose_perturb", True):
            continue
        actor = entry["actor"]
        pose = _get_actor_pose(actor)
        delta_xy = rng.normal(loc=0.0, scale=xy_std, size=2)
        yaw_delta = float(rng.uniform(-yaw_limit_rad, yaw_limit_rad))
        new_pose = sapien.Pose(
            p=[pose.p[0] + delta_xy[0], pose.p[1] + delta_xy[1], pose.p[2]],
            q=_apply_yaw_delta(np.array(pose.q, dtype=np.float64), yaw_delta),
        )
        _set_actor_pose(actor, new_pose)
        applied_actors.append(
            {
                "name": _get_actor_name(actor),
                "role": entry.get("role"),
                "delta_xy": np.round(delta_xy, 6).tolist(),
                "yaw_delta_deg": float(np.degrees(yaw_delta)),
            }
        )

    summary["applied_actor_count"] = len(applied_actors)
    summary["applied_actors"] = applied_actors
    return summary
