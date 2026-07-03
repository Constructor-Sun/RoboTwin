import numpy as np


DEFAULT_ROBOT_INIT_STATE_CONFIG = {
    "enabled": False,
    "joint_noise_std": 0.1,
    "joint_noise_clip": 0.225,
    "gripper_extreme_prob": 0.25,
    "gripper_extreme_values": [0.05, 0.95],
}


def merge_robot_init_state_config(config):
    merged = dict(DEFAULT_ROBOT_INIT_STATE_CONFIG)
    merged.update(config or {})
    return merged


def sample_joint_noise(num_joints, config, rng):
    if not config.get("enabled", False):
        return np.zeros(num_joints, dtype=np.float64)

    std = float(config.get("joint_noise_std", 0.1))
    clip = abs(float(config.get("joint_noise_clip", 0.225)))
    noise = rng.normal(loc=0.0, scale=std, size=num_joints)
    return np.clip(noise, -clip, clip)


def sample_gripper_extreme(config, rng):
    if not config.get("enabled", False):
        return None

    prob = float(config.get("gripper_extreme_prob", 0.25))
    if rng.random() >= prob:
        return None

    values = config.get("gripper_extreme_values", [0.05, 0.95])
    return float(values[int(rng.integers(0, len(values)))])
