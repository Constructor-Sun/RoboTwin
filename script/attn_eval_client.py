"""Single-episode RoboTwin eval client for attention-capture server.

Usage::

    cd RoboTwin/script
    python attn_eval_client.py --task click_alarmclock --seed 10033 --port 1106

The attention-capture server must be running on the given port first.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Paths — must be set BEFORE importing RoboTwin modules
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # counterfactual root
_ROBOTWIN_ROOT = _REPO_ROOT / "external" / "RoboTwin"
_LINGBOT_ROOT = _REPO_ROOT / "external" / "lingbot-va"

for _p in (_LINGBOT_ROOT, _ROBOTWIN_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.chdir(_ROBOTWIN_ROOT)  # RoboTwin modules expect CWD to be RoboTwin root

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from evaluation.robotwin.geometry import euler2quat  # noqa: E402
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from description.utils.generate_episode_instructions import generate_episode_descriptions  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (minimal copies from mask_prune client)
# ---------------------------------------------------------------------------
def _class_decorator(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    return getattr(mod, task_name)()

def _get_camera_config(camera_type):
    path = os.path.join(_ROBOTWIN_ROOT, "task_config", "_camera_config.yml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if camera_type not in cfg:
        raise KeyError(f"camera {camera_type} not in {path}")
    return cfg[camera_type]

def _get_embodiment_config(robot_file):
    path = os.path.join(robot_file, "config.yml")
    with open(path, encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)

def _format_obs(observation, prompt):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }

def _add_eef_pose(new_pose, init_pose):
    new_r = R.from_quat(new_pose[3:7][None])
    init_r = R.from_quat(init_pose[3:7][None])
    out_rot = (init_r * new_r).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def _add_init_pose(new_pose, init_pose):
    left = _add_eef_pose(new_pose[:8], init_pose[:8])
    right = _add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left, right])

def _build_ee_action(raw_action, init_eef_pose):
    """Convert 14-dim or 16-dim raw action to end-effector action."""
    if raw_action.shape[0] == 14:
        return np.concatenate([
            raw_action[:3],
            euler2quat(raw_action[3], raw_action[4], raw_action[5]),
            raw_action[6:10],
            euler2quat(raw_action[10], raw_action[11], raw_action[12]),
            raw_action[13:14],
        ])
    if raw_action.shape[0] == 16:
        ee = _add_init_pose(raw_action, init_eef_pose)
        return np.concatenate([
            ee[:3],
            ee[3:7] / np.linalg.norm(ee[3:7]),
            ee[7:11],
            ee[11:15] / np.linalg.norm(ee[11:15]),
            ee[15:16],
        ])
    raise NotImplementedError(f"unexpected action dim {raw_action.shape[0]}")

# ---------------------------------------------------------------------------
# Main eval logic
# ---------------------------------------------------------------------------
def run_episode(task_env, args, task_name, model, seed, episode_name):
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)

    # Generate instruction
    info = task_env.play_once()
    task_env.close_env()
    if not (info["info"]):
        raise RuntimeError("no episode info from expert")
    results = generate_episode_descriptions(task_name, [info["info"]], 1)
    instruction = np.random.choice(results[0]["seen"])

    # Setup with the actual episode
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=instruction)

    # Reset server
    prompt = task_env.get_instruction()
    model.infer({"reset": True, "prompt": prompt, "episode_name": episode_name})

    # Get initial observation
    init_obs = task_env.get_obs()
    init_eef_pose = np.array(
        init_obs["endpose"]["left_endpose"]
        + [init_obs["endpose"]["left_gripper"]]
        + init_obs["endpose"]["right_endpose"]
        + [init_obs["endpose"]["right_gripper"]],
        dtype=np.float64,
    )
    first_obs = _format_obs(init_obs, prompt)
    action_per_frame = task_env.job_config.get("action_per_frame", 20) if hasattr(task_env, "job_config") else 20

    first = True
    while task_env.take_action_cnt < task_env.step_lim:
        if first:
            obs_for_infer = first_obs
        else:
            obs_for_infer = _format_obs(task_env.get_obs(), prompt)

        ret = model.infer({"obs": obs_for_infer, "prompt": prompt})
        action = ret["action"]

        # Execute actions — accumulate key frames then send all at once.
        # The Wan VAE streaming encoder requires >= 2 frames on subsequent
        # _compute_kv_cache calls for its temporal downsample conv (kernel=3,
        # stride=2).  Sending frames individually triggers "Calculated padded
        # input size per channel: (2 …). Kernel size: (3 …)".
        start = 1 if first else 0
        key_frame_list = []
        for i in range(start, action.shape[1]):
            for j in range(action.shape[2]):
                ee = _build_ee_action(action[:, i, j].flatten(), init_eef_pose)
                task_env.take_action(ee, action_type="ee")

                if (j + 1) % action_per_frame == 0:
                    raw = task_env.get_obs()
                    key_frame_list.append(_format_obs(raw, prompt))

        if key_frame_list:
            kv_req = {
                "obs": key_frame_list,
                "compute_kv_cache": True,
                "imagine": False,
                "state": action,
            }
            model.infer(kv_req)

        first = False
        if task_env.eval_success:
            return True

    return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="e.g. click_alarmclock")
    p.add_argument("--config", default="demo_clean.yml",
                   help="Task config under RoboTwin/task_config/ (default: demo_clean.yml)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--port", type=int, default=29536)
    cargs = p.parse_args()

    # Resolve and load the task config YAML
    config_path = Path(cargs.config)
    if not config_path.is_file():
        config_path = _ROBOTWIN_ROOT / "task_config" / cargs.config
    with open(config_path, encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_name = cargs.task

    # Camera & embodiment setup
    cam_type = task_args["camera"]["head_camera_type"]
    cam_cfg = _get_camera_config(cam_type)
    task_args["head_camera_h"] = cam_cfg["h"]
    task_args["head_camera_w"] = cam_cfg["w"]

    with open(CONFIGS_PATH + "_embodiment_config.yml", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    emb = task_args.get("embodiment", ["aloha-agilex"])
    if len(emb) == 1:
        robot_file = emb_types[emb[0]]["file_path"]
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        task_args["left_robot_file"] = emb_types[emb[0]]["file_path"]
        task_args["right_robot_file"] = emb_types[emb[1]]["file_path"]
        task_args["embodiment_dis"] = emb[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment should have 1 or 3 entries, got {len(emb)}")
    task_args["left_embodiment_config"] = _get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = _get_embodiment_config(task_args["right_robot_file"])

    # Build env
    task_args["task_name"] = task_name
    task_args["eval_mode"] = True
    task_env = _class_decorator(task_name)

    # Connect to attention-capture server
    model = WebsocketClientPolicy(port=cargs.port)

    episode_name = f"seed{cargs.seed:05d}_{Path(cargs.config).stem}"
    try:
        ok = run_episode(task_env, task_args, task_name, model, cargs.seed, episode_name)
    except UnStableError:
        print(f"Unstable environment at seed {cargs.seed}, exiting")
        return
    except Exception:
        traceback.print_exc()
        return
    finally:
        task_env.close_env()

    # Trigger episode finalization on server side
    model.infer({"reset": True, "prompt": "", "episode_name": f"{episode_name}_final"})

    print(f"\n{'SUCCESS' if ok else 'FAIL'}  seed={cargs.seed}")


if __name__ == "__main__":
    main()
