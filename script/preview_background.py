import argparse
import importlib
import os
import sys


def configure_headless_rendering():
    if os.environ.get("ROBOTWIN_PREVIEW_USE_OSMESA") != "1":
        return
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "450")


configure_headless_rendering()

sys.path.append("./")

import cv2
import numpy as np
import yaml
from PIL import Image

from envs import CONFIGS_PATH


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_task_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["need_plan"] = False
    args["save_data"] = False
    args["render_freq"] = 0
    args["camera_shader_dir"] = os.environ.get("ROBOTWIN_CAMERA_SHADER_DIR", "default")
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args.setdefault("background_appearance", {})
    args["background_appearance"]["enabled"] = True

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment):
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def save_contact_sheet(frames, output_path, columns=5):
    if not frames:
        return

    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / columns))
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        sheet[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame
    Image.fromarray(sheet).save(output_path)


def save_video(frames, output_path, fps):
    if not frames:
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _rounded_or_none(value):
    if value is None:
        return None
    return np.round(np.array(value, dtype=np.float64), 3).tolist()


def preview_background(task_name, task_config, seed, episodes, output_dir, camera_name, fps):
    os.makedirs(output_dir, exist_ok=True)
    task = class_decorator(task_name)
    args = load_task_args(task_name, task_config)

    frames = []
    for episode_idx in range(episodes):
        frame_seed = seed + episode_idx
        task.setup_demo(now_ep_num=episode_idx, seed=frame_seed, **args)
        obs = task.get_obs()
        frame = obs["observation"][camera_name]["rgb"]
        frames.append(frame)

        summary = task.background_appearance_summary
        image_path = os.path.join(output_dir, f"episode{episode_idx:03d}_{camera_name}_background.png")
        Image.fromarray(frame).save(image_path)

        print(
            f"episode {episode_idx}: "
            f"wall={_rounded_or_none(summary.get('wall_rgb_multiplier'))}, "
            f"floor={_rounded_or_none(summary.get('floor_color'))}, "
            f"table_tint={_rounded_or_none(summary.get('table_color_tint'))}, "
            f"table_metallic={summary.get('table_metallic')}, "
            f"table_roughness={summary.get('table_roughness')}"
        )

        task.close_env(clear_cache=(episode_idx == episodes - 1))

    save_contact_sheet(frames, os.path.join(output_dir, f"{camera_name}_background_contact_sheet.png"))
    save_video(frames, os.path.join(output_dir, f"{camera_name}_background_preview.mp4"), fps)
    print(f"Saved background preview to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="click_alarmclock")
    parser.add_argument("--task_config", type=str, default="demo_background")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="background_preview")
    parser.add_argument("--camera_name", type=str, default="head_camera")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--render_test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.render_test:
        from test_render import Sapien_TEST

        Sapien_TEST()

    preview_background(
        task_name=parsed_args.task_name,
        task_config=parsed_args.task_config,
        seed=parsed_args.seed,
        episodes=parsed_args.episodes,
        output_dir=parsed_args.output_dir,
        camera_name=parsed_args.camera_name,
        fps=parsed_args.fps,
    )
