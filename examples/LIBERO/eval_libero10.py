# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate all Libero-10 tasks and compute overall success rate.

Usage (run with the LIBERO venv python, server must be running):

uv run python gr00t/eval/run_gr00t_server.py     --model-path /home/jzai/weights/fine_tuned/libero_10_sf/checkpoint-10000     --embodiment-tag LIBERO_PANDA     --use-sim-policy-wrapper

    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python examples/LIBERO/eval_libero10.py \
        --policy-client-host 127.0.0.1 \
        --policy-client-port 5555 \
        --n-episodes 500 \
        --n-envs 5

Or with a local model (no server):

    python examples/LIBERO/eval_libero10.py \
        --model-path /path/to/checkpoint \
        --n-episodes 20 \
        --n-envs 5
"""

import numpy as np
import tyro
from dataclasses import dataclass
from pathlib import Path

LIBERO_10_TASKS = [
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

VIDEO_ROOT = Path(__file__).resolve().parent / "sim_videos"


@dataclass
class EvalConfig:
    n_episodes: int = 500
    """Number of episodes per task."""

    max_episode_steps: int = 720
    """Maximum steps per episode."""

    n_envs: int = 3
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    model_path: str = ""
    """Path to model checkpoint (mutually exclusive with client mode)."""

    policy_client_host: str = ""
    """Policy server host."""

    policy_client_port: int | None = None
    """Policy server port."""

    video_dir: str | None = None
    """Root directory for videos. Defaults to examples/LIBERO/sim_videos."""


def main():
    args = tyro.cli(EvalConfig)

    # Validate: must provide either model_path or client config, not both
    has_model = bool(args.model_path)
    has_client = bool(args.policy_client_host and args.policy_client_port is not None)
    assert has_model or has_client, (
        "Must provide either --model-path or (--policy-client-host & --policy-client-port)"
    )

    video_root = Path(args.video_dir) if args.video_dir else VIDEO_ROOT

    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    task_results = {}

    for task_name in LIBERO_10_TASKS:
        short_name = task_name.split("/", 1)[1]
        task_video_dir = str(video_root / short_name)

        print(f"\n{'='*60}")
        print(f"Evaluating: {short_name}")
        print(f"{'='*60}")

        env_name, successes, infos = run_gr00t_sim_policy(
            env_name=task_name,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            model_path=args.model_path,
            policy_client_host=args.policy_client_host,
            policy_client_port=args.policy_client_port,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            video_dir=task_video_dir,
        )

        sr = np.mean(successes)
        task_results[short_name] = (sr, len(successes), int(sum(successes)))
        print(f"  Success: {int(sum(successes))}/{len(successes)} = {sr:.2%}")

    # Summary
    print(f"\n{'='*60}")
    print("Libero-10 Overall Results")
    print(f"{'='*60}")
    print(f"{'Task':<80} {'SR':>8}")
    print("-" * 90)

    all_sr = []
    for short_name, (sr, total, n_success) in task_results.items():
        all_sr.append(sr)
        print(f"{short_name:<80} {n_success}/{total} ({sr:.2%})")

    overall_sr = np.mean(all_sr)
    print("-" * 90)
    print(f"{'Overall (mean over tasks)':<80} {overall_sr:.2%}")
    print(f"\nVideos saved to: {video_root}")


if __name__ == "__main__":
    main()
