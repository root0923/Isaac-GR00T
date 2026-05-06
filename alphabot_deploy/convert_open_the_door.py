#!/usr/bin/env python3
"""
Convert custom H5 format to LeRobot V2 format for GR00T fine-tuning (Single Arm Version)

Input format (from /home/ysy/data/teleop_data/20260413/*.h5):
  - observations/left_arm_joint_status (T, 7) - 7 joint angles (degrees)
  - observations/left_arm_gripper_width (T, 1) - gripper width
  - observations/images/{cam_name} (T,) - compressed images (JPEG/PNG)

Output format (LeRobot V2):
  - meta/info.json - dataset metadata
  - meta/episodes.jsonl - episode metadata
  - meta/tasks.jsonl - task descriptions
  - meta/modality.json - modality configuration
  - data/chunk-XXX/episode_XXXXXX.parquet - state/action data
  - videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4 - video files
"""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm

# Default camera names and their output aliases
DEFAULT_CAMERAS = ["camera_head_image", "camera_left_image", "camera_chest_image"]
CAMERA_NAME_MAP = {
    "camera_head_image": "front",
    "camera_left_image": "wrist_left",
    "camera_chest_image": "wrist_right",
}
DEFAULT_CHUNK_SIZE = 50


def decompress_image(compressed_data):
    """Decompress JPEG/PNG image and convert to RGB"""
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Returns BGR
    if img is None:
        raise ValueError("Failed to decompress image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img


def load_h5_episode(h5_path: Path, camera_names: list[str]) -> dict[str, Any] | None:
    """
    Load a single H5 episode with error handling.

    Returns:
        Dictionary containing episode data or None if loading failed
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Check required fields
            if "observations/left_arm_joint_status" not in f:
                print(f"  Error: Missing left_arm_joint_status")
                return None
            if "observations/left_arm_gripper_width" not in f:
                print(f"  Error: Missing left_arm_gripper_width")
                return None

            # Read joint and gripper data
            joint_status = f["observations/left_arm_joint_status"][()]  # (T, 7)
            gripper_width = f["observations/left_arm_gripper_width"][()]  # (T, 1)

            if joint_status.shape[0] == 0:
                print(f"  Skip: Empty left_arm_joint_status")
                return None

            # Determine episode length
            episode_len = joint_status.shape[0]

            # Load and decompress images
            images = {}
            available_cams = []

            for cam_name in camera_names:
                cam_path = f"observations/images/{cam_name}"
                if cam_path not in f:
                    print(f"  Warning: Camera {cam_name} not found, skipping")
                    continue

                compressed_imgs = f[cam_path]
                img_len = compressed_imgs.shape[0]

                # Align with shortest sequence
                if img_len < episode_len:
                    print(f"  Align: joint frames({episode_len}) > image frames({img_len}), truncating")
                    episode_len = img_len

                # Decompress images
                if compressed_imgs.dtype == object:
                    decompressed = []
                    for i in range(episode_len):
                        try:
                            img = decompress_image(compressed_imgs[i])
                            decompressed.append(img)
                        except Exception as e:
                            print(f"  Error: Failed to decompress {cam_name}[{i}]: {e}")
                            return None
                    images[cam_name] = np.array(decompressed, dtype=np.uint8)  # (T, H, W, 3)
                else:
                    images[cam_name] = compressed_imgs[:episode_len]

                available_cams.append(cam_name)

            if len(available_cams) == 0:
                print(f"  Error: No available camera data")
                return None

            # Truncate to aligned length
            joint_status = joint_status[:episode_len]
            gripper_width = gripper_width[:episode_len]

            # Combine qpos: [7 joints + 1 gripper] = 8D
            qpos = np.concatenate([
                joint_status,
                gripper_width,
            ], axis=-1).astype(np.float32)

            # Action = next timestep qpos (teacher forcing)
            action = np.zeros_like(qpos)
            action[:-1] = qpos[1:]
            action[-1] = qpos[-1]  # Last frame stays same

            return {
                "qpos": qpos,
                "action": action,
                "images": {CAMERA_NAME_MAP.get(k, k): v for k, v in images.items()},
                "length": episode_len,
            }

    except Exception as e:
        print(f"  Error: Exception while reading H5 file (possibly corrupted): {e}")
        return None


def save_episode_video(images: np.ndarray, output_path: Path, fps: int = 30):
    """Save image array as MP4 video using OpenCV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img in images:
        # Convert RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr_img)

    writer.release()


def compute_episode_statistics(qpos: np.ndarray, action: np.ndarray) -> dict:
    """Compute per-episode statistics for normalization (LeRobot format)"""
    num_frames = len(qpos)
    stats = {
        "observation.state": {
            "mean": qpos.mean(axis=0).tolist(),
            "std": qpos.std(axis=0).tolist(),
            "min": qpos.min(axis=0).tolist(),
            "max": qpos.max(axis=0).tolist(),
            "q01": np.percentile(qpos, 1, axis=0).tolist(),
            "q99": np.percentile(qpos, 99, axis=0).tolist(),
            "count": [num_frames],
        },
        "action": {
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "q01": np.percentile(action, 1, axis=0).tolist(),
            "q99": np.percentile(action, 99, axis=0).tolist(),
            "count": [num_frames],
        },
    }
    return stats


def compute_statistics(all_qpos: list[np.ndarray], all_actions: list[np.ndarray]) -> dict:
    """Compute dataset statistics for normalization"""
    # Stack all episodes
    qpos_stack = np.concatenate(all_qpos, axis=0)  # (N, 8)
    action_stack = np.concatenate(all_actions, axis=0)  # (N, 8)

    stats = {
        "observation.state": {
            "mean": qpos_stack.mean(axis=0).tolist(),
            "std": qpos_stack.std(axis=0).tolist(),
            "min": qpos_stack.min(axis=0).tolist(),
            "max": qpos_stack.max(axis=0).tolist(),
            "q01": np.percentile(qpos_stack, 1, axis=0).tolist(),
            "q99": np.percentile(qpos_stack, 99, axis=0).tolist(),
        },
        "action": {
            "mean": action_stack.mean(axis=0).tolist(),
            "std": action_stack.std(axis=0).tolist(),
            "min": action_stack.min(axis=0).tolist(),
            "max": action_stack.max(axis=0).tolist(),
            "q01": np.percentile(action_stack, 1, axis=0).tolist(),
            "q99": np.percentile(action_stack, 99, axis=0).tolist(),
        },
    }

    return stats


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    camera_names: list[str],
    task_description: str = "Pick and place task with 7-axis manipulator",
    fps: int = 30,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Convert H5 dataset to LeRobot V2 format"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos"

    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Get all H5 files
    h5_files = sorted(input_dir.glob("*.h5"))

    if len(h5_files) == 0:
        raise ValueError(f"No .h5 files found in {input_dir}")

    print(f"Found {len(h5_files)} H5 files")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cameras: {camera_names}")
    print("=" * 80)

    episodes_metadata = []
    episodes_stats = []
    all_qpos = []
    all_actions = []
    success_count = 0

    for episode_idx, h5_path in enumerate(tqdm(h5_files, desc="Converting episodes")):
        print(f"\n[{episode_idx}] {h5_path.name}")

        # Load episode
        episode_data = load_h5_episode(h5_path, camera_names)
        if episode_data is None:
            print(f"  Skipped due to errors")
            continue

        qpos = episode_data["qpos"]
        action = episode_data["action"]
        images = episode_data["images"]
        length = episode_data["length"]

        # Create parquet data with all required columns
        num_frames = len(qpos)
        df = pd.DataFrame({
            "observation.state": list(qpos),
            "action": list(action),
            "timestamp": np.arange(num_frames) / fps,  # Generate timestamps
            "annotation.human.action.task_description": [0] * num_frames,  # All frames use task_index 0
            "task_index": [0] * num_frames,
            "episode_index": [episode_idx] * num_frames,
            "index": list(range(episode_idx * 10000, episode_idx * 10000 + num_frames)),  # Global index
            "next.reward": [0] * num_frames,
            "next.done": [False] * (num_frames - 1) + [True],  # Last frame is done
        })

        # Save parquet file
        chunk_idx = episode_idx // chunk_size
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        # Save videos
        for cam_name, img_data in images.items():
            video_key = f"observation.images.{cam_name}"
            video_path = video_dir / f"chunk-{chunk_idx:03d}" / video_key / f"episode_{episode_idx:06d}.mp4"
            save_episode_video(img_data, video_path, fps=fps)

        # Store episode metadata
        episodes_metadata.append({
            "episode_index": episode_idx,
            "length": length,
            "tasks": [task_description],
        })

        # Compute and store per-episode statistics
        episode_stats = compute_episode_statistics(qpos, action)
        episodes_stats.append({
            "episode_index": episode_idx,
            "stats": episode_stats,
        })

        # Collect for statistics
        all_qpos.append(qpos)
        all_actions.append(action)

        success_count += 1
        print(f"  ✓ Success: {length} frames, {len(images)} cameras")

    if success_count == 0:
        raise ValueError("No episodes were successfully converted!")

    print(f"\n{'=' * 80}")
    print(f"Successfully converted: {success_count}/{len(h5_files)} episodes")

    # Compute statistics
    print("\nComputing dataset statistics...")
    stats = compute_statistics(all_qpos, all_actions)

    # Save statistics
    stats_path = meta_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save episodes metadata
    episodes_path = meta_dir / "episodes.jsonl"
    with jsonlines.open(episodes_path, mode="w") as writer:
        for ep_meta in episodes_metadata:
            writer.write(ep_meta)

    # Save per-episode statistics (LeRobot will convert to numpy when loading)
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    with jsonlines.open(episodes_stats_path, mode="w") as writer:
        for ep_stats in episodes_stats:
            writer.write(ep_stats)

    # Save tasks metadata
    tasks_path = meta_dir / "tasks.jsonl"
    with jsonlines.open(tasks_path, mode="w") as writer:
        writer.write({
            "task_index": 0,
            "task": task_description,
        })

    # Save modality configuration (following GR00T LeRobot format)
    # Use mapped camera names (front, wrist, right_wrist)
    mapped_camera_names = [CAMERA_NAME_MAP.get(cam, cam) for cam in camera_names]
    modality_config = {
        "state": {
            "arm": {
                "start": 0,
                "end": 7,
            },
            "gripper": {
                "start": 7,
                "end": 8,
            },
        },
        "action": {
            "arm": {
                "start": 0,
                "end": 7,
            },
            "gripper": {
                "start": 7,
                "end": 8,
            },
        },
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in mapped_camera_names},
        "annotation": {
            "human.action.task_description": {}  # Empty dict as per schema
        },
    }

    modality_path = meta_dir / "modality.json"
    with open(modality_path, "w") as f:
        json.dump(modality_config, f, indent=2)

    # Generate info.json
    # Use mapped camera names (front, wrist, right_wrist) instead of original names
    mapped_camera_names = [CAMERA_NAME_MAP.get(cam, cam) for cam in camera_names]

    first_image = None
    for cam_name in mapped_camera_names:
        video_key = f"observation.images.{cam_name}"
        first_video = list((video_dir / f"chunk-000" / video_key).glob("*.mp4"))
        if first_video:
            cap = cv2.VideoCapture(str(first_video[0]))
            ret, frame = cap.read()
            if ret:
                first_image = frame
                cap.release()
                break

    img_height, img_width = first_image.shape[:2] if first_image is not None else (480, 640)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": None,
        },
    }

    for cam_name in mapped_camera_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [img_height, img_width, 3],
            "names": ["height", "width", "channel"],
            "fps": fps,
        }

    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "total_episodes": success_count,
        "total_frames": sum(ep["length"] for ep in episodes_metadata),
        "chunks_size": chunk_size,
        "total_chunks": (success_count + chunk_size - 1) // chunk_size,
        "total_videos": success_count * len(camera_names),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"  Output: {output_dir}")
    print(f"  Total episodes: {success_count}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"\nNext steps:")
    print(f"  1. Verify the dataset loads correctly")
    print(f"  2. Update embodiment config in gr00t/configs/data/embodiment_configs.py")
    print(f"  3. Run fine-tuning with --dataset-path {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert H5 to LeRobot V2 format (Single Arm)")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing .h5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERAS,
        help="Camera names to use",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="Single arm manipulation task",
        help="Task description for annotations",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of episodes per chunk",
    )

    args = parser.parse_args()

    convert_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        camera_names=args.cameras,
        task_description=args.task_description,
        fps=args.fps,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
