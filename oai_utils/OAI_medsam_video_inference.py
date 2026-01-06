#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference runner for MedSAM2-style video segmentation.

- Iterates over case directories inside an input base directory.
- For each case, finds frames under a frames subdir (default: imgs_jpeg)
  and mask inputs under a masks subdir.
- Initializes the SAM2 predictor once per case and propagates masks.
- Writes predictions to the specified output base directory, mirroring case names.

Expected project layout per case (customizable via CLI flags):
    <input_base_dir>/<case_name>/imgs_jpeg/*.jpg
    <input_base_dir>/<case_name>/masks/<...mask files...>

This script assumes the following functions are available in your environment:
- build_sam2_video_predictor(...)
- load_masks_from_dir(input_mask_path: str) -> (dict[obj_id->np.ndarray], palette or None)
- save_predictions_to_dir(output_mask_dir, video_name, frame_name, per_obj_output_mask, height, width)

Author: you :)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

import numpy as np

# ---- Configuration defaults (override via CLI) --------------------------------

DEFAULT_MODEL_CONFIG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_FRAMES_SUBDIR = "imgs_jpeg"   # directory under each case with input JPGs
DEFAULT_MASKS_SUBDIR  = "masks"       # directory under each case with input masks
OUTPUT_VIDEO_NAME     = "imgs"        # used by save_predictions_to_dir

# ---- Utilities ----------------------------------------------------------------

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def list_images(img_dir: Path) -> List[Path]:
    if not img_dir.is_dir():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix in VALID_IMG_EXTS])

def get_middle_image_filename(img_dir: Path) -> Optional[str]:
    """Return the filename (basename) of the middle image, or None if none exist."""
    imgs = list_images(img_dir)
    if not imgs:
        return None
    mid_idx = len(imgs) // 2
    return imgs[mid_idx].name

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


# ---- Core inference ------------------------------------------------------------

def load_masks_from_dir(input_mask_path):
    input_mask, input_palette = load_ann_png(input_mask_path)
    per_obj_input_mask = get_per_obj_mask(input_mask)

    return per_obj_input_mask, input_palette

def save_predictions_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

    output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
    output_mask_path = os.path.join(
        output_mask_dir, video_name, f"{frame_name}.png"
    )
    assert output_mask.dtype == np.uint8
    assert output_mask.ndim == 2
    output_mask = Image.fromarray(output_mask)
    output_mask.save(output_mask_path)
    
def run_inference_for_case(
    case_dir: Path,
    out_case_dir: Path,
    model_checkpoint: Path,
    model_config: Path,
    frames_subdir: str,
    masks_subdir: str, ) -> None:
    """
    Runs inference for a single case directory.

    Args:
        case_dir: path to the specific case (contains frames & masks subfolders).
        out_case_dir: output directory for this case (will be created).
        model_checkpoint: path to model weights.
        model_config: path to model config YAML.
        frames_subdir: subdirectory name under case_dir that contains frames.
        masks_subdir: subdirectory name under case_dir that contains masks.
    """

    frames_dir = case_dir / frames_subdir
    masks_dir  = case_dir / masks_subdir
    

    if not frames_dir.is_dir():
        print(f"[WARN] Skipping '{case_dir.name}': frames dir not found -> {frames_dir}")
        return
    if not masks_dir.is_dir():
        print(f"[WARN] Skipping '{case_dir.name}': masks dir not found -> {masks_dir}")
        return

    # For logging/debugging; not strictly required by predictor
    middle_frame_index = get_middle_image_filename(masks_dir)
    middle_frame= f"{masks_dir}/{middle_frame_index}"
    if middle_frame:
        print(f"[INFO] {case_dir.name}: middle frame = {middle_frame}")
    else:
        print(f"[WARN] {case_dir.name}: no frames found in {frames_dir}")
        return

    # Build predictor
    predictor = build_sam2_video_predictor(
        config_file=str(model_config),
        ckpt_path=str(model_checkpoint),
        apply_postprocessing=True,
        vos_optimized=True,
    )

    # Gather frame basenames (without extension) in sorted order
    frame_paths = [p for p in frames_dir.iterdir() if p.suffix in VALID_IMG_EXTS]
    if not frame_paths:
        print(f"[WARN] {case_dir.name}: no valid images in {frames_dir}")
        return

    frame_paths = sorted(frame_paths)
    frame_names_noext = [p.stem for p in frame_paths]

    # Initialize predictor state for this "video" (frames directory)
    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        async_loading_frames=False,
    )
    height = inference_state["video_height"]
    width  = inference_state["video_width"]

    # Load input masks (assumed to be per-object binary masks keyed by object_id)
    # NOTE: You previously set INITIAL_MASK_PROMPT to a file path — the loader here
    # is assumed to accept a directory where it can discover needed masks.
    try:
        per_obj_input_mask, input_palette = load_masks_from_dir(
            input_mask_path=str(middle_frame)
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"In VIDEO='{frames_dir.name}', failed to load input masks for frame 0. "
            "If objects appear later in the sequence, enable tracking for later-appearing objects."
        ) from e

    # Sanity check: at least one object to track
    object_ids_set: Optional[Set[int]] = set(per_obj_input_mask) if per_obj_input_mask else None
    if not object_ids_set:
        raise RuntimeError(
            f"In VIDEO='{frames_dir.name}', found no object ids on frame 0. "
            "Please verify your input masks."
        )

    # Add masks for frame 0 to the predictor
    for object_id, object_mask in per_obj_input_mask.items():
        if object_id not in object_ids_set:
            raise RuntimeError(
                f"In VIDEO='{frames_dir.name}', unexpected object_id={object_id} "
                "appears only in a later frame. Enable tracking for later-appearing objects."
            )
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            mask=object_mask,
        )

    # Prepare output directory
    out_video_dir = out_case_dir / OUTPUT_VIDEO_NAME
    ensure_dir(out_video_dir)

    # Propagate masks across the sequence
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    # Write predictions
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        # Guard against any index drift
        if out_frame_idx < 0 or out_frame_idx >= len(frame_names_noext):
            continue

        save_predictions_to_dir(
            output_mask_dir=str(out_case_dir),
            video_name=OUTPUT_VIDEO_NAME,
            frame_name=frame_names_noext[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
        )

    print(f"[OK] {case_dir.name} -> {out_case_dir}")

# ---- CLI / Main ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference over cases for MedSAM2.")
    p.add_argument(
        "--input_base_dir",
        required=True,
        type=Path,
        help="Directory containing per-case subfolders (each with frames & masks).",
    )
    p.add_argument(
        "--output_base_dir",
        required=True,
        type=Path,
        help="Directory where per-case prediction folders will be written.",
    )
    p.add_argument(
        "--model_checkpoint",
        required=True,
        type=Path,
        help="Path to the model checkpoint (.pth/.pt).",
    )
    p.add_argument(
        "--model_config",
        default=DEFAULT_MODEL_CONFIG,
        type=Path,
        help=f"Path to the model config yaml (default: {DEFAULT_MODEL_CONFIG}).",
    )
    p.add_argument(
        "--frames_subdir",
        default=DEFAULT_FRAMES_SUBDIR,
        help=f"Subdirectory under each case containing frames (default: {DEFAULT_FRAMES_SUBDIR}).",
    )
    p.add_argument(
        "--masks_subdir",
        default=DEFAULT_MASKS_SUBDIR,
        help=f"Subdirectory under each case containing input masks (default: {DEFAULT_MASKS_SUBDIR}).",
    )
    p.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional explicit list of case names to run. If omitted, runs all subdirs of input_base_dir.",
    )
    return p.parse_args()

def discover_cases(input_base_dir: Path) -> List[Path]:
    return sorted([p for p in input_base_dir.iterdir() if p.is_dir()])

def main() -> None:
    args = parse_args()

    input_base_dir: Path  = args.input_base_dir
    output_base_dir: Path = args.output_base_dir
    ensure_dir(output_base_dir)

    # Determine which case directories to process
    if args.cases:
        case_dirs = [input_base_dir / c for c in args.cases]
    else:
        case_dirs = discover_cases(input_base_dir)

    if not case_dirs:
        print(f"[ERROR] No case directories found under: {input_base_dir}")
        return

    print(f"[INFO] Found {len(case_dirs)} case(s) under {input_base_dir}")

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        out_case_dir = output_base_dir / case_dir.name
        ensure_dir(out_case_dir)

        try:
            run_inference_for_case(
                case_dir=case_dir,
                out_case_dir=out_case_dir,
                model_checkpoint=args.model_checkpoint,
                model_config=args.model_config,
                frames_subdir=args.frames_subdir,
                masks_subdir=args.masks_subdir,
            )
        except Exception as e:
            print(f"[FAIL] {case_dir.name}: {e}")

if __name__ == "__main__":
    main()

"""
Example:
    python OAI_medsam_video_inference.py   \
        --input_base_dir /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test \
        --output_base_dir /gpfs/home/machlm03/Segmentation/OAI_demo/Inference \
        --model_checkpoint /gpfs/home/machlm03/Segmentation/MedSAM2/exp_log/MedSAM2_FLARE25_RECIST_OAI/fold0/checkpoints/checkpoint_best.pt \
        --model_config configs/sam2.1_hiera_t512.yaml \
        --frames_subdir imgs_jpeg \
        --masks_subdir masks

To run a subset of cases by name:
    python run_inference.py ... --cases caseA caseB
"""
