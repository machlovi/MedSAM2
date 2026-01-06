#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-class Dice / IoU / Hausdorff (HD & HD95) with two separate roots (GT and Pred),
case-wise folders, and recursive file discovery. Parallelized per file.

Your layout (example):
  gt_base_dir/
    <CASE>/masks/*.png   (or any substructure; pass --gt_subdir if fixed)
  pred_base_dir/
    <CASE>/imgs/*.png    (or any substructure; pass --pred_subdir if fixed)

This script:
  • Matches cases (folder names) present in BOTH roots
  • Matches files by stem (without extension) inside each case (recursive)
  • Remaps grayscale→class IDs using your GRAY_TO_CLASS LUT
  • Computes per-class Dice, IoU, HD, HD95 for each file
  • Aggregates per-case means and overall means
  • Saves CSVs

Key fixes vs your draft:
  • Correct HD/HD95 using surface distances (EDT if SciPy available; brute-force fallback)
  • Removed unused directed_hausdorff import & function; no dependency needed
  • Fixed per-file mean accumulators (previously never appended)
  • Added HD to CSVs and summaries
  • Fixed bug printing file_csv_path when it's commented out
  • Safer handling for empty masks (NaN for Dice/IoU, Inf for HD; configurable averaging)

Usage examples
--------------
python seg_metrics_patientwise.py \
  --gt_base /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0/ \
  --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference/fold0/ \
  --out_csv ./oai_Inference_fold0.csv \
  --gt_subdir masks --pred_subdir imgs \
  --classes 1 2 3 4 5 6 7 \
  --hd_ignore_inf

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image

# -------------------------------
# Grayscale → class-ID mapping and names
# -------------------------------
GRAY_TO_CLASS: Dict[int, int] = {0:0, 60:1, 120:2, 180:3, 240:4, 44:5, 164:6, 104:7}
CLASS_NAMES: Dict[int, str] = {0:"Background",1:"Femur",2:"Tibia",3:"Patella",4:"Femoral Cartilage",5:"Tibial Cartilage",6:"Patellar Cartilage",7:"Meniscus"}

# Prebuilt LUT (shared to workers efficiently)
_LUT = np.zeros(256, dtype=np.uint8)
for g, c in GRAY_TO_CLASS.items():
    _LUT[g] = c

# Try SciPy for fast morphology/EDT
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _HAVE_SCIPY = True
except Exception:  # SciPy not available
    binary_erosion = None  # type: ignore
    distance_transform_edt = None  # type: ignore
    _HAVE_SCIPY = False

# -------------------------------
# Low-level helpers
# -------------------------------

def remap_with_lut(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return _LUT[arr]


def dice_iou_binary(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    inter = np.logical_and(gt, pr).sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    pr_sum = pr.sum(dtype=np.int64)
    union = gt_sum + pr_sum - inter
    dice = np.nan if (gt_sum + pr_sum == 0) else (2.0*inter)/(gt_sum + pr_sum)
    iou  = np.nan if (union == 0)            else (inter/union)
    return float(dice), float(iou)


def _edges(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if _HAVE_SCIPY:
        eroded = binary_erosion(mask)
        return np.logical_and(mask, ~eroded)
    # Fallback 4-neighborhood erosion
    up = np.zeros_like(mask); up[1:] = mask[:-1]
    down = np.zeros_like(mask); down[:-1] = mask[1:]
    left = np.zeros_like(mask); left[:,1:] = mask[:,:-1]
    right = np.zeros_like(mask); right[:,:-1] = mask[:,1:]
    eroded4 = mask & up & down & left & right
    return mask & (~eroded4)


def _surface_distances(a_edge: np.ndarray, b_edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Return directed distances from edges of A→B and B→A
    if a_edge.sum() == 0 and b_edge.sum() == 0:
        return np.array([]), np.array([])
    if _HAVE_SCIPY and distance_transform_edt is not None:
        dt_b = distance_transform_edt(~b_edge)
        dt_a = distance_transform_edt(~a_edge)
        return dt_b[a_edge].astype(float), dt_a[b_edge].astype(float)
    # Brute-force fallback
    a_pts = np.column_stack(np.nonzero(a_edge))
    b_pts = np.column_stack(np.nonzero(b_edge))
    if a_pts.size == 0 and b_pts.size == 0:
        return np.array([]), np.array([])
    if a_pts.size == 0:
        return np.array([]), np.zeros(len(b_pts))
    if b_pts.size == 0:
        return np.zeros(len(a_pts)), np.array([])
    def nn(src: np.ndarray, dst: np.ndarray, chunk: int = 5000) -> np.ndarray:
        out = np.empty(len(src), dtype=float)
        for i in range(0, len(src), chunk):
            s = src[i:i+chunk]
            diffs = s[:, None, :] - dst[None, :, :]
            d2 = np.sum(diffs*diffs, axis=2)
            out[i:i+chunk] = np.sqrt(np.min(d2, axis=1))
        return out
    return nn(a_pts, b_pts), nn(b_pts, a_pts)


def hd_and_hd95(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    if gt.sum() == 0 and pr.sum() == 0:
        return float("nan"), float("nan")
    if gt.sum() == 0 or pr.sum() == 0:
        return float("inf"), float("inf")
    a = _edges(gt)
    b = _edges(pr)
    d_ab, d_ba = _surface_distances(a, b)
    if d_ab.size == 0 and d_ba.size == 0:
        return float("nan"), float("nan")
    all_d = np.concatenate([d_ab, d_ba]) if d_ab.size or d_ba.size else np.array([])
    return float(np.max(all_d)), float(np.percentile(all_d, 95))


def list_images_recursive(directory: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg"}
    files: List[str] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    return files

# -------------------------------
# Workers
# -------------------------------

def _process_one_file(args):
    """Compute per-class metrics for a single file pair."""
    filename, gt_file, pred_file, labels = args
    try:
        with Image.open(gt_file) as im:
            gt_ids = remap_with_lut(im)
        with Image.open(pred_file) as im:
            pr_ids = remap_with_lut(im)

        class_results = {}
        file_dice, file_iou, file_hd = [], [], []

        class_stats = {c: {'gt_pixels': 0, 'pred_pixels': 0,
                           'files_present_gt': 0, 'files_present_pred': 0}
                       for c in labels}

        for c in labels:
            gt_c = (gt_ids == c)
            pr_c = (pr_ids == c)

            # stats
            gp, pp = int(gt_c.sum()), int(pr_c.sum())
            class_stats[c]['gt_pixels'] += gp
            class_stats[c]['pred_pixels'] += pp
            if gp > 0: class_stats[c]['files_present_gt'] += 1
            if pp > 0: class_stats[c]['files_present_pred'] += 1

            dice, iou = dice_iou_binary(gt_c, pr_c)
            hd, hd95 = hd_and_hd95(gt_c, pr_c)

            class_results[c] = {'dice': float(dice), 'iou': float(iou), 'hd': float(hd), 'hd95': float(hd95)}

            if not np.isnan(dice): file_dice.append(dice)
            if not np.isnan(iou):  file_iou.append(iou)
            # For HD, include finite only in per-file mean (more stable). You can change this if desired.
            if np.isfinite(hd):    file_hd.append(hd)

        mean_dice = float(np.mean(file_dice)) if file_dice else float('nan')
        mean_iou  = float(np.mean(file_iou))  if file_iou  else float('nan')
        mean_hd   = float(np.mean(file_hd))   if file_hd   else float('nan')

        return {
            'ok': True,
            'filename': filename,
            'mean_dice': mean_dice,
            'mean_iou':  mean_iou,
            'mean_hd':   mean_hd,
            'class_results': class_results,
            'class_stats': class_stats,
        }
    except Exception as e:
        return {'ok': False, 'filename': filename, 'error': str(e)}

# -------------------------------
# Per-case (directory) evaluation
# -------------------------------

def compute_comprehensive_metrics_parallel(
    gt_dir: Path,
    pred_dir: Path,
    labels: List[int],
    verbose: bool = True,
    workers: Optional[int] = None,
    gt_subdir: Optional[str] = None,
    pred_subdir: Optional[str] = None,
    match_by: str = "stem",
):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if gt_subdir:
        gt_dir = gt_dir / gt_subdir
    if pred_subdir:
        pred_dir = pred_dir / pred_subdir

    gt_files = list_images_recursive(gt_dir)
    pr_files = list_images_recursive(pred_dir)

    if match_by == "stem":
        gt_map = {Path(f).stem: f for f in gt_files}
        pr_map = {Path(f).stem: f for f in pr_files}
    else:  # name
        gt_map = {Path(f).name: f for f in gt_files}
        pr_map = {Path(f).name: f for f in pr_files}

    common = sorted(set(gt_map) & set(pr_map))
    if not common:
        raise RuntimeError(f"No matching files between {gt_dir} and {pred_dir} (matched by {match_by})")

    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            workers = os.cpu_count() or 1

    tasks = [(k, Path(gt_map[k]), Path(pr_map[k]), labels) for k in common]

    all_dice, all_iou, all_hd = [], [], []
    from collections import defaultdict
    per_class_dice = defaultdict(list)
    per_class_iou  = defaultdict(list)
    per_class_hd   = defaultdict(list)
    per_file_results = []
    class_pixel_stats = defaultdict(lambda: {'gt_pixels': 0, 'pred_pixels': 0,
                                             'files_present_gt': 0, 'files_present_pred': 0})

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one_file, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if verbose and (i % 50 == 0):
                print(f"Processed {i}/{len(tasks)} files")
            if not r.get('ok'):
                if verbose:
                    print(f"  ERROR {r['filename']}: {r.get('error')}")
                continue

            per_file_results.append({'filename': r['filename'], 'mean_dice': r['mean_dice'], 'mean_iou': r['mean_iou'], 'mean_hd': r['mean_hd'], 'class_results': r['class_results']})
            if not np.isnan(r['mean_dice']): all_dice.append(r['mean_dice'])
            if not np.isnan(r['mean_iou']):  all_iou.append(r['mean_iou'])
            if not np.isnan(r['mean_hd']):   all_hd.append(r['mean_hd'])

            for c, m in r['class_results'].items():
                if not np.isnan(m['dice']): per_class_dice[c].append(m['dice'])
                if not np.isnan(m['iou']):  per_class_iou[c].append(m['iou'])
                if np.isfinite(m['hd']):    per_class_hd[c].append(m['hd'])

            for c, s in r['class_stats'].items():
                cs = class_pixel_stats[c]
                cs['gt_pixels']         += s['gt_pixels']
                cs['pred_pixels']       += s['pred_pixels']
                cs['files_present_gt']  += s['files_present_gt']
                cs['files_present_pred']+= s['files_present_pred']

    def _mean(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else float('nan')
    def _std(arr: List[float]) -> float:
        return float(np.std(arr)) if arr else float('nan')

    per_class_results = {}
    for c in labels:
        per_class_results[c] = {
            'class_name': CLASS_NAMES.get(c, str(c)),
            'mean_dice': _mean(per_class_dice[c]), 'std_dice': _std(per_class_dice[c]),
            'mean_iou' : _mean(per_class_iou[c]),  'std_iou' : _std(per_class_iou[c]),
            'mean_hd'  : _mean(per_class_hd[c]),   'std_hd'  : _std(per_class_hd[c]),
        }

    overall_metrics = {
        'mean_dice': _mean(all_dice), 'std_dice': _std(all_dice),
        'mean_iou' : _mean(all_iou),  'std_iou' : _std(all_iou),
        'mean_hd'  : _mean(all_hd),   'std_hd'  : _std(all_hd),
        'num_files': len(per_file_results), 'num_classes': len(labels)
    }

    return {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_results,
        'per_file_metrics': per_file_results,
        'class_statistics': dict(class_pixel_stats),
    }

# -------------------------------
# Cross-case evaluation (two roots)
# -------------------------------

def evaluate_nested_cases_parallel(
    gt_base_dir: Path,
    pred_base_dir: Path,
    labels: Optional[List[int]] = None,
    save_csv: Optional[Path] = None,
    verbose: bool = True,
    workers: Optional[int] = None,
    gt_subdir: Optional[str] = None,
    pred_subdir: Optional[str] = None,
    match_by: str = "stem",
):
    gt_base_dir = Path(gt_base_dir)
    pred_base_dir = Path(pred_base_dir)
    if labels is None:
        labels = [1,2,3,4,5,6,7]

    gt_cases = {d.name for d in gt_base_dir.iterdir() if d.is_dir()}
    pr_cases = {d.name for d in pred_base_dir.iterdir() if d.is_dir()}
    common_cases = sorted(gt_cases & pr_cases)
    if not common_cases:
        raise RuntimeError(f"No common cases between {gt_base_dir} and {pred_base_dir}")

    if verbose:
        print(f"Found {len(common_cases)} common cases")

    all_case_means_dice, all_case_means_iou, all_case_means_hd = [], [], []
    per_case_metrics: List[Dict] = []
    per_file_metrics: List[Dict] = []

    # NEW: accumulate per-class metrics across ALL cases/files
    from collections import defaultdict
    per_class_agg = defaultdict(lambda: {'dice': [], 'iou': [], 'hd': []})

    for case in common_cases:
        if verbose:
            print(f"--- Case: {case} ---")
        try:
            case_res = compute_comprehensive_metrics_parallel(
                gt_dir=gt_base_dir / case,
                pred_dir=pred_base_dir / case,
                labels=labels,
                verbose=False,
                workers=workers,
                gt_subdir=gt_subdir,
                pred_subdir=pred_subdir,
                match_by=match_by,
            )
        except RuntimeError as e:
            if verbose:
                print(f"  Skipping {case}: {e}")
            continue

        n_files = case_res['overall_metrics']['num_files']
        m_dice  = case_res['overall_metrics']['mean_dice']
        m_iou   = case_res['overall_metrics']['mean_iou']
        m_hd    = case_res['overall_metrics']['mean_hd']
        if not np.isnan(m_dice): all_case_means_dice.append(m_dice)
        if not np.isnan(m_iou):  all_case_means_iou.append(m_iou)
        if not np.isnan(m_hd):   all_case_means_hd.append(m_hd)

        per_case_metrics.append({'case_name': case, 'num_files': n_files, 'mean_dice': m_dice, 'mean_iou': m_iou, 'mean_hd': m_hd})

        for f in case_res['per_file_metrics']:
            f = dict(f)  # copy
            f['case_name'] = case
            per_file_metrics.append(f)

            # NEW: accumulate per-class metrics for global per-class means
            for c, m in f['class_results'].items():
                if not np.isnan(m.get('dice', np.nan)):
                    per_class_agg[c]['dice'].append(m['dice'])
                if not np.isnan(m.get('iou', np.nan)):
                    per_class_agg[c]['iou'].append(m['iou'])
                if np.isfinite(m.get('hd', np.nan)):
                    per_class_agg[c]['hd'].append(m['hd'])
    def _mean(arr: List[float]) -> float: return float(np.mean(arr)) if arr else float('nan')

    overall = {
        'mean_dice': _mean(all_case_means_dice),
        'mean_iou' : _mean(all_case_means_iou),
        'mean_hd'  : _mean(all_case_means_hd),
        'num_cases': len(per_case_metrics),
        'num_files': sum(m['num_files'] for m in per_case_metrics),
    }

    # NEW: finalize per-class means across ALL cases/files
    per_class_summary = []
    for c in labels:
        d = float(np.mean(per_class_agg[c]['dice'])) if per_class_agg[c]['dice'] else float('nan')
        i = float(np.mean(per_class_agg[c]['iou'])) if per_class_agg[c]['iou'] else float('nan')
        h = float(np.mean(per_class_agg[c]['hd']))  if per_class_agg[c]['hd']  else float('nan')
        per_class_summary.append({'class': c, 'class_name': CLASS_NAMES.get(c, str(c)), 'mean_dice': d, 'mean_iou': i, 'mean_hd': h})

    results = {
        'overall_metrics': overall,
        'per_case_metrics': per_case_metrics,
        'per_file_metrics': per_file_metrics,
    }
    results['per_class_summary'] = per_class_summary

    if save_csv:
        save_results_to_csv(results, Path(save_csv), labels)

    return results



def save_results_to_csv(results: Dict, save_path: Path, labels: List[int]) -> None:
    from pathlib import Path
    import pandas as pd

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Per-case summary
    # -------------------------------
    df_case = pd.DataFrame(results['per_case_metrics'])

    # -------------------------------
    # Per-class summary
    # -------------------------------
    df_class = pd.DataFrame(results.get('per_class_summary', []))

    if not df_class.empty:
        # Rename 'class_name' to 'case_name' to align with df_case
        df_class = df_class.rename(columns={'class_name': 'case_name'})
        # Add missing columns to match df_case structure
        for col in df_case.columns:
            if col not in df_class.columns:
                df_class[col] = None
        # Reorder columns to match df_case
        df_class = df_class[df_case.columns]

    # -------------------------------
    # Overall summary
    # -------------------------------
    df_overall = pd.DataFrame([results['overall_metrics']])
    # Add 'case_name' column to overall summary
    df_overall.insert(0, 'case_name', 'Overall')

    # -------------------------------
    # Combine all sections
    # -------------------------------
    combined_df = pd.concat([df_case, df_class, df_overall], ignore_index=True)
    combined_df.to_csv(save_path, index=False)

    print(f"✅ Saved all results to: {save_path}")




def save_results_to_csv(results: Dict, save_path: Path, labels: List[int]) -> None:
    from pathlib import Path
    import pandas as pd
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------
    # Save the main summary CSV (case and overall metrics)
    # -------------------------------
    df_case = pd.DataFrame(results['per_case_metrics'])
    df_class = pd.DataFrame(results.get('per_class_summary', []))
    if not df_class.empty:
        df_class = df_class.rename(columns={'class_name': 'case_name'})
        for col in df_case.columns:
            if col not in df_class.columns:
                df_class[col] = None
        df_class = df_class[df_case.columns]
    
    df_overall = pd.DataFrame([results['overall_metrics']])
    df_overall.insert(0, 'case_name', 'Overall')
    
    combined_df = pd.concat([df_case, df_class, df_overall], ignore_index=True)
    combined_df.to_csv(save_path, index=False)
    print(f"✅ Saved summary results to: {save_path}")
    
    # -------------------------------
    # Save per-slice metrics CSV with all cases in a single file
    # -------------------------------
    per_slice_path = save_path.with_stem(f"{save_path.stem}_per_slice")
    
    # Extract and format per-slice metrics
    slice_rows = []
    for file_metric in results['per_file_metrics']:
        case_name = file_metric['case_name']
        filename = file_metric['filename']
        # Extract just the basename rather than full path for cleaner output
        basename = Path(filename).name
        
        # Add overall metrics for this slice
        base_row = {
            'case': case_name,
            'slice': basename,
            'mean_dice': file_metric.get('mean_dice', float('nan')),
            'mean_iou': file_metric.get('mean_iou', float('nan')),
            'mean_hd': file_metric.get('mean_hd', float('nan'))
        }
        
        # Add per-class metrics
        for class_id in labels:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            if class_id in file_metric['class_results']:
                class_results = file_metric['class_results'][class_id]
                # Add class-specific metrics with clear column names
                base_row[f'{class_name}_dice'] = class_results.get('dice', float('nan'))
                base_row[f'{class_name}_iou'] = class_results.get('iou', float('nan'))
                base_row[f'{class_name}_hd'] = class_results.get('hd', float('nan'))
                base_row[f'{class_name}_hd95'] = class_results.get('hd95', float('nan'))
            else:
                # Handle missing classes in this slice
                base_row[f'{class_name}_dice'] = float('nan')
                base_row[f'{class_name}_iou'] = float('nan')
                base_row[f'{class_name}_hd'] = float('nan')
                base_row[f'{class_name}_hd95'] = float('nan')
        
        slice_rows.append(base_row)
    
    # Create and save the per-slice DataFrame
    if slice_rows:
        df_slices = pd.DataFrame(slice_rows)
        # Sort by case and then by slice name for better readability
        df_slices = df_slices.sort_values(['case', 'slice'])
        df_slices.to_csv(per_slice_path, index=False, float_format="%.4f")
        print(f"✅ Saved per-slice metrics to: {per_slice_path}")
    else:
        print("⚠️ No per-slice data available to save")


def parse_args():
    ap = argparse.ArgumentParser(description="Dice/IoU/HD for multi-class PNG masks (two-root, case-wise)")
    ap.add_argument('--gt_base', type=Path, required=True)
    ap.add_argument('--pred_base', type=Path, required=True)
    ap.add_argument('--out_csv', type=Path, required=True, help='Output CSV prefix (no _per_file suffix)')
    ap.add_argument('--gt_subdir', type=str, default=None)
    ap.add_argument('--pred_subdir', type=str, default=None)
    ap.add_argument('--classes', type=int, nargs='*', default=[1,2,3,4,5,6,7])
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--match_by', type=str, default='stem', choices=['stem','name'])
    ap.add_argument('--hd_ignore_inf', action='store_true', help='(Deprecated) kept for compatibility; HD means already ignore inf for per-file averages')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    res = evaluate_nested_cases_parallel(
        gt_base_dir=args.gt_base,
        pred_base_dir=args.pred_base,
        labels=args.classes,
        save_csv=args.out_csv,
        workers=args.workers,
        gt_subdir=args.gt_subdir,
        pred_subdir=args.pred_subdir,
        match_by=args.match_by,
        verbose=True,
    )
    print("Overall:")
    print(res['overall_metrics'])

# python metrices.py \
# --gt_base /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0/ \
# --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference/fold5 \
# --out_csv /gpfs/home/machlm03/Segmentation/MedSAM2/oai_utils/updated_results/OAI_IWOAI_Inference2.2.csv \
# --gt_subdir masks \
# --pred_subdir imgs \
# --classes 1 2 3 4 5 6 7 \ 
# --workers 40 


# python metrices.py \
# --gt_base /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0/ \
# --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference_2.3/fold5/ \
# --out_csv /gpfs/home/machlm03/Segmentation/MedSAM2/oai_utils/results/OAI_IWOAI_Inference2.3.csv \
# --gt_subdir masks \
# --pred_subdir imgs \
# --classes 1 2 3 4 5 6 7 \
# --workers 16  # Set this to your number of CPUs or preferred value


# python metrices.py \
# --gt_base /gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/test/imgs/ \
# --pred_base /gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/Inference_test/ \
# --out_csv /gpfs/home/machlm03/Segmentation/MedSAM2/oai_utils/results/OAI_IWOAI_Inference.csv \
# --gt_subdir masks \
# --pred_subdir imgs \
# --classes 1 2 3 4 \
# --workers 2 \


# python metrices.py \
# --gt_base /gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/test/imgs/ \
# --pred_base /gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/mask_medsam_inference/ \
# --out_csv /gpfs/home/machlm03/Segmentation/MedSAM2/oai_utils/results/MedSAM2_IWOAI_Inference.csv \
# --gt_subdir masks \
# --pred_subdir imgs \
# --classes 1 2 3 4 \
# --workers 2
