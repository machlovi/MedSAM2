#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-class Dice / IoU / Hausdorff (HD & HD95) with NPZ files.
Supports 2D slices and 3D volumes.

Your layout (example):
  gt_base_dir/
    case1.npz, case2.npz, ...
  pred_base_dir/
    case1.npz, case2.npz, ...

Each NPZ file should contain:
  - 'gts': ground truth array (2D or 3D)
  - 'pre' or 'pred': prediction array (2D or 3D)

Usage examples
--------------
python npz_metrics.py \
  --gt_base /path/to/gt/ \
  --pred_base /path/to/pred/ \
  --out_csv ./results.csv \
  --classes 1 2 3 4 5 6 7 \
  --gt_key gts \
  --pred_key pre \
  --mode 3d

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# -------------------------------
# Class names (adjust as needed)
# -------------------------------
CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Femur",
    2: "Tibia", 
    3: "Patella",
    4: "Femoral Cartilage",
    5: "Tibial Cartilage",
    6: "Patellar Cartilage",
    7: "Meniscus"
}

# Try SciPy for fast morphology/EDT
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    binary_erosion = None  # type: ignore
    distance_transform_edt = None  # type: ignore
    _HAVE_SCIPY = False

# -------------------------------
# Low-level helpers
# -------------------------------

def dice_iou_binary(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    """Compute Dice and IoU for binary masks"""
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    inter = np.logical_and(gt, pr).sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    pr_sum = pr.sum(dtype=np.int64)
    union = gt_sum + pr_sum - inter
    dice = np.nan if (gt_sum + pr_sum == 0) else (2.0*inter)/(gt_sum + pr_sum)
    iou  = np.nan if (union == 0) else (inter/union)
    return float(dice), float(iou)

def _edges(mask: np.ndarray) -> np.ndarray:
    """Extract edges from binary mask"""
    mask = mask.astype(bool)
    if len(mask.shape) == 2:  # 2D
        if _HAVE_SCIPY:
            eroded = binary_erosion(mask)
            return np.logical_and(mask, ~eroded)
        # Fallback 4-neighborhood erosion for 2D
        up = np.zeros_like(mask); up[1:] = mask[:-1]
        down = np.zeros_like(mask); down[:-1] = mask[1:]
        left = np.zeros_like(mask); left[:,1:] = mask[:,:-1]
        right = np.zeros_like(mask); right[:,:-1] = mask[:,1:]
        eroded4 = mask & up & down & left & right
        return mask & (~eroded4)
    else:  # 3D
        if _HAVE_SCIPY:
            eroded = binary_erosion(mask)
            return np.logical_and(mask, ~eroded)
        # Simple 3D erosion fallback
        eroded = mask.copy()
        if mask.shape[0] > 1:
            eroded[1:] &= mask[:-1]
            eroded[:-1] &= mask[1:]
        if mask.shape[1] > 1:
            eroded[:, 1:] &= mask[:, :-1]
            eroded[:, :-1] &= mask[:, 1:]
        if mask.shape[2] > 1:
            eroded[:, :, 1:] &= mask[:, :, :-1]
            eroded[:, :, :-1] &= mask[:, :, 1:]
        return mask & (~eroded)

def _surface_distances(a_edge: np.ndarray, b_edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return directed distances from edges of A→B and B→A"""
    if a_edge.sum() == 0 and b_edge.sum() == 0:
        return np.array([]), np.array([])
    
    if _HAVE_SCIPY and distance_transform_edt is not None:
        dt_b = distance_transform_edt(~b_edge)
        dt_a = distance_transform_edt(~a_edge)
        return dt_b[a_edge].astype(float), dt_a[b_edge].astype(float)
    
    # Brute-force fallback
    if len(a_edge.shape) == 2:
        a_pts = np.column_stack(np.nonzero(a_edge))
        b_pts = np.column_stack(np.nonzero(b_edge))
    else:  # 3D
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
    """Compute Hausdorff distance and HD95"""
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

def list_npz_files(directory: Path) -> List[str]:
    """List all NPZ files in directory"""
    files = []
    for p in directory.glob("*.npz"):
        if p.is_file():
            files.append(str(p))
    return files

# -------------------------------
# Workers
# -------------------------------

def _process_one_npz_file(args):
    """Compute per-class metrics for a single NPZ file pair"""
    filename, gt_file, pred_file, labels, gt_key, pred_key, mode = args
    
    try:
        # Load data
        gt_data = np.load(gt_file)
        pred_data = np.load(pred_file)
        
        gt_array = gt_data[gt_key]
        pred_array = pred_data[pred_key]
        
        if gt_array.shape != pred_array.shape:
            raise ValueError(f"Shape mismatch: GT {gt_array.shape} vs Pred {pred_array.shape}")
        
        class_results = {}
        file_dice, file_iou, file_hd = [], [], []
        
        # Process based on mode
        if mode == '2d':
            # Process each slice separately and average
            if len(gt_array.shape) == 3:  # [D, H, W]
                print("length is 3")
                slice_dice, slice_iou, slice_hd = [], [], []
                
                for slice_idx in range(gt_array.shape[0]):
                    gt_slice = gt_array[slice_idx]
                    pred_slice = pred_array[slice_idx]
                    
                    # Check if slice has any annotations
                    if not np.any(np.isin(gt_slice, labels)):
                        continue
                    
                    slice_dice_scores, slice_iou_scores, slice_hd_scores = [], [], []
                    
                    for c in labels:
                        gt_c = (gt_slice == c)
                        pr_c = (pred_slice == c)
                        
                        dice, iou = dice_iou_binary(gt_c, pr_c)
                        hd, hd95 = hd_and_hd95(gt_c, pr_c)
                        
                        if not np.isnan(dice):
                            slice_dice_scores.append(dice)
                        if not np.isnan(iou):
                            slice_iou_scores.append(iou)
                        if np.isfinite(hd):
                            slice_hd_scores.append(hd)
                    
                    if slice_dice_scores:
                        slice_dice.append(np.mean(slice_dice_scores))
                    if slice_iou_scores:
                        slice_iou.append(np.mean(slice_iou_scores))
                    if slice_hd_scores:
                        slice_hd.append(np.mean(slice_hd_scores))
                
                # Average across slices
                for c in labels:
                    class_dice, class_iou, class_hd = [], [], []
                    
                    for slice_idx in range(gt_array.shape[0]):
                        gt_c = (gt_array[slice_idx] == c)
                        pr_c = (pred_array[slice_idx] == c)
                        
                        dice, iou = dice_iou_binary(gt_c, pr_c)
                        hd, hd95 = hd_and_hd95(gt_c, pr_c)
                        
                        if not np.isnan(dice):
                            class_dice.append(dice)
                        if not np.isnan(iou):
                            class_iou.append(iou)
                        if np.isfinite(hd):
                            class_hd.append(hd)
                    
                    class_results[c] = {
                        'dice': float(np.mean(class_dice)) if class_dice else float('nan'),
                        'iou': float(np.mean(class_iou)) if class_iou else float('nan'),
                        'hd': float(np.mean(class_hd)) if class_hd else float('nan'),
                        'hd95': float('nan')  # Not computed for 2D mode
                    }
                
                mean_dice = float(np.mean(slice_dice)) if slice_dice else float('nan')
                mean_iou = float(np.mean(slice_iou)) if slice_iou else float('nan')
                mean_hd = float(np.mean(slice_hd)) if slice_hd else float('nan')
                
            else:  # Single 2D slice
                print("2d slice")
                for c in labels:
                    gt_c = (gt_array == c)
                    pr_c = (pred_array == c)
                    
                    dice, iou = dice_iou_binary(gt_c, pr_c)
                    hd, hd95 = hd_and_hd95(gt_c, pr_c)
                    
                    class_results[c] = {
                        'dice': float(dice),
                        'iou': float(iou),
                        'hd': float(hd),
                        'hd95': float(hd95)
                    }
                    
                    if not np.isnan(dice):
                        file_dice.append(dice)
                    if not np.isnan(iou):
                        file_iou.append(iou)
                    if np.isfinite(hd):
                        file_hd.append(hd)
                
                mean_dice = float(np.mean(file_dice)) if file_dice else float('nan')
                mean_iou = float(np.mean(file_iou)) if file_iou else float('nan')
                mean_hd = float(np.mean(file_hd)) if file_hd else float('nan')
                
        else:  # 3D mode
            print("3d mode")
            for c in labels:
                gt_c = (gt_array == c)
                pr_c = (pred_array == c)
                
                dice, iou = dice_iou_binary(gt_c, pr_c)
                hd, hd95 = hd_and_hd95(gt_c, pr_c)
                
                class_results[c] = {
                    'dice': float(dice),
                    'iou': float(iou), 
                    'hd': float(hd),
                    'hd95': float(hd95)
                }
                
                if not np.isnan(dice):
                    file_dice.append(dice)
                if not np.isnan(iou):
                    file_iou.append(iou)
                if np.isfinite(hd):
                    file_hd.append(hd)
            
            mean_dice = float(np.mean(file_dice)) if file_dice else float('nan')
            mean_iou = float(np.mean(file_iou)) if file_iou else float('nan')
            mean_hd = float(np.mean(file_hd)) if file_hd else float('nan')
        
        return {
            'ok': True,
            'filename': filename,
            'mean_dice': mean_dice,
            'mean_iou': mean_iou,
            'mean_hd': mean_hd,
            'class_results': class_results,
        }
        
    except Exception as e:
        return {'ok': False, 'filename': filename, 'error': str(e)}

# -------------------------------
# Main evaluation function
# -------------------------------

def evaluate_npz_files_parallel(
    gt_base_dir: Path,
    pred_base_dir: Path,
    labels: Optional[List[int]] = None,
    save_csv: Optional[Path] = None,
    verbose: bool = True,
    workers: Optional[int] = None,
    gt_key: str = 'gts',
    pred_key: str = 'pre',
    mode: str = '3d',
):
    """
    Evaluate NPZ files containing segmentation masks
    
    Args:
        gt_base_dir: Directory containing ground truth NPZ files
        pred_base_dir: Directory containing prediction NPZ files
        labels: List of class labels to evaluate
        save_csv: Path to save CSV results
        verbose: Print progress
        workers: Number of parallel workers
        gt_key: Key for ground truth data in NPZ files
        pred_key: Key for prediction data in NPZ files
        mode: '2d' or '3d' evaluation mode
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    gt_base_dir = Path(gt_base_dir)
    pred_base_dir = Path(pred_base_dir)
    
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]
    
    # Find matching NPZ files
    gt_files = list_npz_files(gt_base_dir)
    pred_files = list_npz_files(pred_base_dir)
    
    gt_map = {Path(f).stem: f for f in gt_files}
    pred_map = {Path(f).stem: f for f in pred_files}
    
    common = sorted(set(gt_map) & set(pred_map))
    if not common:
        raise RuntimeError(f"No matching NPZ files between {gt_base_dir} and {pred_base_dir}")
    
    if verbose:
        print(f"Found {len(common)} matching NPZ files")
    
    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            workers = os.cpu_count() or 1
    
    tasks = [(k, Path(gt_map[k]), Path(pred_map[k]), labels, gt_key, pred_key, mode) for k in common]
    
    all_dice, all_iou, all_hd = [], [], []
    from collections import defaultdict
    per_class_dice = defaultdict(list)
    per_class_iou = defaultdict(list)
    per_class_hd = defaultdict(list)
    per_file_results = []
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one_npz_file, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if verbose and (i % 10 == 0):
                print(f"Processed {i}/{len(tasks)} files")
            
            if not r.get('ok'):
                if verbose:
                    print(f"  ERROR {r['filename']}: {r.get('error')}")
                continue
            
            per_file_results.append(r)
            
            if not np.isnan(r['mean_dice']):
                all_dice.append(r['mean_dice'])
            if not np.isnan(r['mean_iou']):
                all_iou.append(r['mean_iou'])
            if not np.isnan(r['mean_hd']):
                all_hd.append(r['mean_hd'])
            
            for c, m in r['class_results'].items():
                if not np.isnan(m['dice']):
                    per_class_dice[c].append(m['dice'])
                if not np.isnan(m['iou']):
                    per_class_iou[c].append(m['iou'])
                if np.isfinite(m['hd']):
                    per_class_hd[c].append(m['hd'])
    
    def _mean(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else float('nan')
    
    def _std(arr: List[float]) -> float:
        return float(np.std(arr)) if arr else float('nan')
    
    # Overall metrics
    overall_metrics = {
        'mean_dice': _mean(all_dice),
        'std_dice': _std(all_dice),
        'mean_iou': _mean(all_iou),
        'std_iou': _std(all_iou),
        'mean_hd': _mean(all_hd),
        'std_hd': _std(all_hd),
        'num_files': len(per_file_results),
        'num_classes': len(labels)
    }
    
    # Per-class metrics
    per_class_summary = []
    for c in labels:
        per_class_summary.append({
            'class': c,
            'class_name': CLASS_NAMES.get(c, str(c)),
            'mean_dice': _mean(per_class_dice[c]),
            'std_dice': _std(per_class_dice[c]),
            'mean_iou': _mean(per_class_iou[c]),
            'std_iou': _std(per_class_iou[c]),
            'mean_hd': _mean(per_class_hd[c]),
            'std_hd': _std(per_class_hd[c]),
        })
    
    results = {
        'overall_metrics': overall_metrics,
        'per_class_summary': per_class_summary,
        'per_file_metrics': per_file_results,
    }
    
    if save_csv:
        save_results_to_csv(results, Path(save_csv))
    
    return results

def save_results_to_csv(results: Dict, save_path: Path) -> None:
    """Save results to CSV"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Per-file results
    file_data = []
    for r in results['per_file_metrics']:
        row = {
            'filename': r['filename'],
            'mean_dice': r['mean_dice'],
            'mean_iou': r['mean_iou'],
            'mean_hd': r['mean_hd']
        }
        # Add per-class metrics
        for c, metrics in r['class_results'].items():
            row[f'class_{c}_dice'] = metrics['dice']
            row[f'class_{c}_iou'] = metrics['iou']
            row[f'class_{c}_hd'] = metrics['hd']
        file_data.append(row)
    
    # Per-class summary
    class_data = []
    for c in results['per_class_summary']:
        class_data.append({
            'class': c['class'],
            'class_name': c['class_name'],
            'mean_dice': c['mean_dice'],
            'std_dice': c['std_dice'],
            'mean_iou': c['mean_iou'],
            'std_iou': c['std_iou'],
            'mean_hd': c['mean_hd'],
            'std_hd': c['std_hd'],
        })
    
    # Overall summary
    overall_data = [{
        'metric': 'Overall',
        'mean_dice': results['overall_metrics']['mean_dice'],
        'std_dice': results['overall_metrics']['std_dice'],
        'mean_iou': results['overall_metrics']['mean_iou'],
        'std_iou': results['overall_metrics']['std_iou'],
        'mean_hd': results['overall_metrics']['mean_hd'],
        'std_hd': results['overall_metrics']['std_hd'],
        'num_files': results['overall_metrics']['num_files'],
    }]
    
    # Save separate CSVs
    pd.DataFrame(file_data).to_csv(save_path.with_suffix('.per_file.csv'), index=False)
    pd.DataFrame(class_data).to_csv(save_path.with_suffix('.per_class.csv'), index=False)
    pd.DataFrame(overall_data).to_csv(save_path.with_suffix('.overall.csv'), index=False)
    
    print(f"✅ Saved results to:")
    print(f"  Per-file: {save_path.with_suffix('.per_file.csv')}")
    print(f"  Per-class: {save_path.with_suffix('.per_class.csv')}")
    print(f"  Overall: {save_path.with_suffix('.overall.csv')}")

def parse_args():
    ap = argparse.ArgumentParser(description="Dice/IoU/HD for multi-class NPZ masks")
    ap.add_argument('--gt_base', type=Path, required=True, help='Directory with GT NPZ files')
    ap.add_argument('--pred_base', type=Path, required=True, help='Directory with Pred NPZ files')
    ap.add_argument('--out_csv', type=Path, required=True, help='Output CSV prefix')
    ap.add_argument('--classes', type=int, nargs='*', default=[1,2,3,4,5,6,7], help='Class labels to evaluate')
    ap.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    ap.add_argument('--gt_key', type=str, default='gts', help='Key for GT data in NPZ files')
    ap.add_argument('--pred_key', type=str, default='pre', help='Key for prediction data in NPZ files')
    ap.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'], help='Evaluation mode')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    res = evaluate_npz_files_parallel(
        gt_base_dir=args.gt_base,
        pred_base_dir=args.pred_base,
        labels=args.classes,
        save_csv=args.out_csv,
        workers=args.workers,
        gt_key=args.gt_key,
        pred_key=args.pred_key,
        mode=args.mode,
        verbose=True,
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print("Overall metrics:")
    for k, v in res['overall_metrics'].items():
        print(f"  {k}: {v}")
    
    print("\nPer-class summary:")
    for c in res['per_class_summary']:
        print(f"  {c['class_name']}: Dice={c['mean_dice']:.4f}, IoU={c['mean_iou']:.4f}, HD={c['mean_hd']:.4f}")

# Example usage:
# python metrices_npz.py \
#   --gt_base /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_MultiClass/npz/ \
#   --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSam2_Inference_bbx/ \
#   --out_csv /gpfs/home/machlm03/Segmentation/MedSAM2/oai_utils/bbx_results/medsam_oai_ \
#   --classes 1 2 3 4 5 6 7 \
#   --gt_key gts \
#   --pred_key pre \
#   --mode 2d \
#   --workers 40