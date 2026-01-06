
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# Explicit mapping from grayscale to class-ID
GRAY_TO_CLASS = {
    0: 0,    # background
    60: 1,   # class 1
    120: 2,  # class 2
    180: 3,  # class 3
    240: 4,  # class 4
    44: 5,   # class 5
    164: 6,  # class 6
    104: 7,  # class 7
}

# Define class colors for visualization
CLASS_COLORS = {
    0: (0, 0, 0),       # background - black
    1: (255, 0, 0),     # class 1 - red
    2: (0, 255, 0),     # class 2 - green
    3: (0, 0, 255),     # class 3 - blue
    4: (255, 255, 0),   # class 4 - yellow
    # 5: (255, 0, 255),   # class 5 - magenta
    # 6: (0, 255, 255),   # class 6 - cyan
    # 7: (128, 128, 128), # class 7 - gray
}

# Class names for better reporting
# CLASS_NAMES = {
#     0: "Background",
#     1: "Femur", 
#     2: "Tibia",
#     3: "Patella",
#     4: "Femoral Cartilage",
#     5: "Tibial Cartilage", 
#     6: "Patellar Cartilage",
#     7: "Meniscus"
# }

CLASS_NAMES = {
    0: "Background",
    1: "Femoral Cartilage",
    2: "Tibial Cartilage", 
    3: "Patellar Cartilage",
    4: "Meniscus"
}

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Put near your constants ---
# Prebuild a 256-LUT once so workers can reuse it (sent by pickle efficiently)
_LUT = np.zeros(256, dtype=np.uint8)
for g, c in GRAY_TO_CLASS.items():
    _LUT[g] = c

def _remap_with_lut(arr_uint8):
    # arr_uint8 is np.uint8 array
    return _LUT[arr_uint8]

def hausdorff_distance_from_png(gt_mask, pred_mask):
    """
    Compute symmetric Hausdorff distance between two PNG masks.
    
    Parameters:
    - gt_path: Path to ground truth PNG mask
    - pred_path: Path to predicted PNG mask

    Returns:
    - float: Hausdorff distance
    """
    # Get foreground coordinates
    gt_coords = np.argwhere(gt_mask)
    pred_coords = np.argwhere(pred_mask)

    if gt_coords.size == 0 or pred_coords.size == 0:
        return float('inf')  # No foreground to compare

    # Directed distances
    d1 = directed_hausdorff(gt_coords, pred_coords)[0]
    d2 = directed_hausdorff(pred_coords, gt_coords)[0]

    return max(d1, d2)



def _process_one_file(args):
    """
    Worker: compute all metrics for a single (gt_file, pred_file).
    Returns a dict with per-file summary + per-class metrics + pixel stats.
    """
    filename, gt_file, pred_file, labels = args
    try:
        # Ensure files are closed in worker
        with Image.open(gt_file).convert("L") as gt_img:
            gt_arr = np.array(gt_img, dtype=np.uint8)
        with Image.open(pred_file).convert("L") as pr_img:
            pr_arr = np.array(pr_img, dtype=np.uint8)

        gt_ids = _remap_with_lut(gt_arr)
        pr_ids = _remap_with_lut(pr_arr)

        file_dice = []
        file_hdscore = []
        file_hdscore=[]
        class_results = {}
        class_stats = {c: {'gt_pixels': 0, 'pred_pixels': 0,
                           'files_present_gt': 0, 'files_present_pred': 0}
                       for c in labels}

        for c in labels:
            gt_bool = (gt_ids == c)
            pr_bool = (pr_ids == c)
            class_stats[c]['gt_pixels'] += int(gt_bool.sum())
            class_stats[c]['pred_pixels'] += int(pr_bool.sum())
            if gt_bool.any():
                class_stats[c]['files_present_gt'] += 1
            if pr_bool.any():
                class_stats[c]['files_present_pred'] += 1

            inter = np.logical_and(gt_bool, pr_bool).sum()
            union = np.logical_or(gt_bool, pr_bool).sum()
            gt_sum = gt_bool.sum()
            pr_sum = pr_bool.sum()

            if gt_sum == 0 and pr_sum == 0:
                dice = 1.0; iou = 1.0
                hd = 0.0
            elif union == 0:
                dice = 0.0; iou = 0.0
                hd = float('inf')
            else:
                dice = (2.0 * inter) / (gt_sum + pr_sum) if (gt_sum + pr_sum) > 0 else 0.0
                iou  = inter / union if union > 0 else 0.0
                hd = hausdorff_distance_from_png(gt_bool, pr_bool)

            
            class_results[c] = {
                'dice': float(dice),
                'iou': float(iou),
                'hd': float(hd)
            }

        file_mean_dice = float(np.mean(file_dice))
        file_mean_iou  = float(np.mean(file_iou))
        file_mean_hdscore=float(np.mean(file_hdscore))

        return {
            'ok': True,
            'filename': filename,
            'mean_dice': file_mean_dice,
            'mean_iou': file_mean_iou,
            'mean_hd_score':file_mean_hdscore,
            'class_results': class_results,
            'class_stats': class_stats,
        }

    except Exception as e:
        return {'ok': False, 'filename': filename, 'error': str(e)}










def list_images(directory):
    """List image files in directory"""
    extensions = ['.png', '.jpg', '.jpeg']
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return [str(f) for f in files]


def save_results_to_csv(results, save_path, labels):
    """Save detailed results to CSV files"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Per-file results
    file_data = []
    for file_result in results['per_file_metrics']:
        row = {
            'filename': file_result['filename'],
            'mean_dice': file_result['mean_dice'],
            'mean_iou': file_result['mean_iou']
        }
        # Add per-class results
        for class_label in labels:
            class_name = CLASS_NAMES[class_label].replace(' ', '_')
            if class_label in file_result['class_results']:
                row[f'{class_name}_dice'] = file_result['class_results'][class_label]['dice']
                row[f'{class_name}_iou'] = file_result['class_results'][class_label]['iou']
            else:
                row[f'{class_name}_dice'] = 0.0
                row[f'{class_name}_iou'] = 0.0
        file_data.append(row)
    
    df = pd.DataFrame(file_data)
    df.to_csv(save_path, index=False)
    print(f"\nDetailed results saved to: {save_path}")



def save_nested_results_to_csv(results, save_path, labels):
    """Save nested results to CSV files"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save per-file results (with case information)
    file_data = []
    for file_result in results['per_file_metrics']:
        row = {
            'case_name': file_result['case_name'],
            'filename': file_result['filename'],
            'mean_dice': file_result['mean_dice'],
            'mean_iou': file_result['mean_iou']
        }
        # Add per-class results
        for class_label in labels:
            class_name = CLASS_NAMES[class_label].replace(' ', '_')
            if class_label in file_result['class_results']:
                row[f'{class_name}_dice'] = file_result['class_results'][class_label]['dice']
                row[f'{class_name}_iou'] = file_result['class_results'][class_label]['iou']
            else:
                row[f'{class_name}_dice'] = 0.0
                row[f'{class_name}_iou'] = 0.0
        file_data.append(row)
    
    df_files = pd.DataFrame(file_data)
    file_csv_path = save_path.parent / f"{save_path.stem}_per_file.csv"
    df_files.to_csv(file_csv_path, index=False)
    
    # Save per-case summary
    case_data = []
    for case_result in results['per_case_metrics']:
        case_data.append(case_result)
    
    df_cases = pd.DataFrame(case_data)
    case_csv_path = save_path.parent / f"{save_path.stem}_per_case.csv"
    df_cases.to_csv(case_csv_path, index=False)
    
    print(f"\nDetailed results saved to:")
    print(f"  Per-file results: {file_csv_path}")
    print(f"  Per-case summary: {case_csv_path}")




def compute_comprehensive_metrics_parallel(
    gt_dir,
    pred_dir,
    labels=None,
    verbose=True,
    save_results=None,
    workers=None,
    chunksize=16,
    gt_subdir=None,      # <- NEW: optional, e.g. "masks"
    pred_subdir=None,    # <- NEW: optional, e.g. "imgs"
    match_by="stem"      # "stem" (default) or "name" (same) — keep simple & robust
):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    gt_dir  = Path(gt_dir)
    pred_dir = Path(pred_dir)
    if gt_subdir:
        gt_dir = gt_dir / gt_subdir
    if pred_subdir:
        pred_dir = pred_dir / pred_subdir

    if labels is None:
        labels = [1,2,3,4]

    # RECURSIVE file discovery (any nesting)
    gt_files = list_images_recursive(gt_dir)
    pred_files = list_images_recursive(pred_dir)

    # Build maps by filename stem (robust across nested paths)
    if match_by == "stem":
        gt_map  = {Path(f).stem: f for f in gt_files}
        pr_map  = {Path(f).stem: f for f in pred_files}
    else:  # alias "name"
        gt_map  = {Path(f).name: f for f in gt_files}
        pr_map  = {Path(f).name: f for f in pred_files}

    common_keys = sorted(set(gt_map.keys()) & set(pr_map.keys()))
    if not common_keys:
        raise RuntimeError(
            f"No matching files found between\n GT: {gt_dir}\n PRED: {pred_dir}\n"
            f"(matched by {match_by}; searched recursively)"
        )

    if verbose:
        print(f"Found {len(common_keys)} matching files (recursive search)")
        print(f"Evaluating classes: {labels}")
        print(f"Class names: {[CLASS_NAMES[l] for l in labels]}")

    # containers (same as before)
    all_dice, all_iou = [], []
    from collections import defaultdict
    per_class_dice = defaultdict(list)
    per_class_iou  = defaultdict(list)
    per_file_results = []
    class_pixel_stats = defaultdict(lambda: {'gt_pixels': 0, 'pred_pixels': 0,
                                             'files_present_gt': 0, 'files_present_pred': 0})

    tasks = [(k, Path(gt_map[k]), Path(pr_map[k]), labels) for k in common_keys]

    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            workers = os.cpu_count() or 1

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one_file, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if verbose and (i % 50 == 0):
                print(f"Processed {i}/{len(tasks)} files")

            if not r.get('ok', False):
                if verbose:
                    print(f"Error processing {r['filename']}: {r.get('error')}")
                continue

            per_file_results.append({
                'filename': r['filename'],
                'mean_dice': r['mean_dice'],
                'mean_iou' : r['mean_iou'],
                'class_results': r['class_results']
            })
            all_dice.append(r['mean_dice']); all_iou.append(r['mean_iou'])

            for c, m in r['class_results'].items():
                per_class_dice[c].append(m['dice'])
                per_class_iou[c].append(m['iou'])

            for c, s in r['class_stats'].items():
                cs = class_pixel_stats[c]
                cs['gt_pixels']         += s['gt_pixels']
                cs['pred_pixels']       += s['pred_pixels']
                cs['files_present_gt']  += s['files_present_gt']
                cs['files_present_pred']+= s['files_present_pred']

    # reductions (unchanged)
    overall_mean_dice = float(np.mean(all_dice))
    overall_mean_iou  = float(np.mean(all_iou))
    overall_std_dice  = float(np.std(all_dice))
    overall_std_iou   = float(np.std(all_iou))

    per_class_results = {}
    for c in labels:
        if per_class_dice[c]:
            d = np.array(per_class_dice[c], dtype=float)
            i = np.array(per_class_iou[c] , dtype=float)
            per_class_results[c] = {
                'mean_dice': float(d.mean()), 'mean_iou': float(i.mean()),
                'std_dice' : float(d.std()) , 'std_iou' : float(i.std()),
                'class_name': CLASS_NAMES[c]
            }
        else:
            per_class_results[c] = {
                'mean_dice': 0.0, 'mean_iou': 0.0,
                'std_dice' : 0.0, 'std_iou' : 0.0,
                'class_name': CLASS_NAMES[c]
            }

    results = {
        'overall_metrics': {
            'mean_dice': overall_mean_dice, 'mean_iou': overall_mean_iou,
            'std_dice': overall_std_dice,   'std_iou': overall_std_iou,
            'num_files': len(per_file_results), 'num_classes': len(labels)
        },
        'per_class_metrics': per_class_results,
        'per_file_metrics': per_file_results,
        'class_statistics': dict(class_pixel_stats)
    }


    if save_results:
        save_results_to_csv(results, save_results, labels)
    return results


def list_images_recursive(directory: Path):
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for p in directory.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            files.append(str(p))
    return files


def evaluate_nested_cases_parallel(
    gt_base_dir,
    pred_base_dir,
    labels=None,
    save_csv=None,
    verbose=True,
    # parallel & discovery options passed down to the file-level evaluator
    workers=None,          # e.g., 40
    chunksize=32,
    gt_subdir=None,        # e.g., "masks" if each case has that folder
    pred_subdir=None,      # e.g., "imgs"  if each case has that folder
    match_by="stem",       # or "name" if you need exact filename matching
):
    """
    Parallel, recursive evaluator across multiple *case* subdirectories.

    Expected layout (flexible):
      gt_base_dir/
        case1/(... images possibly nested ...)
        case2/(... images possibly nested ...)
      pred_base_dir/
        case1/(... images possibly nested ...)
        case2/(... images possibly nested ...)

    If every case has fixed inner folders, pass gt_subdir="masks", pred_subdir="imgs".
    Otherwise leave them as None and recursive discovery will find images anywhere under each case.
    """
    gt_base_dir = Path(gt_base_dir)
    pred_base_dir = Path(pred_base_dir)

    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]

    # Discover case directories (one level under each base)
    gt_case_dirs = [d for d in gt_base_dir.iterdir() if d.is_dir()]
    pred_case_dirs = [d for d in pred_base_dir.iterdir() if d.is_dir()]
    gt_cases = {d.name for d in gt_case_dirs}
    pred_cases = {d.name for d in pred_case_dirs}
    common_cases = sorted(gt_cases & pred_cases)

    if not common_cases:
        raise RuntimeError(f"No common case directories found between {gt_base_dir} and {pred_base_dir}")

    if verbose:
        print(f"Found {len(common_cases)} common cases: {common_cases}")

    # Aggregation containers (same as your original)
    all_dice_scores = []
    all_iou_scores = []
    per_class_dice = defaultdict(list)
    per_class_iou = defaultdict(list)
    per_case_results = []
    per_file_results = []
    class_pixel_stats = defaultdict(
        lambda: {'gt_pixels': 0, 'pred_pixels': 0, 'files_present_gt': 0, 'files_present_pred': 0}
    )

    total_files_processed = 0

    # Process each case (sequentially at the case level; parallelism happens per-file inside)
    for case_name in common_cases:
        if verbose:
            print(f"--- Processing Case: {case_name} ---")

        gt_case_dir = gt_base_dir / case_name
        pred_case_dir = pred_base_dir / case_name

        # IMPORTANT:
        # We do NOT append "/masks" or "/imgs" here.
        # If you have those subfolders inside every case, pass gt_subdir="masks", pred_subdir="imgs".
        try:
            case_results = compute_comprehensive_metrics_parallel(
                gt_dir=gt_case_dir,
                pred_dir=pred_case_dir,
                labels=labels,
                verbose=False,     # keep per-case quiet; we print a summary below
                save_results=None,
                workers=workers,   # e.g. 40
                chunksize=chunksize,
                gt_subdir=gt_subdir,
                pred_subdir=pred_subdir,
                match_by=match_by
            )

        except RuntimeError as e:
        # 👇 if no files found, just warn and skip
            if verbose:
                print(f"Skipping case {case_name}: {e}")
            continue

        # Aggregate case summary
        case_files = len(case_results['per_file_metrics'])
        total_files_processed += case_files

        if verbose:
            print(f"  Files processed: {case_files}")
            print(f"  Case Mean Dice: {case_results['overall_metrics']['mean_dice']:.4f}")
            print(f" Case Mean IoU:  {case_results['overall_metrics']['mean_iou']:.4f}")


        
        per_case_results.append({
            'case_name': case_name,
            'num_files': case_files,
            'mean_dice': case_results['overall_metrics']['mean_dice'],
            'mean_iou': case_results['overall_metrics']['mean_iou'],
            'std_dice': case_results['overall_metrics']['std_dice'],
            'std_iou': case_results['overall_metrics']['std_iou']
        })

        # Aggregate per-file
        for file_result in case_results['per_file_metrics']:
            file_result['case_name'] = case_name
            per_file_results.append(file_result)
            all_dice_scores.append(file_result['mean_dice'])
            all_iou_scores.append(file_result['mean_iou'])

        # Aggregate per-class
        for class_label in labels:
            if class_label in case_results['per_class_metrics']:
                for file_result in case_results['per_file_metrics']:
                    if class_label in file_result['class_results']:
                        per_class_dice[class_label].append(file_result['class_results'][class_label]['dice'])
                        per_class_iou[class_label].append(file_result['class_results'][class_label]['iou'])

        # Aggregate pixel stats
        for class_label in labels:
            if class_label in case_results['class_statistics']:
                class_pixel_stats[class_label]['gt_pixels']        += case_results['class_statistics'][class_label]['gt_pixels']
                class_pixel_stats[class_label]['pred_pixels']      += case_results['class_statistics'][class_label]['pred_pixels']
                class_pixel_stats[class_label]['files_present_gt'] += case_results['class_statistics'][class_label]['files_present_gt']
                class_pixel_stats[class_label]['files_present_pred'] += case_results['class_statistics'][class_label]['files_present_pred']

    # Final reductions (same as your original)
    overall_mean_dice = float(np.mean(all_dice_scores)) if all_dice_scores else 0.0
    overall_mean_iou  = float(np.mean(all_iou_scores))  if all_iou_scores  else 0.0
    overall_std_dice  = float(np.std(all_dice_scores))  if all_dice_scores else 0.0
    overall_std_iou   = float(np.std(all_iou_scores))   if all_iou_scores  else 0.0

    per_class_results = {}
    for class_label in labels:
        if per_class_dice[class_label]:
            d = np.array(per_class_dice[class_label], dtype=float)
            i = np.array(per_class_iou[class_label] , dtype=float)
            per_class_results[class_label] = {
                'mean_dice': float(d.mean()),
                'mean_iou' : float(i.mean()),
                'std_dice' : float(d.std()),
                'std_iou'  : float(i.std()),
                'class_name': CLASS_NAMES[class_label]
            }
        else:
            per_class_results[class_label] = {
                'mean_dice': 0.0,
                'mean_iou' : 0.0,
                'std_dice' : 0.0,
                'std_iou'  : 0.0,
                'class_name': CLASS_NAMES[class_label]
            }

    results = {
        'overall_metrics': {
            'mean_dice': overall_mean_dice,
            'mean_iou' : overall_mean_iou,
            'std_dice' : overall_std_dice,
            'std_iou'  : overall_std_iou,
            'num_files': total_files_processed,
            'num_cases': len(common_cases),
            'num_classes': len(labels)
        },
        'per_class_metrics': per_class_results,
        'per_case_metrics': per_case_results,
        'per_file_metrics': per_file_results,
        'class_statistics': dict(class_pixel_stats)
    }


    if save_csv:
        save_nested_results_to_csv(results, save_csv, labels)

    return results




# USAGE EXAMPLES:
if __name__ == "__main__":
    
#     # OPTION 1: Nested Case Structure (NEW - for your use case)
#     # Directory structure:
    fold=0

    # gt_base_dir = "/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test/"      # Contains case1/, case2/, etc.
    gt_base_dir='/gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/test/imgs/'
    # pred_base_dir = "/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_Inference/fold{fold}/"      # Contains case1/, case2/, etc.
    pred_base_dir="/gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/Inference_test/imgs/"


    results = evaluate_nested_cases_parallel(
    gt_base_dir=gt_base_dir,
    pred_base_dir=pred_base_dir,
    # save_csv=f"./oai_Inference_fold{fold}.csv",
    save_csv=f"./IWOAI_Inference.csv",

    labels=[1,2,3,4],
    gt_subdir="masks",
    pred_subdir="imgs",
    workers=20,
    chunksize=32,
    verbose=True
)


    # Access results
    print(f"Overall Mean Dice across all cases: {results['overall_metrics']['mean_dice']:.4f}")
    print(f"Total files processed: {results['overall_metrics']['num_files']}")
    print(f"Total cases processed: {results['overall_metrics']['num_cases']}")
    


