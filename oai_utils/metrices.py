
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
    5: (255, 0, 255),   # class 5 - magenta
    6: (0, 255, 255),   # class 6 - cyan
    7: (128, 128, 128), # class 7 - gray
}

# Class names for better reporting
CLASS_NAMES = {
    0: "Background",
    1: "Femur", 
    2: "Tibia",
    3: "Patella",
    4: "Femoral Cartilage",
    5: "Tibial Cartilage", 
    6: "Patellar Cartilage",
    7: "Meniscus"
}



def remap_with_lookup(mask_img):
    """Convert grayscale mask to class IDs using lookup table"""
    arr = np.array(mask_img, dtype=np.int16)
    lut = np.zeros(256, dtype=np.uint8)
    for gray, cls in GRAY_TO_CLASS.items():
        lut[gray] = cls
    return lut[arr]

def dice_iou(gt_bool, pred_bool):
    """Calculate Dice and IoU scores for boolean masks"""
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    gt_sum = gt_bool.sum()
    pred_sum = pred_bool.sum()
    
    # Handle edge cases
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0  # Perfect match when both are empty
    if union == 0:
        return 0.0, 0.0  # No union means no overlap
    
    dice = 2 * intersection / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    
    return dice, iou

def list_images(directory):
    """List image files in directory"""
    extensions = ['.png', '.jpg', '.jpeg']
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return [str(f) for f in files]

def compute_comprehensive_metrics(gt_dir, pred_dir, labels=None, verbose=True, save_results=None):
    """
    Compute comprehensive evaluation metrics for all files in directories.
    
    Parameters:
    -----------
    gt_dir : str or Path
        Directory containing ground truth masks
    pred_dir : str or Path  
        Directory containing predicted masks
    labels : list of int, optional
        List of class labels to evaluate. If None, evaluates all classes 1-7
    verbose : bool
        Whether to print detailed progress and results
    save_results : str or Path, optional
        Path to save detailed results CSV file
        
    Returns:
    --------
    dict : Comprehensive results dictionary containing:
        - overall_metrics: Mean Dice/IoU across all files and classes
        - per_class_metrics: Mean Dice/IoU per class
        - per_file_metrics: Dice/IoU for each file
        - class_statistics: Pixel counts and presence statistics per class
    # """
    
    gt_dir = Path(f"{gt_dir}/masks")
    pred_dir = Path(f"{pred_dir}/imgs")


    
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]  # All non-background classes
    
    # Match files between GT and prediction directories
    gt_files = list_images(gt_dir)
    pred_files = list_images(pred_dir)
    
    gt_map = {Path(f).stem: f for f in gt_files}
    pred_map = {Path(f).stem: f for f in pred_files}
    common_files = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    
    if not common_files:
        raise RuntimeError("No matching files found between GT and prediction directories")
    
    if verbose:
        print(f"Found {len(common_files)} matching files")
        print(f"Evaluating classes: {labels}")
        print(f"Class names: {[CLASS_NAMES[l] for l in labels]}")
    
    # Initialize result storage
    all_dice_scores = []
    all_iou_scores = []
    per_class_dice = defaultdict(list)
    per_class_iou = defaultdict(list)
    per_file_results = []
    class_pixel_stats = defaultdict(lambda: {'gt_pixels': 0, 'pred_pixels': 0, 'files_present_gt': 0, 'files_present_pred': 0})
    
    # Process each file
    for i, filename in enumerate(common_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"Processing file {i+1}/{len(common_files)}: {filename}")
            
        gt_file = gt_dir / gt_map[filename]
        pred_file = pred_dir / pred_map[filename]
        
        # Load and convert masks
        try:
            gt_mask_img = Image.open(gt_file).convert("L")
            pred_mask_img = Image.open(pred_file).convert("L")
            
            gt_ids = remap_with_lookup(gt_mask_img)
            pred_ids = remap_with_lookup(pred_mask_img)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {filename}: {e}")
            continue
        
        # File-level metrics storage
        file_dice_scores = []
        file_iou_scores = []
        file_class_results = {}
        
        # Process each class
        for class_label in labels:
            gt_bool = (gt_ids == class_label)
            pred_bool = (pred_ids == class_label)
            
            # Update pixel statistics
            class_pixel_stats[class_label]['gt_pixels'] += gt_bool.sum()
            class_pixel_stats[class_label]['pred_pixels'] += pred_bool.sum()
            if gt_bool.any():
                class_pixel_stats[class_label]['files_present_gt'] += 1
            if pred_bool.any():
                class_pixel_stats[class_label]['files_present_pred'] += 1
            
            # Calculate metrics
            dice, iou = dice_iou(gt_bool, pred_bool)
            
            # Store results
            per_class_dice[class_label].append(dice)
            per_class_iou[class_label].append(iou)
            file_dice_scores.append(dice)
            file_iou_scores.append(iou)
            file_class_results[class_label] = {'dice': dice, 'iou': iou}
        
        # Store file-level results
        file_mean_dice = np.mean(file_dice_scores)
        file_mean_iou = np.mean(file_iou_scores)
        all_dice_scores.append(file_mean_dice)
        all_iou_scores.append(file_mean_iou)
        
        per_file_results.append({
            'filename': filename,
            'mean_dice': file_mean_dice,
            'mean_iou': file_mean_iou,
            'class_results': file_class_results
        })
    
    # Calculate final metrics
    overall_mean_dice = np.mean(all_dice_scores)
    overall_mean_iou = np.mean(all_iou_scores)
    overall_std_dice = np.std(all_dice_scores)
    overall_std_iou = np.std(all_iou_scores)
    
    # Per-class metrics
    per_class_results = {}
    for class_label in labels:
        if per_class_dice[class_label]:
            class_mean_dice = np.mean(per_class_dice[class_label])
            class_mean_iou = np.mean(per_class_iou[class_label])
            class_std_dice = np.std(per_class_dice[class_label])
            class_std_iou = np.std(per_class_iou[class_label])
        else:
            class_mean_dice = class_mean_iou = class_std_dice = class_std_iou = 0.0
            
        per_class_results[class_label] = {
            'mean_dice': class_mean_dice,
            'mean_iou': class_mean_iou,
            'std_dice': class_std_dice,
            'std_iou': class_std_iou,
            'class_name': CLASS_NAMES[class_label]
        }
    
    # Compile final results
    results = {
        'overall_metrics': {
            'mean_dice': overall_mean_dice,
            'mean_iou': overall_mean_iou,
            'std_dice': overall_std_dice,
            'std_iou': overall_std_iou,
            'num_files': len(common_files),
            'num_classes': len(labels)
        },
        'per_class_metrics': per_class_results,
        'per_file_metrics': per_file_results,
        'class_statistics': dict(class_pixel_stats)
    }
    
    # Print results
    if verbose:
        print_comprehensive_results(results, labels)
    
    # Save results to CSV if requested
    if save_results:
        save_results_to_csv(results, save_results, labels)
    
    return results

def print_comprehensive_results(results, labels):
    """Print comprehensive results in a formatted way"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    overall = results['overall_metrics']
    print(f"\nOVERALL METRICS ({overall['num_files']} files, {overall['num_classes']} classes):")
    print(f"Mean Dice: {overall['mean_dice']:.4f} ± {overall['std_dice']:.4f}")
    print(f"Mean IoU:  {overall['mean_iou']:.4f} ± {overall['std_iou']:.4f}")
    
    # Per-class metrics
    print(f"\nPER-CLASS METRICS:")
    print(f"{'Class':<20} {'Dice':<12} {'IoU':<12} {'GT Files':<10} {'Pred Files':<10}")
    print("-" * 70)
    
    for class_label in labels:
        class_result = results['per_class_metrics'][class_label]
        class_stats = results['class_statistics'][class_label]
        
        print(f"{class_result['class_name']:<20} "
              f"{class_result['mean_dice']:.4f}±{class_result['std_dice']:.3f}  "
              f"{class_result['mean_iou']:.4f}±{class_result['std_iou']:.3f}  "
              f"{class_stats['files_present_gt']:<10} "
              f"{class_stats['files_present_pred']:<10}")
    
    # Class statistics
    print(f"\nCLASS PIXEL STATISTICS:")
    print(f"{'Class':<20} {'GT Pixels':<12} {'Pred Pixels':<12} {'Ratio':<10}")
    print("-" * 60)
    
    for class_label in labels:
        class_stats = results['class_statistics'][class_label]
        class_name = results['per_class_metrics'][class_label]['class_name']
        ratio = class_stats['pred_pixels'] / max(class_stats['gt_pixels'], 1)
        
        print(f"{class_name:<20} "
              f"{class_stats['gt_pixels']:<12} "
              f"{class_stats['pred_pixels']:<12} "
              f"{ratio:.3f}")

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

# Example usage function
def evaluate_nested_cases(gt_base_dir, pred_base_dir, labels=None, save_csv=None, verbose=True):
    """
    Evaluate segmentation results across multiple nested case directories
    
    Directory structure expected:
    gt_base_dir/
        case1/
            1.png, 2.png, ...
        case2/
            1.png, 2.png, ...
    pred_base_dir/
        case1/
            1.png, 2.png, ...
        case2/
            1.png, 2.png, ...
    
    Parameters:
    -----------
    gt_base_dir : str or Path
        Base directory containing case subdirectories with GT masks
    pred_base_dir : str or Path
        Base directory containing case subdirectories with predicted masks
    labels : list, optional
        Classes to evaluate. Default: [1,2,3,4,5,6,7]
    save_csv : str, optional
        Path to save CSV results
    verbose : bool
        Whether to print detailed progress
    
    Returns:
    --------
    dict : Comprehensive evaluation results across all cases
    """
    
    gt_base_dir = Path(gt_base_dir)
    pred_base_dir = Path(pred_base_dir)
    
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]
    
    # Find all case directories
    gt_case_dirs = [d for d in gt_base_dir.iterdir() if d.is_dir()]
    pred_case_dirs = [d for d in pred_base_dir.iterdir() if d.is_dir()]
    
    gt_cases = {d.name for d in gt_case_dirs}
    pred_cases = {d.name for d in pred_case_dirs}
    common_cases = sorted(gt_cases & pred_cases)
    
    if not common_cases:
        raise RuntimeError(f"No common case directories found between {gt_base_dir} and {pred_base_dir}")
    
    if verbose:
        print(f"Found {len(common_cases)} common cases: {common_cases}")
    
    # Initialize aggregated results
    all_dice_scores = []
    all_iou_scores = []
    per_class_dice = defaultdict(list)
    per_class_iou = defaultdict(list)
    per_case_results = []
    per_file_results = []
    class_pixel_stats = defaultdict(lambda: {'gt_pixels': 0, 'pred_pixels': 0, 'files_present_gt': 0, 'files_present_pred': 0})
    
    total_files_processed = 0
    
    # Process each case
    for case_name in common_cases:
        if verbose:
            print(f"\n--- Processing Case: {case_name} ---")
        
        gt_case_dir = gt_base_dir / case_name 
        pred_case_dir = pred_base_dir / case_name
        
        # Get case-specific results
        case_results = compute_comprehensive_metrics(
            gt_dir=gt_case_dir,
            pred_dir=pred_case_dir,
            labels=labels,
            verbose=False,  # Suppress per-case verbose output
            save_results=None
        )
        
        # Aggregate case results
        case_files = len(case_results['per_file_metrics'])
        total_files_processed += case_files
        
        if verbose:
            print(f"  Files processed: {case_files}")
            print(f"  Case Mean Dice: {case_results['overall_metrics']['mean_dice']:.4f}")
            print(f"  Case Mean IoU: {case_results['overall_metrics']['mean_iou']:.4f}")
        
        # Store per-case summary
        per_case_results.append({
            'case_name': case_name,
            'num_files': case_files,
            'mean_dice': case_results['overall_metrics']['mean_dice'],
            'mean_iou': case_results['overall_metrics']['mean_iou'],
            'std_dice': case_results['overall_metrics']['std_dice'],
            'std_iou': case_results['overall_metrics']['std_iou']
        })
        
        # Aggregate all file-level results
        for file_result in case_results['per_file_metrics']:
            file_result['case_name'] = case_name  # Add case info
            per_file_results.append(file_result)
            all_dice_scores.append(file_result['mean_dice'])
            all_iou_scores.append(file_result['mean_iou'])
        
        # Aggregate per-class results
        for class_label in labels:
            if class_label in case_results['per_class_metrics']:
                # Get all individual scores for this class from this case
                case_class_dice = []
                case_class_iou = []
                for file_result in case_results['per_file_metrics']:
                    if class_label in file_result['class_results']:
                        case_class_dice.append(file_result['class_results'][class_label]['dice'])
                        case_class_iou.append(file_result['class_results'][class_label]['iou'])
                
                per_class_dice[class_label].extend(case_class_dice)
                per_class_iou[class_label].extend(case_class_iou)
        
        # Aggregate pixel statistics
        for class_label in labels:
            if class_label in case_results['class_statistics']:
                class_pixel_stats[class_label]['gt_pixels'] += case_results['class_statistics'][class_label]['gt_pixels']
                class_pixel_stats[class_label]['pred_pixels'] += case_results['class_statistics'][class_label]['pred_pixels']
                class_pixel_stats[class_label]['files_present_gt'] += case_results['class_statistics'][class_label]['files_present_gt']
                class_pixel_stats[class_label]['files_present_pred'] += case_results['class_statistics'][class_label]['files_present_pred']
    
    # Calculate overall metrics across all cases
    overall_mean_dice = np.mean(all_dice_scores)
    overall_mean_iou = np.mean(all_iou_scores)
    overall_std_dice = np.std(all_dice_scores)
    overall_std_iou = np.std(all_iou_scores)
    
    # Calculate per-class metrics across all cases
    per_class_results = {}
    for class_label in labels:
        if per_class_dice[class_label]:
            class_mean_dice = np.mean(per_class_dice[class_label])
            class_mean_iou = np.mean(per_class_iou[class_label])
            class_std_dice = np.std(per_class_dice[class_label])
            class_std_iou = np.std(per_class_iou[class_label])
        else:
            class_mean_dice = class_mean_iou = class_std_dice = class_std_iou = 0.0
            
        per_class_results[class_label] = {
            'mean_dice': class_mean_dice,
            'mean_iou': class_mean_iou,
            'std_dice': class_std_dice,
            'std_iou': class_std_iou,
            'class_name': CLASS_NAMES[class_label]
        }
    
    # Compile final results
    results = {
        'overall_metrics': {
            'mean_dice': overall_mean_dice,
            'mean_iou': overall_mean_iou,
            'std_dice': overall_std_dice,
            'std_iou': overall_std_iou,
            'num_files': total_files_processed,
            'num_cases': len(common_cases),
            'num_classes': len(labels)
        },
        'per_class_metrics': per_class_results,
        'per_case_metrics': per_case_results,
        'per_file_metrics': per_file_results,
        'class_statistics': dict(class_pixel_stats)
    }
    
    # Print results
    if verbose:
        print_nested_results(results, labels)
    
    # Save results to CSV if requested
    if save_csv:
        save_nested_results_to_csv(results, save_csv, labels)
    
    return results

def print_nested_results(results, labels):
    """Print comprehensive results for nested case structure"""
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS - ALL CASES")
    print("="*100)
    
    # Overall metrics
    overall = results['overall_metrics']
    print(f"\nOVERALL METRICS ({overall['num_cases']} cases, {overall['num_files']} files, {overall['num_classes']} classes):")
    print(f"Mean Dice: {overall['mean_dice']:.4f} ± {overall['std_dice']:.4f}")
    print(f"Mean IoU:  {overall['mean_iou']:.4f} ± {overall['std_iou']:.4f}")
    
    # Per-case summary
    print(f"\nPER-CASE SUMMARY:")
    print(f"{'Case':<15} {'Files':<8} {'Mean Dice':<12} {'Mean IoU':<12}")
    print("-" * 55)
    
    for case_result in results['per_case_metrics']:
        print(f"{case_result['case_name']:<15} "
              f"{case_result['num_files']:<8} "
              f"{case_result['mean_dice']:.4f}±{case_result['std_dice']:.3f}  "
              f"{case_result['mean_iou']:.4f}±{case_result['std_iou']:.3f}")
    
    # Per-class metrics (same as before)
    print(f"\nPER-CLASS METRICS (ACROSS ALL CASES):")
    print(f"{'Class':<20} {'Dice':<12} {'IoU':<12} {'GT Files':<10} {'Pred Files':<10}")
    print("-" * 70)
    
    for class_label in labels:
        class_result = results['per_class_metrics'][class_label]
        class_stats = results['class_statistics'][class_label]
        
        print(f"{class_result['class_name']:<20} "
              f"{class_result['mean_dice']:.4f}±{class_result['std_dice']:.3f}  "
              f"{class_result['mean_iou']:.4f}±{class_result['std_iou']:.3f}  "
              f"{class_stats['files_present_gt']:<10} "
              f"{class_stats['files_present_pred']:<10}")

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

def evaluate_segmentation_results(gt_base_dir, pred_base_dir, labels=None, save_csv=None):
    """
    Legacy function - for single directory structure
    """
    
    gt_dir = Path(f"{gt_base_dir}")
    pred_dir = Path(f"{pred_base_dir}")
    
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]
    
    print(f"GT Directory: {gt_dir}")
    print(f"Prediction Directory: {pred_dir}")
    
    results = compute_comprehensive_metrics(
        gt_dir=gt_dir,
        pred_dir=pred_dir, 
        labels=labels,
        verbose=True,
        save_results=save_csv
    )
    
    return results


# USAGE EXAMPLES:
if __name__ == "__main__":
    
#     # OPTION 1: Nested Case Structure (NEW - for your use case)
#     # Directory structure:
#     # gt_base_dir/case1/1.png, gt_base_dir/case2/1.png, etc.
#     # pred_base_dir/case1/1.png, pred_base_dir/case2/1.png, etc.
    
    gt_base_dir = "/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test/"      # Contains case1/, case2/, etc.
    pred_base_dir = "/gpfs/home/machlm03/Segmentation/OAI_demo/Inference/"      # Contains case1/, case2/, etc.
    
    # # Evaluate across ALL cases and get overall statistics
    results = evaluate_nested_cases(
        gt_base_dir=gt_base_dir,
        pred_base_dir=pred_base_dir,
        labels=[1, 2, 3, 4, 5, 6, 7],  # All classes
        save_csv="all_cases_evaluation_results.csv",
        verbose=True
    )
    
    # Access results
    print(f"Overall Mean Dice across all cases: {results['overall_metrics']['mean_dice']:.4f}")
    print(f"Total files processed: {results['overall_metrics']['num_files']}")
    print(f"Total cases processed: {results['overall_metrics']['num_cases']}")
    
    # # #OPTION 2: Single Directory Structure (LEGACY - for flat structure)

    # # /gpfs/home/machlm03/Segmentation/OAI_demo/Inference/9002411_00m_LEFT_SAG_3D_DESS_WE/imgs/000.png
    # results = evaluate_segmentation_results(

    #     gt_base_dir="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test/9003380_00m_LEFT_SAG_3D_DESS_WE/",
    #     pred_base_dir="/gpfs/home/machlm03/Segmentation/OAI_demo/Inference/9003380_00m_LEFT_SAG_3D_DESS_WE/",
    #     labels=[1, 2, 3, 4, 5, 6, 7],
    #     save_csv="single_dir_results.csv"
    # )
    
    # OPTION 3: Specific classes only
    # results = evaluate_nested_cases(
    #     gt_base_dir="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test/9000099_00m_LEFT_SAG_3D_DESS_WE/",
    #     pred_base_dir="/gpfs/home/machlm03/Segmentation/OAI_demo/Inference/9000099_00m_LEFT_SAG_3D_DESS_WE/",
    #     labels=[1, 2, 4],  # Only Femur, Tibia, Femoral Cartilage
    #     save_csv="specific_classes_results.csv"
    # )

