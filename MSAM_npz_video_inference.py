from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
import SimpleITK as sitk
from skimage import measure, morphology
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import torch
from os.path import join

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)




def get_percentile_indices(total_frames, percentiles=(25, 50, 75)):
    """Get frame indices at specified percentiles"""
    indices = []
    for p in percentiles:
        idx = int(np.round((p / 100.0) * (total_frames - 1)))
        indices.append(idx)
    return sorted(indices)

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def get_percentile_slices_from_npz(npz_path, method, percentiles=None, class_ids=None):
    data = np.load(npz_path)
    gt = data['gts']  # shape: [D, H, W]
    selected_slices = []
    indexs = []
    
    # Set default class_ids if not provided - moved to top
    if class_ids is None:
        class_ids = [c for c in np.unique(gt) if c > 0]  # Exclude background
    
    if method == 'percentile':
        if percentiles is None:
            raise ValueError("percentiles must be provided for 'percentile' method")
            
        slice_foreground_counts = np.sum(gt > 0, axis=(1, 2))
        valid_slices = np.where(slice_foreground_counts > 0)[0]
    
        if len(valid_slices) == 0:
            raise ValueError(f"No foreground found in {npz_path}")
    
        for p in percentiles:
            # Fix: use percentile of indices, not values
            percentile_idx = int(len(valid_slices) * p / 100)
            percentile_idx = min(percentile_idx, len(valid_slices) - 1)  # Ensure within bounds
            idx = valid_slices[percentile_idx]
            selected_slices.append(gt[idx])
            indexs.append(idx)
        print(f"Selected slice {idx}")

            
    elif method == 'max_classes':
        # Select slice with maximum number of different classes
        D = gt.shape[0]
        print(f"Total slices: {D}")
        slice_scores = []
        for i in range(D):
            slice_data = gt[i]
            # Count number of different classes present
            present_classes = len([c for c in class_ids if np.sum(slice_data == c) > 0])
            slice_scores.append(present_classes)
        
        # key_slice_idx = np.argmax(slice_scores)+5

        max_val = max(slice_scores)
        max_indices = [i for i, val in enumerate(slice_scores) if val == max_val]
        key_slice_idx = max_indices[len(max_indices) // 2]



        max_classes = slice_scores[key_slice_idx]
        selected_slices.append(gt[key_slice_idx])
        indexs.append(key_slice_idx)
        print(f"Selected slice {key_slice_idx} with {max_classes} classes")
        
    return selected_slices, indexs

def get_object_ids_from_mask(mask):
    """Extract unique object IDs and their masks"""
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]  # Remove background
    
    per_obj_mask = {}
    for obj_id in unique_ids:
        per_obj_mask[int(obj_id)] = (mask == obj_id).astype(np.uint8)
    
    return per_obj_mask


def infer(test_files,imgs_path,model_cfg,checkpoint,method,percentiles,propagate_with_box,pred_save_dir):
    with open(test_files, "r") as f:
        allowed_cases = {line.strip() for line in f if line.strip()}
        allowed_cases = {case + ".npz" for case in allowed_cases}
    npz_fnames = sorted(os.listdir(imgs_path))
    npz_fnames = [i for i in npz_fnames if i.endswith('.npz')]
    npz_fnames = [i for i in npz_fnames if not i.startswith('._')]
    npz_fnames = [i for i in npz_fnames if i in allowed_cases]
    
    print(f'Processing {len(npz_fnames)} nii files')
    
    # initialized predictor
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    all_prompt_slices = []
    # percentiles = [33]
    
    # npz_files = ["9000099_00m_LEFT_SAG_3D_DESS_WE.npz"]
    
    for npz_file in tqdm(npz_fnames):
        # Load data
        data = np.load(join(imgs_path, npz_file))
        nii_image_data = data['imgs']  # shape: [D, H, W]
        gt_data = data['gts']          # shape: [D, H, W]
    
        video_height, video_width = nii_image_data[0].shape
        # Initialize multi-class segmentation array
        segs_3D = np.zeros_like(nii_image_data, dtype=np.uint8)
        img_3D_ori = nii_image_data

        slices, indices = get_percentile_slices_from_npz(join(imgs_path, npz_file), method,percentiles=percentiles,class_ids=None)

        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
        img_resized -= img_mean
        img_resized /= img_std
        
        start_time = time.time()


        inference_state = predictor.init_state(
            img_resized,video_height, video_width
        )

        initial_prompts = get_percentile_indices(nii_image_data.shape[0], percentiles)

        # print(object_ids_set)
        # predictor.reset_state(inference_state)
        for frame_idx in initial_prompts:
            object_ids_set = None
            input_frame_idx = frame_idx
            # print(f"Adding mask from frame {frame_idx}")
            
            gt_mask = gt_data[frame_idx]  # (384, 384)
            per_obj_input_mask = get_object_ids_from_mask(gt_mask)
            
            if len(per_obj_input_mask) == 0:
                print(f"  Warning: No objects in frame {frame_idx}")
                continue
            
            # Initialize object_ids_set from first frame
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask.keys())
            # print(f"  Initialized tracking with objects: {object_ids_set}")
            
            # Iterate over object_id and object_mask
            for object_id, object_mask in per_obj_input_mask.items():

                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
                
            # print(f"  Added object {object_id}")
            # print(f"\nTracking {len(object_ids_set)} objects: {sorted(object_ids_set)}")

                

        # Check we have objects to track
        if object_ids_set is None or len(object_ids_set) == 0:
            raise RuntimeError("No objects found in the selected frames!")
        

        img_3D_ori = nii_image_data
        predicted_masks_3d = np.zeros_like(img_3D_ori, dtype=np.uint8)

        video_segments = {}  # Store the per-frame segmentation results


        for reverse_flag in [True, False]:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=initial_prompts[len(initial_prompts)//2],
                reverse=reverse_flag,
            ):
                # Build per-object masks
                per_obj_output_mask = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

                # Merge results efficiently
                if out_frame_idx in video_segments:
                    video_segments[out_frame_idx].update(per_obj_output_mask)
                else:
                    video_segments[out_frame_idx] = per_obj_output_mask




        # Convert dictionary to 3D numpy array
        predicted_masks_3d = np.zeros((nii_image_data.shape[0],video_height, video_width), dtype=np.uint8)
        for out_frame_idx, per_obj_output_mask in video_segments.items():
            frame_mask = np.zeros((video_height, video_width), dtype=np.uint8)
            
            for obj_id, obj_mask in per_obj_output_mask.items():
                # Squeeze and handle dimension mismatch
                obj_mask_squeezed = obj_mask.squeeze()
                
                frame_mask[obj_mask_squeezed] = obj_id
            
            predicted_masks_3d[out_frame_idx] = frame_mask


        # Save as NPZ
        os.makedirs(pred_save_dir, exist_ok=True)

        # npz_file = "inference_results.npz"

        np.savez_compressed(
            os.path.join(pred_save_dir, npz_file),
            imgs=img_3D_ori.astype(np.uint8),
            pre=predicted_masks_3d.astype(np.uint8),  # Changed from video_segments
            gts=gt_data.astype(np.uint8)
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f'Finished {npz_file} in {duration:.2f} seconds')
        print(f'Final predicted classes: {np.unique(predicted_masks_3d)}')

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        type=str,
        default="checkpoints/MedSAM2_latest.pt",
        help='checkpoint path',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help='model config',
    )

    parser.add_argument(
        '--imgs_path',
        type=str,
        default="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_MultiClass/npz/",
        help='imgs path',
    )
    parser.add_argument(
        '--gts_path',
        type=str,
        default="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_MultiClass/npz/",
        help='simulate prompts based on ground truth',
    )



    parser.add_argument(
        '--test_files',
        type=str,
        default="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/cv_txt_files/test/test_split.txt",
        help='whether to propagate with box'
    )




    parser.add_argument(
        '--pred_save_dir',
        type=str,
        default="/gpfs/home/machlm03/Segmentation/OAI_demo/MedSam2_OAI_Inference_bbx_test/",
        help='path to save segmentation results',
    )
    # add option to propagate with either box or mask
    parser.add_argument(
        '--propagate_with_box',
        default=True,
        action='store_true',
        help='whether to propagate with box'
    )




    parser.add_argument(
        '--method',
        type=str,
        choices=["percentiles", "max_classes"],  # restrict valid options
        default="max_classes",
        help='select method from percentile or max_classes',
    )

    parser.add_argument(
        "--percentile",
        type=int,
        nargs="+",
        default=[25,50,75],
        help="Percentiles to select prompt masks from (e.g., 25 50 75)"
    )


    args = parser.parse_args()
    # args, _ = parser.parse_known_args()

    args.checkpoint="/gpfs/home/machlm03/Segmentation/MedSAM2/MSAM2.4/checkpoints/checkpoint.pt"
    # args.checkpoint= "/gpfs/home/machlm03/Segmentation/MedSAM2/MSAM_IWOAI/checkpoints/checkpoint_130.pt"
    args.imgs_path = "/gpfs/home/machlm03/Segmentation/IWOAI_demo/data/NPZ/test/"
    args.gts_path = "/gpfs/home/machlm03/Segmentation/IWOAI_demo/data/NPZ/test/"
    args.pred_save_dir = '/gpfs/home/machlm03/Segmentation/IWOAI_demo/OAI_on_IWOAI_npz_inference/npz/'
    args.test_files = "/gpfs/home/machlm03/Segmentation/IWOAI_demo/data/test.txt"



    # args.checkpoint="/gpfs/home/machlm03/Segmentation/MedSAM2/MSAM2.4/checkpoints/checkpoint.pt"
    # # args.checkpoint= "/gpfs/home/machlm03/Segmentation/MedSAM2/MSAM_IWOAI/checkpoints/checkpoint_130.pt"
    # args.imgs_path = "/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest_Sample/OAI_TrainTest/V00_00m_MultiClass/npz/"
    # args.gts_path = "/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest_Sample/OAI_TrainTest/V00_00m_MultiClass/npz/"
    # args.pred_save_dir ='/gpfs/home/machlm03/Segmentation/OAI_ZIB/MSAM_OAI/npz/'
    # args.test_files = "/gpfs/home/machlm03/Segmentation/OAI_ZIB/test.txt"

    checkpoint = args.checkpoint
    model_cfg = args.cfg
    imgs_path = args.imgs_path
    gts_path = args.gts_path
    pred_save_dir = args.pred_save_dir
    method = args.method
    percentile = args.percentile
    test_files = args.test_files
    propagate_with_box = args.propagate_with_box

    if args.method == "percentiles":
        print("Using percentiles:", args.percentile)
    elif args.method == "max_classes":
        print("Using max_class by scanning the slices with max labels, and selecting it as keyslice")


    # args.checkpoint = "/gpfs/home/machlm03/Segmentation/MedSAM2/checkpoints/"
                        
    print(checkpoint,test_files,imgs_path)
    infer(test_files,imgs_path,model_cfg,checkpoint,method,percentile,propagate_with_box,pred_save_dir)


import sys
if __name__ == "__main__":
    # main()
    sys.argv = [
  
    " MSAM_npz_video_inference.py ",
    "--propagate_with_box", ### uncomment it to inlcude the bbx 
    "--method", "percentiles",
    # "--percentile", "25","50","75",   
    

    ]
    main()

