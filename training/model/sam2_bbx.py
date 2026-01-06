# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


import logging

import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False



    def forward(self, input: BatchedVideoDatapoint):

        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes




    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Aggregate bounding boxes by FRAME + VIDEO, then reverse map
        """
        print("\n=== FRAME+VIDEO AGGREGATED BOUNDING BOX MODE ===")
        print(f"Batch shape - masks: {input.masks.shape}")
        print(f"Batch shape - obj_to_frame_idx: {input.obj_to_frame_idx.shape}")

        # Create individual bounding boxes first, then aggregate
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [O, 1, H, W]
            for stage_id, masks in enumerate(input.masks)
        }

        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames
        
        # Training parameters (keep existing)
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        
        if num_frames == 1:
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        
        assert num_init_cond_frames >= 1
        
        # Frame selection logic
        if rand_init_cond_frames and num_init_cond_frames > 1:
            num_init_cond_frames = self.rng.integers(1, num_init_cond_frames, endpoint=True)
        
        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        
        print(init_cond_frames)
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        
        # FORCE BOUNDING BOX INPUT ONLY WITH FRAME+VIDEO AGGREGATION
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}
        
        print("\n=== AGGREGATING BOUNDING BOXES BY (FRAME + VIDEO) ===")
        for t in init_cond_frames:
            # Step 1: Generate individual bounding boxes for all objects

            individual_points, individual_labels = sample_box_points(gt_masks_per_frame[t])
            
            print(f"\nStage {t} - Individual bounding boxes:")
            print(f"  points shape: {individual_points.shape}")  # [O, 2, 2] - O objects
            print(f"  labels shape: {individual_labels.shape}")  # [O, 2]
            
            # Step 2: Group objects by (FRAME_ID + VIDEO_ID) combination
            obj_to_frame_idx = input.obj_to_frame_idx[t]  # [O, 2]
            frame_video_objects = {}  # Group by (frame_id, video_id) tuple
            
            print(f"Stage {t} - Object mappings:")
            for obj_idx, (frame_idx, video_idx) in enumerate(obj_to_frame_idx):
                frame_idx = frame_idx.item()
                video_idx = video_idx.item()
                print(f"  Object {obj_idx}: frame={frame_idx}, video={video_idx}")
                
                # Create (frame_id, video_id) key
                frame_video_key = (frame_idx, video_idx)
                
                if frame_video_key not in frame_video_objects:
                    frame_video_objects[frame_video_key] = []
                frame_video_objects[frame_video_key].append(obj_idx)
            
            print(f"Stage {t} - Frame+Video grouping:")
            for (frame_idx, video_idx) in sorted(frame_video_objects.keys()):
                objects = frame_video_objects[(frame_idx, video_idx)]
                print(f"  Frame {frame_idx} + Video {video_idx}: objects {objects}")
            
            # Step 3: Aggregate bounding boxes for each (FRAME + VIDEO) combination
            aggregated_boxes_per_frame_video = {}
            
            for (frame_idx, video_idx) in sorted(frame_video_objects.keys()):
                object_indices = frame_video_objects[(frame_idx, video_idx)]
                
                if len(object_indices) > 0:
                    # Get all bounding boxes for this (frame, video) combination
                    frame_video_boxes = individual_points[object_indices]  # [num_objects_in_frame_video, 2, 2]
                    
                    # Extract all corners
                    all_top_lefts = frame_video_boxes[:, 0, :]     # [num_objects, 2] - all top-left corners
                    all_bottom_rights = frame_video_boxes[:, 1, :] # [num_objects, 2] - all bottom-right corners
                    
                    # Find the compound bounding box (min of top-lefts, max of bottom-rights)
                    compound_top_left = torch.min(all_top_lefts, dim=0)[0]     # [2] - min x,y of top-left
                    compound_bottom_right = torch.max(all_bottom_rights, dim=0)[0]  # [2] - max x,y of bottom-right
                    
                    # Create compound bounding box
                    compound_box = torch.stack([compound_top_left, compound_bottom_right])  # [2, 2]
                    aggregated_boxes_per_frame_video[(frame_idx, video_idx)] = compound_box
                    
                    # Debug info
                    width = compound_bottom_right[0] - compound_top_left[0]
                    height = compound_bottom_right[1] - compound_top_left[1]
                    print(f"    Compound box: ({compound_top_left[0]:.0f},{compound_top_left[1]:.0f}) → ({compound_bottom_right[0]:.0f},{compound_bottom_right[1]:.0f})")
                    print(f"      Size: {width:.0f} x {height:.0f} pixels (covers {len(object_indices)} objects)")
            
            print(f"Stage {t} - Created {len(aggregated_boxes_per_frame_video)} unique compoun bboxes")
            
            # Step 4: REVERSE MAPPING - Map back to original objects
            final_points = []
            final_labels = []
            
            for obj_idx, (frame_idx, video_idx) in enumerate(obj_to_frame_idx):
                frame_idx = frame_idx.item()
                video_idx = video_idx.item()
                
                # Get the compound box for this object's (frame_id, video_id)
                frame_video_key = (frame_idx, video_idx)
                compound_box = aggregated_boxes_per_frame_video[frame_video_key]  # [2, 2]
                
                # Keep original labels
                original_labels = individual_labels[obj_idx]  # [2] - corner labels [2, 3]
                
                final_points.append(compound_box)
                final_labels.append(original_labels)
            
            # Stack into final tensors
            final_points = torch.stack(final_points)  # [O, 2, 2] - same count as original
            final_labels = torch.stack(final_labels)  # [O, 2]
            
            print(f"\nStage {t} - Reverse mapping complete:")
            print(f"  final points shape: {final_points.shape}")  # [O, 2, 2]
            print(f"  final labels shape: {final_labels.shape}")   # [O, 2]
            
            # Verification: Check that objects from same (frame, video) have same compound box
            for (frame_idx, video_idx), object_indices in frame_video_objects.items():
                if len(object_indices) > 1:
                    first_box = final_points[object_indices[0]]
                    for obj_idx in object_indices[1:]:
                        assert torch.allclose(final_points[obj_idx], first_box), \
                            f"Objects in frame {frame_idx}, video {video_idx} don't have same compound box!"
            
            print(f"  ✓ Verification passed: All objects from same (frame, video) have identical compound boxes")
            
            point_inputs = {"point_coords": final_points, "point_labels": final_labels}
            backbone_out["point_inputs_per_frame"][t] = point_inputs
        
        # No correction points
        backbone_out["frames_to_add_correction_pt"] = []
        
        return backbone_out




    # def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
    #     """
    #     Aggregate bounding boxes by FRAME + VIDEO, then reverse map
    #     """
    #     print("\n=== FRAME+VIDEO AGGREGATED BOUNDING BOX MODE ===")
    #     print(f"Batch shape - masks: {input.masks.shape}")
    #     print(f"Batch shape - obj_to_frame_idx: {input.obj_to_frame_idx.shape}")

    #     # Create individual bounding boxes first, then aggregate
    #     gt_masks_per_frame = {
    #         stage_id: masks.unsqueeze(1)  # [O, 1, H, W]
    #         for stage_id, masks in enumerate(input.masks)
    #     }

    #     backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
    #     num_frames = input.num_frames
    #     backbone_out["num_frames"] = num_frames

        
    #     # Training parameters (keep existing)
    #     if self.training:
    #         prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
    #         prob_to_use_box_input = self.prob_to_use_box_input_for_train
    #         num_frames_to_correct = self.num_frames_to_correct_for_train
    #         rand_frames_to_correct = self.rand_frames_to_correct_for_train
    #         num_init_cond_frames = self.num_init_cond_frames_for_train
    #         rand_init_cond_frames = self.rand_init_cond_frames_for_train
    #     else:
    #         prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
    #         prob_to_use_box_input = self.prob_to_use_box_input_for_eval
    #         num_frames_to_correct = self.num_frames_to_correct_for_eval
    #         rand_frames_to_correct = self.rand_frames_to_correct_for_eval
    #         num_init_cond_frames = self.num_init_cond_frames_for_eval
    #         rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        
    #     if num_frames == 1:
    #         prob_to_use_pt_input = 1.0
    #         num_frames_to_correct = 1
    #         num_init_cond_frames = 1
        
    #     assert num_init_cond_frames >= 1
        
    #     # Frame selection logic
    #     if rand_init_cond_frames and num_init_cond_frames > 1:
    #         num_init_cond_frames = self.rng.integers(1, num_init_cond_frames, endpoint=True)
        
    #     # Sample initial conditioning frames
    #     if num_init_cond_frames == 1:
    #         init_cond_frames = [start_frame_idx]
    #     else:
    #         init_cond_frames = [start_frame_idx] + self.rng.choice(
    #             range(start_frame_idx + 1, num_frames),
    #             num_init_cond_frames - 1,
    #             replace=False,
    #         ).tolist()
        
    #     backbone_out["init_cond_frames"] = init_cond_frames
    #     backbone_out["frames_not_in_init_cond"] = [
    #         t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
    #     ]
        
    #     # FORCE BOUNDING BOX INPUT ONLY WITH FRAME+VIDEO AGGREGATION
    #     backbone_out["mask_inputs_per_frame"] = {}
    #     backbone_out["point_inputs_per_frame"] = {}
        

    #     print("\n=== CREATING ONE COMPOUND BBX PER SLICE ===")
    #     for t in init_cond_frames:
    #         # Get the masks for this stage
    #         masks = gt_masks_per_frame[t]  # [O, 1, H, W]
    #         print(f"Stage {t} - Mask shape: {masks.shape}")
            
    #         # Step 1: Generate individual bounding boxes for all objects
    #         individual_points, individual_labels = sample_box_points(masks)
            
    #         print(f"Stage {t} - Individual bounding boxes:")
    #         print(f"  points shape: {individual_points.shape}")  # [O, 2, 2] - O objects
    #         print(f"  labels shape: {individual_labels.shape}")  # [O, 2]
            

    #         # Step 2: Group objects by frame within each video
    #         obj_to_frame_idx = input.obj_to_frame_idx[t]  # [O, 2]
    #         frame_objects_per_video = {}  # Map from (video_idx, frame_idx) to list of object indices
            
    #         print(f"Stage {t} - Object mappings:")
    #         for obj_idx, (frame_idx, video_idx) in enumerate(obj_to_frame_idx):
    #             frame_idx = frame_idx.item()
    #             video_idx = video_idx.item()
                
    #             # Create key for this frame+video combination
    #             frame_video_key = (video_idx, frame_idx)
                
    #             if frame_video_key not in frame_objects_per_video:
    #                 frame_objects_per_video[frame_video_key] = []
                
    #             frame_objects_per_video[frame_video_key].append(obj_idx)
    #             print(f"  Object {obj_idx}: frame={frame_idx}, video={video_idx}")
            
    #         print(f"Stage {t} - Frame+Video grouping:")
    #         num_unique_frames = 0
    #         for (video_idx, frame_idx) in sorted(frame_objects_per_video.keys()):
    #             objects = frame_objects_per_video[(video_idx, frame_idx)]
    #             print(f"  Video {video_idx}, Frame {frame_idx}: objects {objects}")
    #             num_unique_frames += 1
            
    #         print(f"Stage {t} - Found {num_unique_frames} unique frames across all videos")
            
    #         # Step 3: Aggregate bounding boxes for each frame within each video
    #         aggregated_boxes = {}  # (video_idx, frame_idx) -> compound_box
            
    #         for (video_idx, frame_idx), object_indices in frame_objects_per_video.items():
    #             if len(object_indices) > 0:
    #                 # Get all bounding boxes for this frame in this video
    #                 frame_boxes = individual_points[object_indices]  # [num_objects_in_frame, 2, 2]
                    
    #                 # Extract all corners
    #                 all_top_lefts = frame_boxes[:, 0, :]     # [num_objects, 2] - all top-left corners
    #                 all_bottom_rights = frame_boxes[:, 1, :] # [num_objects, 2] - all bottom-right corners
                    
    #                 # Find the compound bounding box (min of top-lefts, max of bottom-rights)
    #                 compound_top_left = torch.min(all_top_lefts, dim=0)[0]     # [2] - min x,y of top-left
    #                 compound_bottom_right = torch.max(all_bottom_rights, dim=0)[0]  # [2] - max x,y of bottom-right
                    
    #                 # Create compound bounding box
    #                 compound_box = torch.stack([compound_top_left, compound_bottom_right])  # [2, 2]
    #                 aggregated_boxes[(video_idx, frame_idx)] = compound_box
                    
    #                 # Debug info
    #                 width = compound_bottom_right[0] - compound_top_left[0]
    #                 height = compound_bottom_right[1] - compound_top_left[1]
    #                 print(f"    Video {video_idx}, Frame {frame_idx} - Compound box: ({compound_top_left[0]:.0f},{compound_top_left[1]:.0f}) → ({compound_bottom_right[0]:.0f},{compound_bottom_right[1]:.0f})")
    #                 print(f"      Size: {width:.0f} x {height:.0f} pixels (covers {len(object_indices)} objects)")
            
    #         print(f"Stage {t} - Created {len(aggregated_boxes)} unique compound bboxes")
            
    #         # Step 4: REVERSE MAPPING - Map back to original objects
    #         final_points = []
    #         final_labels = []
            
    #         for obj_idx, (frame_idx, video_idx) in enumerate(obj_to_frame_idx):
    #             frame_idx = frame_idx.item()
    #             video_idx = video_idx.item()
                
    #             # Get the compound box for this object's frame in this video
    #             frame_video_key = (video_idx, frame_idx)
    #             compound_box = aggregated_boxes[frame_video_key]  # [2, 2]
                
    #             # Keep original labels
    #             original_labels = individual_labels[obj_idx]  # [2] - corner labels [2, 3]
                
    #             final_points.append(compound_box)
    #             final_labels.append(original_labels)
            
    #         # Stack into final tensors
    #         final_points = torch.stack(final_points)  # [O, 2, 2] - same count as original
    #         final_labels = torch.stack(final_labels)  # [O, 2]
            
    #         print(f"\nStage {t} - Reverse mapping complete:")
    #         print(f"  final points shape: {final_points.shape}")  # [O, 2, 2]
    #         print(f"  final labels shape: {final_labels.shape}")   # [O, 2]
            
    #         # Verification: Check that objects from same (video, frame) have same compound box
    #         for (video_idx, frame_idx), object_indices in frame_objects_per_video.items():
    #             if len(object_indices) > 1:
    #                 first_box = final_points[object_indices[0]]
    #                 for obj_idx in object_indices[1:]:
    #                     assert torch.allclose(final_points[obj_idx], first_box), \
    #                         f"Objects in video {video_idx}, frame {frame_idx} don't have same compound box!"
            
    #         print(f"  ✓ Verification passed: All objects from same frame within a video have identical compound boxes")
            
    #         point_inputs = {"point_coords": final_points, "point_labels": final_labels}
    #         backbone_out["point_inputs_per_frame"][t] = point_inputs

    #     # No correction points
    #     backbone_out["frames_to_add_correction_pt"] = []

    #     return backbone_out


    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]

            #     # DEBUG: Print the actual indices
            # print(f"Frame {stage_id} img_ids: {img_ids}")
            # print(f"Frame {stage_id} img_ids shape: {img_ids.shape}")
            
            # # Check if all img_ids point to the same frame
            # if hasattr(img_ids, 'tolist'):
            #     ids_list = img_ids.tolist()
            #     print(f"Frame {stage_id} first few img_ids: {ids_list[:5]}")

            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
                
                # DEBUG: Check how many objects per frame
            num_objects_in_frame = len(img_ids) if isinstance(img_ids, (list, tuple)) else img_ids.numel()
            print(f"Frame {stage_id}: Processing {num_objects_in_frame} objects simultaneously")

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs
