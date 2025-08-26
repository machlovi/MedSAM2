#!/bin/bash

source ~/.bashrc
conda activate medsam2
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# -------------------------
# Move to project directory
# -------------------------
cd /gpfs/data/arsoyd01lab/machlm03/Segmentation/MedSAM2

# -------------------------
# Config and output base
# -------------------------
config=configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml
output_base="./exp_log/MedSAM2_FLARE25_RECIST_OAI"

export WANDB_PROJECT="medsam2-finetune"
export CUDA_VISIBLE_DEVICES=0

# -------------------------
# Loop over folds
# -------------------------
N_FOLDS=2  # Adjust as needed

for fold in $(seq 0 $N_FOLDS); do
    train_txt="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/cv_txt_files/train/train_fold${fold}.txt"
    val_txt="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/cv_txt_files/train/val_fold${fold}.txt"
    output_path="${output_base}/fold${fold}"
    export WANDB_NAME="fold${fold}"

    echo "=========================="
    echo "Running fold ${fold}"
    echo "Train file: $train_txt"
    echo "Val file: $val_txt"
    echo "Output path: $output_path"
    echo "=========================="

    python training/train.py \
        -c $config \
        --output-path "$output_path" \
        --num-gpus 1 \
        --train-txt "$train_txt" \
        --val-txt "$val_txt" \
        --use-cluster 0 \
        --num-nodes 1

    echo "Fold ${fold} training done"
    echo ""
done

echo "All folds completed."
