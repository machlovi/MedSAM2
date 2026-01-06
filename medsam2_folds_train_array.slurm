#!/bin/bash
#SBATCH --job-name=MSAM2.4
#SBATCH --partition=a100_short        # or a100_long if >8h runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G          # 4 GB per CPU core → 40 GB total
#SBATCH --cpus-per-task=5               # threads per fold
#SBATCH --gres=gpu:2                     # 1 GPU per fold
#SBATCH --time=08:00:00
#SBATCH --array=5            #  folds: 0,1,2,3,4
#SBATCH --output=MSAM2.4_logs/%x_fold%a_%j.out
#SBATCH --error=MSAM2.4_logs/%x_fold%a_%j.err

# -------------------------
# Activate environment
# -------------------------
source ~/.bashrc
conda activate medsam2
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# -------------------------
# Move to project directory
# -------------------------
cd /gpfs/data/arsoyd01lab/machlm03/Segmentation/MedSAM2

# -------------------------
# Config and output
# -------------------------
config=configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml

output_base="./MSAM2.4"

# WandB project (optional)
export WANDB_PROJECT="MSAM2.4"

# Threading limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------
# Fold index from SLURM_ARRAY_TASK_ID
# -------------------------
fold=${SLURM_ARRAY_TASK_ID}

train_txt="/gpfs/home/machlm03/Segmentation/SAM_OAI_noteboooks/cv_txt_files/train/train_fold${fold}.txt"

val_txt="/gpfs/home/machlm03/Segmentation/SAM_OAI_noteboooks/cv_txt_files/train/val_fold${fold}.txt"
# /gpfs/home/machlm03/Segmentation/SAM_OAI_noteboooks/cv_txt_files/train/val_fold${fold}.txt

# train_txt="/gpfs/home/machlm03/Segmentation/IWOAI_demo/data/train.txt"
# val_txt="/gpfs/home/machlm03/Segmentation/IWOAI_demo/data/valid.txt"



output_path="${output_base}"
# output_path="${output_base}/MSAM_ecnoderfreeze"

export WANDB_NAME="MSAM2.4"

echo "=========================="
# echo "Running fold ${fold}"
echo "Train file: $train_txt"
echo "Val file: $val_txt"
echo "Output path: $output_path"
echo "=========================="





python training/train.py \
    -c $config \
    --output-path "$output_path" \
    --num-gpus 2 \
    --train-txt "$train_txt" \
    --val-txt "$val_txt" \
    --use-cluster 0 \
    --num-nodes 1


    # --val-txt "$val_txt" \

# python /gpfs/home/machlm03/Segmentation/MedSAM2/training/train.py \
#     -c configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
#     --output-path ./medsam2.3_MSAM_ecnoderfreeze/5 \
#     --num-gpus 1 \
#     --train-txt /gpfs/home/machlm03/Segmentation/IWOAI_Segmentation_Challenge/train/train.txt \
