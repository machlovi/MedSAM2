# !/bin/bash
# SBATCH --job-name=MedSAM2_folds_val
# SBATCH --partition=a100_dev          # or a100_long if >8h runtime
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --mem-per-cpu=16G          # 4 GB per CPU core → 40 GB total
# SBATCH --cpus-per-task=10               # threads per fold
# SBATCH --gres=gpu:3                     # 1 GPU per fold
# SBATCH --time=04:00:00
# SBATCH --array=0               #  folds: 0,1,2,3,4
# SBATCH --output=val_logs/%x_fold%a_%j.out
# SBATCH --error=val_logs/%x_fold%a_%j.err


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
output_base="./val_log"

# WandB project (optional)
export WANDB_PROJECT="medsam2-finetune_val"

# Threading limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------
# Fold index from SLURM_ARRAY_TASK_ID
# -------------------------
fold=${SLURM_ARRAY_TASK_ID}

# train_txt="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/cv_txt_files/train/train_fold${fold}.txt"
val_txt="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/cv_txt_files/test/test_split.txt"


# /gpfs/home/machlm03/Segmentation/MedSAM2/checkpoints/medsam_oai_checkpoints/MedSam2_OAI_fold0_bestcheckpoint.pt
# /gpfs/home/machlm03/Segmentation/MedSAM2/checkpoints/MedSam2_OAI_fold0.pt
CKPT="/gpfs/home/machlm03/Segmentation/MedSAM2/checkpoints/medsam_oai_checkpoints2.2/MedSam2_OAI_fold${fold}_bestcheckpoint.pt"

val="val"

output_path="${output_base}/fold${fold}"
export WANDB_NAME="val_fold${fold}"

echo "=========================="
echo "Validation ONLY"
echo "Running fold ${fold}"
echo "Val file: $val_txt"
echo "Output path: $output_path"
echo "=========================="



# --checkpoint "$CKPT" \
# --train-txt "$train_txt" \

python training/train.py \
    -c $config \
    --output-path "$output_path" \
    --num-gpus 3 \
    --checkpoint "$CKPT" \
    --val-txt "$val_txt" \
    --mode "$val" \
    --use-cluster 0 \
    --num-nodes 1
