#!/bin/bash
#SBATCH --job-name=medsam2_infer_arr
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G          # 4 GB per CPU core → 40 GB total
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=5   
#SBATCH --output=val_logs2/%x_%A_%a.out
#SBATCH --error=val_logs2/%x_%A_%a.err

# set -euo pipefail
source ~/.bashrc
conda activate medsam2
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

cd /gpfs/data/arsoyd01lab/machlm03/Segmentation/MedSAM2


fold=${SLURM_ARRAY_TASK_ID}


INPUT_BASE_DIR="/gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0"
# /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0/9000099_00m_LEFT_SAG_3D_DESS_WE
# MODEL_CKPT="/gpfs/home/machlm03/Segmentation/MedSAM2/checkpoints/medsam_oai_checkpoints2.2/MedSam2_OAI_fold${fold}_bestcheckpoint.pt"
MODEL_CKPT='/gpfs/home/machlm03/Segmentation/MedSAM2/medsam2.3_exp_log/5/checkpoints/checkpoint_75.pt'


MODEL_CFG="configs/sam2.1_hiera_t512.yaml"
FRAMES_SUBDIR="imgs_jpeg"
MASKS_SUBDIR="masks"
PERCENTILES=(25 50 75)  # ✅ Bash array
  # ✅ split into separate strings


OUT_CASE_DIR="/gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference_2.3/fold${fold}" \


mkdir -p "$OUT_CASE_DIR" logs

echo "=========================="
echo "Inference by loading MedSam2_OAI_fold${fold}_bestcheckpoint.pt"
echo "MedSAM2 Inference (array)"
echo "Output case: $OUT_CASE_DIR"
echo "=========================="

# Run the same script but restrict to this case name
python OAI_medsam_video_inference.py \
  --input_base_dir "$INPUT_BASE_DIR" \
  --output_base_dir "$OUT_CASE_DIR" \
  --model_checkpoint "$MODEL_CKPT" \
  --percentiles "${PERCENTILES[@]}" \
  --model_config "$MODEL_CFG" \
  --frames_subdir "$FRAMES_SUBDIR" \
  --masks_subdir "$MASKS_SUBDIR" \
  --skip_until 9489320_00m_LEFT_SAG_3D_DESS_WE \

