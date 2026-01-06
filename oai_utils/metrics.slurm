#!/bin/bash
#SBATCH --job-name=medsam_metrices
#SBATCH --partition=a100_short         # or a100_long if >8h runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48               # threads per fold
#SBATCH --mem-per-cpu=5G          # 4 GB per CPU core → 40 GB total
#SBATCH --gres=gpu:1                     # 1 GPU per fold
#SBATCH --time=01:00:00
#SBATCH --array=0            #  folds: 0,1,2,3,4
#SBATCH --output=logs/%x_fold%a_%j.out
#SBATCH --error=logs/%x_fold%a_%j.err

# -------------------------
# Activate environment
# -------------------------
source ~/.bashrc
conda activate medsam2
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)


# Threading limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------
# Fold index from SLURM_ARRAY_TASK_ID
# -------------------------
fold=${SLURM_ARRAY_TASK_ID}




echo "=========================="
echo "Running fold ${fold}"
echo "=========================="

    # --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference/fold${fold}/ \
    # --out_csv ./results/fold${fold}.csv \


    # --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSam2_OAI_Inference/ \
    # --out_csv ./results/MedSam2.csv \

python metrices.py \
    --gt_base /gpfs/home/machlm03/Segmentation/OAI_demo/OAI_TrainTest/V00_00m_test_1.0/ \
    --pred_base /gpfs/home/machlm03/Segmentation/OAI_demo/MedSAM_Finetune_OAI_Inference/fold${fold}/ \
    --out_csv ./results/fold${fold}.csv \
    --gt_subdir masks \
    --pred_subdir imgs \
    --classes 1 2 3 4 5 6 7 \
    --workers ${SLURM_CPUS_PER_TASK-4}



