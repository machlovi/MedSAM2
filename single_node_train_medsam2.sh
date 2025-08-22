#!/bin/bash

source ~/.bashrc
conda activate medsam2
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)


cd /gpfs/data/arsoyd01lab/machlm03/Segmentation/MedSAM2


# Use absolute path or verify the relative path is correct
config=configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml
output_path=./exp_log/MedSAM2_FLARE25_RECIST_OAI


export WANDB_PROJECT="medsam2-finetune"
export WANDB_NAME="fold1"
export CUDA_VISIBLE_DEVICES=0,1

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python - <<'EOF'
import torch, os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch sees:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
EOF


cd /gpfs/data/arsoyd01lab/machlm03/Segmentation/MedSAM2

pwd
# Run training
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
    -c $config \
    --output-path $output_path \
    --use-cluster 0 \
    --num-gpus 2 \
    --num-nodes 1 \

echo "training done"