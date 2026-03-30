#!/bin/bash

#SBATCH --job-name=mdoel3.1
#SBATCH --output=mdoel3.1.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=06:00:00

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate TensoIR
DATASET_NAME="1"
SCENE_ID="1"

python train_tensoIR_general_multi_lights.py \
  --expname "HPC-$DATASET_NAME-$SCENE_ID" \
  --config ./configs/multi_light_general/vsr.txt \
  --datadir "$HOME/data/studio_test5/scene$DATASET_NAME/" \
  --hdrdir "$HOME/data/studio_test5/scene$DATASET_NAMEE/meta/backgrounds" \
  --scene $SCENE_ID\
  --relight_chunk_size 10000 
