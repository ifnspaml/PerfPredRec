#!/bin/bash -l

#SBATCH --job-name=trn_SN
#SBATCH --partition=gpu
#SBATCH --time=100:20:00
#SBATCH --begin=now
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --cpus-per-task=2

export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../experiments/"

conda activate swiftnet-pp

python train_swiftnet.py \
--model_name SwiftNet \
--savedir swiftnet_baseline \
--dataset cityscapes \
--zeromean 1 \
