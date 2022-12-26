#!/bin/bash -l

#SBATCH --job-name=trn_SN
#SBATCH --partition=gpu         # gpu or debug
#SBATCH --time=100:20:00        # 100:20:00 (gpu) or 00:20:00 (debug)
#SBATCH --begin=now
#SBATCH --gres=gpu:1
#SBATCH --mem=20000M		    # was 50000/20000/15000
#SBATCH --cpus-per-task=4       # 2 or 4
#SBATCH --exclude=gpu06

export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../experiments/"

conda activate swiftnet-pp_cr2

python train_swiftnet.py \
--model_name SwiftNet \
--savedir swiftnet_baseline \
--dataset cityscapes \
--zeromean 1 \
