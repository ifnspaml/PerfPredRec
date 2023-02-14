#!/bin/bash -l

#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --time=100:20:00
#SBATCH --begin=now
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M
#SBATCH --cpus-per-task=4

export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../../experiments/"
export PYTHONPATH="${PWD}/../"

conda activate swiftnet-pp

python eval_attacks_n_noise.py \
--model_name SwiftNetRec \
--rec_decoder swiftnet \
--lateral 0 \
--model_state_name decoder_swiftnet \
--weights_epoch 10 \
--dataset kitti_2015 \
--subset val \
--zeroMean 1 \
--epsilon 0.25 0.5 0 1 2 4 8 12 16 20 24 28 32