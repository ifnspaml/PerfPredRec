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

python train_swiftnet_rec.py \
--model_name SwiftNetRec \
--savedir swiftnet_rec_baseline \
--dataset cityscapes \
--zeromean 1 \
--batch_size_train 8 \
--num_epochs 10 \
--rec_decoder swiftnet \
--lateral 1 \
--load_model_state_name ../SwiftNet_ss_common/swiftnet_baseline_cvpr2022/
