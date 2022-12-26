#!/bin/bash -l

#SBATCH --job-name=attack
#SBATCH --partition=gpu,gpub        # gpu or debug
#SBATCH --time=100:20:00        # 100:20:00 (gpu) or 00:20:00 (debug)
#SBATCH --begin=now
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M		    # was 50000/20000/15000
#SBATCH --cpus-per-task=2       # 2 or 4
#SBATCH --exclude=gpu06

export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../../experiments/"
#export PYTHONPATH=/beegfs/work/$(whoami)/semantic_segmentation
#export PYTHONPATH="${PYTHONPATH}:/beegfs/work/$(whoami)/semantic_segmentation/swiftnet-pp_code_release/"

conda activate swiftnet-pp

# --model_state decoder:resnet10_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder resnet10

# --model_state decoder:resnet18_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder resnet18

# --model_state decoder:resnet18_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10_lateral:1
# --rec_decoder resnet18

# --model_state decoder:resnet26_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder resnet26

# --model_state decoder:basic_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder swiftnet

# --model_state decoder:basic_noskip_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder swiftnet_noskip

# --model_state decoder:basic_nospp_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10
# --rec_decoder swiftnet_nospp

python eval_attacks_n_noise.py \
--model_name SwiftNetRec \
--rec_decoder swiftnet \
--lateral 0 \
--model_state_name decoder:resnet10_encoder:pretrained_frozen:1_skip:concatenate-000_epochs:10\
--weights_epoch 10 \
--dataset kitti_2015 \
--subset val \
--zeroMean 1 \
--epsilon 0.25 0.5 0 1 2 4 8 12 16 20 24 28 32 \