from __future__ import absolute_import, division, print_function

import argparse
from models.swiftnet_rec.rec_decoders import SUPPORTED_REC_DECODER


class SwiftNetOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SemSeg Evaluator options")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 type=int,
                                 choices=[0, 1],
                                 default=0,
                                 help="if set disables CUDA")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=2)
        self.parser.add_argument("--verbose",
                                 type=int,
                                 choices=[0, 1],
                                 default=0,
                                 help="If set, script will be more talkative...")
        self.parser.add_argument("--deterministic",
                                 type=int,
                                 choices=[0, 1],
                                 default=1,
                                 help="If set, training is performed deterministic")
        self.parser.add_argument("--global_seed",
                                 type=int,
                                 help="Set seed for initialization",
                                 default=1234)
        self.parser.add_argument("--worker_seed",
                                 type=int,
                                 help="Set seed for initialization",
                                 default=3333)

        # DATA options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to use",
                                 choices=['kitti_2015', 'cityscapes'],
                                 default='cityscapes')
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=1024)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=2048)
        self.parser.add_argument("--crop_height",
                                 type=int,
                                 help="input crop image height",
                                 default=768)
        self.parser.add_argument("--crop_width",
                                 type=int,
                                 help="input crop image width",
                                 default=768)
        self.parser.add_argument("--trainvaltest_split",
                                 type=str,
                                 help="Which split of dataset to load",
                                 choices=['train','validation','test'],
                                 default='train')
        self.parser.add_argument("--zeromean",
                                 type=int,
                                 choices=[0, 1],
                                 default=0,
                                 help="Input data is normalized to zero mean/var")

        # MODEL options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the model architecture this network is based on",
                                 default="SwiftNet")

        # TRAINING options
        self.parser.add_argument("--classweighing",
                                 help="If set, use classweighing for dataset",
                                 type=int,
                                 choices=[0,1],
                                 default=0)
        self.parser.add_argument("--LRscheduler",
                                 type=str,
                                 help="specify LR scheduler",
                                 default="CosineAnnealing",
                                 choices=["CosineAnnealing"])

        # OPTIMIZATION options
        self.parser.add_argument("--optimizer",
                                 type=str,
                                 help="optimizer for training",
                                 default="Adam",
                                 choices=["Adam"])
        self.parser.add_argument("--batch_size_train",
                                 type=int,
                                 help="batch size for training",
                                 default=12)
        self.parser.add_argument("--learning_rate_fine_tune",
                                 type=float,
                                 help="initial learning rate for fine_tune params",
                                 default=1e-4)
        self.parser.add_argument("--learning_rate_random_init",
                                 type=float,
                                 help="initial learning rate for random_init params",
                                 default=4e-4)
        self.parser.add_argument("--eta_min_fine_tune",
                                 type=float,
                                 help="eta min for fine_tune params",
                                 default=1e-7)
        self.parser.add_argument("--eta_min_random_init",
                                 type=float,
                                 help="eta min for random init params",
                                 default=1e-6)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs for training",
                                 default=200)

        # EVALUATION options
        self.parser.add_argument("--val_frequency",
                                 type=int,
                                 help="number of epochs between each validation. For standalone, any number > 0 will produce an output",
                                 default=1)
        self.parser.add_argument("--batch_size_val",
                                 type=int,
                                 help="batch size for validation",
                                 default=1)

        # SAVE & LOGGING options
        self.parser.add_argument("--savedir",
                                 type=str,
                                 default='')
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=10)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

class SwiftNetRecOptions(SwiftNetOptions):
    def __init__(self):
        super().__init__()
        # OPTIMIZATION options
        self.parser.add_argument("--rec_decoder",
                                 type=str,
                                 help="Choose the reconstruction decoder for SwiftNetRec",
                                 default="resnet18",
                                 choices=SUPPORTED_REC_DECODER)

        # Lateral connection between encoder and decoder (ResNet only)
        self.parser.add_argument("--lateral",
                                 type=int,
                                 help="Whether to connect encoder to decoder (ResNet only)",
                                 choices=[0, 1],
                                 default=0)

        self.parser.add_argument("--load_model_state_name",
                                 type=str,
                                 help="name of model state checkpoint to load",
                                 default=None)
