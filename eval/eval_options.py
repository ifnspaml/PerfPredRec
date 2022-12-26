from __future__ import absolute_import, division, print_function

import argparse

from models.swiftnet_rec.rec_decoders import SUPPORTED_REC_DECODER


class EvalOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SemSeg Evaluator options")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 type=int,
                                 choices=[0, 1],
                                 default=0)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)
        self.parser.add_argument("--verbose",
                                 help="If set, script will be more talkative...",
                                 type=int,
                                 choices=[0, 1],
                                 default=0)

        # DATA options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to use",
                                 choices=['kitti_2015', 'cityscapes'],
                                 default='cityscapes')
        self.parser.add_argument('--subset',
                                 type=str,
                                 choices=['val', 'test'],
                                 help="Subset on which to perform inference on.")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=1024)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=2048)
        self.parser.add_argument("--zeroMean",
                                 help="Input data is normalized to zero mean/var",
                                 type=int,
                                 choices=[0, 1],
                                 default=0)

        # MODEL options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the model architecture this network is based on",
                                 default="SwiftNetRec",
                                 choices=["SwiftNet", "SwiftNetRec"])

        # LOADING options
        self.parser.add_argument("--model_state_name",
                                 type=str,
                                 help="name of model state checkpoint to load",
                                 default='none')
        self.parser.add_argument("--weights_epoch",
                                 type=int,
                                 help="Specify epoch of which weights to load",
                                 default=0)

        # RECONSTRUCTION DECODER options
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

        # ATTACK STRENGTH options
        self.parser.add_argument('--epsilon',
                                 type=float,
                                 nargs='+',
                                 help="(Infinity) norm bound."
                                      "Supported by: FGSM, IterativeFGSM, GDUAP.")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
