# AE Decoder (ResNet-18 based) for Image Upsampling
# Uses the Resnet-18 Encoder Output after complete Downsampling (without(!) SPP)
# This Code is loosely based on a VAE-ResNet18 from Julian Stastny:
# https://github.com/julianstastny/VAE-ResNet18-PyTorch

import torch
import torch.nn.functional as F
from torch import nn

from models.util import BasicBlockDec, ResizeConv2D

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')


class ResNetDecoder(nn.Module):

    def __init__(self, num_Blocks=[1, 1, 1, 1], nc=3, efficient=True, use_bn=True, lateral=False):
        super().__init__()
        self.use_bn = use_bn
        self.inplanes = 512
        self.efficient = efficient
        self.lateral = lateral

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2, lateral=self.lateral)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2, lateral=self.lateral)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1, lateral=self.lateral)

        self.conv1 = ResizeConv2D(64, nc, kernel_size=3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, lateral=False):
        layers = []

        # Create residual units.
        if blocks == 1:
            # Add the one and only residual unit. Pass the concatenation information. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn,
                             lateral=lateral)]
        elif blocks == 2:
            # Add the first residual unit. Pass the concatenation information.
            layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn,
                             lateral=lateral)]
            # Add the last residual unit. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn)]
        elif blocks > 2:
            # Add the first residual unit.
            layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn,
                             lateral=lateral)]
            # Add additional residual units with standard configuration.
            for i in range(1, blocks - 1):
                layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn)]
            # Add the last residual unit. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn)]

        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, features_resnet=None):
        x = self.layer4(x)
        if self.lateral:  # No connection between decoders
            x = self.layer3([x, features_resnet[2]])
            x = self.layer2([x, features_resnet[1]])
            x = self.layer1([x, features_resnet[0]])
        else:
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)

        # Interpolation after the four layers needs less resources
        x = F.interpolate(x, scale_factor=2)

        # the Sigmoid-function transforms the input to [0, 1]-space
        x = torch.sigmoid(self.conv1(x))

        # Perform zero-mean normalization to have the space as the input
        x = x - MEAN
        x = x / STD

        return x
