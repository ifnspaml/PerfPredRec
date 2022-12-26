import torch.nn as nn
from itertools import chain

from .. util import upsample, _BNActConv

class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, freeze_complete_backbone=False, freeze_backbone_bn=False,
                 freeze_backbone_bn_affine=False, use_bn=True):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.target_size = (2048, 1024)
        self.num_classes = num_classes

        self.logits = _BNActConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)
        self.freeze_complete_backbone = freeze_complete_backbone
        self.freeze_backbone_bn = freeze_backbone_bn
        self.freeze_backbone_bn_affine = freeze_backbone_bn_affine

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN and/or entire backbone parameters
        """
        super(SemsegModel, self).train(mode)
        if self.freeze_complete_backbone:
            print("Freezing ResNet backbone via eval() and param.requires_grad = False.")
        if self.freeze_backbone_bn:
            print("Freezing Mean/Var of BatchNorm2D in backbone.")
            if self.freeze_backbone_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D backbone.")

        if self.freeze_complete_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        if self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_backbone_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def forward(self, batch):
        feats, additional = zip(*[self.backbone(batch)])

        logits = self.logits.forward(feats[0])
        # During training deterministic, in evaluation "non-deterministic"
        upsampled = upsample(self.training, logits, batch.shape[2:4])
        return upsampled

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
