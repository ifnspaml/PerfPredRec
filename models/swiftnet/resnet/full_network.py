import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain

from ... util import _Upsample, SpatialPyramidPooling, BasicBlock

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True, spp_grids=(8, 4, 2, 1),
                 spp_square_grid=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        upsamples = []

        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient,
                        use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_down_only(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [skip]
        return features

    def forward_up(self, features):
        features = features[::-1]
        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model
