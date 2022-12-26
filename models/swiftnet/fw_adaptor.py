import torch.nn as nn
from .semseg import SemsegModel
from .resnet.full_network import resnet18


# --------------------------------------------------------------------------------
#  Custom function for the specific model architecture to load/update state_dict
# --------------------------------------------------------------------------------
def load_state_dict_into_model(model, pretrained_dict):
    model_dict = model.state_dict()
    if list(pretrained_dict.keys())[0] == 'backbone.conv1.weight':
        pretrained_dict = {k.replace('backbone', 'loaded_model.backbone'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('logits', 'loaded_model.logits'): v for k, v in pretrained_dict.items()}
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            print("State_dict mismatch!", flush=True)
            continue
        model_dict[name].copy_(param)
    model.load_state_dict(pretrained_dict, strict=True)


class SwiftNet(nn.Module):
    def __init__(self, num_classes_wo_bg, **kwargs):
        super().__init__()
        use_bn = True
        resnet = resnet18(pretrained=True, efficient=False, use_bn=use_bn, **kwargs)
        self.loaded_model = SemsegModel(resnet, num_classes_wo_bg, use_bn=use_bn, freeze_complete_backbone=False,
                                        freeze_backbone_bn=False, freeze_backbone_bn_affine=False)
        self.loaded_model.eval()

    def forward(self, batch):
        output = self.loaded_model.forward(batch)
        return output

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()
