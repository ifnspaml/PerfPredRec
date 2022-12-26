import torch.nn as nn

from .resnet.full_network import resnet18
from .semseg import SemsegModel


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
    model.load_state_dict(pretrained_dict, strict=False)


# Class SwiftNetRec based on SwiftNet and an separate Autoencoder-Decoder for Image-Upsampling
class SwiftNetRec(nn.Module):
    def __init__(self, num_classes_wo_bg, **kwargs):
        super().__init__()
        # Create the SwiftNet model
        model = resnet18(use_bn=True,
                         efficient=False,
                         **kwargs)

        # Pass the SwiftNet model and create a image reconstruction on top
        self.loaded_model = SemsegModel(model, num_classes_wo_bg,
                                        use_bn=True,
                                        efficient=False,
                                        **kwargs)
        self.loaded_model.eval()

    def forward(self, batch):
        output_dict = self.loaded_model.forward(batch)
        return output_dict

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()
