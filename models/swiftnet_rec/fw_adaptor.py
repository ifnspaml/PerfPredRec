import torch.nn as nn

from .resnet.full_network import resnet18
from .semseg import SemsegModel


# --------------------------------------------------------------------------------
#  Custom function for the specific model architecture to load/update state_dict
# --------------------------------------------------------------------------------
def load_state_dict_into_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # For compatibility with models uploaded to Google Drive
    pretrained_dict = {k.replace('spp_ae', 'spp_rec'): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('ae_decoder', 'rec_decoder'): v for k, v in pretrained_dict.items()}
    if list(pretrained_dict.keys())[0] == 'backbone.conv1.weight':
        pretrained_dict = {k.replace('backbone', 'loaded_model.backbone'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('logits', 'loaded_model.logits'): v for k, v in pretrained_dict.items()}
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            if "onexone_conv" in name:
                pass # Old legacy code created some modules which were not used by default. They are ignored here.
            else:
                print(f"State_dict mismatch! Entry {name} not found.", flush=True)
            continue
        try:
            model_dict[name].copy_(param)
        except:
            print(f"Tensor size mismatch for weight tensor {name}! Model dict: {model_dict[name].size()}, Pretrained dict: {param.size()}.", flush=True)
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
