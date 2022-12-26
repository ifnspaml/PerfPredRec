"""Collection of losses with corresponding functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """ Deterministic variant of the cross entropy loss"""
    def __init__(self, weight=None, ignore_index=-100, ignore_background=False, train_id_0=0, reduction='mean',
                 device=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight has to be None or type torch.tensor"
        assert isinstance(ignore_index, int), "ignore_index has to be of type int"
        assert isinstance(ignore_background, bool), "ignore_background has to be of type bool"
        # With train_id_0 != 0, a slice of outputs can be passed to this loss even if train_ids > #classes. train_id_0
        # will make sure that train_ids < #classes is valid (iff class train_ids are consecutive!).
        assert isinstance(train_id_0, int), "train_id_0 has to be of type int"
        assert reduction in ('mean', 'sum', 'none'), "reduction only supports 'mean' (default), 'sum' and 'none'"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background
        self.train_id_0 = train_id_0
        self.reduction = reduction
        self.device = device
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_index, reduction='none')

    def forward(self, outputs_from_net, targets):

        #outputs_from_net = outputs_from_net['output_SemSeg']    # new line for the correct indexing of the output dict
        outputs = torch.nn.functional.log_softmax(outputs_from_net, dim=1)
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1], \
            "either provide weights for all classes or none"

        # Cast class trainIDs of targets into [0, #classes -1] by subtracting smallest trainID, background is taken
        # care of afterwards.
        targets = torch.add(targets, -self.train_id_0)
        bg = 255 - self.train_id_0

        # Background treatment
        if self.ignore_background:
            targets[targets == bg] = self.ignore_index
        else:
            targets[targets == bg] = outputs.shape[1] - 1

        # Calculate unreduced loss
        loss = self.loss(outputs, targets)          # =Cross-Entropy

        # Apply reduction manually since NLLLoss shows non-deterministic behaviour on GPU
        if self.reduction == 'mean':
            denom = 0
            if self.weight is not None:
                for i in range(outputs.shape[1]):
                    denom += torch.sum((targets == i).int()) * self.weight[i]
            else:
                if self.ignore_background:
                    denom = torch.numel(targets) - int(torch.sum((targets == self.ignore_index).int()))
                else:
                    denom = torch.numel(targets)
            return torch.sum(loss) / denom
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
