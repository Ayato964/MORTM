from torch import nn
import torch

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)

        cross_loss = self.cross_entropy(inputs, targets)
        cross_loss = cross_loss * mask

        return cross_loss.sum() / (mask.sum() + 1e-12)