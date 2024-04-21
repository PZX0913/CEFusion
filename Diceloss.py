import torch
import numpy
import torch.nn as nn
import cv2 
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre/255 * tar/255).sum(-1).sum() 
        union = (pre/255).sum(-1).sum() + (tar/255).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score
