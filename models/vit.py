import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
import math
import numpy as np
from torch.autograd import Variable

from timm.models.layers import trunc_normal_
import os
import logging
logger = logging.getLogger(__name__)
import random
from torch.nn import init
import thop
import cv2

from torchvision import transforms
from .archs import Baseline
import random

from torchvision import transforms

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

class SILogLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SILogLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


@register('vit')
class ViT(nn.Module):
    def __init__(self, arch=None, inp_size=None, encoder_mode=None, weights=None, depth_config=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.arch = arch
        self.depth_config = depth_config
        self.inp_size = inp_size

        self.is_depth = depth_config['depth_loss']

        self.encoder = Baseline(encoder_mode['backbone'], depth_config=depth_config)

        model_total_params = sum(p.numel() for p in self.encoder.parameters())
        print('model_total_params:' + str(model_total_params))

        if weights!=None:
            print('loading pretrained weights from ' + weights)
            self.encoder.load_state_dict(torch.load(weights), strict=False)

        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        self.criterionDepth = SILogLoss()

    def set_input(self, image, flow, gt_mask, depth):
        self.image = image.to(self.device)
        self.flow = flow.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.depth = depth.to(self.device)

    def forward(self):
        if self.is_depth:
            self.pred, self.pred_depth = self.encoder(self.image, self.flow)
        else:
            self.pred = self.encoder(self.image, self.flow)

    def backward(self):
        if self.is_depth:
            depth = self.pred_depth
            depth = F.interpolate(depth, size=self.depth.shape[2:], mode='bilinear', align_corners=False)

            depth = torch.sigmoid(depth)
            depth_loss = 0.1 * self.criterionDepth(depth, self.depth)

            self.loss = self.criterionBCE(
                F.interpolate(self.pred, size=self.image.shape[2:], mode='bilinear', align_corners=False), self.gt_mask)

            self.loss = self.loss + depth_loss

        else:
            self.loss = self.criterionBCE(
                F.interpolate(self.pred, size=self.image.shape[2:], mode='bilinear', align_corners=False), self.gt_mask)

        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights
