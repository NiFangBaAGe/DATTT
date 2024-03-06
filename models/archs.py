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

from .swin_transformer import swin_tiny, swin_small, swin_base, swin_base_384

from torchvision import transforms


from .mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


def build_backbone(backbone):
    if backbone == 'swin_tiny':
        b = swin_tiny()
        saved_state_dict = torch.load('./swin_tiny_patch4_window7_224_22k.pth')
        b.load_state_dict(saved_state_dict['model'], strict=False)
        feature_channels = [96, 192, 384, 768]
        embedding_dim = 256
    elif backbone == 'mit_b1':
        b = mit_b1()
        saved_state_dict = torch.load('./mit_b1.pth')
        b.load_state_dict(saved_state_dict, strict=False)
        feature_channels = [64, 128, 320, 512]
        embedding_dim = 256

    return b, feature_channels, embedding_dim


class Baseline(nn.Module):
    def __init__(self, backbone, depth_config):
        super().__init__()
        self.depth_config = depth_config
        self.encoder_image, feature_channels, embedding_dim = build_backbone(backbone)
        self.encoder_flow, _, _ = build_backbone(backbone)

        self.decode_head = SegformerHead(feature_channels, embedding_dim)

        if self.depth_config['depth_loss']:
            self.decode_head = DepthAssistHead(feature_channels, embedding_dim, 1,
                                               depth_arch=self.depth_config['depth_arch'])

    def get_depth(self, x):
        x = self.encoder_image(x)
        x = self.decode_head.depth_head(x)

        return x

    def encode_image(self, x):
        return self.encoder_image(x)

    def get_curr(self):
        return [self.x_0+self.y_0, self.x_1+self.y_1, self.x_2+self.y_2, self.x_3+self.y_3]

    def get_image_feat(self):
        return [self.x_0, self.x_1, self.x_2, self.x_3]

    def get_flow_feat(self):
        return [self.y_0, self.y_1, self.y_2, self.y_3]

    def get_out(self, feat):
        return self.decode_head(feat)

    def forward(self, x, y):
        self.x_0, self.x_1, self.x_2, self.x_3 = self.encoder_image(x)
        self.y_0, self.y_1, self.y_2, self.y_3 = self.encoder_flow(y)
        feat = [self.x_0+self.y_0, self.x_1+self.y_1, self.x_2+self.y_2, self.x_3+self.y_3]

        if self.depth_config['depth_loss']:
            z = self.decode_head([feat, [self.x_0, self.x_1, self.x_2, self.x_3]])
        else:
            z = self.decode_head(feat)

        return z


import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F
import attr
from mmcv.cnn import ConvModule


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class SegformerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, embedding_dim, out_chan=1):
        super(SegformerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, out_chan, kernel_size=1)


    def get_feat(self, x):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        return _c1, _c2, _c3, _c4

    def get_result(self, x):
        _c1, _c2, _c3, _c4 = x
        _c4 = resize(_c4, size=_c1.size()[2:],mode='bilinear',align_corners=False)
        _c3 = resize(_c3, size=_c1.size()[2:],mode='bilinear',align_corners=False)
        _c2 = resize(_c2, size=_c1.size()[2:],mode='bilinear',align_corners=False)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))


        # x = self.dropout(_c)
        x = self.linear_pred(_c)
        return x

    def forward(self, x):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])


        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # x = self.dropout(_c)
        self.decoder_feat = _c
        x = self.linear_pred(_c)

        return x


def disp_to_depth(disp, min_depth=1e-3, max_depth=80):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class DepthAssistHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, embedding_dim, out_chan=1, depth_arch='basic'):
        super(DepthAssistHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        embedding_dim = embedding_dim
        self.depth_arch = depth_arch
        self.depth_head = SegformerHead(in_channels, embedding_dim, out_chan)
        self.vos_head = SegformerHead(in_channels, embedding_dim, out_chan)


        print('depth arch : ', self.depth_arch)


        if self.depth_arch == 'basic':
            1
        elif self.depth_arch == "alpha+beta":
            self.shared1 = nn.Sequential(nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1), nn.ReLU())
            self.shared2 = nn.Sequential(nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1), nn.ReLU())
            self.shared3 = nn.Sequential(nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1), nn.ReLU())
            self.shared4 = nn.Sequential(nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1), nn.ReLU())
            self.alpha1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.alpha2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.alpha3 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.alpha4 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.beta1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.beta2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.beta3 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            self.beta4 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        elif self.depth_arch == "alpha":
            self.alpha1 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.alpha2 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.alpha3 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.alpha4 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
        elif self.depth_arch == "beta":
            self.beta1 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.beta2 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.beta3 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
            self.beta4 = nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1)
        else:
            raise ValueError('wrong depth arch.')

    def get_depth(self, image_feat):
        depth_feat = self.depth_head.get_feat(image_feat)
        depth_out = self.depth_head.get_result(depth_feat)

        return depth_out

    def forward(self, x):
        vos_feat = x[0]
        image_feat = x[1]

        if self.depth_arch=='basic':
            vos_feat = self.vos_head.get_feat(vos_feat)
            vos_out = self.vos_head.get_result(vos_feat)
            depth_feat = self.depth_head.get_feat(image_feat)
            depth_out = self.depth_head.get_result(depth_feat)

        elif self.depth_arch=='alpha+beta':
            vos_feat = self.vos_head.get_feat(vos_feat)
            v1, v2, v3, v4 = vos_feat
            depth_feat = self.depth_head.get_feat(image_feat)
            d1, d2, d3, d4 = depth_feat
            shared4 = self.shared4(torch.cat([d4, v4], dim=1))
            shared3 = self.shared3(torch.cat([d3, v3], dim=1))
            shared2 = self.shared2(torch.cat([d2, v2], dim=1))
            shared1 = self.shared1(torch.cat([d1, v1], dim=1))
            alpha4, beta4 = self.alpha4(shared4), self.beta4(shared4)
            alpha3, beta3 = self.alpha3(shared3), self.beta3(shared3)
            alpha2, beta2 = self.alpha2(shared2), self.beta2(shared2)
            alpha1, beta1 = self.alpha1(shared1), self.beta1(shared1)

            vos_feat = [alpha1 * v1 + beta1, alpha2 * v2 + beta2, alpha3 * v3 + beta3, alpha4 * v4 + beta4]

            depth_out = self.depth_head.get_result(depth_feat)
            vos_out = self.vos_head.get_result(vos_feat)

        elif self.depth_arch=='alpha':
            vos_feat = self.vos_head.get_feat(vos_feat)
            v1, v2, v3, v4 = vos_feat
            depth_feat = self.depth_head.get_feat(image_feat)
            d1, d2, d3, d4 = depth_feat
            alpha4 = self.alpha4(torch.cat([d4, v4], dim=1))
            alpha3 = self.alpha3(torch.cat([d3, v3], dim=1))
            alpha2 = self.alpha2(torch.cat([d2, v2], dim=1))
            alpha1 = self.alpha1(torch.cat([d1, v1], dim=1))

            vos_feat = [alpha1 * v1, alpha2 * v2, alpha3 * v3, alpha4 * v4]

            depth_out = self.depth_head.get_result(depth_feat)
            vos_out = self.vos_head.get_result(vos_feat)

        elif self.depth_arch=='beta':
            vos_feat = self.vos_head.get_feat(vos_feat)
            v1, v2, v3, v4 = vos_feat
            depth_feat = self.depth_head.get_feat(image_feat)
            d1, d2, d3, d4 = depth_feat
            beta4 = self.beta4(torch.cat([d4, v4], dim=1))
            beta3 = self.beta3(torch.cat([d3, v3], dim=1))
            beta2 = self.beta2(torch.cat([d2, v2], dim=1))
            beta1 = self.beta1(torch.cat([d1, v1], dim=1))

            vos_feat = [v1 + beta1, v2 + beta2, v3 + beta3, v4 + beta4]

            depth_out = self.depth_head.get_result(depth_feat)
            vos_out = self.vos_head.get_result(vos_feat)

        return vos_out, depth_out
