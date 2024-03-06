import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import torch.nn.functional as F

import numpy as np
from PIL import Image
from torchvision import transforms
from torchsummary import summary

import cv2
from utils import vis_depth


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    metric_fn = utils.calc_vos
    metric1, metric2 = 'jaccard', 'fmeasure'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:

        image = batch['image'].cuda()
        gt_flow = batch['flow'].cuda()
        gt = batch['gt'].cuda()
        depth = batch['depth'].cuda()

        with torch.no_grad():
            image_ = F.interpolate(image, size=(model.inp_size, model.inp_size), mode='bilinear', align_corners=False)
            gt_flow_ = F.interpolate(gt_flow, size=(model.inp_size, model.inp_size), mode='bilinear', align_corners=False)
            depth_ = F.interpolate(depth, size=(model.inp_size, model.inp_size), mode='bilinear', align_corners=False)

            if model.is_depth:
                pred, depth = model.encoder(image_, gt_flow_)
            else:
                pred = model.encoder(image_, gt_flow_)

            # if model.is_depth:
            #     depth = torch.sigmoid(depth)
            #     im = vis_depth(depth[0].squeeze(0).cpu())
            #     im.save('depth.png')

        pred = F.interpolate(pred, size=image.shape[2:], mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)

        # transforms.ToPILImage()(pred[0].cpu()).save('result.png')
        # transforms.ToPILImage()(inverse_transform(image[0].cpu())).save('input.png')
        # transforms.ToPILImage()(gt[0].cpu()).save('gt.png')

        result1, result2, result3, result4 = metric_fn(pred, gt)
        val_metric1.add(result1.item(), image.shape[0])
        val_metric2.add(result2.item(), image.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))

    return val_metric1.item(), val_metric2.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['val_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()

    model.encoder.load_state_dict(torch.load(args.model), strict=True)


    metric1, metric2 = eval(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('jaccard : {:.4f}'.format(metric1))
    print('fmeasure: {:.4f}'.format(metric2))
