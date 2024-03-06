import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import itertools
import models
import yaml
import cv2
import numpy as np
import time
from torchsummary import  summary
from datasets.wrappers import rgb_loader, binary_loader
from utils import calc_vos
import random
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR, ExponentialLR
from augmentation import PhotoMetricDistortion, RandomResize, RandomCrop
from utils import ensure_path
import torch.nn.functional as F
from utils import vis_depth
import copy

def read_files(path):
    files = []
    filenames = sorted(os.listdir(path))
    for filename in filenames:
        file = os.path.join(path, filename)
        files.append(file)

    return files

def split_files(files):

    videos = {}
    for filename in files:
        cat = filename.split('/')[-1].split('_')[0]
        # if cat == 'breakdance':
        #     continue
        if cat not in videos.keys():
            videos[cat] = []
        videos[cat].append(filename)

    return videos

def unpatchify(x):
    """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
    """
    p = 16
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def ttt_loss(image_aug, image, model):

    model.encoder.train()

    depth_pred = model.encoder.get_depth(image)
    depth_pred = torch.sigmoid(depth_pred)

    aug_depth_pred = model.encoder.get_depth(image_aug)
    aug_depth_pred = torch.sigmoid(aug_depth_pred)

    # if depth_pred.shape[0] != aug_depth_pred.shape[0]:
    #     depth_pred = depth_pred.repeat(aug_depth_pred.shape[0], 1, 1, 1)
    loss = model.criterionDepth(aug_depth_pred, depth_pred)

    return loss


def videottt_ttt_n():
    model = models.make(config['model']).cuda()
    for k, p in model.encoder.named_parameters():
        if 'encoder_image' not in k:
            p.requires_grad = False
        else:
            p.requires_grad = True

    parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]

        jaccard[video_name], fmeasure[video_name] = {}, {}
        for epoch in range(epochs):
            jaccard[video_name][epoch], fmeasure[video_name][epoch] = [], []

        for i in frame_order:
            model.encoder.load_state_dict(torch.load(model_dir))
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)

            for epoch in range(epochs):
                image = rgb_loader(videos[video_name][i])
                gt = binary_loader(gts[video_name][i])

                sample_image = []
                sample_image_aug = []
                for b in range(batch_size):
                    image_ = image
                    if random.random() < 0.5:
                        image_ = image_.transpose(Image.FLIP_LEFT_RIGHT)
                    image_ = resize_aug(np.array(image_))
                    image_ = crop_aug(image_)
                    sample_image.append(np.array(image_))
                    sample_image_aug.append(np.array(image_))
                sample_image = pmd_aug(sample_image)
                sample_image_aug = pmd_aug(sample_image_aug)
                ia, ii = [], []
                for b in range(batch_size):
                    ii.append(img_transform(sample_image[b]).unsqueeze(0).cuda())
                    ia.append(img_transform(sample_image_aug[b]).unsqueeze(0).cuda())
                image_aug = torch.cat(ia, dim=0)
                image = torch.cat(ii, dim=0)

                image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                image_aug = F.interpolate(image_aug, size=(inp_size, inp_size), mode='bilinear',
                                               align_corners=False)

                loss = ttt_loss(image_aug, image, model)

                if loss != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    image = rgb_loader(videos[video_name][i])
                    flow = rgb_loader(flows[video_name][i])
                    gt = binary_loader(gts[video_name][i])
                    image = img_transform(image).unsqueeze(0).cuda()
                    flow = img_transform(flow).unsqueeze(0).cuda()
                    gt = mask_transform(gt).unsqueeze(0).cuda()

                    image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    flow = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    pred, _ = model.encoder(image, flow)

                pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred).cpu()

                j, f, _, _ = metric_func(pred.cuda(), gt)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1

                jaccard[video_name][epoch].append(j)
                fmeasure[video_name][epoch].append(f)

        # break
    for epoch in range(epochs):
        j_mean, f_mean = [], []
        for video_name in videos.keys():
            j_mean.append(np.mean(jaccard[video_name][epoch]))
            f_mean.append(np.mean(fmeasure[video_name][epoch]))
            if epoch == epochs - 1:
                print(epoch, video_name, np.mean(jaccard[video_name][epoch]), np.mean(fmeasure[video_name][epoch]))
            # break
        print(epoch, np.mean(j_mean), np.mean(f_mean))



def videottt_ttt_mwi():
    model = models.make(config['model']).cuda()
    for k, p in model.encoder.named_parameters():
        if 'encoder_image' not in k:
            p.requires_grad = False
        else:
            p.requires_grad = True

    parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]

        jaccard[video_name], fmeasure[video_name] = {}, {}
        for epoch in range(epochs):
            jaccard[video_name][epoch], fmeasure[video_name][epoch] = [], []

        model.encoder.load_state_dict(torch.load(model_dir))
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        for i in frame_order:

            for epoch in range(epochs):
                image = rgb_loader(videos[video_name][i])
                gt = binary_loader(gts[video_name][i])

                sample_image = []
                sample_image_aug = []
                for b in range(batch_size):
                    image_ = image
                    if random.random() < 0.5:
                        image_ = image_.transpose(Image.FLIP_LEFT_RIGHT)
                    image_ = resize_aug(np.array(image_))
                    image_ = crop_aug(image_)
                    sample_image.append(np.array(image_))
                    sample_image_aug.append(np.array(image_))
                sample_image = pmd_aug(sample_image)
                sample_image_aug = pmd_aug(sample_image_aug)
                ia, ii = [], []
                for b in range(batch_size):
                    ii.append(img_transform(sample_image[b]).unsqueeze(0).cuda())
                    ia.append(img_transform(sample_image_aug[b]).unsqueeze(0).cuda())
                image_aug = torch.cat(ia, dim=0)
                image = torch.cat(ii, dim=0)

                image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                image_aug = F.interpolate(image_aug, size=(inp_size, inp_size), mode='bilinear',
                                               align_corners=False)

                loss = ttt_loss(image_aug, image, model)

                if loss != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    image = rgb_loader(videos[video_name][i])
                    flow = rgb_loader(flows[video_name][i])
                    gt = binary_loader(gts[video_name][i])
                    image = img_transform(image).unsqueeze(0).cuda()
                    flow = img_transform(flow).unsqueeze(0).cuda()
                    gt = mask_transform(gt).unsqueeze(0).cuda()

                    image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    flow = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    pred, _ = model.encoder(image, flow)

                pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred)

                j, f, _, _ = metric_func(pred, gt)
                pred[pred<0.5]=0
                pred[pred>=0.5]=1

                jaccard[video_name][epoch].append(j)
                fmeasure[video_name][epoch].append(f)

    for epoch in range(epochs):
        j_mean, f_mean = [], []
        for video_name in videos.keys():
            j_mean.append(np.mean(jaccard[video_name][epoch]))
            f_mean.append(np.mean(fmeasure[video_name][epoch]))
            if epoch == epochs - 1:
                print(epoch, video_name, np.mean(jaccard[video_name][epoch]), np.mean(fmeasure[video_name][epoch]))
            # break
        print(epoch, np.mean(j_mean), np.mean(f_mean))


def videottt_ttt_ltv():
    model = models.make(config['model']).cuda()
    for k, p in model.encoder.named_parameters():
        if 'encoder_image' not in k:
            p.requires_grad = False
        else:
            p.requires_grad = True

    parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]

        jaccard[video_name], fmeasure[video_name] = {}, {}
        for epoch in range(epochs):
            jaccard[video_name][epoch], fmeasure[video_name][epoch] = [], []

        model.encoder.load_state_dict(torch.load(model_dir))
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        for epoch in range(epochs):
            for i in frame_order:
                image = rgb_loader(videos[video_name][i])
                gt = binary_loader(gts[video_name][i])

                sample_image = []
                sample_image_aug = []

                for b in range(batch_size):
                    image_ = image
                    if random.random() < 0.5:
                        image_ = image_.transpose(Image.FLIP_LEFT_RIGHT)
                    image_ = resize_aug(np.array(image_))
                    image_ = crop_aug(image_)
                    sample_image.append(np.array(image_))
                    sample_image_aug.append(np.array(image_))
                sample_image = pmd_aug(sample_image)
                sample_image_aug = pmd_aug(sample_image_aug)
                ia, ii = [], []
                for b in range(batch_size):
                    ii.append(img_transform(sample_image[b]).unsqueeze(0).cuda())
                    ia.append(img_transform(sample_image_aug[b]).unsqueeze(0).cuda())
                image_aug = torch.cat(ia, dim=0)
                image = torch.cat(ii, dim=0)

                image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                image_aug = F.interpolate(image_aug, size=(inp_size, inp_size), mode='bilinear',
                                               align_corners=False)

                loss = ttt_loss(image_aug, image, model)

                if loss != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    image = rgb_loader(videos[video_name][i])
                    flow = rgb_loader(flows[video_name][i])
                    gt = binary_loader(gts[video_name][i])
                    image = img_transform(image).unsqueeze(0).cuda()
                    flow = img_transform(flow).unsqueeze(0).cuda()
                    gt = mask_transform(gt).unsqueeze(0).cuda()

                    image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    flow = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    pred, _ = model.encoder(image, flow)

                pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred)

                j, f, _, _ = metric_func(pred, gt)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1

                jaccard[video_name][epoch].append(j)
                fmeasure[video_name][epoch].append(f)

    for epoch in range(epochs):
        j_mean, f_mean = [], []
        for video_name in videos.keys():
            j_mean.append(np.mean(jaccard[video_name][epoch]))
            f_mean.append(np.mean(fmeasure[video_name][epoch]))
            if epoch == epochs - 1:
                print(epoch, video_name, np.mean(jaccard[video_name][epoch]), np.mean(fmeasure[video_name][epoch]))
            # break
        print(epoch, np.mean(j_mean), np.mean(f_mean))


def videottt_group_ttt_ltv():
    model = models.make(config['model']).cuda()
    for k, p in model.encoder.named_parameters():
        if 'encoder_image' not in k:
            p.requires_grad = False
        else:
            p.requires_grad = True
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]

        jaccard[video_name], fmeasure[video_name] = {}, {}
        for epoch in range(epochs):
            jaccard[video_name][epoch], fmeasure[video_name][epoch] = [], []

        groups = []
        for i in frame_order:
            if i < frame_gap:
                groups.append([])
                groups[-1].append(i)
            else:
                groups[i % frame_gap].append(i)

        for group in groups:
            model.encoder.load_state_dict(torch.load(model_dir))
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            for epoch in range(epochs):
                for i in group:
                    image = rgb_loader(videos[video_name][i])
                    gt = binary_loader(gts[video_name][i])

                    sample_image = []
                    sample_image_aug = []
                    for b in range(batch_size):
                        image_ = image
                        if random.random() < 0.5:
                            image_ = image_.transpose(Image.FLIP_LEFT_RIGHT)
                        image_ = resize_aug(np.array(image_))
                        image_ = crop_aug(image_)
                        sample_image.append(np.array(image_))
                        sample_image_aug.append(np.array(image_))
                    sample_image = pmd_aug(sample_image)
                    sample_image_aug = pmd_aug(sample_image_aug)
                    ia, ii = [], []
                    for b in range(batch_size):
                        ii.append(img_transform(sample_image[b]).unsqueeze(0).cuda())
                        ia.append(img_transform(sample_image_aug[b]).unsqueeze(0).cuda())
                    image_aug = torch.cat(ia, dim=0)
                    image = torch.cat(ii, dim=0)

                    image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    image_aug = F.interpolate(image_aug, size=(inp_size, inp_size), mode='bilinear',
                                               align_corners=False)

                    loss = ttt_loss(image_aug, image, model)

                    if loss != 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        image = rgb_loader(videos[video_name][i])
                        flow = rgb_loader(flows[video_name][i])
                        gt = binary_loader(gts[video_name][i])
                        image = img_transform(image).unsqueeze(0).cuda()
                        flow = img_transform(flow).unsqueeze(0).cuda()
                        gt = mask_transform(gt).unsqueeze(0).cuda()

                        image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                        flow = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                        pred, _ = model.encoder(image, flow)

                    pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    pred = torch.sigmoid(pred)

                    j, f, _, _ = metric_func(pred, gt)
                    pred[pred < 0.5] = 0
                    pred[pred >= 0.5] = 1

                    jaccard[video_name][epoch].append(j)
                    fmeasure[video_name][epoch].append(f)

    for epoch in range(epochs):
        j_mean, f_mean = [], []
        for video_name in videos.keys():
            j_mean.append(np.mean(jaccard[video_name][epoch]))
            f_mean.append(np.mean(fmeasure[video_name][epoch]))
            if epoch == epochs - 1:
                print(epoch, video_name, np.mean(jaccard[video_name][epoch]), np.mean(fmeasure[video_name][epoch]))
        print(epoch, np.mean(j_mean), np.mean(f_mean))


def videottt_group_ttt_mwi():
    model = models.make(config['model']).cuda()
    for k, p in model.encoder.named_parameters():
        if 'encoder_image' not in k:
            p.requires_grad = False
        else:
            p.requires_grad = True
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]

        jaccard[video_name], fmeasure[video_name] = {}, {}
        for epoch in range(epochs):
            jaccard[video_name][epoch], fmeasure[video_name][epoch] = [], []

        groups = []
        for i in frame_order:
            if i < frame_gap:
                groups.append([])
                groups[-1].append(i)
            else:
                groups[i % frame_gap].append(i)

        for group in groups:
            model.encoder.load_state_dict(torch.load(model_dir))
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            for i in group:
                for epoch in range(epochs):
                    image = rgb_loader(videos[video_name][i])
                    gt = binary_loader(gts[video_name][i])

                    sample_image = []
                    sample_image_aug = []
                    for b in range(batch_size):
                        image_ = image
                        if random.random() < 0.5:
                            image_ = image_.transpose(Image.FLIP_LEFT_RIGHT)
                        image_ = resize_aug(np.array(image_))
                        image_ = crop_aug(image_)
                        sample_image.append(np.array(image_))
                        sample_image_aug.append(np.array(image_))
                    sample_image = pmd_aug(sample_image)
                    sample_image_aug = pmd_aug(sample_image_aug)
                    ia, ii = [], []
                    for b in range(batch_size):
                        ii.append(img_transform(sample_image[b]).unsqueeze(0).cuda())
                        ia.append(img_transform(sample_image_aug[b]).unsqueeze(0).cuda())
                    image_aug = torch.cat(ia, dim=0)
                    image = torch.cat(ii, dim=0)

                    image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                    image_aug = F.interpolate(image_aug, size=(inp_size, inp_size), mode='bilinear',
                                               align_corners=False)

                    loss = ttt_loss(image_aug, image, model)

                    if loss != 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        image = rgb_loader(videos[video_name][i])
                        flow = rgb_loader(flows[video_name][i])
                        gt = binary_loader(gts[video_name][i])
                        image = img_transform(image).unsqueeze(0).cuda()
                        flow = img_transform(flow).unsqueeze(0).cuda()
                        gt = mask_transform(gt).unsqueeze(0).cuda()

                        image = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                        flow = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                        pred, _ = model.encoder(image, flow)

                    pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    pred = torch.sigmoid(pred).cpu()

                    j, f, _, _ = calc_vos(pred, gt)
                    pred[pred<0.5]=0
                    pred[pred>=0.5]=1

                    jaccard[video_name][epoch].append(j)
                    fmeasure[video_name][epoch].append(f)

    for epoch in range(epochs):
        j_mean, f_mean = [], []
        for video_name in videos.keys():
            j_mean.append(np.mean(jaccard[video_name][epoch]))
            f_mean.append(np.mean(fmeasure[video_name][epoch]))
            if epoch == epochs - 1:
                print(epoch, video_name, np.mean(jaccard[video_name][epoch]), np.mean(fmeasure[video_name][epoch]))
            # break
        print(epoch, np.mean(j_mean), np.mean(f_mean))



def baseline():
    for video_name in videos.keys():
        print(video_name)

        frame_order = [i for i in range(len(videos[video_name]))]
        jaccard[video_name], fmeasure[video_name] = [], []

        for i in frame_order:

            image = rgb_loader(videos[video_name][i])
            flow = rgb_loader(flows[video_name][i])
            gt = binary_loader(gts[video_name][i])

            image = img_transform(image)
            flow = img_transform(flow)
            gt = mask_transform(gt)
            image = image.to(device).unsqueeze(0)
            flow = flow.to(device).unsqueeze(0)
            gt = gt.to(device).unsqueeze(0)

            image_ = F.interpolate(image, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
            flow_ = F.interpolate(flow, size=(inp_size, inp_size), mode='bilinear', align_corners=False)

            model.encoder.eval()
            with torch.no_grad():
                if model.is_depth:
                    pred, _ = model.encoder(image_, flow_)
                else:
                    pred = model.encoder(image_, flow_)

            pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred)
            j, f, _, _ = metric_func(pred, gt)
            pred[pred<0.5]=0
            pred[pred>=0.5]=1

            jaccard[video_name].append(j)
            fmeasure[video_name].append(f)
        # break
    j_mean, f_mean = [], []
    for video_name in videos.keys():
        j_mean.append(np.mean(jaccard[video_name]))
        f_mean.append(np.mean(fmeasure[video_name]))
        # break
        print(video_name, np.mean(jaccard[video_name]), np.mean(fmeasure[video_name]))
    print(np.mean(j_mean), np.mean(f_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--eval_type', choices=['base', 'TTT-N', 'TTT-MWI', 'TTT-LTV'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_dir = args.model
    eval_type = args.eval_type

    frame_path = '/data/video_segmentation_dataset/LV2SEG/frame'
    flow_path = '/data/video_segmentation_dataset/LV2SEG/flow'
    gt_path = '/data/video_segmentation_dataset/LV2SEG/mask'

    metric_func = calc_vos

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(model_dir), strict=True)

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1, 1, 1])
    ])

    inp_size = config['model']['args']['inp_size']
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    pmd_aug = PhotoMetricDistortion()
    resize_aug = RandomResize(int(inp_size * 1.05), int(inp_size * 1.5))
    crop_aug = RandomCrop((inp_size, inp_size))

    videos = split_files(read_files(frame_path))
    flows = split_files(read_files(flow_path))
    gts = split_files(read_files(gt_path))

    device = torch.device("cuda")
    jaccard, fmeasure = {}, {}

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = 10
    learning_rate = 1e-5
    frame_gap = 10
    batch_size = 8
    print('config : epoch-' + str(epochs) + ' learning rate-' + str(learning_rate) +
          ' frame gap-' + str(frame_gap) + ' batch size-' + str(batch_size))

    if eval_type == 'base':
        baseline()
    elif eval_type == 'TTT-N':
        videottt_ttt_n()
    elif eval_type == 'TTT-MWI':
        videottt_ttt_mwi()
        # for davis/stv2
        # videottt_group_ttt_mwi()
    elif eval_type == 'TTT-LTV':
        videottt_ttt_ltv()
        # for davis/stv2
        # videottt_group_ttt_ltv()


