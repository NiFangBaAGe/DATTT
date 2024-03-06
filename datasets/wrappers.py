import logging
class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        image, flow, gt = sample['image'], sample['flow'], sample['gt']
        depth = sample['depth']
        h, w = gt.shape

        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)),
                                interpolation=cv2.INTER_CUBIC)
                flow = cv2.resize(flow, (int(w * ratio), int(h * ratio)),
                                   interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth, (int(w * ratio), int(h * ratio)),
                                interpolation=cv2.INTER_CUBIC)

        left_top = (np.random.randint(0, h - self.output_size[0] + 1), np.random.randint(0, w - self.output_size[1] + 1))

        image_crop = image[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        flow_crop = flow[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        gt_crop = gt[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        depth_crop = depth[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]


        sample.update({'image': image_crop, 'flow': flow_crop, 'gt': gt_crop, 'depth': depth_crop})
        return sample
#

class RandomResize(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):
        h, w = random.randint(self.min, self.max), random.randint(self.min, self.max)

        sample['image'] = cv2.resize(sample['image'], (w, h), interpolation=cv2.INTER_CUBIC)
        sample['flow'] = cv2.resize(sample['flow'], (w, h), interpolation=cv2.INTER_CUBIC)
        sample['gt'] = cv2.resize(sample['gt'], (w, h), interpolation=cv2.INTER_NEAREST)
        sample['depth'] = cv2.resize(sample['depth'], (w, h), interpolation=cv2.INTER_CUBIC)

        return sample

import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
import torch.nn.functional as F
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def flow_loader(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # print('Reading %d x %d flo file' % (w, h))
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
        return data2D

@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                # transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                # transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

        self.image = dataset.image
        self.flow = dataset.flow
        self.gt = dataset.gt
        self.depth = dataset.depth

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = rgb_loader(self.image[idx])
        flow = rgb_loader(self.flow[idx])
        gt = binary_loader(self.gt[idx])
        depth = binary_loader(self.depth[idx])

        image = self.img_transform(image)
        flow = self.img_transform(flow)
        gt = self.mask_transform(gt)
        depth = transforms.ToTensor()(depth)
        depth = transforms.Resize((self.inp_size, self.inp_size), interpolation=Image.NEAREST)(depth)

        return {
            'image': image,
            'flow': flow,
            'gt': gt,
            'depth': depth
        }



@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

        self.pmd = PhotoMetricDistortion()
        self.random_resize = RandomResize(int(self.inp_size*1.05), int(self.inp_size*1.5))
        self.random_crop = RandomCrop((self.inp_size, self.inp_size))

        self.image = dataset.image
        self.flow = dataset.flow
        self.gt = dataset.gt
        self.depth = dataset.depth

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = rgb_loader(self.image[idx])
        flow = rgb_loader(self.flow[idx])
        gt = binary_loader(self.gt[idx])
        depth = binary_loader(self.depth[idx])

        # random filp
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        # sample = {'image': np.array(image), 'flow': np.array(flow), 'gt': np.array(gt),
        #           'depth': np.array(depth)}
        # sample = self.random_resize(sample)
        # sample = self.random_crop(sample)
        # image = Image.fromarray(sample['image'])
        # gt = Image.fromarray(sample['gt'])
        # flow = Image.fromarray(sample['flow'])
        # depth = Image.fromarray(sample['depth'])

        sample = [np.array(image)]
        sample = self.pmd(sample)
        image = Image.fromarray(sample[0])

        image = self.img_transform(image)
        flow = self.img_transform(flow)
        gt = self.mask_transform(gt)

        depth = transforms.ToTensor()(depth)
        depth = transforms.Resize((self.inp_size, self.inp_size), interpolation=Image.BICUBIC)(depth)

        return {
            'image': image,
            'flow': flow,
            'gt': gt,
            'depth': depth
        }

import mmcv
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
    # def __init__(self,
    #              brightness_delta=10,
    #              contrast_range=(0.2, 1.2),
    #              saturation_range=(0.2, 1.2),
    #              hue_delta=10):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_para(self):
        self.brightness_para = random.uniform(-self.brightness_delta, self.brightness_delta)
        self.contrast_para = random.uniform(self.contrast_lower, self.contrast_upper)
        self.saturation_para = random.uniform(self.saturation_lower, self.saturation_upper)
        self.hue_para = random.randint(-self.hue_delta, self.hue_delta)

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.uniform(0, 1) < 0.5:
            return self.convert(
                img,
                beta=self.brightness_para)
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.uniform(0, 1) < 0.5:
            return self.convert(
                img,
                alpha=self.contrast_para)
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.uniform(0, 1) < 0.5:
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=self.saturation_para)
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.uniform(0, 1) < 0.5:
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      self.hue_para) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        self.get_para()

        # random brightness
        for i in range(len(sample)):
            sample[i] = self.brightness(sample[i])


        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.uniform(0, 1)
        if mode < 0.5:
            for i in range(len(sample)):
                sample[i] = self.contrast(sample[i])

        # random saturation
        for i in range(len(sample)):
            sample[i] = self.saturation(sample[i])

        # random hue
        for i in range(len(sample)):
            sample[i] = self.hue(sample[i])

        # random contrast
        if mode > 0.5:
            for i in range(len(sample)):
                sample[i] = self.contrast(sample[i])

        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str