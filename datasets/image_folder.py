
import os
import json
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register
import cv2

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path, split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        videos = {}
        for filename in filenames:
            cat = filename.split('/')[-1].split('_')[0]
            if cat not in videos.keys():
                videos[cat] = []
            videos[cat].append(os.path.join(path, filename))

        self.files = []
        for video_name in videos.keys():
            for frame in videos[video_name]:
                self.append_file(frame)

    def append_file(self, file):
        self.files.append(file)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x

@register('paired-image-folders')
class PairedImageFolders(Dataset):
    def __init__(self, image, flow, gt, depth, **kwargs):
        self.image = ImageFolder(image, **kwargs)
        self.flow = ImageFolder(flow, **kwargs)
        self.gt = ImageFolder(gt, **kwargs)
        self.depth = ImageFolder(depth, **kwargs)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.flow[idx], self.gt[idx], self.depth[idx]