import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data



import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_edge(image, sigma):
    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))

    return edge, gray_image

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class GTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_env = None  # environment for lmdb
        self.GT_size = opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        print("dataset length: {}".format(len(self.GT_paths)))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.GT_env is None:
                self._init_lmdb()

        GT_path = None
        GT_size = self.opt["GT_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        if self.opt["phase"] == "train":
            H, W, C = img_GT.shape

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment(
                img_GT,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )
        else:
            img_GT = cv2.resize(
                np.copy(img_GT), (256, 256), interpolation=cv2.INTER_LINEAR
            )

        # change color space if necessary
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]
        print(GT_path)
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        
        GT_edge,GT_gray = image_to_edge(img_GT, sigma=2.)
        return {"GT": img_GT, "GT_path": GT_path, "GT_edge": GT_edge, "GT_gray": GT_gray}

    def __len__(self):
        return len(self.GT_paths)
