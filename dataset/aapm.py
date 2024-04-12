# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import glob

import cv2
import os
import numpy as np
import torch

# ----------------------------------------------------------------------------
import utils


class AAPMDataset:
    def __init__(self, root,  filenames, resolution=512, use_labels=False, transform=True):
        self._root = root
        self._filenames = [
            _.strip().split('|') for _ in sum(
              [open(f).readlines() for f in glob.glob(root + '/' + filenames)], []
            )
        ]
        self._resolution = resolution
        self._name = 'aapm'
        self._use_labels = use_labels
        self._transform = transform

    @property
    def name(self):
        return self._name

    @property
    def has_labels(self):
        return True

    def get_label(self, idx):
        filename = self._filenames[idx]
        if 'body' in filename[0]:
            label = 1
        elif 'head' in filename[0]:
            label = 0
        else:
            raise NotImplementedError
        return label

    @property
    def resolution(self):
        return self._resolution

    @property
    def label_dim(self):
        if self._use_labels:
            return 2
        else:
            return 0

    @property
    def num_channels(self):
        return 3
    @property
    def image_shape(self):
        return 3, 512, 512

    def __getitem__(self, idx):

        filename = self._filenames[idx]
        if 'body' in filename[0]:
            l = 1
        elif 'head' in filename[0]:
            l = 0
        else:
            raise NotImplementedError
        metalart, mask, nometal = [utils.load(f'{self._root}/{_}', self._resolution) for _ in filename]
        if self._transform:
            if np.random.randint(0, 2) == 1:
                metalart, mask, nometal = [cv2.flip(_, 1) for _ in [metalart, mask, nometal]]
            rotate = [
                None, cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
            ][np.random.randint(0, 4)]
            if rotate is not None:
                metalart, mask, nometal = [cv2.rotate(_, rotate) for _ in [metalart, mask, nometal]]

        x = utils.normalize(metalart)
        mask = mask.astype("bool")
        x[..., mask] = 0
        y = utils.normalize(nometal)
        return x, y, l

    def __len__(self):
        return len(self._filenames)
