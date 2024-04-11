# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np
import torch

# ----------------------------------------------------------------------------
import utils


class AAPMDataset:
    def __init__(self, root,  filenames, resolution=512):
        self._path = root
        self._filenames = [_.strip().split('|') for _ in open(filenames, 'r').readlines()]
        self._resolution = (resolution, resolution)

    def __getitem__(self, idx):

        filename = self._filenames[idx]
        if 'body' in filename[0]:
            l = 1
        elif 'head' in filename[0]:
            l = 0
        else:
            raise NotImplementedError
        metalart, mask, nometal = [utils.load(_, self._resolution) for _ in filename]
        mask = mask.astype('bool')
        x = utils.normalize(metalart)
        x[..., mask] = 0
        y = utils.normalize(nometal)
        return x, y, l

    def __len__(self):
        return len(self._filenames)
