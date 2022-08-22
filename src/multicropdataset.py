# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        s = 1.0
        for i in range(len(size_crops)):
            randomresizedcrop = A.RandomResizedCrop(
                size_crops[i], size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i])
            )

            main_transform = [
                randomresizedcrop,
                # Color Jitter
                A.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=5, sigma_limit=(0.1, 2.), p=0.5), 
                A.HorizontalFlip(),
                A.ToFloat(255),
                A.Normalize(mean=mean, std=std, max_pixel_value=1),
                ToTensorV2(),
            ]

            trans.extend([A.Compose(main_transform)] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(np.array(image)), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops
