"""
Created on 2020/9/8

@author: Boyun Li
"""
import os
from torch.utils.data import Dataset
import random
from PIL import Image
import glob
import numpy as np
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from utils.image_utils import random_augmentation, crop_img
from utils.imresize import np_imresize


class TrainDataset(Dataset):
    def __init__(self, opt, length, patch_size, mode=0):
        super(TrainDataset, self).__init__()
        self._init_ids(opt, mode)
        self.length = length
        self.patch_size = patch_size
        self.mode = mode
        self.transform = Compose([
            ToPILImage(),
            RandomCrop(patch_size),
            ToTensor()
        ])
        self.grayscale = Compose([
            ToPILImage(),
            Grayscale(),
            ToTensor()
        ])

    def _init_ids(self, opt, mode):
        """
        This function aims to define the input of
        :param opt: options.
        :param mode: This param aims to define the mode of add haze.
                    Specially, if mode == 0: Use both indoor and outdoor clean images.
                               if mode == 1: Only use indoor clean images.
                               if mode == 2: Only use outdoor clean images.
        :return:
        """
        haze_ids = glob.glob(os.path.join(opt.rw_path + '*'))
        indoor_clear_ids = glob.glob(os.path.join(opt.indoor_path + '*'))
        outdoor_clear_ids = glob.glob(os.path.join(opt.outdoor_path + '*'))

        self.num_indoor = len(indoor_clear_ids)
        self.num_outdoor = len(outdoor_clear_ids)

        if mode == 0:
            self.clean_ids = indoor_clear_ids + outdoor_clear_ids
            self.haze_ids = haze_ids
        elif mode == 1:
            self.clean_ids = indoor_clear_ids
            self.haze_ids = haze_ids
        elif mode == 2:
            self.clean_ids = outdoor_clear_ids
            self.haze_ids = haze_ids

    def __getitem__(self, _):
        # id_ = self.ids[index].split('.')[0] + '.jpg'
        short_edge = 0
        while short_edge < self.patch_size:
            clean_id = random.randint(0, len(self.clean_ids) - 1)
            haze_id = random.randint(0, len(self.haze_ids) - 1)

            haze_img = crop_img(np.array(Image.open(self.haze_ids[haze_id]).convert('RGB')), base=16)
            haze_name = self.haze_ids[haze_id].split("/")[-1].split(".")[0]
            hazy_short_edge = np.min(haze_img.shape[:2])

            clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
            clean_short_edge = np.min(clean_img.shape[:2])
            short_edge = np.min([hazy_short_edge, clean_short_edge])

        name_tuple = ["indoor_", "outdoor_"]
        if self.mode == 0:
            if clean_id < self.num_indoor:
                clean_name = name_tuple[0] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
            else:
                clean_name = name_tuple[1] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
        elif self.mode == 1:
            clean_name = name_tuple[0] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
        elif self.mode == 2:
            clean_name = name_tuple[1] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]

        hazy_patch, clean_patch = random_augmentation(haze_img, clean_img)

        hazy_patch = self.transform(hazy_patch)
        clean_patch = self.transform(clean_patch)
        gray_patch = self.grayscale(clean_patch)

        return [haze_name, clean_name], hazy_patch, clean_patch, gray_patch

    def __len__(self):
        return self.length


class CleanDataset(Dataset):
    def __init__(self, opt, mode=2):
        super(CleanDataset, self).__init__()
        self._init_ids(opt, mode)
        self.mode = mode
        self.transform = ToTensor()

    def _init_ids(self, opt, mode):
        """
        This function aims to define the input of
        :param opt: options.
        :param mode: This param aims to define the mode of add haze.
                    Specially, if mode == 0: Use both indoor and outdoor clean images.
                               if mode == 1: Only use indoor clean images.
                               if mode == 2: Only use outdoor clean images.
        :return:
        """
        indoor_clear_ids = glob.glob(os.path.join(opt.indoor_path + '*'))
        outdoor_clear_ids = glob.glob(os.path.join(opt.outdoor_path + '*'))

        self.num_indoor = len(indoor_clear_ids)
        self.num_outdoor = len(outdoor_clear_ids)

        if mode == 0:
            self.clean_ids = indoor_clear_ids + outdoor_clear_ids
        elif mode == 1:
            self.clean_ids = indoor_clear_ids
        elif mode == 2:
            self.clean_ids = outdoor_clear_ids

    def __getitem__(self, clean_id):
        # clean_id = random.randint(0, len(self.clean_ids) - 1)
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)

        name_tuple = ["indoor_", "outdoor_"]
        if self.mode == 0:
            if clean_id < self.num_indoor:
                clean_name = name_tuple[0] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
            else:
                clean_name = name_tuple[1] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
        elif self.mode == 1:
            clean_name = name_tuple[0] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]
        elif self.mode == 2:
            clean_name = name_tuple[1] + self.clean_ids[clean_id].split("/")[-1].split(".")[0]

        return clean_name, self.transform(clean_img)

    def __len__(self):
        return len(self.clean_ids)


class HazyDataset(Dataset):
    def __init__(self, opt, length):
        super(HazyDataset, self).__init__()
        self._init_ids(opt)
        self.length = length
        self.transform = ToTensor()
        self.haze_id = 0

    def _init_ids(self, opt):
        """
        This function aims to define the input of
        :param opt: options.
        :return:
        """
        haze_ids = glob.glob(os.path.join(opt.rw_path + '*'))

        self.haze_ids = haze_ids

    def __getitem__(self, _):
        haze_img = crop_img(np.array(Image.open(self.haze_ids[self.haze_id]).convert('RGB')), base=16)
        haze_name = self.haze_ids[self.haze_id].split("/")[-1].split(".")[0]
        self.haze_id += 1
        if self.haze_id == len(self.haze_ids):
            self.haze_id = 0

        return haze_name, self.transform(haze_img)

    def __len__(self):
        return self.length
