import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class FireSmokeDataset(Dataset):
    def __init__(self,img_paths, mask_paths, transform=None):
        #self.root = root
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # iterate all possible paths containing masks and images
        self.imgs = []
        self.masks = []
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            img_names = glob.glob(img_path + '*/*')
            self.imgs.extend(img_names)
            mask_names = glob.glob(mask_path + '*/*')
            self.masks.extend(mask_names)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # get paths
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        # load image and mask
        img = cv2.imread(img_path)
        img = np.array(img)
        mask = cv2.imread(mask_path)

        # transform mask into an array and collapse channels
        mask = np.array(mask)
        mask = self.collapse_channel(mask)

        # resize_images
        img = self.resize(img)
        mask = self.resize(mask)

        # normalize
        img = self.normalize(img)

        # # augmentation with albumentations
        # if self.transform is not None:
        #     augmentations = self.transform(image=img, mask=mask)
        #     img = augmentations['images']
        #     mask = augmentations['mask']

        #TODO: add image name
        return img.astype(np.float32), mask.astype(np.float32)



    def collapse_channel(self, mask):
        additional_encoding = np.array([[128, 0, 0], [128, 128, 128],
                                       [128, 0, 128], [128, 128, 0]])
        # create an array that will host the transformed mask
        collapsed = np.zeros((mask.shape[0], mask.shape[1]))
        # find all unique elements in the mask
        unique_arr = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        for idx, channel in enumerate(unique_arr):
            indices = np.where(np.all(mask==channel, axis=-1))
            # see if the current mask has additional encoding
            present = np.where(np.all(additional_encoding == channel, axis=-1))
            if len(present[0]) != 0:
                collapsed[indices] = 2
            else:
                collapsed[indices] = idx

        return collapsed

    def resize(self, img,dim=256):
        return cv2.resize(img, (dim, dim))

    def normalize(self, img):
        im = img.astype(float)
        return im/255.0


