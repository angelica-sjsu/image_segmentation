import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage import io


class FireSmokeDataset(Dataset):
    def __init__(self,img_paths, mask_paths, transform=None, isbin=False):
        #self.root = root
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.rgb_map = {
            0: (0, 0, 0),
            1: (0, 0, 128),
            2: (0, 128, 0),
            3: (0, 128, 128),
            4: (128, 0, 0),
            5: (128, 0, 128),
            6: (128, 128, 0),
            7: (128, 128, 128)
        }
        self.isbin = isbin

        # iterate all possible paths containing masks and images
        self.imgs = []
        self.masks = []
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            img_names = sorted(list(glob.glob(img_path + '*/*')))
            self.imgs.extend(img_names)
            mask_names = sorted(list(glob.glob(mask_path + '*/*')))
            self.masks.extend(mask_names)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # get paths
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        # load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))


        #mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path))

        if self.isbin:
         mask = self.two_classes_encoding(mask)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


    def two_classes_encoding(self, mask):
        collapsed = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
        indices = np.where(mask!=0)
        collapsed[indices] = 1

        return collapsed


