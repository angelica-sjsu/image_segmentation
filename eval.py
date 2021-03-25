# @Angelica
import torch
import torchvision
import albumentations as A   #pip install albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
from unet_model import UNET as nn
import os
import numpy as np
import torch.optim as optim
import sys


class Segmentation:

    def __init__(self):
        print('Parent Segmentation obj created...')

    def transform(self, read_from_path, write_to_path):
        print('Segmentation transform() not set!')


class UNET(Segmentation):
    def __init__(self, unet, path_to_model=''):
        self.model = unet
        self.model_path = path_to_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        self.load_checkpoint(torch.load(path_to_model, map_location='cpu'), self.model)

    def transform(self, read_from_path, write_to_path):
        if not os.path.exists(write_to_path):
            os.mkdir(write_to_path)

        transforms = A.Compose([
            #A.Resize(height=640, width=360),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

        # load image
        image = Image.open(read_from_path)
        image = np.array(image)
        transf = transforms(image=image)
        image = transf['image'].unsqueeze(0)

        # make model predict!!
        # self.model.eval()
        #torch.cuda.empty_cache()
        with torch.no_grad():
            image = image.to(device='cuda')
            preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds,
                                     f"{write_to_path}/unet_pred.png")


    def load_checkpoint(self,checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])




# TODO: test code!!

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn(in_channels=3, out_channels=1).to(DEVICE)
unet = UNET(model, 'checkpoint.pth.tar')
#unet.transform('Ognisko_ubt_0126.jpeg.jpeg','eval_images')
unet.transform('rawImage.png','eval_images')
sys.exit()