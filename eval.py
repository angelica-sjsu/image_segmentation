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
from time import time
import matplotlib.pyplot as plt
from skimage import io
import cv2



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
            # preds = torch.sigmoid(model(image))
            # preds = (preds > 0.5).float()
            preds = torch.argmax(model(image), dim=1)

        # torchvision.utils.save_image(preds,
        #                            f"{write_to_path}/3c_unet_pred.png")
        preds = preds.to(device='cpu')
        image = image.to(device='cpu')
        for p in preds:
            # plt.imshow(p)
            # plt.show()
            plt.imsave(f"{write_to_path}/3cscratch_1_pred.png", p)

        image1 = Image.open(read_from_path)
        image2 = Image.open(f"{write_to_path}/3cscratch_1_pred.png")

        image1 = image1.convert('RGBA')
        plt.imshow(image1)
        plt.show()
        image2 = image2.convert('RGBA')
        plt.imshow(image2)
        plt.show()

        # When two images are merged, follow the formula: blended_img = img1 * (1 – alpha) + img2* alpha
        image = Image.blend(image1, image2, 0.3)
        image.save("test.png")
        # image.show()
        plt.imshow(image)
        plt.show()

    def transform_frame(self, frame, write_to_path,idx):

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
        # image = Image.open(read_from_path)
        image = np.array(frame)
        cp_image = np.copy(image)
        transf = transforms(image=image)
        image = transf['image'].unsqueeze(0)

        # make model predict!!
        # self.model.eval()
        #torch.cuda.empty_cache()
        with torch.no_grad():
            image = image.to(device='cuda')
            # preds = torch.sigmoid(model(image))
            # preds = (preds > 0.5).float()
            preds = torch.argmax(model(image), dim=1)

        # torchvision.utils.save_image(preds,
        #                            f"{write_to_path}/3c_unet_pred.png")
        preds = preds.to(device='cpu')
        image = image.to(device='cpu')
        for p, img in zip(preds, image):
            # plt.imshow(p)
            # plt.show()
            plt.imsave(f'{write_to_path}/{idx:05}_frame.png', cp_image)
            plt.imsave(f"{write_to_path}/{idx:05}.png", p)

        image1 = Image.open(f'{write_to_path}/{idx:05}_frame.png')
        image2 = Image.open(f"{write_to_path}/{idx:05}.png")

        image1 = image1.convert('RGBA')
        image2 = image2.convert('RGBA')

        # When two images are merged, follow the formula: blended_img = img1 * (1 – alpha) + img2* alpha
        image = Image.blend(image1, image2, 0.3)
        image.save(f"eval/{idx:05}.png")
        # image.show()
        # plt.imshow(image)
        # plt.show()


    def load_checkpoint(self,checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])




# TODO: evaluate model with a video

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn(in_channels=3, out_channels=4).to(DEVICE)
#unet = UNET(model, 'checkpoint.pth.tar')
unet = UNET(model, 'mc_loadtraine.pth.tar')
#unet = UNET(model, '3c_scratch.pth.tar')

start = time()
#unet.transform('Ognisko_ubt_0126.jpeg.jpeg','eval_images')
unet.transform('5406.jpg','eval_images')
end = time()
print(f'Inference time: {end-start} seconds')
sys.exit()
# video = cv2.VideoCapture('train-3.avi')
# idx = 0
# while(True):
#     # Capture frame-by-frame
#     ret, frame = video.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     pil_image = Image.fromarray(gray)
#     start = time()
#     unet.transform_frame(pil_image, 'eval_images', idx)
#     end = time()
#     print(f'Inferece time for {idx}: {end-start}')
#     idx += 1
# sys.exit()