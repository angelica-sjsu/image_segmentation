import os
import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# paths
train_images_path = ['ss_train_voc/angelica/JPEGImages',
                     'ss_train_voc/courtney/JPEGImages',
                     'ss_train_voc/olivia/JPEGImages',
                     'ss_train_voc/tim/JPEGImages']
train_masks_path = ['ss_train_voc/angelica/SegmentationClassPNG',
                    'ss_train_voc/courtney/SegmentationClassPNG',
                    'ss_train_voc/olivia/SegmentationClassPNG',
                    'ss_train_voc/tim/SegmentationClassPNG']

test_images_path = ['ss_test_voc/angelica/JPEGImages',
                    'ss_test_voc/courtney/JPEGImages',
                    'ss_test_voc/olivia/JPEGImages',
                    'ss_test_voc/tim/JPEGImages']
test_masks_path = ['ss_test_voc/angelica/SegmentationClassPNG',
                   'ss_test_voc/courtney/SegmentationClassPNG',
                   'ss_test_voc/olivia/SegmentationClassPNG',
                   'ss_test_voc/tim/SegmentationClassPNG', ]

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 #32
NUM_EPOCHS = 10 #100
NUM_WORKERS = 2
IMG_HEIGHT = 256
IMG_WIDTH = 256
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIRS = train_images_path
TRAIN_MASK_DIRS = train_masks_path
VAL_IMG_DIRS = test_images_path
VAL_MASK_DIRS = test_masks_path


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)

        #forward: float16
        # with torch.cuda.amp.autocast():
        data = data.permute(0,3,2,1)
        predictions = model(data)
        loss = loss_fn(predictions,targets)

        #backward
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())


def main():
    # change out_channels for additional channels and change loss_fn to BCELoss
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIRS,
                                           TRAIN_MASK_DIRS,
                                           VAL_IMG_DIRS,
                                           VAL_MASK_DIRS,
                                           BATCH_SIZE,
                                           NUM_WORKERS,
                                           PIN_MEMORY)


    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check_Acc
        check_accuracy(val_loader, model, device=DEVICE)

        # print_samples
        folder_name = 'saved_images'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        save_predictions_as_imgs(val_loader, model, folder='saved_images/', device=DEVICE)


if __name__=="__main__":
    main()


