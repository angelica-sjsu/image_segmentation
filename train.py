import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
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

# flags
LOAD_MODEL = False
LOAD_RETRAIN = True
MULTICLASS = True
# PATHS
TRAIN_IMG_DIRS = train_images_path
TRAIN_MASK_DIRS = train_masks_path
VAL_IMG_DIRS = test_images_path
VAL_MASK_DIRS = test_masks_path
# hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOSS = nn.CrossEntropyLoss() if MULTICLASS else nn.BCEWithLogitsLoss()
OUTCHANNELS = 8 if MULTICLASS else 1
BIN_CHECKPOINT = 'checkpoint.pth.tar'
MC_CHECKPOINT = 'mc_checkpoint.pth.tar'
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMG_HEIGHT = 640
IMG_WIDTH = 360
PIN_MEMORY = True


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        if MULTICLASS:
            targets = targets.to(device=DEVICE)
        else:
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward: float16
        with torch.cuda.amp.autocast():
            predictions = model(data)
            if MULTICLASS:
                loss = loss_fn(predictions,targets.long())
            else:
                loss = loss_fn(predictions, targets)


        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())


def main():
    # transformations
    train_transforms = A.Compose([
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIRS,
                                           TRAIN_MASK_DIRS,
                                           VAL_IMG_DIRS,
                                           VAL_MASK_DIRS,
                                           train_transforms,
                                           val_transforms,
                                           BATCH_SIZE,
                                           NUM_WORKERS,
                                           PIN_MEMORY)

    scale = torch.cuda.amp.GradScaler()
    current_acc = 0
    current_dice = 0
    if LOAD_RETRAIN:
        load_checkpoint(torch.load(BIN_CHECKPOINT), model)
        layers = list(model.modules())
        #change final layer
        layers[0].final_conv = nn.Conv2d(64, OUTCHANNELS, kernel_size=1)
        # move model to device: useful when GPU is available
        model.to(device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, LOSS, scale)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # check_Acc
        dice, acc = check_accuracy(val_loader, model, device=DEVICE, multiclass=MULTICLASS)

        if acc >= current_acc and dice >= current_dice:
            current_acc = acc
            current_dice = dice
            # print_sample
            if MULTICLASS:
                name = MC_CHECKPOINT
            elif not MULTICLASS:
                name = BIN_CHECKPOINT
            else:
                name = 'mc_loadtraine.pth.tar'
            save_checkpoint(checkpoint, fname=name)

            folder_name = f'saved_images/'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            save_predictions_as_imgs(val_loader, model, folder=folder_name, device=DEVICE, multiclass=MULTICLASS)


if __name__ == "__main__":
    main()


