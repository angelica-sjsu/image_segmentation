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
                    ]
test_masks_path = ['ss_test_voc/angelica/SegmentationClassPNG',
                   'ss_test_voc/courtney/SegmentationClassPNG',
                   'ss_test_voc/olivia/SegmentationClassPNG',
                   ]

# flags
LOAD_RETRAIN = True

# PATHS
TRAIN_IMG_DIRS = train_images_path
TRAIN_MASK_DIRS = train_masks_path
VAL_IMG_DIRS = test_images_path
VAL_MASK_DIRS = test_masks_path
# hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LOSS = nn.CrossEntropyLoss()
NUM_CLASSES = 4
BIN_CHECKPOINT = 'bin_checkpoint.pth.tar'
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
        targets = targets.to(device=DEVICE)

        # forward: float16
        # with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions,targets.long())


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


    train_loader, val_loader = get_loaders(TRAIN_IMG_DIRS,
                                           TRAIN_MASK_DIRS,
                                           VAL_IMG_DIRS,
                                           VAL_MASK_DIRS,
                                           train_transforms,
                                           val_transforms,
                                           BATCH_SIZE,
                                           NUM_WORKERS,
                                           PIN_MEMORY,
                                           NUM_CLASSES)

    scale = torch.cuda.amp.GradScaler()
    current_acc = 0
    current_dice = 0
    if LOAD_RETRAIN:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(torch.load(BIN_CHECKPOINT), model)
        layers = list(model.modules())
        #change final layer
        layers[0].final_conv = nn.Conv2d(64, NUM_CLASSES, kernel_size=1).to(DEVICE)
    else:
        model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, LOSS, scale)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # check_Acc
        dice, acc = check_accuracy(val_loader, model, device=DEVICE, num_classes=NUM_CLASSES)

        if acc >= current_acc and dice >= current_dice:
            current_acc = acc
            current_dice = dice
            # print_sample

            # TODO: check the naming of saved models
            if LOAD_RETRAIN:
                name = 'macro_retrain.pth.tar'
            else:
                name = 'macro.pth.tar'
            save_checkpoint(checkpoint, fname=name)

            folder_name = f'saved_images/'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            save_predictions_as_imgs(val_loader, model, folder=folder_name, device=DEVICE, multiclass=True)


if __name__ == "__main__":
    main()
    # TODO: make confusion matrix
    # TODO: evaluate model with videos
    # TODO:

