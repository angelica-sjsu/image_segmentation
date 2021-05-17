# @Angelica
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet_model import UNET
from utils import single_loader, load_checkpoint
import numpy as np
import matplotlib.pyplot as plt
from utils import single_loader, load_checkpoint, check_accuracy, save_predictions_as_imgs





DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BIN_CHK = 'bin_checkpoint.pth.tar'
MACRO_CHK = 'macro.pth.tar'
MACRO_RETRAIN_CHK = 'macro_retrain.pth.tar'
# MICRO_CHK = 'micro.pth.tar'
# MICRO_RETRAIN_CHK = 'micro_retrain.pth.tar'
IMG_DIR = ['ss_test_voc/tim/JPEGImages']
MASK_DIR = ['ss_test_voc/tim/SegmentationClassPNG']

# model = UNET(in_channels=3, out_channels=4).to(DEVICE)
# load_checkpoint(torch.load(BIN_CHECKPOINT), model)

transforms = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

checkpoints = [BIN_CHK, MACRO_CHK, MACRO_RETRAIN_CHK]
names = ['BIN', '4-CL', '4CL-RT']

# metrics
accuracies = []
dices = []
ious = []

for chk in checkpoints:
    if 'bin' in chk:
        num_classes = 1
        MULTICLASS = False
    elif 'micro' in chk:
        num_classes = 6
        MULTICLASS = True
    else:
        num_classes = 4
        MULTICLASS = True


    # load model
    model = UNET(in_channels=3, out_channels=num_classes)
    load_checkpoint(torch.load(chk, map_location=DEVICE), model)
    model = model.to(DEVICE)

    # dataloader
    loader = single_loader(IMG_DIR, MASK_DIR, set_transforms=transforms, batch_size=1, num_classes=num_classes)

    #check accuracy
    acc, dice, iou = check_accuracy(loader, model, DEVICE, num_classes, training=False)
    accuracies.append(acc)
    dices.append(dice)
    ious.append(iou)

    save_predictions_as_imgs(loader, model, f'{chk}_images', DEVICE, training=False, multiclass=MULTICLASS)

#visualize
# N = 3
# ind = np.arange(len(checkpoints))
# width = 0.25
#
# bar1 = plt.bar(ind, accuracies, width, color='r')
# bar2 = plt.bar(ind + width, dices, width, color='g')
# bar3 = plt.bar(ind + width * 2, ious, width, color='b')
#
# plt.xlabel("Models")
# plt.ylabel('Scores')
# plt.title("Model Scores \n RT=retrained model from loaded BIN MODEL")
#
# plt.xticks(ind + width, names)
# plt.legend((bar1, bar2, bar3), ('Accuracy', 'Dice', 'IOU'))
# plt.show()

print(names, accuracies, dices, ious)


