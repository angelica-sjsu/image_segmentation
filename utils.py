import torch
import torchvision
from dataset import FireSmokeDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def save_checkpoint(state, fname):
    print("=> Saving checkpoint")
    torch.save(state, fname)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dirs, train_mask_dirs, val_dirs, val_mask_dirs,
                train_transforms, val_transforms,
                batch_size, num_workers=4, pin_memory=True,num_classes=1 ):

    train_ds = FireSmokeDataset(train_dirs, train_mask_dirs, transform=train_transforms,num_classes=num_classes)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    val_ds = FireSmokeDataset(val_dirs, val_mask_dirs, transform=val_transforms,num_classes=num_classes)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader

def single_loader(img_dirs, mask_dirs, set_transforms,
                batch_size, num_workers=4, pin_memory=True, num_classes=1):


    ds = FireSmokeDataset(img_dirs, mask_dirs, transform=set_transforms,num_classes=num_classes)
    loader = DataLoader(ds,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)

    return loader


def check_accuracy(loader, model, device='cuda', num_classes=1, training=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    iou_elements = 0
    dice_elements = 0
    acc_elements = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if num_classes > 1:
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
            else:
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            intersection = torch.sum(torch.logical_and(preds, y))
            union = torch.sum(torch.logical_or(preds, y))
            # mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
            mask_sum = torch.sum(torch.abs(y)) + torch.sum(torch.abs(preds))

            smooth = .001
            iou_elements += (intersection + smooth) / (union + smooth)
            dice_elements += 2 * (intersection + smooth) / (mask_sum + smooth)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)


        mean_accuracy = num_correct/num_pixels
        mean_dice = dice_elements/len(loader)
        mean_iou = iou_elements/len(loader)
        print(f'Accuracy: {mean_accuracy:.2f}')
        print(f'Dice: {mean_dice:.2f}')
        print(f'IOU: {mean_iou:.2f}')
        if training:
            model.train()
            return mean_accuracy, mean_dice
        else:
            return mean_accuracy, mean_dice, mean_iou


def save_predictions_as_imgs(loader, model, folder='saved_images/', device="cuda", multiclass=False, training=True):
    model.eval()
    print("=> Saving Images")

    if not os.path.exists(folder):
        os.mkdir(folder)

    for idx, (x,y) in enumerate(loader):
        x =x.to(device=device)
        with torch.no_grad():
            if multiclass:
                preds = torch.argmax(model(x), dim=1)
            else:
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
        # copy preds to CPU for viz
        if multiclass:
            preds = preds.to(device='cpu')
            for p,tg in zip(preds, y):
                #plt.imshow(p)
                #plt.show()
                plt.imsave(f"{folder}/{idx:05}_pred.png", p)
                plt.imsave(f"{folder}/{idx:05}_target.png", tg)
        else:
            torchvision.utils.save_image(preds,
                                         f"{folder}/{idx:05}_pred.png")
            torchvision.utils.save_image(y.unsqueeze(1),
                                         f"{folder}/{idx:05}_target.png")
    if training:
        model.train()
