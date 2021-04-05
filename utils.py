import torch
import torchvision
from dataset import FireSmokeDataset
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt


def save_checkpoint(state, fname):
    print("=> Saving checkpoint")
    torch.save(state, fname)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dirs, train_mask_dirs, val_dirs, val_mask_dirs,
                train_transforms, val_transforms,
                batch_size, num_workers=4, pin_memory=True,is_bin=True):

    train_ds = FireSmokeDataset(train_dirs, train_mask_dirs, transform=train_transforms,isbin=is_bin)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    val_ds = FireSmokeDataset(val_dirs, val_mask_dirs, transform=val_transforms,isbin=is_bin)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda', multiclass=False):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if multiclass:
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
            else:
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum()+1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score/len(loader), num_correct/num_pixels


def save_predictions_as_imgs(loader, model, folder='saved_images/', device="cuda", multiclass=False):
    model.eval()
    print("=> Saving Images")
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
    model.train()
