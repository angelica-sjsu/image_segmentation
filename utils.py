import torch
import torchvision
from dataset import FireSmokeDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dirs, train_mask_dirs, val_dirs, val_mask_dirs,
                batch_size, num_workers=4, pin_memory=True):

    train_ds = FireSmokeDataset(train_dirs, train_mask_dirs)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    val_ds = FireSmokeDataset(val_dirs, val_mask_dirs)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def check_accuracy(loader, model, device='cpu'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.permute(0,3,2,1)
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            #TODO: is this necessary??
            #preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score = (2*(preds*y).sum()) / ((preds+y).sum()+1e-8)

    # TODO: fix accuracy
    print(f"Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels)*100:.2f}")
    # dice
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder='saved_images/', device="cpu"):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.permute(0,3,2,1)
        x =x.to(device=device)
        with torch.no_grad():
            #preds = torch.sigmoid(model(x))
            preds = model(x)
            preds = preds.float()
        torchvision.utils.save_image(preds,
                                     f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1),
                                     f"{folder}/pred_{idx}.png")
        model.train()
