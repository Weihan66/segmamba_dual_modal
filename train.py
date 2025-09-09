# train.py
import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from configs import TRAIN
from data_loader import create_loaders
from utils import set_seed, save_checkpoint, dice_coeff, bce_dice_loss
from models.dual_net import DualModalNet

def train_epoch(model, loader, opt, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        ct = batch["ct"].to(device)
        pet = batch["pet"].to(device)
        gt = batch["gt"].to(device)
        opt.zero_grad()
        pred = model(ct, pet)
        loss = bce_dice_loss(pred, gt)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
    return running_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    dices = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for batch in pbar:
            ct = batch["ct"].to(device)
            pet = batch["pet"].to(device)
            gt = batch["gt"].to(device)
            pred = model(ct, pet)
            loss = bce_dice_loss(pred, gt)
            running_loss += loss.item()
            dices.append(dice_coeff(pred, gt))
    return running_loss / len(loader), sum(dices) / len(dices)

def main():
    set_seed(TRAIN["seed"])
    device = TRAIN["device"]
    os.makedirs(TRAIN["save_dir"], exist_ok=True)
    model = DualModalNet(ct_in_ch=1, pet_in_ch=1, base_ch=16, depth=4).to(device)
    opt = optim.Adam(model.parameters(), lr=TRAIN["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, verbose=True)
    train_loader, val_loader = create_loaders()
    best_dice = 0.0
    for epoch in range(1, TRAIN["epochs"]+1):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss, val_dice = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{TRAIN['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}")
        # save
        ckpt_path = os.path.join(TRAIN["save_dir"], f"epoch_{epoch}.pth")
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "val_dice": val_dice
        }, ckpt_path)
        # keep best
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
                "val_dice": val_dice
            }, os.path.join(TRAIN["save_dir"], "best.pth"))
    print("Training finished. Best val dice:", best_dice)

if __name__ == "__main__":
    main()
