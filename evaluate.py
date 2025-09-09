# evaluate.py
import argparse
import os
import torch
from data_loader import create_loaders
from models.dual_net import DualModalNet
from utils import dice_coeff
from configs import TRAIN

def load_model(ckpt_path, device):
    model = DualModalNet(ct_in_ch=1, pet_in_ch=1, base_ch=16, depth=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def evaluate(ckpt, device):
    model = load_model(ckpt, device)
    _, val_loader = create_loaders()
    dices = []
    with torch.no_grad():
        for batch in val_loader:
            ct = batch["ct"].to(device)
            pet = batch["pet"].to(device)
            gt = batch["gt"].to(device)
            pred = model(ct, pet)
            dices.append(dice_coeff(pred, gt))
    print("Mean Dice on val:", sum(dices)/len(dices))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    device = TRAIN["device"]
    evaluate(args.ckpt, device)
