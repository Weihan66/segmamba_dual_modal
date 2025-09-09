# data_loader.py
import os
import random
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from configs import DATA, TRAIN

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float32)

def normalize_intensity(x, clip=(0, 1000)):
    x = np.clip(x, clip[0], clip[1])
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

class PairedCTPETDataset(Dataset):
    """
    expects files like:
      <id>_CT.nii.gz
      <id>_PET.nii.gz
      <id>_GT.nii.gz
    in the dataset folder.
    """
    def __init__(self, root_dir, patch_size=(96,96,96), mode="train"):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        # find ids by CT files
        self.ct_files = sorted(glob(os.path.join(root_dir, "*_CT.nii*")))
        self.ids = [os.path.basename(p).replace("_CT.nii","").replace("_CT.nii.gz","") for p in self.ct_files]

    def __len__(self):
        return len(self.ids)

    def random_crop(self, ct, pet, gt):
        # ct, pet, gt: numpy arrays (D,H,W)
        dz, dy, dx = self.patch_size
        D, H, W = ct.shape
        if D <= dz:
            z0 = 0
            dz = D
        else:
            z0 = random.randint(0, D - dz)
        if H <= dy:
            y0 = 0; dy = H
        else:
            y0 = random.randint(0, H - dy)
        if W <= dx:
            x0 = 0; dx = W
        else:
            x0 = random.randint(0, W - dx)
        ct_c = ct[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        pet_c = pet[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        gt_c = gt[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        return ct_c, pet_c, gt_c

    def __getitem__(self, idx):
        idd = self.ids[idx]
        ct_path = os.path.join(self.root_dir, f"{idd}_CT.nii.gz")
        pet_path = os.path.join(self.root_dir, f"{idd}_PET.nii.gz")
        gt_path = os.path.join(self.root_dir, f"{idd}_GT.nii.gz")
        ct = load_nifti(ct_path)
        pet = load_nifti(pet_path)
        gt = load_nifti(gt_path)
        # Normalize
        ct = normalize_intensity(ct, clip=( -1000, 1000))  # CT Hounsfield typical clip
        pet = normalize_intensity(pet, clip=(0, np.percentile(pet, 99)))  # PET scale
        gt = (gt > 0.5).astype(np.float32)
        # crop
        if self.mode == "train":
            ct, pet, gt = self.random_crop(ct, pet, gt)
        # to tensor C,D,H,W
        ct = torch.from_numpy(ct)[None, ...].float()
        pet = torch.from_numpy(pet)[None, ...].float()
        gt = torch.from_numpy(gt)[None, ...].float()
        return {"id": idd, "ct": ct, "pet": pet, "gt": gt}

def create_loaders():
    train_ds = PairedCTPETDataset(DATA["train_dir"], patch_size=TRAIN["patch_size"], mode="train")
    val_ds = PairedCTPETDataset(DATA["val_dir"], patch_size=TRAIN["patch_size"], mode="val")
    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True, num_workers=TRAIN["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

if __name__ == "__main__":
    tr, va = create_loaders()
    print("Train size:", len(tr.dataset), "Val size:", len(va.dataset))
    batch = next(iter(tr))
    print(batch["ct"].shape, batch["pet"].shape, batch["gt"].shape)
