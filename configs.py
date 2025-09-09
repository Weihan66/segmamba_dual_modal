# configs.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = {
    "train_dir": str(ROOT / "data" / "train"),
    "val_dir": str(ROOT / "data" / "val"),
}

TRAIN = {
    "epochs": 200,
    "batch_size": 1,
    "lr": 1e-4,
    "num_workers": 4,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "save_dir": str(ROOT / "checkpoints"),
    "patch_size": (96, 96, 96),  # crop size for training
    "seed": 42,
}
