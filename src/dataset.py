import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def apply_clahe(gray_np: np.ndarray) -> np.ndarray:
    """Enhance lung tissue contrast on grayscale X-rays."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_np)


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, eval_tf


class ChestXRayDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gray = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to load image: {row['path']}")

        enhanced = apply_clahe(gray)
        pil_img = Image.fromarray(enhanced, mode="L")

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def preprocess_pil_for_inference(pil_img: Image.Image, img_size: int = 224):
    """Returns (input_tensor [1,1,H,W], enhanced_gray_224 [H,W] uint8)."""
    gray = np.array(pil_img.convert("L"))
    enhanced = apply_clahe(gray)
    enhanced_resized = cv2.resize(enhanced, (img_size, img_size), interpolation=cv2.INTER_AREA)

    _, eval_tf = get_transforms(img_size)
    tensor = eval_tf(Image.fromarray(enhanced, mode="L")).unsqueeze(0)
    return tensor, enhanced_resized