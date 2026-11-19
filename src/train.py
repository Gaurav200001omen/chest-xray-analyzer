from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import ChestXRayDataset, get_transforms, CLASSES
from model import ChestXRayResNetSE

IMG_SIZE = 224
BATCH_SIZE = 16       # Safe for 4 GB VRAM
ACCUM_STEPS = 2       # Effective batch size = 32
EPOCHS = 25
LR = 1e-3
NUM_WORKERS = 2       # Stable on Windows
OUTPUT_DIR = Path("outputs")


def compute_class_weights(train_csv: str, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index().values.astype(np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Training on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""))

    train_tf, eval_tf = get_transforms(IMG_SIZE)
    train_ds = ChestXRayDataset("data/splits/train.csv", transform=train_tf)
    val_ds = ChestXRayDataset("data/splits/val.csv", transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=True
    )

    model = ChestXRayResNetSE(num_classes=len(CLASSES), dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Custom ResNet-SE Parameters: {total_params:,}")

    class_weights = compute_class_weights("data/splits/train.csv", device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch:02d}/{EPOCHS}")
        for step, (images, labels) in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=f"{running_loss / step:.4f}")

        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Epoch {epoch:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = OUTPUT_DIR / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "img_size": IMG_SIZE,
                "val_f1": best_f1,
            }, ckpt_path)
            print(f"  -> Saved new best model to {ckpt_path} (Macro-F1: {best_f1:.4f})")


if __name__ == "__main__":
    main()