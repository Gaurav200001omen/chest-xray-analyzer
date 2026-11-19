from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ChestXRayDataset, get_transforms
from model import ChestXRayResNetSE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("outputs/best_model.pth", map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    _, eval_tf = get_transforms(img_size)
    test_ds = ChestXRayDataset("data/splits/test.csv", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = ChestXRayResNetSE(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1))
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("\n=== Test Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Chest X-Ray Analyzer - Test Confusion Matrix")
    plt.tight_layout()

    out_path = Path("outputs/confusion_matrix.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()