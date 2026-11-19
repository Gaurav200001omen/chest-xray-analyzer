from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw/COVID-19_Radiography_Dataset")
SPLIT_DIR = Path("data/splits")
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for label_idx, cls_name in enumerate(CLASSES):
        img_dir = RAW_DIR / cls_name / "images"
        if not img_dir.exists():
            img_dir = RAW_DIR / cls_name  # fallback if no nested /images folder

        if not img_dir.exists():
            raise FileNotFoundError(f"Could not find class folder: {img_dir}")

        files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        print(f"Found {len(files):>5} images for {cls_name}")
        for p in files:
            records.append({"path": str(p), "label": label_idx, "class_name": cls_name})

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No images found. Check your data/raw path.")

    # 70% train, 30% temp -> split temp 50/50 into 15% val and 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print(
        f"\nSaved splits to {SPLIT_DIR}/ -> "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )


if __name__ == "__main__":
    main()