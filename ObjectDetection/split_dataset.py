import os
import shutil
import random
import yaml
from pathlib import Path

def main():
    dataset_dir = Path("dataset").resolve()
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print("Dataset not found! Make sure you run make_dataset.py first.")
        return

    # Get all flats images in the directory (ignore if already split)
    images = [f for f in img_dir.glob("*.jpg") if f.is_file()]
    if not images:
        print("No images found to split (they might already be in train/val/test folders).")
        return

    print(f"Found {len(images)} images to split.")
    
    # Shuffle dataset
    random.seed(42)
    random.shuffle(images)

    # Split sizes (80% train, 10% val, 10% test)
    total = len(images)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        split_img_dir = img_dir / split_name
        split_lbl_dir = lbl_dir / split_name
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            # Move image
            shutil.move(str(img_path), str(split_img_dir / img_path.name))
            
            # Find and move corresponding label
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(split_lbl_dir / lbl_path.name))

        print(f"Moved {len(split_images)} files to {split_name} split.")

    # Create the data.yaml file needed for YOLO training
    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 7,
        "names": {
            0: "PuzzleBase",
            1: "PuzzleCircle",
            2: "PuzzleSquare",
            3: "PuzzleTriangle",
            4: "cup",
            5: "book",
            6: "pen"
        }
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated successfully! YOLO configuration file saved at: {yaml_path}")

if __name__ == "__main__":
    main()
