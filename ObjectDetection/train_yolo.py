import os
from ultralytics import YOLO

def main():
    # 1. Load a pre-trained YOLO model (nano size is fastest for testing, change to yolov8s.pt or yolov8m.pt if you need more accuracy)
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolo26n.pt")
    
    # 2. Define the path to the dataset
    data_yaml_path = os.path.abspath("dataset/data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: Could not find {data_yaml_path}. Make sure you ran 'split_dataset.py' first.")
        return

    print(f"\nStarting Transfer Learning using dataset config: {data_yaml_path}\n")

    # 3. Train the model using the synthetic dataset
    results = model.train(
        data=data_yaml_path,
        epochs=300,           # Set higher (e.g., 100) for better final performance
        imgsz=640,           # Model input resolution
        batch=16,            # Adjust batch size based on your GPU VRAM (decrease if you OOM)
        device=0,            # Force train on GPU (if set up)
        project="yolo_runs", # Output directory for logs and weights
        name="grippy_train"  # Name of this training run
    )
    
    print("\nTraining Complete! Best model weights saved at: yolo_runs/grippy_train/weights/best.pt")

if __name__ == "__main__":
    main()
