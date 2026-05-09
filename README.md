# The Grippy Grabber

The Grippy Grabber is a simulated robotic pick-and-place system using the Franka Emika Panda arm. It utilizes MuJoCo for physics simulation, IKFast for inverse kinematics, and YOLO for object detection and pose estimation. The system orchestrates obstacle avoidance, grasping, and placement using a finite state machine.

## Prerequisites

- Python 3.10 or higher
- Git

## Installation

It is strongly recommended to use a Python virtual environment to manage dependencies locally.

1. **Navigate to the project directory:**
   ```bash
   cd The-Grippy-Grabber
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install the required packages:**
   Install the necessary libraries (`mujoco`, `opencv-python`, `ultralytics`) by running:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation

Ensure your virtual environment is activated, then launch the main task manager and simulation by running:

```bash
python main.py
```

This will initialize the MuJoCo viewer and simultaneously open a window displaying the perception camera feeds. The arm will automatically begin exploring and grasping objects according to the finite state machine logic.

## Object Detection Training Pipeline

The project includes an automated pipeline for generating synthetic data from the MuJoCo simulation environment and training a YOLO object detection model. All relevant scripts are located in the `ObjectDetection/` directory.

### 1. Generating the Dataset

To generate synthetic images and auto-labeled bounding boxes using the simulation:

```bash
cd ObjectDetection
python make_dataset.py
```

This script will run the simulation, place randomized objects throughout the workspace, and capture images. Bounding box data is calculated automatically from the geometry properties and exported into YOLO annotation format.

### 2. Splitting the Dataset

After creating the dataset, divide the collected images and label files into proper train and validate splits:

```bash
python split_dataset.py
```

Outputs will be correctly distributed into the `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val` directories.

### 3. Training the Model

Train the YOLO model on the newly generated dataset:

```bash
python train_yolo.py
```

This script invokes the Ultralytics training procedure outlined in `dataset/data.yaml`. Once training completes, the best performing model weights will be saved to the `runs/` directory output path. You can move the resulting `best.pt` file to the root project directory to implement it in the primary simulation pipeline.