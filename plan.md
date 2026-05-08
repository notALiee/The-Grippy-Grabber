# 🤖 The Grippy Grabber: Autonomous Manipulation Pipeline

## 🎯 Project Goal
To develop a fully autonomous robotic pipeline in MuJoCo that uses a Panda arm to:
1. Grasp and stack books.
2. Pick up pens and place them into cups.
3. Group puzzle pieces tightly together.

## 🏗️ Architecture & Implementation Plan

### Phase 1: Perception & 3D Localization
**Goal:** Translate 2D YOLO bounding boxes into 3D world coordinates.
- [ ] **2D Object Detection:** Run YOLO inference on the `gripper_camera` RGB feed to find objects and their center pixels `(u, v)`.
- [ ] **Depth Extraction:** Sample the MuJoCo depth buffer at `(u, v)` to get the object's distance from the camera `(z)`.
- [ ] **De-projection:** Convert the `(u, v, z)` pixel data into 3D camera-relative coordinates using the camera's intrinsic matrix (Focal Length / FOV).
- [ ] **Coordinate Transformation:** Apply the arm's Forward Kinematics (using MuJoCo's `data.cam_xpos` and `data.cam_xmat`) to transform the camera coordinates into absolute global World Coordinates `(x, y, z)`.

### Phase 2: Grasp Pose Synthesis
**Goal:** Determine the 6-DoF (Degrees of Freedom) orientation required to pick up an object safely.
- [ ] **Heuristic Grasping (Iterative Step 1):** 
  - Books & Cups: Top-down grasp (Pitch = -90°).
  - Pens: Side grasp aligned with the longest axis of the YOLO bounding box.
- [ ] **Deep Learning Grasping (Iterative Step 2):** Integrate *GraspAnything* or *Contact-GraspNet* taking the cropped RGB-D object as input and outputting exact gripper quaternions.

### Phase 3: Task Logic & State Machine
**Goal:** Program the logic to complete specific chores autonomously.
- [ ] **Build a Finite State Machine (FSM):** states include `SCANNING`, `PLANNING`, `APPROACHING`, `GRASPING`, `TRANSPORTING`, `PLACING`.
- [ ] **Book Stacking Logic:** Identify the lowest book in the Z-axis (Base Book) and sequentially stack other books on top `(Base_Z + Book_Thickness)`.
- [ ] **Pen & Cup Logic:** Pair a Pen target with a Cup target; drop the pen slightly above the cup's Z-centroid.
- [ ] **Puzzle Assembly Logic:** Find the `PuzzleBase`, calculate a bounding box around it, and calculate drop-off offsets for the Circle, Square, and Triangle pieces.

### Phase 4: Motion Planning (Collision Avoidance)
**Goal:** Compute a safe joint-trajectory from the current arm pose to the Target Grasp Pose without knocking over other objects.
- [ ] **Integrate Path Planner:** Setup Rapidly-exploring Random Trees (RRT) using an external library like OMPL (Open Motion Planning Library) or `roboticstoolbox-python`.
- [ ] **Collision Checking:** Hook the planner's state validity checker into MuJoCo's native collision detection engine.
- [ ] **Waypoint Generation:** Define safe traversal points: `Current Pose` ➔ `Pre-Grasp (10cm above)` ➔ `Grasp` ➔ `Pre-Grasp` ➔ `Place location`.

### Phase 5: Execution & Inverse Kinematics (IK)
**Goal:** Actuate the MuJoCo simulated robot along the planned path.
- [ ] **Connect IK Solver:** Pass the RRT generated 3D Cartesian waypoints into the IK algorithm to yield joint target angles.
- [ ] **Actuation loop:** Send joint position commands via MuJoCo's `data.ctrl` to the Panda arm actuators smoothly over time frame timesteps.