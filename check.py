import os
import cv2
import mujoco
import numpy as np

# Load model
xml_path = os.path.join(os.path.dirname(__file__), "world.xml")

model = mujoco.MjModel.from_xml_path("/home/eevee/The-Grippy-Grabber/panda_mujoco/world.xml")
data = mujoco.MjData(model)

# Set initial Panda pose
qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

for i in range(7):
    joint_name = f"panda_joint{i+1}"
    data.joint(joint_name).qpos = qpos0[i]

mujoco.mj_forward(model, data)

# Create renderer
renderer = mujoco.Renderer(model, 640, 480)

# Camera name from XML
camera_name = "gripper_camera"

cv2.namedWindow("MuJoCo Camera", cv2.WINDOW_NORMAL)

while True:

    # Step simulation
    mujoco.mj_step(model, data)

    # Render camera
    renderer.update_scene(data, camera=camera_name)

    frame = renderer.render()

    # Convert RGB -> BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("MuJoCo Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()