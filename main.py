import mujoco
import mujoco.viewer
import time
from camera_streamer import CameraStreamer

def main():
    # Load the MuJoCo model and data from the XML scene
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    # Initialize the background camera streamer
    streamer = CameraStreamer(model, camera_name="gripper_depth_camera", fps=30)

    print("Launching viewer. Close the window to exit.")
    
    last_sync_time = data.time

    # Launch the passive MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            
            # Step the simulation forward
            mujoco.mj_step(model, data)
            
            # Sync the interactive viewer at 60 FPS
            if data.time - last_sync_time >= 1.0 / 60.0:
                viewer.sync()
                last_sync_time = data.time

            # Update the lightweight external camera stream
            streamer.update(data)

if __name__ == "__main__":
    main()