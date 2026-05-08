import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # Load YOLO Model
    print("Loading YOLO model...")
    yolo_model = YOLO("best.pt")  # Use the trained model

    # Load the MuJoCo model and data from the XML scene
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    # Initialize offscreen renderers to extract RGB-D data
    width, height = 640, 480
    rgb_renderer = mujoco.Renderer(model, height=height, width=width)
    
    depth_renderer = mujoco.Renderer(model, height=height, width=width)
    depth_renderer.enable_depth_rendering()

    print("Launching native MuJoCo viewer. Close the window to exit.")
    print("Tip: Expand the right side panel in the viewer to change the camera view.")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (width, height)) # 10 FPS matching render_fps

    # Launch the passive MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_render_time = data.time
        render_fps = 10 # Extract RGB-D at 10 frames per second
        
        while viewer.is_running():
            step_start = time.time()
            
            # Step the simulation forward
            mujoco.mj_step(model, data)
            
            # Sync the interactive viewer
            viewer.sync()
            
            # Extract RGB-D Data at realistic camera framerate
            if data.time - last_render_time >= 1.0 / render_fps:
                
                # RGB output (height, width, 3)
                rgb_renderer.update_scene(data, camera="gripper_camera")
                rgb_image = rgb_renderer.render()
                
                # Convert RGB to BGR for OpenCV and YOLO
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Run YOLO inference
                results = yolo_model(bgr_image, verbose=False)
                
                # Annotate and show the frame
                annotated_frame = results[0].plot()
                cv2.imshow("Gripper Camera YOLO", annotated_frame)
                cv2.waitKey(1)
                
                # Save the frame to the video video
                out.write(annotated_frame)
                
                # Depth output (height, width) mapping distances in meters
                depth_renderer.update_scene(data, camera="gripper_camera")
                depth_image = depth_renderer.render()
                
                last_render_time = data.time
            
            # Simple time keeping to avoid running too fast 
            # (MuJoCo default timestep is usually 0.002s)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    # Release the video writer and close windows
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
