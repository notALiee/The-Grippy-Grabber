import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2

from perception import PerceptionSystem
from control import PandaController

def main():
    # Load YOLO Model using Perception Class
    print("Initializing perception system...")
    perception = PerceptionSystem("best.pt")

    # Load the MuJoCo model and data from the XML scene
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    # Initialize the Robot Controller
    robot = PandaController(model, data)
    robot.set_initial_pose()
    mujoco.mj_forward(model, data)

    print("Initialising IK target...")
    xpos0 = data.body("panda_hand").xpos.copy()
    search_center = xpos0.copy()
    search_rad_x = 0.67505
    search_rad_y = 0.4625
    search_speed = 0.5
    search_height = xpos0[2] + 0.1

    # Initialize offscreen renderers to extract RGB-D data
    width, height = 640, 480
    rgb_renderer = mujoco.Renderer(model, height=height, width=width)
    
    depth_renderer = mujoco.Renderer(model, height=height, width=width)
    depth_renderer.enable_depth_rendering()
    
    # Get camera properties for 3D de-projection
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_camera")
    fovy = model.cam_fovy[cam_id]

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
            t = time.time() - start_time
            
            # Control the robot
            target_pos, target_quat = robot.circular_search_motion(search_center, search_rad_x, search_rad_y, search_speed, search_height, t)
            robot.control(target_pos, target_quat)
            
            # Step the simulation forward
            mujoco.mj_step(model, data)
            
            # Sync the interactive viewer
            viewer.sync()
            
            # Extract RGB-D Data at realistic camera framerate
            if data.time - last_render_time >= 1.0 / render_fps:
                
                # RGB output (height, width, 3)
                rgb_renderer.update_scene(data, camera="gripper_camera")
                rgb_image = rgb_renderer.render()
                
                # Depth output (height, width) mapping distances in meters
                depth_renderer.update_scene(data, camera="gripper_camera")
                depth_image = depth_renderer.render()
                
                # Get camera transforms
                cam_xpos = data.cam_xpos[cam_id].copy()
                cam_xmat = data.cam_xmat[cam_id].copy()
                
                # Run YOLO inference via PerceptionSystem
                results, annotated_frame = perception.detect(rgb_image)
                
                # Phase 1: 3D Localization implementation
                # Extract first detected object's 3D coordinates if any boxes exist
                if len(results[0].boxes) > 0:
                    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                        # Center of the bounding box
                        u = int((box[0] + box[2]) / 2)
                        v = int((box[1] + box[3]) / 2)
                        
                        # Get 3D global position
                        obj_pos = perception.get_3d_point(
                            u, v, depth_image, fovy, width, height, cam_xpos, cam_xmat
                        )
                        
                        class_name = perception.yolo.names[int(cls)] if hasattr(perception.yolo, 'names') else str(int(cls))
                        print(f"Detected {class_name} at 3D World Pos: {obj_pos}")
                
                # Annotate and show the frame
                cv2.imshow("Gripper Camera YOLO", annotated_frame)
                cv2.waitKey(1)
                
                # Save the frame to the video video
                out.write(annotated_frame)
                
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
