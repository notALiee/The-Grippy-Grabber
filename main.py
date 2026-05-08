import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.append('ikfast')
from ikfast import ikfast_panda_arm as ikfast

# Controller gains (Cartesian operational-space control)
K = np.array([600, 600, 600, 30, 30, 30])

def set_initial_pose(data):
    qpos0 = [0, 0, 0, 0, 0, 2.2, 0.785]
    for i in range(7):
        data.joint(f"panda_joint{i+1}").qpos = qpos0[i]

def control(model, data, xpos_d, xquat_d):
    xpos = data.body("panda_hand").xpos
    xquat = data.body("panda_hand").xquat

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    bodyid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "panda_hand"
    )

    mujoco.mj_jacBody(model, data, jacp, jacr, bodyid)

    error = np.zeros(6)

    # position error
    error[:3] = xpos_d - xpos

    # orientation error
    quat_error = np.zeros(3)
    mujoco.mju_subQuat(quat_error, xquat, xquat_d)
    mujoco.mju_rotVecQuat(quat_error, quat_error, xquat)
    error[3:] = -quat_error

    J = np.vstack([jacp, jacr])

    v = J @ data.qvel

    damping = 2 * np.sqrt(K)

    # operational space torque
    tau = J.T @ (K * error - damping * v)

    # apply only first 7 joints (Panda arm)
    data.ctrl[:7] = tau[:7]

def panda_ik(R_G, p_G, model, max_iters=100):
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i+1}")
        for i in range(7)
    ]
    free_joint_id = joint_ids[6]
    free_min = model.jnt_range[free_joint_id, 0]
    free_max = model.jnt_range[free_joint_id, 1]

    for _ in range(max_iters):
        free_joint = np.random.uniform(free_min, free_max)
        solutions = ikfast.get_ik(R_G, p_G.tolist(), [free_joint])

        if solutions is None:
            continue

        for sol in solutions:
            valid = True
            for i in range(7):
                jmin = model.jnt_range[joint_ids[i], 0]
                jmax = model.jnt_range[joint_ids[i], 1]
                if sol[i] < jmin or sol[i] > jmax:
                    valid = False
                    break
            if valid:
                return np.array(sol)

    return None

def circular_search_motion(center, rad_x, rad_y, speed, height, t):
    """
    Generates a circular Cartesian trajectory for desk scanning.
    """
    x = 0.42505 + rad_x * np.cos(speed * t)
    y = -0.2125 + rad_y * np.sin(speed * t)
    z = height

    pos = np.array([x, y, z])

    # Gripper facing downward
    R = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    mat = np.zeros(9)
    mat[:] = R.flatten()

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)

    return pos, quat

def main():
    # Load YOLO Model
    print("Loading YOLO model...")
    yolo_model = YOLO("best.pt")  # Use the trained model

    # Load the MuJoCo model and data from the XML scene
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    set_initial_pose(data)
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
            target_pos, target_quat = circular_search_motion(search_center, search_rad_x, search_rad_y, search_speed, search_height, t)
            control(model, data, target_pos, target_quat)
            
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
