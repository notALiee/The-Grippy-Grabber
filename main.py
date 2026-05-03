import mujoco
import mujoco.viewer
import time

def main():
    # Load the MuJoCo model and data from the XML scene
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    print("Launching viewer. Close the window to exit.")
    
    # Launch the passive MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Step the simulation forward
            mujoco.mj_step(model, data)
            
            # Sync the viewer with the updated physics state
            viewer.sync()
            
            # Run at roughly real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()