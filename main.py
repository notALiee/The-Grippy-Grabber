import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from control import PandaController
from perception import PerceptionSystem, get_intrinsics
from task_manager import TaskFSM, _SceneRenderers


# Per project requirement: ALL perception happens through the gripper camera.
# We keep one static "scene_camera" rendered in the debug video for visual
# context (so the user can see what the arm is doing), but YOLO is never run
# on it.
DEBUG_SCENE_CAM = "scene_camera"


def _make_scene_option(model):
    """
    MjvOption that DISABLES geom group 2. Every panda link/hand/finger geom
    has been tagged group=2 in panda.xml, so passing this option to
    Renderer.update_scene(...) renders the scene without the robot. We use
    this when capturing depth/RGB from the gripper camera so the fingers
    don't occlude the table.
    """
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0
    return opt


def _label_frame(frame_bgr, text, color=(255, 255, 255)):
    cv2.putText(frame_bgr, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)
    return frame_bgr


def main():
    print("Initializing perception system...")
    perception = PerceptionSystem("best.pt")

    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
    data = mujoco.MjData(model)

    robot = PandaController(model, data)
    robot.set_initial_pose()
    mujoco.mj_forward(model, data)

    width, height = 640, 480

    # Gripper camera: this is the ONLY perception input. Used by the FSM during
    # EXPLORING (active sweep + per-tick perception) and CENTERING (final depth refinement).
    rgb_gripper = mujoco.Renderer(model, height=height, width=width)
    depth_gripper = mujoco.Renderer(model, height=height, width=width)
    depth_gripper.enable_depth_rendering()
    gripper_intrinsics = get_intrinsics(model, "gripper_camera", width, height)

    # Debug-only static scene camera (not used for perception).
    rgb_debug = mujoco.Renderer(model, height=height, width=width)

    scene_option = _make_scene_option(model)

    fsm = TaskFSM()
    renderers = _SceneRenderers(
        gripper_rgb_renderer=rgb_gripper,
        gripper_depth_renderer=depth_gripper,
        gripper_intrinsics=gripper_intrinsics,
        scene_option=scene_option,
    )

    print("Launching native MuJoCo viewer. Close the window to exit.")
    print("Tip: Expand the right side panel in the viewer to change the camera view.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Layout: side-by-side: [scene_camera (debug)] [gripper_camera (live)]
    out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (width * 2, height))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_render_time = data.time
        render_fps = 10

        nan_recoveries = 0
        while viewer.is_running():
            step_start = time.time()
            t_sim = data.time

            # Defensive: if a previous physics step produced NaN/Inf in the
            # state (e.g. an arm/object contact transient went unstable),
            # zero velocities and reset the arm to the home pose so we can
            # keep going. The objects on the table aren't reset, but this
            # at least lets the mission continue instead of crashing.
            if not (np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))):
                nan_recoveries += 1
                print(f"[main] WARNING: NaN/Inf in state at t={t_sim:.2f}s "
                      f"(recovery #{nan_recoveries}). Resetting arm.")
                # Replace any NaN/Inf entries; preserve finite ones.
                data.qpos[:] = np.where(np.isfinite(data.qpos), data.qpos, 0.0)
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
                data.ctrl[:] = 0.0
                robot.set_initial_pose()
                mujoco.mj_forward(model, data)
                continue

            target_pos, target_quat = fsm.step(
                t_sim, model, data, perception, robot, renderers,
            )
            robot.control(target_pos, target_quat)
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_render_time >= 1.0 / render_fps:
                # Debug scene cam (no YOLO, robot visible so we see the arm).
                rgb_debug.update_scene(data, camera=DEBUG_SCENE_CAM)
                debug_rgb = rgb_debug.render()
                debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
                debug_panel = _label_frame(
                    debug_bgr, f"scene_camera (debug) | state={fsm.state.name}",
                    color=(255, 255, 255),
                )

                # Gripper cam: render live (robot visible) for the user-facing
                # video so they can see what the camera actually sees. Note
                # this is purely for display — perception renders this same
                # camera with scene_option set, separately.
                rgb_gripper.update_scene(data, camera="gripper_camera")
                gripper_rgb = rgb_gripper.render()
                gripper_bgr = cv2.cvtColor(gripper_rgb, cv2.COLOR_RGB2BGR)
                gripper_panel = _label_frame(
                    gripper_bgr, "gripper_camera (live, perception input)",
                    color=(0, 255, 255),
                )

                combined = np.hstack([debug_panel, gripper_panel])
                cv2.imshow("Grippy Grabber", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                out.write(combined)

                last_render_time = data.time

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
