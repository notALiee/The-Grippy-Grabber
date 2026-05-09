import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from control import PandaController
from perception import PerceptionSystem, get_intrinsics
from task_manager import GRIPPER_CAM_SIDE_NAMES, TaskFSM, _SceneRenderers


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

    # Gripper-mounted perception: center (top-down) + two side-facing RGB-D
    # cameras. The FSM merges detections across all three (task_manager
    # scan_multi).
    rgb_gripper = mujoco.Renderer(model, height=height, width=width)
    depth_gripper = mujoco.Renderer(model, height=height, width=width)
    depth_gripper.enable_depth_rendering()
    gripper_intrinsics = get_intrinsics(model, "gripper_camera", width, height)

    side_rgb_renderers = []
    side_depth_renderers = []
    side_intrinsics_list = []
    for cam_name in GRIPPER_CAM_SIDE_NAMES:
        r_side = mujoco.Renderer(model, height=height, width=width)
        d_side = mujoco.Renderer(model, height=height, width=width)
        d_side.enable_depth_rendering()
        side_rgb_renderers.append(r_side)
        side_depth_renderers.append(d_side)
        side_intrinsics_list.append(get_intrinsics(model, cam_name, width, height))

    scene_option = _make_scene_option(model)

    fsm = TaskFSM()
    renderers = _SceneRenderers(
        gripper_rgb_renderer=rgb_gripper,
        gripper_depth_renderer=depth_gripper,
        gripper_intrinsics=gripper_intrinsics,
        side_rgb_renderers=(side_rgb_renderers[0], side_rgb_renderers[1]),
        side_depth_renderers=(side_depth_renderers[0], side_depth_renderers[1]),
        side_intrinsics=(side_intrinsics_list[0], side_intrinsics_list[1]),
        scene_option=scene_option,
    )

    print("Launching native MuJoCo viewer. Close the window to exit.")
    print("Tip: Expand the right side panel in the viewer to change the camera view.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Recording: only gripper-mounted RGB (center + two side cams). No world cameras.
    out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (width * 3, height))

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
                rgb_gripper.update_scene(data, camera="gripper_camera")
                ctr = cv2.cvtColor(rgb_gripper.render(), cv2.COLOR_RGB2BGR)
                ctr = _label_frame(
                    ctr, f"gripper_center | {fsm.state.name}",
                    color=(0, 255, 255),
                )
                panels = [ctr]
                for i, cam_name in enumerate(GRIPPER_CAM_SIDE_NAMES):
                    side_rgb_renderers[i].update_scene(data, camera=cam_name)
                    sb = cv2.cvtColor(side_rgb_renderers[i].render(),
                                      cv2.COLOR_RGB2BGR)
                    panels.append(_label_frame(
                        sb, f"gripper_side_{i} ({cam_name})",
                        color=(0, 200, 255),
                    ))
                combined = np.hstack(panels)
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
