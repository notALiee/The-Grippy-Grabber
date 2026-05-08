import mujoco
import mujoco.viewer
import numpy as np
import time
import sys

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

    center : np.array([x,y,z])
        center of the search circle

    radius : float
        circle radius

    speed : float
        angular speed (rad/sec)

    height : float
        fixed z-height above desk

    t : float
        current simulation time
    """

    x = 0.42505 + rad_x * np.cos(speed * t)
    y = -0.2125 + rad_y * np.sin(speed * t)
    z = height

    pos = np.array([x, y, z])

    # Gripper facing downward
    #
    # End effector Z-axis points downward toward desk.
    #
    # Rotation:
    # X stays X
    # Y flips
    # Z flips
    #
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
    search_height = xpos0[2]+0.1

    print("Launching controller...")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        start_time = time.time()

        while viewer.is_running():

            t = time.time() - start_time

            target_pos, target_quat = circular_search_motion(search_center,search_rad_x,search_rad_y ,search_speed,search_height,t)

            control(model, data, target_pos, target_quat)

            mujoco.mj_step(model, data)

            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()