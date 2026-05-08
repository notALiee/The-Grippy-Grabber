import mujoco
import numpy as np
import sys
sys.path.append('ikfast')
try:
    from ikfast import ikfast_panda_arm as ikfast
except ImportError:
    ikfast = None

class PandaController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.K = np.array([600, 600, 600, 30, 30, 30])
        
    def set_initial_pose(self):
        qpos0 = [0, 0, 0, 0, 0, 2.2, 0.785]
        for i in range(7):
            self.data.joint(f"panda_joint{i+1}").qpos = qpos0[i]

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)

        # position error
        error[:3] = xpos_d - xpos

        # orientation error
        quat_error = np.zeros(3)
        mujoco.mju_subQuat(quat_error, xquat, xquat_d)
        mujoco.mju_rotVecQuat(quat_error, quat_error, xquat)
        error[3:] = -quat_error

        J = np.vstack([jacp, jacr])
        v = J @ self.data.qvel
        damping = 2 * np.sqrt(self.K)

        # operational space torque
        tau = J.T @ (self.K * error - damping * v)

        # apply only first 7 joints (Panda arm)
        self.data.ctrl[:7] = tau[:7]

    def panda_ik(self, R_G, p_G, max_iters=100):
        if not ikfast:
            return None
            
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i+1}")
            for i in range(7)
        ]
        free_joint_id = joint_ids[6]
        free_min = self.model.jnt_range[free_joint_id, 0]
        free_max = self.model.jnt_range[free_joint_id, 1]

        for _ in range(max_iters):
            free_joint = np.random.uniform(free_min, free_max)
            solutions = ikfast.get_ik(R_G, p_G.tolist(), [free_joint])

            if solutions is None:
                continue

            for sol in solutions:
                valid = True
                for i in range(7):
                    jmin = self.model.jnt_range[joint_ids[i], 0]
                    jmax = self.model.jnt_range[joint_ids[i], 1]
                    if sol[i] < jmin or sol[i] > jmax:
                        valid = False
                        break
                if valid:
                    return np.array(sol)

        return None

    def circular_search_motion(self, center, rad_x, rad_y, speed, height, t):
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
