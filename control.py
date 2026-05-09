import mujoco
import numpy as np
import sys
sys.path.append('ikfast')
try:
    from ikfast import ikfast_panda_arm as ikfast
except ImportError:
    ikfast = None


def top_down_quat(yaw_rad=0.0):
    """
    Returns a unit quaternion (w, x, y, z) for a top-down gripper pose:
    gripper +z points along world -z (straight down), rotated around world z by yaw_rad.
    Used by both perception (per-object grasp orientation) and FSM (transport pose)
    so all top-down targets are consistent.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    # Base top-down rotation (matches the matrix used in the original circular_search_motion).
    # Then apply a yaw rotation about world Z on the LEFT.
    R_base = np.array([
        [1, 0,  0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    R_yaw = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    R = R_yaw @ R_base

    mat = np.zeros(9)
    mat[:] = R.flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


class PandaController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.K = np.array([600, 600, 600, 30, 30, 30])

        # Cache finger position- and velocity-actuator indices once. The
        # position actuator sets the target opening; the velocity actuator
        # ADDS active force in the desired direction (since the position
        # actuator alone is too soft when the arm is also moving and getting
        # PDed around -- the fingers don't fully close in time otherwise).
        self._finger_pos_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("pos_panda_finger_joint1", "pos_panda_finger_joint2")
        ]
        self._finger_vel_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("vel_panda_finger_joint1", "vel_panda_finger_joint2")
        ]
        # Backwards-compat alias for any other code reading the old name.
        self._finger_act_ids = self._finger_pos_act_ids

    # Comfortable Panda "ready" pose, well inside every joint limit. Used
    # both for set_initial_pose() and as the null-space bias in control().
    Q_NOMINAL = np.array([0.0, -0.4, 0.0, -2.5, 0.0, 2.0, 0.785])

    def set_initial_pose(self):
        # NOTE: the previous default ([0,0,0,0,0,2.2,0.785]) put joint 4 at
        # qpos=0, which is OUTSIDE its valid range [-3.072, -0.070]. The
        # simulator silently clamped it, leaving the arm in a borderline
        # configuration that pinned joint 4 at its upper limit and made
        # several reach poses physically unreachable.
        for i, q in enumerate(self.Q_NOMINAL):
            self.data.joint(f"panda_joint{i+1}").qpos = float(q)

    def set_rest_pose(self):
        """
        Optional rest pose. Tucks the arm low and to the side so it
        does NOT occlude the table from any of the scene cameras and so it
        cannot accidentally be visible at all (group-2 hiding is the primary
        defense; this is the secondary one). Joints chosen so the hand sits
        roughly behind the base, well below the lowest scene camera angle.
        """
        qpos_rest = [-1.5, -1.0, 0.0, -2.4, 0.0, 1.4, 0.785]
        for i in range(7):
            self.data.joint(f"panda_joint{i+1}").qpos = qpos_rest[i]
        for j in ("panda_finger_joint1", "panda_finger_joint2"):
            self.data.joint(j).qpos = 0.08
        self.data.qvel[:] = 0.0

    def control(self, xpos_d, xquat_d):
        # Defensive: if upstream physics has produced NaN/Inf in qpos/qvel,
        # don't pile on more torque -- just zero the arm ctrl and let the
        # simulator (or the main loop's recovery path) try to recover.
        if not (np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))):
            self.data.ctrl[:7] = 0.0
            return

        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)

        # Position error
        error[:3] = xpos_d - xpos

        # Orientation error
        quat_error = np.zeros(3)
        mujoco.mju_subQuat(quat_error, xquat, xquat_d)
        mujoco.mju_rotVecQuat(quat_error, quat_error, xquat)
        error[3:] = -quat_error

        J = np.vstack([jacp, jacr])
        v = J @ self.data.qvel
        damping = 2 * np.sqrt(self.K)

        # Operational-space PD torque on the arm joints + gravity/Coriolis
        # feedforward. qfrc_bias holds the Coriolis + centrifugal + gravity
        # forces needed to keep each joint stationary; without this
        # feedforward the OS-PD would have to maintain a permanent ~5-8 cm
        # z-error to generate the upward force that holds the arm against
        # gravity.
        os_torque = J.T @ (self.K * error - damping * v)  # shape (nv,)

        # Null-space posture bias: with 7 DoFs and a 6-DoF task, there's
        # one redundant DoF the OS-PD can't constrain. Without a bias the
        # redundant DoF wanders to joint limits depending on starting pose,
        # which makes lots of nominally-reachable targets unreachable in
        # practice. Project a weak joint-space pull toward Q_NOMINAL into
        # the task null-space N = (I - J^+ J) so it doesn't fight the
        # primary task. The damped pseudo-inverse term (1e-3) makes the
        # null-space projector LEAK a small fraction of the posture torque
        # back into the task direction, so Kn must stay tiny -- with
        # Q_NOMINAL biasing the elbow toward a bent posture, larger gains
        # sabotage low-Z reaches by 1.5-3 cm (the "missing by inches"
        # the gripper kept doing). Kn=0.5 gives ~5 mm Z-tracking error
        # at the 11 cm grasp floor and is still enough posture pull to
        # keep the redundant DoF off its limits.
        J_arm = J[:6, :7]
        Kn = 0.5
        Dn = 2.0 * np.sqrt(Kn)
        q_arm = self.data.qpos[:7]
        qd_arm = self.data.qvel[:7]
        tau_posture = Kn * (self.Q_NOMINAL - q_arm) - Dn * qd_arm
        try:
            JJt = J_arm @ J_arm.T + 1e-3 * np.eye(6)
            J_pinv = J_arm.T @ np.linalg.solve(JJt, np.eye(6))
            tau_null = (np.eye(7) - J_pinv @ J_arm) @ tau_posture
        except np.linalg.LinAlgError:
            tau_null = np.zeros(7)

        ctrl = (
            os_torque[:7]
            + self.data.qfrc_bias[:7]
            + tau_null
        )

        # Replace any NaN / Inf with 0 (belt-and-suspenders), then clamp to
        # the actuator's hard ctrl limits before applying. Without this an
        # arm/object contact transient occasionally produced a huge torque
        # that knocked another body to NaN velocity.
        ctrl = np.where(np.isfinite(ctrl), ctrl, 0.0)
        lo = self.model.actuator_ctrlrange[:7, 0]
        hi = self.model.actuator_ctrlrange[:7, 1]
        self.data.ctrl[:7] = np.clip(ctrl, lo, hi)

    def open_gripper(self, opening=0.08):
        """
        Command both fingers to specified opening (default 0.08 m). Uses
        only the position actuator + the velocity actuator's pure damping
        (ctrl=0) -- no active opening force, otherwise the fingers blow
        past the joint upper limit (the joint limit is a soft constraint).
        """
        opening = float(np.clip(opening, 0.0, 0.08))
        for aid in self._finger_pos_act_ids:
            if aid >= 0:
                self.data.ctrl[aid] = opening
        for aid in self._finger_vel_act_ids:
            if aid >= 0:
                self.data.ctrl[aid] = 0.0  # pure damping while opening

    def close_gripper(self):
        """
        Command both fingers fully closed. Drives the position actuator
        (target qpos = 0) with the velocity actuator providing damping
        only (ctrl=0). A small closing assist used to live here, but the
        slam-together force was too strong on small/light objects --
        fingers would punt them sideways instead of pinching them.
        kp=800 on the position actuator is plenty to finish closing on
        its own within the GRASP_DWELL window.
        """
        for aid in self._finger_pos_act_ids:
            if aid >= 0:
                self.data.ctrl[aid] = 0.0
        for aid in self._finger_vel_act_ids:
            if aid >= 0:
                self.data.ctrl[aid] = 0.0  # pure damping, no slam

    def pose_reached(self, target_pos, target_quat, pos_tol=1.5e-2, rot_tol=2.5e-1):
        """
        True iff panda_hand is within (pos_tol meters, rot_tol radians) of the
        target. Uses mj_subQuat in the same way as control() to measure
        rotation error.

        Defaults are intentionally loose: the operational-space PD here doesn't
        have gravity comp, so demanding sub-mm/sub-deg convergence causes the
        FSM to hang in approach states forever. 1.5 cm + ~14 deg is plenty
        accurate for top-down picks of cm-scale objects.
        """
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        if np.linalg.norm(np.asarray(target_pos) - np.asarray(xpos)) > pos_tol:
            return False
        quat_error = np.zeros(3)
        mujoco.mju_subQuat(quat_error, np.asarray(xquat), np.asarray(target_quat))
        return np.linalg.norm(quat_error) <= rot_tol

    def hand_pose(self):
        """Convenience: returns (pos, quat) of the panda_hand in world frame as fresh copies."""
        return (self.data.body("panda_hand").xpos.copy(),
                self.data.body("panda_hand").xquat.copy())

    def get_finger_qpos(self):
        """
        Returns (qpos1, qpos2) for the two finger slide joints, in meters.
        qpos in [0, 0.08]: 0 = fully closed, 0.08 = fully open. Useful for
        debug logging. Don't rely on qpos for grasp verification; use
        is_holding_object() which checks actual contact forces.
        """
        return (float(self.data.joint("panda_finger_joint1").qpos[0]),
                float(self.data.joint("panda_finger_joint2").qpos[0]))

    def is_holding_object(self):
        """
        Robust grasp verification: returns True iff at least one finger geom
        is in contact with something that is NOT the robot itself, NOT the
        table, and NOT the floor. That necessarily means there's an external
        object physically pinched between (or pressed against) the fingers.

        Why this and not just qpos: the position actuator + joint-limit soft
        constraints don't reliably converge fingers to qpos=0 during the
        grasp transient -- they often settle several millimeters open even
        with nothing in the gripper. Contact checking is unambiguous.
        """
        # Lazy-cache: finger geom IDs and the "not-an-object" exclusion set.
        if not hasattr(self, "_grasp_verify_cached"):
            self._finger_geom_ids = set()
            for body_name in ("panda_leftfinger", "panda_rightfinger"):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    continue
                # Iterate this body's geoms.
                for gi in range(self.model.ngeom):
                    if self.model.geom_bodyid[gi] == bid:
                        self._finger_geom_ids.add(gi)
            # Exclude all robot geoms (group 2) + table + floor.
            self._exclude_geom_ids = set()
            for gi in range(self.model.ngeom):
                bname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[gi]
                )
                if self.model.geom_group[gi] == 2:  # robot link/finger
                    self._exclude_geom_ids.add(gi)
                if bname in ("table", "world"):
                    self._exclude_geom_ids.add(gi)
            self._grasp_verify_cached = True

        # Walk active contacts; anything finger<->external counts as a grasp.
        for i in range(self.data.ncon):
            g1 = self.data.contact.geom1[i]
            g2 = self.data.contact.geom2[i]
            if g1 in self._finger_geom_ids and g2 not in self._exclude_geom_ids:
                return True
            if g2 in self._finger_geom_ids and g1 not in self._exclude_geom_ids:
                return True
        return False

    def panda_body_positions(self):
        """
        Returns an (N, 3) array of world positions for every panda body.
        Used by perception's robot-proximity filter to drop any detection that
        landed on the robot itself (last-line defense after geom-group hiding).
        """
        names = [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link6", "panda_link7",
            "panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger",
            "gripper_camera_link",
        ]
        pts = []
        for n in names:
            try:
                pts.append(self.data.body(n).xpos.copy())
            except KeyError:
                pass
        return np.asarray(pts) if pts else np.empty((0, 3))

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
        Kept for backwards-compat / debugging; the FSM no longer uses it.
        """
        x = 0.42505 + rad_x * np.cos(speed * t)
        y = -0.2125 + rad_y * np.sin(speed * t)
        z = height

        pos = np.array([x, y, z])
        quat = top_down_quat(0.0)
        return pos, quat
