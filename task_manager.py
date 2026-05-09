from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from control import top_down_quat
from perception import ObjectPose, CameraView


class State(Enum):
    EXPLORING = auto()       # actively sweep workspace + perceive each tick
    CONFIRMING = auto()      # multi-view re-scan to verify the candidate is real
    PLANNING = auto()        # commit to the detected target
    HOMING = auto()          # lift current pose to a safe transit height
    APPROACHING = auto()     # horizontal move to over the target
    DESCENDING = auto()      # quick coarse drop to a pre-grasp hover height
    LOWERING = auto()        # very slow final descent to grip height (no slamming)
    GRASPING = auto()        # close fingers, dwell
    VERIFY_GRASP = auto()    # contact check to confirm we picked something up
    LIFTING = auto()         # vertical lift back to safe height
    TRANSPORTING = auto()    # horizontal move to over the drop zone
    PLACING = auto()         # open fingers, dwell
    RETREATING = auto()      # small lift, then back to EXPLORING
    DONE = auto()


# ---- Geometry / motion constants -------------------------------------------
# These are panda_hand world positions, NOT object positions. The Panda
# fingertips sit ~10.8 cm below panda_hand when the gripper is top-down.

SAFE_Z = 0.45                         # panda_hand world z used for safe horizontal transit
PRE_GRASP_HOVER_Z = 0.27              # panda_hand z right before the slow final descent
# Geometry: panda_hand z = 0.108 puts the FINGERTIPS exactly at table level
# (z=0). We deliberately COMMAND ~3 cm BELOW that nominal -- the OS-PD
# can't actually drive fingers through the table (the table contact stops
# them), so the extra command shows up as PRESS-DOWN force during the
# grasp. That extra force is what makes the inner finger pads conform to
# table-tops and bite around the object instead of stopping a few mm
# above it. Without this the controller settles too gently and the grip
# closes around air.
GRIP_HAND_Z_FLOOR = 0.075             # panda_hand z floor (fingers press onto table; sim contact bounces them up to ~0.108)
GRIP_HAND_Z_OFFSET = 0.078            # detected-centroid z + this = hand z target;
                                      # with the floor above this also adds a
                                      # 3 cm press-down on tall objects so the
                                      # fingers wrap around their middle, not
                                      # their top edge
DROP_HAND_POS = np.array([0.0, 0.5, 0.30])  # panda_hand release pose; objects fall ~19 cm
RETREAT_OFFSET = 0.15                 # vertical retreat after release
PRE_PLACE_LIFT = 0.20                 # vertical lift before PLACING (gives a clear drop)

# A consistent, comfortable home pose to which the arm always returns at the
# start of a grasp cycle. The OS-PD has 7-DoF redundancy and gets stuck in
# bad joint configurations (e.g. joint 4 pinned at its upper limit) if we
# always APPROACH from wherever we ended up after the last LIFTING. Going
# through this fixed waypoint resets the arm to a comfortable null-space
# configuration before every grasp, which dramatically improves tracking.
HOME_HAND_POS = np.array([0.30, 0.0, SAFE_Z])

# ---- Timing ----------------------------------------------------------------
# All speeds are deliberately conservative. The smoothstep trajectory in
# Trajectory class also tapers velocity at the start/end, so peak velocity
# is ~1.5x average and end velocity is zero -- contact is gentle.
GRASP_DWELL = 2.0                     # s, finger close time (slow position-actuator)
PLACE_DWELL = 0.8                     # s, finger open time
TRANSIT_SPEED = 0.15                  # m/s, horizontal moves at safe altitude
DESCEND_SPEED = 0.10                  # m/s, coarse descent to PRE_GRASP_HOVER_Z
LOWERING_SPEED = 0.025                # m/s, slow final descent onto the object
LIFT_SPEED = 0.12                     # m/s, lift after grasp (gentle so payload doesn't swing)
MIN_TRAJ_DURATION = 1.0               # s, never let trajectories get instantaneous
STATE_TIMEOUT = 20.0                  # s, hard cap per state (raised for slower motion)

# ---- Active exploration (gripper-camera-only perception) -------------------
# Per project requirement: ALL perception comes from the gripper camera, and
# we don't precompute a depth map. Instead the arm continuously sweeps these
# waypoints in a loop, running gripper-cam perception every
# EXPLORE_PERCEIVE_INTERVAL seconds. The FIRST high-confidence detection
# wins -- we abort the sweep, go grasp it, then come back to exploring.
EXPLORE_WAYPOINTS = [
    np.array([0.35,  0.00, 0.60]),
    np.array([0.55,  0.00, 0.60]),
    np.array([0.55, -0.25, 0.60]),
    np.array([0.35, -0.25, 0.60]),
]
EXPLORE_TRAVEL_SPEED = 0.12           # m/s, slow + smooth so YOLO has time + arm doesn't lurch
EXPLORE_PERCEIVE_INTERVAL = 0.30      # s, sim time between gripper-cam perception runs
EXPLORE_MIN_SCORE = 0.80              # min YOLO confidence to even consider a target
EXPLORE_MAX_LAPS = 2                  # full passes with zero detections -> DONE
EXPLORE_GRIPPER_OPENING = 0.04        # m, partial-open during sweep

# ---- Multi-view confirmation ----------------------------------------------
# When EXPLORING spots a high-confidence candidate we don't immediately
# commit -- one bad bbox or one specular reflection in the depth map can
# send the arm crashing into nothing. Instead we orbit the gripper camera
# to a few viewpoints around the candidate and re-run YOLO + depth pose.
# Only if MULTIPLE views independently confirm the SAME class within
# CONFIRM_MAX_XY_DRIFT of the original do we proceed. The averaged XY
# from the confirming hits is also a much better grasp center than any
# single-view detection.
CONFIRM_MIN_SCORE = 0.80              # min YOLO confidence per confirmation view
CONFIRM_REQUIRED_HITS = 2             # at least N matching detections out of N_VIEWPOINTS
CONFIRM_HOVER_Z = 0.35                # m, panda_hand z while orbiting (slightly above PRE_GRASP_HOVER_Z so the whole object fits in frame even after the XY offset)
CONFIRM_SETTLE_TIME = 0.4             # s, wait at each viewpoint for the camera to stabilize before snapping
CONFIRM_MAX_XY_DRIFT = 0.06           # m, max XY distance between hits / from candidate to count as "same object"
CONFIRM_VIEWPOINTS_OFFSETS = [        # XY offsets (world frame, meters) around the candidate
    np.array([0.00, 0.00]),           # straight overhead -- the strongest view
    np.array([0.05, 0.00]),           # camera shifted +X (object appears on -X side of frame)
    np.array([-0.05, 0.00]),          # camera shifted -X
    np.array([0.00, 0.05]),           # camera shifted +Y
]
# Failed-confirmation memory: if a candidate at XY fails confirmation we
# remember it for FAILED_TARGET_TTL seconds and skip any new detections
# within FAILED_TARGET_RADIUS of it. Otherwise EXPLORING re-locks on the
# same false-positive every tick.
FAILED_TARGET_TTL = 30.0              # s
FAILED_TARGET_RADIUS = 0.08           # m

# ---- Grasp verification ----------------------------------------------------
# After GRASP_DWELL we check whether any finger is actually in contact with
# an external object via PandaController.is_holding_object(). Finger qpos
# is logged for debugging only.
GRASP_VERIFY_DELAY = 0.6              # s, wait after close for fingers + arm to settle


# ----------------------------------------------------------------------------
# Trajectory: lerp position, normalized quaternion lerp (nlerp) for orientation
# ----------------------------------------------------------------------------
def _quat_nlerp(q0, q1, alpha):
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    n = np.linalg.norm(q)
    return q1 if n < 1e-9 else q / n


@dataclass
class Trajectory:
    start_pos: np.ndarray
    end_pos: np.ndarray
    start_quat: np.ndarray
    end_quat: np.ndarray
    t0: float
    duration: float

    @classmethod
    def build(cls, start_pos, start_quat, end_pos, end_quat, t0, speed):
        dist = float(np.linalg.norm(np.asarray(end_pos) - np.asarray(start_pos)))
        duration = max(MIN_TRAJ_DURATION, dist / max(speed, 1e-3))
        return cls(np.asarray(start_pos, dtype=float).copy(),
                   np.asarray(end_pos, dtype=float).copy(),
                   np.asarray(start_quat, dtype=float).copy(),
                   np.asarray(end_quat, dtype=float).copy(),
                   t0, duration)

    def alpha(self, t):
        if self.duration <= 0:
            return 1.0
        return float(np.clip((t - self.t0) / self.duration, 0.0, 1.0))

    def pose_at(self, t):
        a = self.alpha(t)
        # Smoothstep: 3a^2 - 2a^3 — gentle accel/decel, removes jerk at endpoints.
        s = a * a * (3.0 - 2.0 * a)
        pos = (1.0 - s) * self.start_pos + s * self.end_pos
        quat = _quat_nlerp(self.start_quat, self.end_quat, s)
        return pos, quat

    def done(self, t):
        return self.alpha(t) >= 1.0


# ----------------------------------------------------------------------------
# Renderer bundle handed to the FSM by main.py
#
# Per project requirement: perception is gripper-camera-only. This bundle
# carries the gripper RGB + depth renderers, gripper intrinsics, and an
# optional MjvOption that hides the robot (group=2) so the fingers don't
# occlude the depth map during scan/centering.
# ----------------------------------------------------------------------------
@dataclass
class _SceneRenderers:
    gripper_rgb_renderer: object
    gripper_depth_renderer: object
    gripper_intrinsics: dict
    scene_option: object = None       # MjvOption hiding robot (group=2) for perception


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _grip_hand_z(target: ObjectPose) -> float:
    """
    panda_hand world Z for the GRASPING state. Detected position[2] is roughly
    the visible top surface of the object; we add a fixed offset so the
    fingertips end up around the object centroid, and we floor at 0.105 so
    the fingers settle ~2 mm above the table even for paper-thin items
    (pens, puzzle pieces). The OS-PD has a small (~5 mm) steady-state
    z-error baked into the floor on purpose: that means the actual hand
    z-on-arrival lands near 0.110, which is the safe sweet spot.
    """
    return max(float(target.position[2]) + GRIP_HAND_Z_OFFSET, GRIP_HAND_Z_FLOOR)


def _get_gripper_opening(target: ObjectPose) -> float:
    """
    Return optimal gripper opening (0.0 to 0.08) based on object class.
    Smaller objects need less opening to avoid grasping air.
    """
    class_openings = {
        'pen': 0.02,           # Small thin object
        'PuzzleCircle': 0.025, # Small puzzle piece  
        'PuzzleSquare': 0.025, # Small puzzle piece
        'PuzzleTriangle': 0.025, # Small puzzle piece
        'cup': 0.06,           # Medium object
        'book': 0.08,          # Large/thick object
        'PuzzleBase': 0.04,    # Medium puzzle piece
    }
    return class_openings.get(target.class_name, 0.08)  # Default to full opening


def _capture_gripper_view(data, renderers: _SceneRenderers,
                          name: str = "gripper") -> CameraView:
    """
    Render a single RGB+depth frame from the gripper camera with the robot
    hidden (group=2) so the fingers/hand don't appear in the depth map. Stamps
    the camera world pose (cam_xpos, cam_xmat) at this exact moment so we can
    de-project pixels into world coordinates.
    """
    rgb_r = renderers.gripper_rgb_renderer
    dep_r = renderers.gripper_depth_renderer
    intr = renderers.gripper_intrinsics
    so = renderers.scene_option

    if so is not None:
        rgb_r.update_scene(data, camera="gripper_camera", scene_option=so)
        dep_r.update_scene(data, camera="gripper_camera", scene_option=so)
    else:
        rgb_r.update_scene(data, camera="gripper_camera")
        dep_r.update_scene(data, camera="gripper_camera")
    rgb = rgb_r.render()
    depth = dep_r.render()
    cam_id = intr["cam_id"]
    return CameraView(
        name=name,
        rgb=rgb,
        depth=depth,
        intrinsics=intr,
        cam_xpos=data.cam_xpos[cam_id].copy(),
        cam_xmat=data.cam_xmat[cam_id].copy(),
    )


# ----------------------------------------------------------------------------
# TaskFSM
# ----------------------------------------------------------------------------
class TaskFSM:
    """
    Pick-and-dump FSM. One scan, then for each detected object:
        HOMING -> APPROACHING -> DESCENDING -> GRASPING -> LIFTING ->
        TRANSPORTING -> PLACING -> RETREATING -> PLANNING -> ...

    All cartesian moves are smooth trajectories (smoothstep position lerp +
    nlerp quaternion). The FSM also separates HORIZONTAL transit from
    VERTICAL descent so we never drag the gripper sideways through an object
    on the table.
    """

    def __init__(self):
        self.state: State = State.EXPLORING
        self.target: Optional[ObjectPose] = None

        self._traj: Optional[Trajectory] = None
        self._dwell_t0: Optional[float] = None
        self._state_t0: Optional[float] = None
        self._desired_pos: np.ndarray = np.array([0.3, 0.0, SAFE_Z])
        self._desired_quat: np.ndarray = top_down_quat(0.0)

        # Active-exploration bookkeeping.
        self._explore_idx: int = 0            # next EXPLORE_WAYPOINTS entry
        self._explore_perceive_t: float = -1e9  # last perception time (sim seconds)
        self._explore_lap: int = 0            # full passes with no detection

        # Grasp verification state. Set by VERIFY_GRASP, consumed by LIFTING.
        self._grasp_success: bool = False

        # Per-step references cached for state callbacks (DESCENDING uses
        # these to take a one-shot perception snapshot at PRE_GRASP_HOVER_Z
        # without having to thread them through every state's signature).
        self._data = None
        self._perception = None
        self._renderers: Optional[_SceneRenderers] = None

    # ------------------------------------------------------------------
    def current_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._desired_pos, self._desired_quat

    def is_done(self) -> bool:
        return self.state == State.DONE

    # ------------------------------------------------------------------
    def step(self, t_sim: float, model, data, perception, robot,
             renderers: _SceneRenderers) -> Tuple[np.ndarray, np.ndarray]:
        if self._state_t0 is None:
            self._state_t0 = t_sim
        elapsed = t_sim - self._state_t0
        timed_out = elapsed > STATE_TIMEOUT

        # Cache for state callbacks that need to invoke perception mid-flight
        # (e.g. _do_descending's one-shot XY refinement at PRE_GRASP_HOVER_Z).
        self._data = data
        self._perception = perception
        self._renderers = renderers

        if self.state == State.EXPLORING:
            self._do_exploring(t_sim, elapsed, model, data, perception, robot, renderers)
        elif self.state == State.PLANNING:
            self._do_planning(t_sim, robot)
        elif self.state == State.HOMING:
            self._do_homing(t_sim, robot, timed_out)
        elif self.state == State.APPROACHING:
            self._do_approaching(t_sim, robot, timed_out)
        elif self.state == State.DESCENDING:
            self._do_descending(t_sim, robot, timed_out)
        elif self.state == State.LOWERING:
            self._do_lowering(t_sim, robot, timed_out)
        elif self.state == State.GRASPING:
            self._do_grasping(t_sim, elapsed, robot)
        elif self.state == State.VERIFY_GRASP:
            self._do_verify_grasp(t_sim, elapsed, robot)
        elif self.state == State.LIFTING:
            self._do_lifting(t_sim, robot, timed_out)
        elif self.state == State.TRANSPORTING:
            self._do_transporting(t_sim, robot, timed_out)
        elif self.state == State.PLACING:
            self._do_placing(t_sim, elapsed, robot)
        elif self.state == State.RETREATING:
            self._do_retreating(t_sim, robot, timed_out)
        elif self.state == State.DONE:
            robot.open_gripper()

        return self._desired_pos, self._desired_quat

    # ------------------------------------------------------------------
    def _enter(self, new_state: State):
        if new_state != self.state:
            print(f"[FSM] {self.state.name} -> {new_state.name}")
            self.state = new_state
        self._state_t0 = None
        self._traj = None
        self._dwell_t0 = None

    def _start_traj(self, t_sim, robot, end_pos, end_quat, speed):
        cur_pos, cur_quat = robot.hand_pose()
        self._traj = Trajectory.build(cur_pos, cur_quat, end_pos, end_quat, t_sim, speed)

    def _track_traj(self, t_sim):
        if self._traj is not None:
            self._desired_pos, self._desired_quat = self._traj.pose_at(t_sim)

    # ------------------------------------------------------------------
    # State implementations
    # ------------------------------------------------------------------
    def _do_exploring(self, t_sim, elapsed, model, data, perception, robot, renderers):
        """
        Active exploration: continuously sweep through EXPLORE_WAYPOINTS while
        running gripper-cam perception every EXPLORE_PERCEIVE_INTERVAL seconds.
        First confident detection wins -- abort the sweep, transition to
        PLANNING. After grasping + placing, return here and resume.

        Two full passes with zero detections -> DONE (workspace empty).
        """
        explore_quat = top_down_quat(0.0)
        robot.open_gripper(EXPLORE_GRIPPER_OPENING)

        # Finished a lap?
        if self._explore_idx >= len(EXPLORE_WAYPOINTS):
            self._explore_idx = 0
            self._explore_lap += 1
            self._traj = None
            print(f"[FSM] EXPLORING: completed lap {self._explore_lap} (no detections)")
            if self._explore_lap >= EXPLORE_MAX_LAPS:
                print(f"[FSM] EXPLORING: {EXPLORE_MAX_LAPS} empty laps -> DONE")
                self._enter(State.DONE)
                return

        # Drive a smooth trajectory to the next waypoint. As soon as we arrive,
        # increment _explore_idx and let the next tick build the next leg.
        target_pos = EXPLORE_WAYPOINTS[self._explore_idx]
        if self._traj is None:
            self._start_traj(t_sim, robot, target_pos, explore_quat, EXPLORE_TRAVEL_SPEED)
            print(f"[FSM] EXPLORING: -> waypoint {self._explore_idx + 1}/"
                  f"{len(EXPLORE_WAYPOINTS)} at {target_pos}")
        self._track_traj(t_sim)

        # Rate-limited gripper-cam perception while moving.
        if (t_sim - self._explore_perceive_t) >= EXPLORE_PERCEIVE_INTERVAL:
            self._explore_perceive_t = t_sim
            target = self._look_for_target(data, perception, robot, renderers)
            if target is not None:
                self.target = target
                self._explore_lap = 0  # we found something, reset lap counter
                self._traj = None
                print(f"[FSM] EXPLORING: spotted {target.class_name} "
                      f"score={target.score:.2f} at {target.position}")
                self._enter(State.PLANNING)
                return

        # Reached this waypoint? Go to the next one on the next tick.
        if self._traj is not None and self._traj.done(t_sim):
            self._explore_idx += 1
            self._traj = None

    def _look_for_target(self, data, perception, robot, renderers) -> Optional[ObjectPose]:
        """
        Capture a single gripper-cam frame, run YOLO + pose estimation, and
        return the best candidate target (or None). Uses scan_scene which
        already handles depth-band masking + per-class quaternion synthesis.
        Also rejects targets that are physically unsafe to grasp:
          - too close to the Panda base (arm has to fold tightly + sweeps
            through neighboring objects, which has caused NaN-level contact
            blowups in mujoco)
          - off-table z range
        """
        view = _capture_gripper_view(data, renderers, name="explore")
        poses = perception.scan_scene(
            view.rgb, view.depth, view.cam_xpos, view.cam_xmat,
            view.intrinsics, min_points=20, min_score=EXPLORE_MIN_SCORE,
        )
        if not poses:
            return None

        # Drop anything that's actually the robot or obviously off-table.
        try:
            robot_pts = robot.panda_body_positions()
            poses = perception.filter_by_robot_proximity(poses, robot_pts, min_dist=0.10)
        except Exception:
            pass
        poses = [p for p in poses if -0.05 <= p.position[2] <= 0.30]

        # Reachable workspace filter. Panda base is at (-0.1, 0.1, 0); with
        # top-down orientation the practical min reach is ~0.30 m and max
        # is ~0.75 m horizontal. Targets outside that ring are unsafe (arm
        # crashes through other objects trying to fold up or stretches into
        # a hard singularity) so we just decline to grasp them.
        BASE_XY = np.array([-0.1, 0.1])
        MIN_REACH = 0.32
        MAX_REACH = 0.75
        in_reach = []
        for p in poses:
            r = float(np.linalg.norm(p.position[:2] - BASE_XY))
            if MIN_REACH <= r <= MAX_REACH:
                in_reach.append(p)
            else:
                print(f"[FSM] EXPLORING: skipping {p.class_name} at "
                      f"{p.position} (reach={r*1000:.0f}mm out of "
                      f"[{MIN_REACH*1000:.0f},{MAX_REACH*1000:.0f}]mm)")
        if not in_reach:
            return None

        # Highest confidence wins. Tie-break on number of supporting depth points.
        in_reach.sort(key=lambda p: (-p.score, -p.n_points))
        return in_reach[0]

    def _refine_target_xy(self, robot):
        """
        One-shot XY refinement, called once when DESCENDING reaches
        PRE_GRASP_HOVER_Z. Uses pure DEPTH (no YOLO) to locate the
        object directly under the gripper:

          1. capture a depth frame from the centered gripper camera,
          2. estimate the table depth as the median of finite pixels,
          3. mask every pixel that's >5 mm above the table -> binary
             "stuff above the table" image,
          4. extract closed contours from that mask,
          5. pick the contour whose centroid is closest to the image
             center (the gripper is ~directly above the target, so the
             nearest-to-center blob IS the target),
          6. take the median depth WITHIN that contour as the object's
             top-surface depth,
          7. de-project the contour-centroid pixel + that depth to
             world coordinates,
          8. snap target XY to that world point and walk straight down.

        This handles big objects far better than a YOLO-bbox centroid
        because the bbox is rectangular and includes table pixels at
        the corners, whereas the depth contour conforms exactly to the
        object's silhouette.

        Failure modes (no contour, jump > 6 cm, etc.) silently keep
        the original XY -- this is best-effort polish.
        """
        if self.target is None:
            return
        if self._data is None or self._renderers is None:
            return
        try:
            view = _capture_gripper_view(self._data, self._renderers, name="refine")
        except Exception as e:
            print(f"[FSM] DESCEND-refine: capture failed ({e})")
            return

        depth = np.asarray(view.depth, dtype=np.float32)
        H, W = depth.shape[:2]
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size < 200:
            print("[FSM] DESCEND-refine: depth frame mostly empty, keeping original XY")
            return
        table_depth = float(np.median(valid))

        # Above-table mask: anything closer to the camera than (table - 5 mm)
        # and farther than 1 cm (sanity: clip near-camera spikes from the
        # finger renderable group, even though it should be hidden).
        margin = 0.005
        above = (
            np.isfinite(depth)
            & (depth > 0.01)
            & (depth < table_depth - margin)
        )
        if int(above.sum()) < 100:
            print("[FSM] DESCEND-refine: nothing above table, keeping original XY")
            return

        # Light morphological close to fill camera-noise pinholes inside the
        # object silhouette so findContours returns a single closed loop.
        mask = above.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[FSM] DESCEND-refine: no closed contours, keeping original XY")
            return

        # Pick the contour whose centroid is closest to the image center;
        # the gripper is roughly above the target so the target's blob is
        # the centermost one. Reject blobs <50 px (specular noise).
        img_center = np.array([W * 0.5, H * 0.5])
        best = None
        best_dist = float("inf")
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 50.0:
                continue
            mom = cv2.moments(c)
            if mom["m00"] == 0.0:
                continue
            cu = mom["m10"] / mom["m00"]
            cv_ = mom["m01"] / mom["m00"]
            d = float(np.linalg.norm([cu - img_center[0], cv_ - img_center[1]]))
            if d < best_dist:
                best_dist = d
                best = (c, cu, cv_, area)

        if best is None:
            print("[FSM] DESCEND-refine: contours all too small, keeping original XY")
            return
        contour, cu, cv_, area = best

        # Median depth INSIDE the contour as the representative object
        # depth (more robust than the centroid pixel which can fall on a
        # noise hole or a hollow center, e.g. cup interior).
        fill = np.zeros_like(mask)
        cv2.drawContours(fill, [contour], -1, 255, -1)
        d_inside = depth[fill > 0]
        d_inside = d_inside[np.isfinite(d_inside) & (d_inside > 0)]
        if d_inside.size == 0:
            print("[FSM] DESCEND-refine: contour has no valid depth, keeping XY")
            return
        obj_depth = float(np.median(d_inside))

        # Deproject (cu, cv_, obj_depth) -> world. Same convention as
        # perception._deproject_bbox: MuJoCo cam looks down its -Z axis,
        # image y-axis points down, so y_cam negates (v - cy)/f * z.
        intr = view.intrinsics
        f = intr["f"]
        cx_intr = intr["cx"]
        cy_intr = intr["cy"]
        x_cam = (cu - cx_intr) * obj_depth / f
        y_cam = -(cv_ - cy_intr) * obj_depth / f
        z_cam = -obj_depth
        R = np.asarray(view.cam_xmat).reshape(3, 3)
        p_world = R @ np.array([x_cam, y_cam, z_cam]) + np.asarray(view.cam_xpos)

        # Sanity: in-workspace XY (don't accept stupid jumps).
        old_xy = self.target.position[:2]
        delta = p_world[:2] - old_xy
        d = float(np.linalg.norm(delta))
        if d > 0.06:
            print(f"[FSM] DESCEND-refine: contour {d*1000:.1f}mm from prior XY, too far - ignoring")
            return

        new_pos = self.target.position.copy()
        new_pos[:2] = p_world[:2]
        self.target = ObjectPose(
            class_name=self.target.class_name,
            position=new_pos,
            quaternion=self.target.quaternion,
            score=self.target.score,
            n_points=int(area),
            principal_axis=self.target.principal_axis,
            source_cam=self.target.source_cam,
        )
        print(f"[FSM] DESCEND-refine: depth contour ctr=({cu:.0f},{cv_:.0f}), "
              f"area={int(area)}px, obj_depth={obj_depth:.3f}m -> XY shift "
              f"({delta[0]*1000:+.1f},{delta[1]*1000:+.1f})mm")

    def _do_planning(self, t_sim, robot):
        # Target was already chosen by EXPLORING; just commit and start moving.
        if self.target is None:
            print("[FSM] PLANNING: no target set, returning to EXPLORING")
            self._enter(State.EXPLORING)
            return
        gripper_opening = _get_gripper_opening(self.target)
        print(f"[FSM] PLANNING: committing to {self.target.class_name} at "
              f"{self.target.position} -> grip_hand_z={_grip_hand_z(self.target):.3f}, "
              f"pre-opening to {gripper_opening*1000:.0f}mm")
        # Pre-size the gripper to the class-specific opening NOW so fingers
        # finish moving long before we reach GRASPING. Otherwise the slow
        # position actuator can't fully close in the grasp dwell window.
        robot.open_gripper(gripper_opening)
        self._grasp_success = False
        self._enter(State.HOMING)

    def _do_homing(self, t_sim, robot, timed_out):
        # Drive the arm through a consistent comfortable HOME_HAND_POS
        # before every grasp. This resets the redundant 7-DoF arm to a
        # known null-space configuration; otherwise APPROACHING from
        # whatever pose we ended up in often gets stuck at a joint limit.
        if self._traj is None:
            assert self.target is not None
            self._start_traj(t_sim, robot, HOME_HAND_POS, self.target.quaternion,
                             TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] HOMING timeout, advancing.")
            self._enter(State.APPROACHING)

    def _do_approaching(self, t_sim, robot, timed_out):
        # Horizontal move to (target.x, target.y, SAFE_Z) at target orientation.
        assert self.target is not None
        if self._traj is None:
            end_pos = np.array([self.target.position[0],
                                self.target.position[1],
                                SAFE_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion, TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] APPROACHING timeout, advancing.")
            self._enter(State.DESCENDING)

    def _do_descending(self, t_sim, robot, timed_out):
        """
        Coarse descent: from SAFE_Z down to PRE_GRASP_HOVER_Z directly above
        the target. Done at DESCEND_SPEED (still moderate). The slow,
        gentle final approach is handled by the next state (LOWERING) so we
        don't slam into the table.

        Once the hand reaches PRE_GRASP_HOVER_Z we take ONE depth snapshot
        from the gripper camera (now physically centered between the
        fingers) and rewrite the target XY to the depth centroid. With a
        centered camera the depth centroid IS the correct grasp point, so
        this corrects any XY drift from the original active-exploration
        detection (which was taken from a much higher viewpoint).
        """
        assert self.target is not None
        if self._traj is None:
            end_pos = np.array([self.target.position[0],
                                self.target.position[1],
                                PRE_GRASP_HOVER_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion,
                             DESCEND_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            self._refine_target_xy(robot)
            if timed_out:
                print("[FSM] DESCENDING timeout, advancing.")
            self._enter(State.LOWERING)

    def _do_lowering(self, t_sim, robot, timed_out):
        """
        Final slow descent from PRE_GRASP_HOVER_Z to grip_hand_z. At
        LOWERING_SPEED (~2.5 cm/s) so contact with the object is gentle and
        nothing gets bumped. The smoothstep trajectory tapers velocity to
        zero at the bottom, giving a soft landing.
        """
        assert self.target is not None
        if self._traj is None:
            end_pos = np.array([self.target.position[0],
                                self.target.position[1],
                                _grip_hand_z(self.target)])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion,
                             LOWERING_SPEED)
            print(f"[FSM] LOWERING: gentle descent to grip_hand_z="
                  f"{end_pos[2]:.3f} at {LOWERING_SPEED*1000:.0f}mm/s")
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] LOWERING timeout, advancing.")
            self._enter(State.GRASPING)

    def _do_grasping(self, t_sim, elapsed, robot):
        # Hold the descended pose, close fingers, dwell.
        assert self.target is not None
        end_pos = np.array([self.target.position[0],
                            self.target.position[1],
                            _grip_hand_z(self.target)])
        self._desired_pos = end_pos
        self._desired_quat = self.target.quaternion
        robot.close_gripper()
        if elapsed >= GRASP_DWELL:
            self._enter(State.VERIFY_GRASP)

    def _do_verify_grasp(self, t_sim, elapsed, robot):
        """
        Robust contact-based grasp verification. Stays at the grasp pose
        with fingers commanded closed; checks whether any finger is in
        physical contact with an external object (not the robot, not the
        table). Sets self._grasp_success which LIFTING reads to decide
        whether to TRANSPORT or just go back to EXPLORING.

        We use contact (mj_data.contact) instead of finger qpos because
        the finger position actuator + joint-limit soft constraints don't
        cleanly converge to qpos=0 in the grasp transient -- empty
        closures often settle several mm open, defeating any qpos
        threshold. Contacts are unambiguous.
        """
        assert self.target is not None
        end_pos = np.array([self.target.position[0],
                            self.target.position[1],
                            _grip_hand_z(self.target)])
        self._desired_pos = end_pos
        self._desired_quat = self.target.quaternion
        robot.close_gripper()  # keep commanding close while we wait

        # Let fingers finish closing + arm settle before sampling contacts.
        if elapsed < GRASP_VERIFY_DELAY:
            return

        try:
            holding = robot.is_holding_object()
            q1, q2 = robot.get_finger_qpos()
        except Exception as e:
            print(f"[FSM] VERIFY_GRASP: contact check failed ({e}), assuming success")
            self._grasp_success = True
            self._enter(State.LIFTING)
            return

        if holding:
            self._grasp_success = True
            print(f"[FSM] VERIFY_GRASP: HOLDING (finger contact w/ external "
                  f"geom; fingers at {q1*1000:.1f}, {q2*1000:.1f} mm)")
        else:
            self._grasp_success = False
            print(f"[FSM] VERIFY_GRASP: EMPTY (no external contact; fingers "
                  f"at {q1*1000:.1f}, {q2*1000:.1f} mm)")

        self._enter(State.LIFTING)

    def _do_lifting(self, t_sim, robot, timed_out):
        # Vertical lift back to SAFE_Z while keeping the same XY. If we
        # actually grabbed something, keep fingers closed and head to
        # TRANSPORTING. If the grasp came up empty, open the fingers and
        # skip straight back to EXPLORING (no point dropping nothing at the
        # drop zone).
        if self._traj is None:
            cur_pos, _ = robot.hand_pose()
            end_pos = np.array([cur_pos[0], cur_pos[1], SAFE_Z])
            quat = (self.target.quaternion if self.target is not None
                    else top_down_quat(0.0))
            self._start_traj(t_sim, robot, end_pos, quat, LIFT_SPEED)
        self._track_traj(t_sim)

        if self._grasp_success:
            robot.close_gripper()
        else:
            robot.open_gripper(EXPLORE_GRIPPER_OPENING)

        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] LIFTING timeout, advancing.")
            if self._grasp_success:
                self._enter(State.TRANSPORTING)
            else:
                # Empty grasp -> abandon target and go back to looking. The
                # object may have shifted, or perception was slightly off;
                # exploration will spot it (or another object) again.
                print("[FSM] LIFTING: empty grasp confirmed -> EXPLORING")
                self.target = None
                self._enter(State.EXPLORING)

    def _do_transporting(self, t_sim, robot, timed_out):
        # Horizontal move to over the drop zone.
        assert self.target is not None
        if self._traj is None:
            end_pos = np.array([DROP_HAND_POS[0], DROP_HAND_POS[1], SAFE_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion, TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.close_gripper()
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] TRANSPORTING timeout, advancing.")
            self._enter(State.PLACING)

    def _do_placing(self, t_sim, elapsed, robot):
        # Hold drop pose, open fingers, dwell.
        self._desired_pos = DROP_HAND_POS
        if self.target is not None:
            self._desired_quat = self.target.quaternion
        robot.open_gripper()
        if elapsed >= PLACE_DWELL:
            self._enter(State.RETREATING)

    def _do_retreating(self, t_sim, robot, timed_out):
        # Successful place complete -> lift up and resume exploring from
        # waypoint 0 with a fresh perception interval.
        if self._traj is None:
            cur_pos, cur_quat = robot.hand_pose()
            end_pos = np.array([cur_pos[0], cur_pos[1], SAFE_Z + RETREAT_OFFSET])
            self._start_traj(t_sim, robot, end_pos, cur_quat, LIFT_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper()
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] RETREATING timeout, advancing.")
            self.target = None
            self._explore_idx = 0
            self._explore_perceive_t = -1e9
            self._enter(State.EXPLORING)
