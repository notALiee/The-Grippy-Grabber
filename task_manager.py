from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from control import top_down_quat
from perception import ObjectPose, CameraView
from planner import CartesianRRT


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
# Class-keyed drop zones. The arm releases each grasped object at the
# panda_hand pose listed for its class; objects fall ~19 cm. Stacking
# inside a zone is fine -- the planner only sees the static walls and
# the FSM uses contact-based grasp verification, so a tower of pens
# doesn't confuse anything. Keep all three zones inside the table
# limits the user gave (X in [-0.25, 1.10], Y in [-0.675, 0.25]).
DROP_HAND_Z = 0.30
DROP_ZONES = {
    "book":           np.array([0.30,  0.20, DROP_HAND_Z]),  # back-left of workspace
    "cup":            np.array([0.90,  0.20, DROP_HAND_Z]),  # back-right (shared w/ pens)
    "pen":            np.array([0.90,  0.20, DROP_HAND_Z]),
    "PuzzleBase":     np.array([0.60, -0.60, DROP_HAND_Z]),  # front-center
    "PuzzleCircle":   np.array([0.60, -0.60, DROP_HAND_Z]),
    "PuzzleSquare":   np.array([0.60, -0.60, DROP_HAND_Z]),
    "PuzzleTriangle": np.array([0.60, -0.60, DROP_HAND_Z]),
}
# Fallback for any unknown class -- well clear of all named zones.
_DROP_DEFAULT = np.array([0.0, 0.50, DROP_HAND_Z])
# Backwards-compat alias for any external code still importing the old name.
DROP_HAND_POS = _DROP_DEFAULT
# Detections within DROP_ZONE_INHIBIT_RADIUS of any drop zone are
# suppressed in _look_for_target so the arm doesn't try to "grasp" the
# object it just placed (or one stacked on top).
DROP_ZONE_INHIBIT_RADIUS = 0.10


def _drop_pos_for(cls: str) -> np.ndarray:
    return DROP_ZONES.get(cls, _DROP_DEFAULT).copy()
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

# ---- Active exploration (gripper-mounted cameras) ---------------------------
# Perception uses the center + four lateral RGB-D cameras on the hand (±Y, ±X).
# The arm continuously sweeps EXPLORE_WAYPOINTS while running merged
# perception every EXPLORE_PERCEIVE_INTERVAL seconds. The FIRST high-confidence detection
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
# Five-view perception (center + 4 horizontal cams): merge radius wider
# than single-view because lateral cameras introduce parallax.
MULTIVIEW_MERGE_RADIUS = 0.062

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

# ---- Quick depth-probe verification (runs at detection time) ---------------
# As soon as EXPLORING spots a candidate we don't immediately commit to it.
# We sample the gripper-cam depth feed at the candidate XY and at +/-
# DEPTH_PROBE_OFFSET meters on both the X and Y axes (in world frame). If
# at least DEPTH_PROBE_MIN_HITS of those probes report a pixel that's
# clearly above the table, the object is real and the confirming probe
# pixels are de-projected back to world space and averaged into a refined
# grasp center (depth-derived, no YOLO bbox bias). Otherwise the candidate
# is treated as a ghost and EXPLORING keeps sweeping.
DEPTH_PROBE_OFFSET = 0.0075           # m, +/-7.5 mm probe arm on X and Y
DEPTH_PROBE_TABLE_MARGIN = 0.005      # m, must sit >=5 mm above the table
DEPTH_PROBE_MIN_HITS = 3              # min confirming probes (out of 5)

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


# Names must match <camera name="..."/> in panda.xml.
GRIPPER_CAM_CENTER = "gripper_camera"
# Lateral RGB-D ring on the hand: ±Y (sides) + ±X (front/back in hand frame).
GRIPPER_CAM_RING_NAMES = (
    "gripper_depth_side_ypos",
    "gripper_depth_side_yneg",
    "gripper_depth_side_xpos",
    "gripper_depth_side_xneg",
)
GRIPPER_CAM_SIDE_NAMES = GRIPPER_CAM_RING_NAMES  # backwards-compat alias


# ----------------------------------------------------------------------------
# Renderer bundle handed to the FSM by main.py
#
# Center gripper cam + four lateral RGB-D cams (hand ±Y, ±X). All use the
# same scene_option (robot hidden) during perception so fingers don’t
# corrupt depth.
# ----------------------------------------------------------------------------
@dataclass
class _SceneRenderers:
    gripper_rgb_renderer: object
    gripper_depth_renderer: object
    gripper_intrinsics: dict
    side_rgb_renderers: Tuple[object, ...]
    side_depth_renderers: Tuple[object, ...]
    side_intrinsics: Tuple[dict, ...]
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


def _depth_probe_verify(view: CameraView, candidate: ObjectPose,
                        offset: float = DEPTH_PROBE_OFFSET,
                        table_margin: float = DEPTH_PROBE_TABLE_MARGIN,
                        min_hits: int = DEPTH_PROBE_MIN_HITS
                        ) -> Optional[ObjectPose]:
    """
    Quick depth-feed sanity check + recenter. Probes the depth image at
    the candidate XY plus +/- `offset` on both X and Y axes (5 probes).
    Each probe pixel must report a depth value clearly closer than the
    table to count as a hit. If at least `min_hits` probes hit, the
    confirming probes are de-projected back to world coordinates and
    averaged into a depth-derived grasp center; otherwise None.
    """
    depth = np.asarray(view.depth, dtype=np.float32)
    H, W = depth.shape[:2]
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size < 200:
        return None
    table_depth = float(np.median(finite))

    intr = view.intrinsics
    f = float(intr["f"])
    cx_intr = float(intr["cx"])
    cy_intr = float(intr["cy"])
    R = np.asarray(view.cam_xmat).reshape(3, 3)
    cam_xpos = np.asarray(view.cam_xpos)
    obj_z = float(candidate.position[2])

    def world_to_pixel(world_xy):
        # Project a world-frame point at the object's z back into the
        # depth image. Returns (u, v) ints or None if off-frame / behind.
        p_cam = R.T @ (np.array([world_xy[0], world_xy[1], obj_z]) - cam_xpos)
        z = -float(p_cam[2])  # MuJoCo cam looks down its -Z axis
        if z <= 0.01:
            return None
        u = p_cam[0] * f / z + cx_intr
        v = -p_cam[1] * f / z + cy_intr
        if not (0 <= u < W and 0 <= v < H):
            return None
        return int(round(u)), int(round(v))

    base_xy = candidate.position[:2]
    probes_xy = [
        base_xy,
        base_xy + np.array([+offset, 0.0]),
        base_xy + np.array([-offset, 0.0]),
        base_xy + np.array([0.0, +offset]),
        base_xy + np.array([0.0, -offset]),
    ]

    hits_world_xy: List[np.ndarray] = []
    for p_xy in probes_xy:
        px = world_to_pixel(p_xy)
        if px is None:
            continue
        u, v = px
        d = float(depth[v, u])
        if not (np.isfinite(d) and d > 0.01 and d < table_depth - table_margin):
            continue
        # De-project the hit pixel back to world (matches perception convention).
        x_cam = (u - cx_intr) * d / f
        y_cam = -(v - cy_intr) * d / f
        z_cam = -d
        p_world = R @ np.array([x_cam, y_cam, z_cam]) + cam_xpos
        hits_world_xy.append(p_world[:2])

    if len(hits_world_xy) < min_hits:
        return None

    refined_xy = np.mean(np.stack(hits_world_xy, axis=0), axis=0)
    refined_pos = candidate.position.copy()
    refined_pos[:2] = refined_xy
    return ObjectPose(
        class_name=candidate.class_name,
        position=refined_pos,
        quaternion=candidate.quaternion,
        score=candidate.score,
        n_points=candidate.n_points,
        principal_axis=candidate.principal_axis,
        source_cam=candidate.source_cam,
    )


def _capture_single_camera_view(
        data, renderers: _SceneRenderers,
        rgb_r, dep_r, cam_name: str, intr: dict, name: str) -> CameraView:
    """One RGB+depth frame from a named MuJoCo camera; robot hidden via scene_option."""
    so = renderers.scene_option
    if so is not None:
        rgb_r.update_scene(data, camera=cam_name, scene_option=so)
        dep_r.update_scene(data, camera=cam_name, scene_option=so)
    else:
        rgb_r.update_scene(data, camera=cam_name)
        dep_r.update_scene(data, camera=cam_name)
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


def _capture_gripper_view(data, renderers: _SceneRenderers,
                          name: str = "gripper") -> CameraView:
    """
    Downward center gripper camera only (used for DESCEND-refine depth
    contour + depth-probe, where we want the unbiased overhead ray).
    """
    return _capture_single_camera_view(
        data, renderers,
        renderers.gripper_rgb_renderer,
        renderers.gripper_depth_renderer,
        GRIPPER_CAM_CENTER,
        renderers.gripper_intrinsics,
        name,
    )


def _capture_ring_gripper_views(data, renderers: _SceneRenderers) -> List[CameraView]:
    """
    Downward center cam + four horizontal views (±Y and ±X in hand frame).
    Fed to perception.scan_multi for all-round gripper-fixed vision.
    """
    views: List[CameraView] = [
        _capture_single_camera_view(
            data, renderers,
            renderers.gripper_rgb_renderer,
            renderers.gripper_depth_renderer,
            GRIPPER_CAM_CENTER,
            renderers.gripper_intrinsics,
            "gripper_center",
        ),
    ]
    for i, cam_name in enumerate(GRIPPER_CAM_RING_NAMES):
        views.append(_capture_single_camera_view(
            data, renderers,
            renderers.side_rgb_renderers[i],
            renderers.side_depth_renderers[i],
            cam_name,
            renderers.side_intrinsics[i],
            f"gripper_ring_{i}",
        ))
    return views



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

        # RRT-planned multi-leg path. _waypoints[0] is the start of the
        # current leg, _waypoints[-1] is the goal. _waypoint_idx points
        # at the END of the leg the current Trajectory is animating; on
        # leg completion we advance the index and build the next
        # Trajectory. Empty list / idx >= len means "no path active".
        self._waypoints: List[np.ndarray] = []
        self._waypoint_idx: int = 0
        self._leg_end_quat: np.ndarray = top_down_quat(0.0)
        self._leg_speed: float = TRANSIT_SPEED

        # One Cartesian RRT instance, reused across all transit moves.
        self._rrt = CartesianRRT()

        # Active-exploration bookkeeping.
        self._explore_idx: int = 0            # next EXPLORE_WAYPOINTS entry
        self._explore_perceive_t: float = -1e9  # last perception time (sim seconds)
        self._explore_lap: int = 0            # full passes with no detection

        # Multi-view CONFIRMING bookkeeping. _confirm_idx walks
        # CONFIRM_VIEWPOINTS_OFFSETS; _confirm_hits accumulates
        # successful re-detections (one per viewpoint at most).
        self._confirm_idx: int = 0
        self._confirm_arrived_t: Optional[float] = None
        self._confirm_hits: List[ObjectPose] = []

        # Failed-target memory: list of (xy, expiry_time) tuples. New
        # candidates within FAILED_TARGET_RADIUS of any unexpired entry
        # are skipped so we don't re-lock onto the same false positive.
        self._failed_targets: List[Tuple[np.ndarray, float]] = []

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
        elif self.state == State.CONFIRMING:
            self._do_confirming(t_sim, elapsed, data, perception, robot, renderers, timed_out)
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
            # Leaving CONFIRMING through any code path (including
            # success/timeout inside _do_confirming) clears the orbit
            # state so the next confirm cycle starts fresh.
            if self.state == State.CONFIRMING and new_state != State.CONFIRMING:
                self._reset_confirm_state()
            self.state = new_state
        self._state_t0 = None
        self._traj = None
        self._dwell_t0 = None
        self._waypoints = []
        self._waypoint_idx = 0

    def _start_traj(self, t_sim, robot, end_pos, end_quat, speed,
                    use_rrt: bool = True):
        """
        Build a (multi-leg if RRT-planned) Cartesian path from the
        current hand pose to `end_pos` at `end_quat`. Each leg is a
        smoothstep Trajectory; on leg completion `_track_traj` rolls
        forward to the next one until the whole path is done. Falls
        back to a straight segment if RRT can't find a route.
        """
        cur_pos, cur_quat = robot.hand_pose()
        end_pos = np.asarray(end_pos, dtype=float)
        end_quat = np.asarray(end_quat, dtype=float)

        path: Optional[List[np.ndarray]] = None
        if use_rrt:
            try:
                path = self._rrt.plan(cur_pos, end_pos)
            except Exception as e:
                print(f"[FSM] RRT planner failed ({e}); using straight segment")
                path = None

        if path is None or len(path) < 2:
            # Either RRT was disabled, planner failed, or already trivial.
            self._waypoints = [cur_pos.copy(), end_pos.copy()]
            if use_rrt and path is None:
                print("[FSM] WARNING: RRT found no path — using straight line "
                      "(collision risk)")
        else:
            self._waypoints = [np.asarray(p, dtype=float).copy() for p in path]

        self._waypoint_idx = 1
        self._leg_end_quat = end_quat
        self._leg_speed = float(speed)

        # First leg: start hand pose -> first intermediate waypoint.
        self._traj = self._build_leg_traj(cur_pos, cur_quat, t_sim)
        if len(self._waypoints) > 2:
            print(f"[FSM] RRT path: {len(self._waypoints)} waypoints "
                  f"({sum(float(np.linalg.norm(self._waypoints[i+1] - self._waypoints[i])) for i in range(len(self._waypoints)-1))*100:.0f}cm total)")

    def _build_leg_traj(self, start_pos: np.ndarray, start_quat: np.ndarray,
                        t_sim: float) -> Trajectory:
        """
        Build the smoothstep Trajectory for the leg ending at
        self._waypoints[self._waypoint_idx]. Quaternion slerps from
        start_quat to a partial-blend toward _leg_end_quat sized by
        this leg's share of the remaining path length, so by the time
        the final leg finishes the orientation has reached
        _leg_end_quat exactly.
        """
        end_pos = self._waypoints[self._waypoint_idx]
        # Leg's share of the remaining-path length (avoids huge orientation
        # snaps on short last-mile legs).
        remaining = sum(
            float(np.linalg.norm(self._waypoints[i + 1] - self._waypoints[i]))
            for i in range(self._waypoint_idx - 1, len(self._waypoints) - 1)
        )
        leg_len = float(np.linalg.norm(end_pos - start_pos))
        if remaining > 1e-6:
            leg_alpha = float(np.clip(leg_len / remaining, 0.0, 1.0))
        else:
            leg_alpha = 1.0
        leg_end_quat = _quat_nlerp(start_quat, self._leg_end_quat, leg_alpha)
        return Trajectory.build(start_pos, start_quat, end_pos, leg_end_quat,
                                t_sim, self._leg_speed)

    def _track_traj(self, t_sim):
        if self._traj is None:
            return
        self._desired_pos, self._desired_quat = self._traj.pose_at(t_sim)
        # Auto-roll to the next leg of an RRT path on leg completion.
        if (self._traj.done(t_sim)
                and self._waypoint_idx + 1 < len(self._waypoints)):
            self._waypoint_idx += 1
            self._traj = self._build_leg_traj(
                self._waypoints[self._waypoint_idx - 1].copy(),
                self._desired_quat.copy(),
                t_sim,
            )

    def _traj_fully_done(self, t_sim: float) -> bool:
        """True iff the current trajectory AND all remaining waypoints
        of the active RRT path are complete. States should use this in
        place of `self._traj.done(t_sim)` so they only advance when the
        full multi-leg path has been animated."""
        if self._traj is None:
            return True
        return (self._traj.done(t_sim)
                and self._waypoint_idx + 1 >= len(self._waypoints))

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
            self._prune_failed_targets(t_sim)
            target = self._look_for_target(data, perception, robot, renderers, t_sim)
            if target is not None:
                self.target = target
                self._explore_lap = 0  # we found something, reset lap counter
                self._traj = None
                print(f"[FSM] EXPLORING: spotted {target.class_name} "
                      f"score={target.score:.2f} at {target.position}")
                self._enter(State.CONFIRMING)
                return

        # Reached this waypoint? Go to the next one on the next tick.
        if self._traj is not None and self._traj_fully_done(t_sim):
            self._explore_idx += 1
            self._traj = None
            self._waypoints = []

    def _look_for_target(self, data, perception, robot, renderers,
                         t_sim: float = 0.0) -> Optional[ObjectPose]:
        """
        Capture gripper RGB-D from the center + four lateral cameras,
        run YOLO + pose per view and merge via scan_multi (handles occlusion
        where the top-down cam is blind).
        Also rejects targets that are physically unsafe to grasp:
          - too close to the Panda base (arm has to fold tightly + sweeps
            through neighboring objects, which has caused NaN-level contact
            blowups in mujoco)
          - off-table z range
        """
        view = _capture_gripper_view(data, renderers, name="explore")
        views = _capture_ring_gripper_views(data, renderers)
        poses = perception.scan_multi(
            views, min_points=20, min_score=EXPLORE_MIN_SCORE,
            merge_radius=MULTIVIEW_MERGE_RADIUS,
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
            if not (MIN_REACH <= r <= MAX_REACH):
                print(f"[FSM] EXPLORING: skipping {p.class_name} at "
                      f"{p.position} (reach={r*1000:.0f}mm out of "
                      f"[{MIN_REACH*1000:.0f},{MAX_REACH*1000:.0f}]mm)")
                continue
            if self._is_failed_target(p.position[:2], t_sim):
                print(f"[FSM] EXPLORING: skipping {p.class_name} at "
                      f"{p.position} (recently failed CONFIRMING)")
                continue
            if self._is_in_drop_zone(p.position[:2]):
                print(f"[FSM] EXPLORING: skipping {p.class_name} at "
                      f"{p.position} (within drop-zone inhibit radius)")
                continue
            in_reach.append(p)
        if not in_reach:
            return None

        # Highest confidence wins. Tie-break on number of supporting depth points.
        in_reach.sort(key=lambda p: (-p.score, -p.n_points))

        # Quick depth-feed verification at +/- DEPTH_PROBE_OFFSET on X and Y.
        # Walk best-first; first candidate that survives the probe wins. Anything
        # that fails is treated as a perception ghost and we keep sweeping.
        for cand in in_reach:
            verified = _depth_probe_verify(view, cand)
            if verified is not None:
                shift_mm = (verified.position[:2] - cand.position[:2]) * 1000.0
                print(f"[FSM] EXPLORING: depth-probe verified {cand.class_name} "
                      f"(+/-{DEPTH_PROBE_OFFSET*1000:.1f}mm probes), "
                      f"recenter shift=({shift_mm[0]:+.1f},{shift_mm[1]:+.1f})mm")
                return verified
            print(f"[FSM] EXPLORING: depth-probe REJECTED {cand.class_name} "
                  f"at {cand.position} (no consistent above-table depth)")
        return None

    # ------------------------------------------------------------------
    # Failed-target memory
    # ------------------------------------------------------------------
    def _prune_failed_targets(self, t_sim: float) -> None:
        """Drop expired entries from `_failed_targets`."""
        if not self._failed_targets:
            return
        self._failed_targets = [(xy, exp) for xy, exp in self._failed_targets
                                if exp > t_sim]

    def _is_failed_target(self, xy: np.ndarray, t_sim: float) -> bool:
        """True iff `xy` is within FAILED_TARGET_RADIUS of an unexpired
        failed-target entry."""
        for fxy, exp in self._failed_targets:
            if exp <= t_sim:
                continue
            if float(np.linalg.norm(np.asarray(xy) - fxy)) < FAILED_TARGET_RADIUS:
                return True
        return False

    def _remember_failed_target(self, xy: np.ndarray, t_sim: float) -> None:
        """Record `xy` as a failed CONFIRMING target with the standard
        TTL so EXPLORING doesn't re-lock onto the same false positive."""
        self._failed_targets.append((np.asarray(xy, dtype=float).copy(),
                                     t_sim + FAILED_TARGET_TTL))

    @staticmethod
    def _is_in_drop_zone(xy: np.ndarray) -> bool:
        """True iff `xy` is within DROP_ZONE_INHIBIT_RADIUS of any drop
        zone XY -- prevents the arm from re-grasping objects it (or a
        previous run) just placed."""
        xy_arr = np.asarray(xy)
        for zone in DROP_ZONES.values():
            if float(np.linalg.norm(xy_arr - zone[:2])) < DROP_ZONE_INHIBIT_RADIUS:
                return True
        return False

    # ------------------------------------------------------------------
    # CONFIRMING: orbit + multi-view re-scan
    # ------------------------------------------------------------------
    def _do_confirming(self, t_sim, elapsed, data, perception, robot,
                       renderers, timed_out):
        """
        After EXPLORING locks onto a candidate, orbit the gripper camera
        through CONFIRM_VIEWPOINTS_OFFSETS and re-run YOLO + depth at
        each viewpoint. The robot physically moves a few cm between
        snapshots; this disambiguates real objects (which appear at the
        same world XY across viewpoints) from depth-noise ghosts
        (which don't), and the parallax also gives the depth-derived
        center much better accuracy than any single bird's-eye frame.

        Decision:
          - >= CONFIRM_REQUIRED_HITS same-class detections within
            CONFIRM_MAX_XY_DRIFT of the original  -> refine
            target.position to the (n_points-weighted) mean of the
            confirming detections, transition to PLANNING.
          - otherwise -> remember as failed target, transition to
            EXPLORING. The TTL keeps us from immediately re-locking.
        """
        if self.target is None:
            self._enter(State.EXPLORING)
            return

        # Pre-size the gripper to the class-specific opening so it's
        # already settled by the time we reach LOWERING.
        robot.open_gripper(_get_gripper_opening(self.target))

        # Move to the next viewpoint if we don't have a trajectory.
        if self._traj is None and self._confirm_idx < len(CONFIRM_VIEWPOINTS_OFFSETS):
            offset = CONFIRM_VIEWPOINTS_OFFSETS[self._confirm_idx]
            end_pos = np.array([self.target.position[0] + offset[0],
                                self.target.position[1] + offset[1],
                                CONFIRM_HOVER_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion,
                             TRANSIT_SPEED)
            self._confirm_arrived_t = None
            print(f"[FSM] CONFIRMING: viewpoint "
                  f"{self._confirm_idx + 1}/{len(CONFIRM_VIEWPOINTS_OFFSETS)} "
                  f"offset=({offset[0]*100:+.0f},{offset[1]*100:+.0f})cm")

        self._track_traj(t_sim)

        if timed_out:
            print(f"[FSM] CONFIRMING timeout with {len(self._confirm_hits)} hits, "
                  f"abandoning {self.target.class_name}")
            self._remember_failed_target(self.target.position[:2], t_sim)
            self._reset_confirm_state()
            self.target = None
            self._enter(State.EXPLORING)
            return

        # Wait for the move to finish, then settle, then snap.
        if self._traj_fully_done(t_sim):
            if self._confirm_arrived_t is None:
                self._confirm_arrived_t = t_sim
            elif (t_sim - self._confirm_arrived_t) >= CONFIRM_SETTLE_TIME:
                # Take a snapshot and try to find the same class within
                # CONFIRM_MAX_XY_DRIFT.
                views = _capture_ring_gripper_views(data, renderers)
                poses = perception.scan_multi(
                    views, min_points=20,
                    min_score=CONFIRM_MIN_SCORE,
                    merge_radius=MULTIVIEW_MERGE_RADIUS,
                )
                hit = self._best_confirm_hit(poses)
                if hit is not None:
                    self._confirm_hits.append(hit)
                    print(f"[FSM] CONFIRMING: hit #{len(self._confirm_hits)} "
                          f"({hit.class_name} score={hit.score:.2f} at "
                          f"{hit.position})")
                else:
                    print(f"[FSM] CONFIRMING: viewpoint "
                          f"{self._confirm_idx + 1} -> no matching detection")

                # Advance to next viewpoint.
                self._confirm_idx += 1
                self._traj = None
                self._waypoints = []
                self._confirm_arrived_t = None

        # All viewpoints visited -> decide.
        if (self._confirm_idx >= len(CONFIRM_VIEWPOINTS_OFFSETS)
                and self._traj is None):
            if len(self._confirm_hits) >= CONFIRM_REQUIRED_HITS:
                refined = self._fuse_confirm_hits()
                self.target = refined
                print(f"[FSM] CONFIRMING: PASSED ({len(self._confirm_hits)} "
                      f"hits) -> refined target {refined.class_name} at "
                      f"{refined.position}")
                self._reset_confirm_state()
                self._enter(State.PLANNING)
            else:
                print(f"[FSM] CONFIRMING: FAILED ({len(self._confirm_hits)} "
                      f"hits, need {CONFIRM_REQUIRED_HITS}) -> EXPLORING")
                self._remember_failed_target(self.target.position[:2], t_sim)
                self._reset_confirm_state()
                self.target = None
                self._enter(State.EXPLORING)

    def _reset_confirm_state(self) -> None:
        self._confirm_idx = 0
        self._confirm_arrived_t = None
        self._confirm_hits = []

    def _best_confirm_hit(self, poses: List[ObjectPose]) -> Optional[ObjectPose]:
        """Pick the highest-confidence detection that matches the
        candidate class and lies within CONFIRM_MAX_XY_DRIFT of the
        candidate XY. Returns None if no detection qualifies."""
        if not poses or self.target is None:
            return None
        cand_xy = self.target.position[:2]
        cls = self.target.class_name
        best = None
        best_key = (0.0, 0)
        for p in poses:
            if p.class_name != cls:
                continue
            if float(np.linalg.norm(p.position[:2] - cand_xy)) > CONFIRM_MAX_XY_DRIFT:
                continue
            key = (p.score, p.n_points)
            if key > best_key:
                best_key = key
                best = p
        return best

    def _fuse_confirm_hits(self) -> ObjectPose:
        """Average the confirming detections' XYs (weighted by
        n_points). Keeps the highest-confidence quaternion + class."""
        assert self._confirm_hits
        weights = np.array([max(h.n_points, 1) for h in self._confirm_hits],
                           dtype=float)
        positions = np.stack([h.position for h in self._confirm_hits], axis=0)
        fused_pos = np.average(positions, axis=0, weights=weights)
        # Z stays as the visible-top depth value already estimated.
        best = max(self._confirm_hits, key=lambda h: (h.score, h.n_points))
        return ObjectPose(
            class_name=best.class_name,
            position=fused_pos,
            quaternion=best.quaternion,
            score=best.score,
            n_points=int(weights.sum()),
            principal_axis=best.principal_axis,
            source_cam=best.source_cam,
        )

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
        # Routed through RRT in case a wall sits between the post-grasp
        # pose and the home pose.
        if self._traj is None:
            assert self.target is not None
            self._start_traj(t_sim, robot, HOME_HAND_POS, self.target.quaternion,
                             TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if self._traj_fully_done(t_sim) or timed_out:
            if timed_out:
                print("[FSM] HOMING timeout, advancing.")
            self._enter(State.APPROACHING)

    def _do_approaching(self, t_sim, robot, timed_out):
        # Horizontal move to (target.x, target.y, SAFE_Z) at target
        # orientation, RRT-planned around walls.
        assert self.target is not None
        if self._traj is None:
            end_pos = np.array([self.target.position[0],
                                self.target.position[1],
                                SAFE_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion, TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.open_gripper(_get_gripper_opening(self.target))
        if self._traj_fully_done(t_sim) or timed_out:
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
            # use_rrt=False: descent is straight-down inside the verified
            # depth-clear column above the target; planning around walls
            # would only invent detours.
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion,
                             DESCEND_SPEED, use_rrt=False)
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
                             LOWERING_SPEED, use_rrt=False)
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
            self._start_traj(t_sim, robot, end_pos, quat, LIFT_SPEED,
                             use_rrt=False)
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
        # RRT-routed move to over the class-specific drop zone.
        assert self.target is not None
        if self._traj is None:
            drop_pos = _drop_pos_for(self.target.class_name)
            end_pos = np.array([drop_pos[0], drop_pos[1], SAFE_Z])
            self._start_traj(t_sim, robot, end_pos, self.target.quaternion,
                             TRANSIT_SPEED)
        self._track_traj(t_sim)
        robot.close_gripper()
        if self._traj_fully_done(t_sim) or timed_out:
            if timed_out:
                print("[FSM] TRANSPORTING timeout, advancing.")
            self._enter(State.PLACING)

    def _do_placing(self, t_sim, elapsed, robot):
        # Hold drop pose, open fingers, dwell.
        if self.target is not None:
            self._desired_pos = _drop_pos_for(self.target.class_name)
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
            self._start_traj(t_sim, robot, end_pos, cur_quat, LIFT_SPEED,
                             use_rrt=False)
        self._track_traj(t_sim)
        robot.open_gripper()
        if (self._traj is not None and self._traj.done(t_sim)) or timed_out:
            if timed_out:
                print("[FSM] RETREATING timeout, advancing.")
            self.target = None
            self._explore_idx = 0
            self._explore_perceive_t = -1e9
            self._enter(State.EXPLORING)
