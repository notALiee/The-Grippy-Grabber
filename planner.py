"""
Fast Cartesian RRT-Connect path planner for the Panda end-effector.

Drop-in replacement for planner.py.

Key changes versus the plain RRT version:
- Bidirectional RRT-Connect instead of one-tree RRT, so wall detours are found faster.
- Safer endpoint handling when the hand starts/ends inside the inflated safety margin.
- Deterministic shortcut smoothing that aggressively removes ugly jittery waypoints.
- Path validation helpers so TaskFSM can avoid accidentally using unsafe straight-line fallbacks.

The planner is still Cartesian: it plans panda_hand XYZ waypoints only. Your OS-PD +
null-space controller in control.py handles tracking those Cartesian targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# World description; keep this in sync with panda_mujoco/world.xml
# ----------------------------------------------------------------------------
OBSTACLE_Z_TOP_PLANNING = 0.72

# AABB format: (x_min, y_min, z_min, x_max, y_max, z_max)
WORLD_OBSTACLES: List[Tuple[float, float, float, float, float, float]] = [
    # wall_a: pos=(0.40, -0.20, 0.20), size=(0.01, 0.20, 0.20)
    (0.39, -0.40, 0.0, 0.41, 0.00, OBSTACLE_Z_TOP_PLANNING),
    # wall_b: pos=(0.75,  0.05, 0.25), size=(0.15, 0.01, 0.25)
    (0.60, 0.04, 0.0, 0.90, 0.06, OBSTACLE_Z_TOP_PLANNING),
    # wall_c: pos=(0.20,  0.10, 0.18), size=(0.01, 0.10, 0.18)
    (0.19, 0.00, 0.0, 0.21, 0.20, OBSTACLE_Z_TOP_PLANNING),
]

# (x_min, x_max, y_min, y_max, z_min, z_max)
WORKSPACE_BOUNDS: Tuple[float, float, float, float, float, float] = (
    -0.25, 1.10,
    -0.675, 0.25,
    0.075, 0.65,
)

XY_SAFETY_MARGIN = 0.04
Z_SAFETY_MARGIN = 0.06
DEFAULT_SAFETY_MARGIN = XY_SAFETY_MARGIN


# ----------------------------------------------------------------------------
# AABB collision math
# ----------------------------------------------------------------------------
def _inflate(box: Sequence[float], xy_margin: float, z_margin: float
             ) -> Tuple[float, float, float, float, float, float]:
    return (
        float(box[0]) - xy_margin,
        float(box[1]) - xy_margin,
        float(box[2]) - z_margin,
        float(box[3]) + xy_margin,
        float(box[4]) + xy_margin,
        float(box[5]) + z_margin,
    )


def point_in_aabb(p: np.ndarray, box: Sequence[float]) -> bool:
    p = np.asarray(p, dtype=float)
    return (
        box[0] <= p[0] <= box[3]
        and box[1] <= p[1] <= box[4]
        and box[2] <= p[2] <= box[5]
    )


def segment_hits_aabb(p0: np.ndarray, p1: np.ndarray,
                      box: Sequence[float]) -> bool:
    """
    Robust slab test for segment-vs-AABB.
    Returns True even for grazing contact, which is what we want for safety.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0

    t_near = -np.inf
    t_far = np.inf
    bmin = np.array([box[0], box[1], box[2]], dtype=float)
    bmax = np.array([box[3], box[4], box[5]], dtype=float)

    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if p0[axis] < bmin[axis] or p0[axis] > bmax[axis]:
                return False
            continue

        inv_d = 1.0 / d[axis]
        t1 = (bmin[axis] - p0[axis]) * inv_d
        t2 = (bmax[axis] - p0[axis]) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1

        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return False

    return t_far >= 0.0 and t_near <= 1.0


def segment_collides(p0: np.ndarray, p1: np.ndarray,
                     obstacles: Sequence[Sequence[float]]) -> bool:
    return any(segment_hits_aabb(p0, p1, box) for box in obstacles)


def path_collides(path: Sequence[np.ndarray],
                  obstacles: Sequence[Sequence[float]]) -> bool:
    if len(path) < 2:
        return False
    return any(segment_collides(path[i], path[i + 1], obstacles)
               for i in range(len(path) - 1))


# ----------------------------------------------------------------------------
# RRT-Connect
# ----------------------------------------------------------------------------
@dataclass
class _Node:
    pos: np.ndarray
    parent: int


class CartesianRRT:
    """
    Fast bidirectional Cartesian RRT-Connect in XYZ space.

    plan(start, goal) returns a list of numpy XYZ waypoints:
        [start, ..., goal]
    or None if no route is found.

    Notes:
    - Collision checks use inflated obstacle boxes for internal planning edges.
    - If start/goal is inside only the inflated margin, not the physical wall,
      the planner adds a short entry/exit connector and checks that connector
      against the raw obstacle boxes. This avoids the common bug where the robot
      starts 1 cm inside the safety margin and RRT refuses to grow at all.
    """

    def __init__(self,
                 bounds: Sequence[float] = WORKSPACE_BOUNDS,
                 obstacles: Sequence[Sequence[float]] = WORLD_OBSTACLES,
                 step_size: float = 0.055,
                 goal_bias: float = 0.12,
                 goal_tol: float = 0.035,
                 max_iters: int = 2500,
                 xy_margin: float = XY_SAFETY_MARGIN,
                 z_margin: float = Z_SAFETY_MARGIN,
                 smooth_passes: int = 80,
                 seed: Optional[int] = None):
        self.bounds = tuple(float(v) for v in bounds)
        self.raw_obstacles = [tuple(float(x) for x in b) for b in obstacles]
        self.obstacles = [_inflate(b, xy_margin, z_margin) for b in obstacles]
        self.step_size = float(step_size)
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        self.goal_tol = float(goal_tol)
        self.max_iters = int(max_iters)
        self.smooth_passes = int(smooth_passes)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, start_xyz: np.ndarray, goal_xyz: np.ndarray
             ) -> Optional[List[np.ndarray]]:
        raw_start = self._checked_point(start_xyz, "start")
        raw_goal = self._checked_point(goal_xyz, "goal")

        start = self._clamp_to_bounds(raw_start)
        goal = self._clamp_to_bounds(raw_goal)

        # If an endpoint is inside only the inflated margin, create a nearby
        # planning point outside the margin. Keep the original endpoint as the
        # actual first/last waypoint so TaskFSM still starts and ends correctly.
        start_plan = self._escape_inflated_if_needed(start, toward=goal)
        goal_plan = self._escape_inflated_if_needed(goal, toward=start)

        if start_plan is None or goal_plan is None:
            return None

        # Connector segments are allowed to pass through the *inflated* margin,
        # but never through the physical/raw obstacle.
        prefix: List[np.ndarray] = [start]
        if not self._same_point(start, start_plan):
            if segment_collides(start, start_plan, self.raw_obstacles):
                return None
            prefix.append(start_plan)

        suffix: List[np.ndarray] = []
        if not self._same_point(goal_plan, goal):
            if segment_collides(goal_plan, goal, self.raw_obstacles):
                return None
            suffix.append(goal)

        # Fast path when the repaired planning endpoints can see each other.
        if not segment_collides(start_plan, goal_plan, self.obstacles):
            return self._dedupe_path(prefix + [goal_plan] + suffix)

        core = self._rrt_connect(start_plan, goal_plan)
        if core is None:
            core = self._deterministic_fallback(start_plan, goal_plan)
        if core is None:
            return None

        core = self._shortcut_smooth(core)
        full_path = self._dedupe_path(prefix + core[1:-1] + [goal_plan] + suffix)

        # Internal/core segments should be inflated-safe. Endpoint connector
        # segments may be only raw-safe when the endpoint is in the safety margin.
        if not self._path_is_acceptable(full_path, start, goal):
            return None
        return full_path

    def validate_path(self, path: Sequence[np.ndarray]) -> bool:
        """Strict validation against inflated planning obstacles."""
        if len(path) < 2:
            return True
        if any(self._point_in_any(p, self.raw_obstacles) for p in path):
            return False
        return not path_collides(path, self.obstacles)

    # ------------------------------------------------------------------
    # RRT-Connect internals
    # ------------------------------------------------------------------
    def _rrt_connect(self, start: np.ndarray, goal: np.ndarray
                     ) -> Optional[List[np.ndarray]]:
        if self._point_in_any(start, self.obstacles):
            return None
        if self._point_in_any(goal, self.obstacles):
            return None

        tree_start: List[_Node] = [_Node(start.copy(), -1)]
        tree_goal: List[_Node] = [_Node(goal.copy(), -1)]

        flipped = False
        for k in range(self.max_iters):
            if self._rng.random() < self.goal_bias:
                sample = goal if not flipped else start
            else:
                sample = self._sample_free()

            tree_a = tree_goal if flipped else tree_start
            tree_b = tree_start if flipped else tree_goal

            status_a, idx_a = self._extend(tree_a, sample)
            if status_a == "trapped":
                flipped = not flipped
                continue

            new_pos = tree_a[idx_a].pos
            status_b, idx_b = self._connect(tree_b, new_pos)
            if status_b == "reached":
                if not flipped:
                    left = self._trace(tree_start, idx_a)          # start -> meet
                    right = self._trace(tree_goal, idx_b)          # goal -> meet
                else:
                    left = self._trace(tree_start, idx_b)          # start -> meet
                    right = self._trace(tree_goal, idx_a)          # goal -> meet
                return self._dedupe_path(left + list(reversed(right)))

            # Alternate which side grows. This keeps the trees balanced.
            flipped = not flipped

        return None

    def _extend(self, tree: List[_Node], target: np.ndarray) -> Tuple[str, int]:
        nearest_idx = self._nearest(tree, target)
        nearest = tree[nearest_idx].pos
        new_pos = self._steer(nearest, target)

        if self._point_in_any(new_pos, self.obstacles):
            return "trapped", nearest_idx
        if segment_collides(nearest, new_pos, self.obstacles):
            return "trapped", nearest_idx

        tree.append(_Node(new_pos, nearest_idx))
        new_idx = len(tree) - 1

        if np.linalg.norm(new_pos - target) <= self.goal_tol:
            return "reached", new_idx
        return "advanced", new_idx

    def _connect(self, tree: List[_Node], target: np.ndarray) -> Tuple[str, int]:
        last_status = "advanced"
        last_idx = self._nearest(tree, target)

        # Hard cap prevents infinite loops if target is almost reachable but
        # numerical tolerances keep the status at advanced forever.
        for _ in range(128):
            status, idx = self._extend(tree, target)
            if status == "trapped":
                return last_status, last_idx
            last_status, last_idx = status, idx
            if status == "reached":
                return status, idx
        return last_status, last_idx

    def _trace(self, tree: List[_Node], idx: int) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        while idx >= 0:
            out.append(tree[idx].pos.copy())
            idx = tree[idx].parent
        out.reverse()
        return out

    # ------------------------------------------------------------------
    # Sampling / geometry helpers
    # ------------------------------------------------------------------
    def _checked_point(self, p: np.ndarray, name: str) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(p)):
            raise ValueError(f"{name} contains NaN/Inf: {p}")
        return p

    def _sample_free(self) -> np.ndarray:
        low = np.array([self.bounds[0], self.bounds[2], self.bounds[4]], dtype=float)
        high = np.array([self.bounds[1], self.bounds[3], self.bounds[5]], dtype=float)
        for _ in range(64):
            p = self._rng.uniform(low=low, high=high)
            if not self._point_in_any(p, self.obstacles):
                return p
        return self._rng.uniform(low=low, high=high)

    def _nearest(self, tree: List[_Node], q: np.ndarray) -> int:
        # Vectorized nearest is much faster than looping in Python once the
        # trees reach hundreds/thousands of nodes.
        pts = np.array([node.pos for node in tree])
        return int(np.argmin(np.sum((pts - q) ** 2, axis=1)))

    def _steer(self, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
        delta = to - frm
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return to.copy()
        return frm + delta * (self.step_size / dist)

    def _clamp_to_bounds(self, p: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(p[0], self.bounds[0], self.bounds[1]),
            np.clip(p[1], self.bounds[2], self.bounds[3]),
            np.clip(p[2], self.bounds[4], self.bounds[5]),
        ], dtype=float)

    def _point_in_any(self, p: np.ndarray,
                      obstacles: Sequence[Sequence[float]]) -> bool:
        return any(point_in_aabb(p, b) for b in obstacles)

    def _escape_inflated_if_needed(self, p: np.ndarray,
                                   toward: np.ndarray) -> Optional[np.ndarray]:
        """
        If p is inside an inflated obstacle, push it to the nearest free face.
        This is for safety-margin issues, not for planning through real walls.
        """
        p = self._clamp_to_bounds(p)
        if not self._point_in_any(p, self.obstacles):
            return p

        # If it is inside the raw physical obstacle, still try to escape. This
        # can happen after a recovery, but we will reject connector segments
        # that cut through raw geometry.
        cand = p.copy()
        eps = 2e-3
        for _ in range(8):
            containing = [b for b in self.obstacles if point_in_aabb(cand, b)]
            if not containing:
                return self._clamp_to_bounds(cand)

            # Push out of the nearest face of the tightest containing box.
            best_move = None
            best_dist = float("inf")
            for b in containing:
                mins = np.array([b[0], b[1], b[2]], dtype=float)
                maxs = np.array([b[3], b[4], b[5]], dtype=float)
                for axis in range(3):
                    lower_dist = abs(cand[axis] - mins[axis])
                    upper_dist = abs(maxs[axis] - cand[axis])

                    # Tie-break: prefer moving in the general direction of
                    # `toward` so the first connector does not U-turn hard.
                    sign_hint = np.sign(toward[axis] - cand[axis])
                    if lower_dist < best_dist:
                        best_dist = lower_dist
                        best_move = (axis, mins[axis] - eps)
                    if upper_dist < best_dist or (
                        abs(upper_dist - best_dist) < 1e-9 and sign_hint >= 0
                    ):
                        best_dist = upper_dist
                        best_move = (axis, maxs[axis] + eps)

            if best_move is None:
                break
            axis, value = best_move
            cand[axis] = value
            cand = self._clamp_to_bounds(cand)

        if not self._point_in_any(cand, self.obstacles):
            return cand

        # Last resort: small radial samples around the endpoint.
        directions = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for radius in np.linspace(0.01, 0.16, 16):
            for d in directions:
                cand = self._clamp_to_bounds(p + d * radius)
                if not self._point_in_any(cand, self.obstacles):
                    return cand
        return None

    # ------------------------------------------------------------------
    # Smoothing / fallback / validation
    # ------------------------------------------------------------------
    def _shortcut_smooth(self, path: List[np.ndarray]) -> List[np.ndarray]:
        if len(path) <= 2:
            return self._dedupe_path(path)

        path = self._dedupe_path(path)

        # Greedy farthest-visible shortcut pass.
        changed = True
        while changed:
            changed = False
            out = [path[0]]
            i = 0
            while i < len(path) - 1:
                j_best = i + 1
                for j in range(len(path) - 1, i, -1):
                    if not segment_collides(path[i], path[j], self.obstacles):
                        j_best = j
                        break
                out.append(path[j_best])
                if j_best > i + 1:
                    changed = True
                i = j_best
            path = self._dedupe_path(out)

        # Random shortcut attempts catch cases the greedy pass misses.
        for _ in range(self.smooth_passes):
            if len(path) <= 2:
                break
            i, j = sorted(self._rng.choice(len(path), size=2, replace=False))
            if j <= i + 1:
                continue
            if not segment_collides(path[i], path[j], self.obstacles):
                path = path[:i + 1] + path[j:]

        return self._dedupe_path(path)

    def _deterministic_fallback(self, start: np.ndarray, goal: np.ndarray
                                ) -> Optional[List[np.ndarray]]:
        """
        Cheap non-random backup: try Manhattan-style doglegs around wall columns.
        This is not a replacement for RRT; it just avoids failing on simple maps.
        """
        z_values = [start[2], goal[2], max(start[2], goal[2]), self.bounds[5] - 0.01]
        x_lanes = [self.bounds[0] + 0.03, self.bounds[1] - 0.03, start[0], goal[0]]
        y_lanes = [self.bounds[2] + 0.03, self.bounds[3] - 0.03, start[1], goal[1]]

        candidates: List[List[np.ndarray]] = []
        for z in z_values:
            z = float(np.clip(z, self.bounds[4], self.bounds[5]))
            s_up = np.array([start[0], start[1], z])
            g_up = np.array([goal[0], goal[1], z])
            candidates.append([start, s_up, g_up, goal])
            for xm in x_lanes:
                xm = float(np.clip(xm, self.bounds[0], self.bounds[1]))
                candidates.append([
                    start, s_up,
                    np.array([xm, start[1], z]),
                    np.array([xm, goal[1], z]),
                    g_up, goal,
                ])
            for ym in y_lanes:
                ym = float(np.clip(ym, self.bounds[2], self.bounds[3]))
                candidates.append([
                    start, s_up,
                    np.array([start[0], ym, z]),
                    np.array([goal[0], ym, z]),
                    g_up, goal,
                ])

        best: Optional[List[np.ndarray]] = None
        best_len = float("inf")
        for cand in candidates:
            cand = self._dedupe_path([self._clamp_to_bounds(p) for p in cand])
            if any(self._point_in_any(p, self.obstacles) for p in cand):
                continue
            if path_collides(cand, self.obstacles):
                continue
            length = self._path_length(cand)
            if length < best_len:
                best_len = length
                best = cand
        return best

    def _path_is_acceptable(self, path: List[np.ndarray],
                            start: np.ndarray, goal: np.ndarray) -> bool:
        if len(path) < 2:
            return False

        # Never allow waypoints inside the physical walls.
        if any(self._point_in_any(p, self.raw_obstacles) for p in path):
            return False

        # Internal edges must be inflated-safe. First/last connector may be
        # only raw-safe if the actual endpoint is inside the inflated margin.
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            is_start_connector = i == 0 and self._same_point(a, start)
            is_goal_connector = i == len(path) - 2 and self._same_point(b, goal)
            obstacles = self.raw_obstacles if (is_start_connector or is_goal_connector) else self.obstacles
            if segment_collides(a, b, obstacles):
                return False
        return True

    def _dedupe_path(self, path: Sequence[np.ndarray], eps: float = 1e-7
                     ) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in path:
            p = np.asarray(p, dtype=float).copy()
            if not out or np.linalg.norm(p - out[-1]) > eps:
                out.append(p)
        return out

    def _path_length(self, path: Sequence[np.ndarray]) -> float:
        return float(sum(np.linalg.norm(path[i + 1] - path[i])
                         for i in range(len(path) - 1)))

    def _same_point(self, a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> bool:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) <= eps