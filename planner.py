"""
Cartesian RRT path planner for the Panda end-effector.

Plans collision-free 3D waypoint sequences for `panda_hand` through an
axis-aligned bounding box (AABB) obstacle field. Used by the FSM in
`task_manager.py` for every transit move (HOMING, APPROACHING,
TRANSPORTING, RETREATING, CONFIRMING orbits, EXPLORING sweep) so the arm
routes around the static wall geoms in `panda_mujoco/world.xml` instead
of trying to fly straight through them.

Design notes:
- Cartesian, not joint-space: simpler, fast enough, and the FSM already
  has working OS-PD + null-space control once given Cartesian targets.
- Thin walls use modest XY inflation plus tall Z columns (`OBSTACLE_Z_TOP_PLANNING`)
  so horizontal chords at SAFE_Z / explore height cannot declare “free sky”
  while the elbow still strikes the physical wall. XY/Z margins are split so
  we don’t balloon a 2 cm slab into a slab that swallows the whole workspace.
- Shortcut-smoothing post-process is critical: raw RRT paths look
  jittery and produce dozens of micro-trajectories. Greedy shortcut
  collapses runs of nodes whose direct connection is collision-free,
  giving the FSM the few clean waypoints it actually needs.

Hand-mirror constants (`WORLD_OBSTACLES`, `WORKSPACE_BOUNDS`) describe
the world used by the FSM. Keep `WORLD_OBSTACLES` in sync with the
geoms in `panda_mujoco/world.xml`.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# World description (kept in sync by hand with panda_mujoco/world.xml)
# ----------------------------------------------------------------------------
# XY footprints MUST match wall bodies in panda_mujoco/world.xml (half-extent
# + center). Z is extended to OBSTACLE_Z_TOP_PLANNING — the real geoms are
# shorter, but the elbow / lower links sweep well below the wrist during
# horizontal moves at SAFE_Z (~0.45) and EXPLORING (~0.60). If we only boxed
# the visible wall height, segment_collides saw "clear sky" at transit Z and
# returned a straight line straight through the obstacle column while the
# physical arm still hit the wall.
OBSTACLE_Z_TOP_PLANNING = 0.72

# AABB format: (x_min, y_min, z_min, x_max, y_max, z_max).
WORLD_OBSTACLES: List[Tuple[float, float, float, float, float, float]] = [
    # wall_a: pos=(0.40, -0.20, 0.20), size=(0.01, 0.20, 0.20)
    (0.39, -0.40, 0.0, 0.41, 0.00, OBSTACLE_Z_TOP_PLANNING),
    # wall_b: pos=(0.75,  0.05, 0.25), size=(0.15, 0.01, 0.25)
    (0.60, 0.04, 0.0, 0.90, 0.06, OBSTACLE_Z_TOP_PLANNING),
    # wall_c: pos=(0.20,  0.10, 0.18), size=(0.01, 0.10, 0.18)
    (0.19, 0.00, 0.0, 0.21, 0.20, OBSTACLE_Z_TOP_PLANNING),
]

# Workspace AABB the planner is allowed to sample inside. X/Y derived
# from the table limits the user gave, with a small inset so we don't
# sample right at the table edge. Z floor is the grasp floor (so RRT
# can plan into a near-table approach), Z ceiling leaves clear sky for
# transit + retreat.
WORKSPACE_BOUNDS: Tuple[float, float, float, float, float, float] = (
    -0.25, 1.10,    # x_min, x_max
    -0.675, 0.25,   # y_min, y_max
    0.075, 0.65,    # z_min, z_max
)

# Thin walls need TALL Z columns (so horizontal moves at SAFE_Z / explore
# height cannot ignore them) but modest XY inflation — isotropic 10 cm+
# margins turn a 2 cm slab into a continent and strand RRT starts inside
# inflated geometry.
XY_SAFETY_MARGIN = 0.04
Z_SAFETY_MARGIN = 0.06

# Legacy single margin kept for API grep; CartesianRRT uses XY/Z split above.
DEFAULT_SAFETY_MARGIN = XY_SAFETY_MARGIN


# ----------------------------------------------------------------------------
# AABB collision math
# ----------------------------------------------------------------------------
def _inflate(box: Sequence[float], xy_margin: float, z_margin: float
             ) -> Tuple[float, float, float, float, float, float]:
    """Expand AABB in XY (thin walls stay thin) and Z (extra above wall top)."""
    return (box[0] - xy_margin, box[1] - xy_margin, box[2] - z_margin,
            box[3] + xy_margin, box[4] + xy_margin, box[5] + z_margin)


def point_in_aabb(p: np.ndarray, box: Sequence[float]) -> bool:
    return (box[0] <= p[0] <= box[3]
            and box[1] <= p[1] <= box[4]
            and box[2] <= p[2] <= box[5])


def segment_hits_aabb(p0: np.ndarray, p1: np.ndarray,
                      box: Sequence[float]) -> bool:
    """
    Slab method: the segment hits the box iff every axis's entry-time
    `t_near` is <= every axis's exit-time `t_far`, and the overall
    intersection interval overlaps [0, 1] (the parameter range of the
    segment). Treats a segment that only grazes the surface as a hit.
    """
    d = p1 - p0
    t_near = -np.inf
    t_far = np.inf
    box_min = (box[0], box[1], box[2])
    box_max = (box[3], box[4], box[5])
    for i in range(3):
        if abs(d[i]) < 1e-9:
            # Parallel to slab i. If origin is outside the slab we miss.
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return False
            continue
        inv = 1.0 / d[i]
        t1 = (box_min[i] - p0[i]) * inv
        t2 = (box_max[i] - p0[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
        if t2 < t_far:
            t_far = t2
        if t_near > t_far:
            return False
    return t_far >= 0.0 and t_near <= 1.0


def segment_collides(p0: np.ndarray, p1: np.ndarray,
                     obstacles: Sequence[Sequence[float]]) -> bool:
    for box in obstacles:
        if segment_hits_aabb(p0, p1, box):
            return True
    return False


# ----------------------------------------------------------------------------
# Cartesian RRT
# ----------------------------------------------------------------------------
@dataclass
class _Node:
    pos: np.ndarray
    parent: int  # index in tree; -1 for the root


class CartesianRRT:
    """
    Plain RRT in 3D Cartesian space for the panda_hand. No RRT-Connect /
    RRT* niceties -- this is an FSM helper, not a research planner. The
    shortcut-smoothing pass is what makes the output look reasonable.
    """

    def __init__(self,
                 bounds: Sequence[float] = WORKSPACE_BOUNDS,
                 obstacles: Sequence[Sequence[float]] = WORLD_OBSTACLES,
                 step_size: float = 0.05,
                 goal_bias: float = 0.20,
                 goal_tol: float = 0.035,
                 max_iters: int = 4500,
                 xy_margin: float = XY_SAFETY_MARGIN,
                 z_margin: float = Z_SAFETY_MARGIN,
                 seed: Optional[int] = 0):
        self.bounds = tuple(bounds)
        self.obstacles = [_inflate(b, xy_margin, z_margin) for b in obstacles]
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.goal_tol = float(goal_tol)
        self.max_iters = int(max_iters)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, start_xyz: np.ndarray, goal_xyz: np.ndarray
             ) -> Optional[List[np.ndarray]]:
        """
        Returns a list of 3D waypoints from `start` to `goal` (inclusive,
        with `start` first and `goal` last) that's collision-free under
        the inflated AABB obstacles. None if no path was found within
        `max_iters`.

        Trivial-path fast path: if the straight segment is already
        collision-free, return [start, goal] without any tree growth.
        """
        start = np.asarray(start_xyz, dtype=float).copy()
        goal = np.asarray(goal_xyz, dtype=float).copy()

        # Clamp endpoints into bounds (callers occasionally hand us a
        # goal outside the workspace box, e.g. drop zones near the
        # edge). Clamping is safer than refusing to plan.
        start = self._clamp_to_bounds(start)
        goal = self._clamp_to_bounds(goal)

        # If the start happens to be inside an inflated obstacle (e.g.
        # we started a leg too close to a wall after a previous
        # collision recovery), nudge it toward the goal until it's
        # free, otherwise the very first edge collides with itself.
        start = self._escape_obstacle(start, goal)

        if not segment_collides(start, goal, self.obstacles):
            return [start, goal]

        tree: List[_Node] = [_Node(pos=start, parent=-1)]
        goal_idx = -1

        for _ in range(self.max_iters):
            sample = (goal if self._rng.random() < self.goal_bias
                      else self._sample_free())
            nearest_idx = self._nearest(tree, sample)
            nearest_pos = tree[nearest_idx].pos
            new_pos = self._steer(nearest_pos, sample)
            if segment_collides(nearest_pos, new_pos, self.obstacles):
                continue
            new_idx = len(tree)
            tree.append(_Node(pos=new_pos, parent=nearest_idx))

            if (np.linalg.norm(new_pos - goal) <= self.goal_tol
                    and not segment_collides(new_pos, goal, self.obstacles)):
                tree.append(_Node(pos=goal, parent=new_idx))
                goal_idx = len(tree) - 1
                break

        if goal_idx < 0:
            return None

        path = self._backtrace(tree, goal_idx)
        return self._shortcut_smooth(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sample_free(self) -> np.ndarray:
        # Try a handful of random samples; bail with a free-but-near-goal
        # fallback if the workspace is very crowded.
        for _ in range(32):
            p = self._rng.uniform(
                low=(self.bounds[0], self.bounds[2], self.bounds[4]),
                high=(self.bounds[1], self.bounds[3], self.bounds[5]),
            )
            if not any(point_in_aabb(p, b) for b in self.obstacles):
                return p
        # Pathological case: just return uniform sample, edge check will
        # reject any colliding edge anyway.
        return self._rng.uniform(
            low=(self.bounds[0], self.bounds[2], self.bounds[4]),
            high=(self.bounds[1], self.bounds[3], self.bounds[5]),
        )

    def _nearest(self, tree: List[_Node], q: np.ndarray) -> int:
        # Linear scan -- fine for a few-thousand-node tree at 3D.
        best = 0
        best_d = float("inf")
        for i, node in enumerate(tree):
            d = float(np.linalg.norm(node.pos - q))
            if d < best_d:
                best_d = d
                best = i
        return best

    def _steer(self, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
        delta = to - frm
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return to.copy()
        return frm + delta * (self.step_size / dist)

    def _backtrace(self, tree: List[_Node], leaf_idx: int) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        i = leaf_idx
        while i >= 0:
            path.append(tree[i].pos.copy())
            i = tree[i].parent
        path.reverse()
        return path

    def _shortcut_smooth(self, path: List[np.ndarray],
                         max_passes: int = 4) -> List[np.ndarray]:
        """
        Greedy shortcut: repeatedly walk the path and drop intermediate
        waypoints whose neighbors can be connected directly without
        hitting an obstacle. Converges quickly; 4 passes is plenty.
        """
        if len(path) <= 2:
            return path
        out = [p.copy() for p in path]
        for _ in range(max_passes):
            i = 0
            changed = False
            while i + 2 < len(out):
                if not segment_collides(out[i], out[i + 2], self.obstacles):
                    del out[i + 1]
                    changed = True
                else:
                    i += 1
            if not changed:
                break
        return out

    def _clamp_to_bounds(self, p: np.ndarray) -> np.ndarray:
        out = p.copy()
        out[0] = float(np.clip(out[0], self.bounds[0], self.bounds[1]))
        out[1] = float(np.clip(out[1], self.bounds[2], self.bounds[3]))
        out[2] = float(np.clip(out[2], self.bounds[4], self.bounds[5]))
        return out

    def _escape_obstacle(self, start: np.ndarray, goal: np.ndarray
                         ) -> np.ndarray:
        """
        If the start point happens to lie inside an inflated obstacle
        (or inside the safety margin around one), step it toward the
        goal in small increments until it's free. Without this the
        very first sampled edge collides with the obstacle the start
        is buried in and RRT can't grow.
        """
        if not any(point_in_aabb(start, b) for b in self.obstacles):
            return start
        delta = goal - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return start
        direction = delta / dist
        step = max(self.step_size * 0.5, 0.02)
        for k in range(1, 21):
            cand = start + direction * (step * k)
            if not any(point_in_aabb(cand, b) for b in self.obstacles):
                return cand
        # Couldn't find a free spot toward the goal; just return original.
        return start
