from dataclasses import dataclass, field
from typing import Optional, List, Sequence

import cv2
import mujoco
import numpy as np
from ultralytics import YOLO

from control import top_down_quat


@dataclass
class ObjectPose:
    """6-DoF grasp target inferred from a single RGB-D detection."""
    class_name: str
    position: np.ndarray              # (3,) world frame
    quaternion: np.ndarray            # (4,) MuJoCo (w, x, y, z), gripper target orientation
    score: float                      # YOLO confidence
    n_points: int                     # masked depth pixels used for the centroid/PCA
    principal_axis: Optional[np.ndarray] = None  # (3,) world frame, only set for pens
    source_cam: str = ""              # name of the camera the detection came from


@dataclass
class CameraView:
    """Bundle of one rendered RGB-D frame plus its intrinsics + extrinsics."""
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: dict
    cam_xpos: np.ndarray
    cam_xmat: np.ndarray


def get_intrinsics(model, cam_name: str, width: int, height: int) -> dict:
    """Returns pinhole intrinsics derived from a MuJoCo camera's vertical FOV."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model.")
    fovy_deg = float(model.cam_fovy[cam_id])
    f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return {
        "cam_id": cam_id,
        "fovy_deg": fovy_deg,
        "f": f,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
    }


class PerceptionSystem:
    def __init__(self, yolo_model_path: str = "best.pt"):
        self.yolo = YOLO(yolo_model_path)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, rgb_image):
        """Runs YOLO on an RGB image, returns (results, annotated BGR frame)."""
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        results = self.yolo(bgr_image, verbose=False)
        annotated_frame = results[0].plot()
        return results, annotated_frame

    # ------------------------------------------------------------------
    # Single-pixel deprojection (kept for backwards-compat with main.py
    # / Phase 1 demos). Prefer estimate_pose / scan_scene below.
    # ------------------------------------------------------------------
    def get_3d_point(self, u, v, depth_image, fovy_deg, width, height,
                     cam_xpos, cam_xmat):
        u = int(np.clip(u, 0, width - 1))
        v = int(np.clip(v, 0, height - 1))
        z_c = float(depth_image[v, u])

        f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        cx, cy = width / 2.0, height / 2.0

        x_c = (u - cx) * z_c / f
        y_c = -(v - cy) * z_c / f
        z_c_mujoco = -z_c

        point_camera = np.array([x_c, y_c, z_c_mujoco])
        cam_xmat = cam_xmat.reshape(3, 3)
        return cam_xpos + cam_xmat @ point_camera

    # ------------------------------------------------------------------
    # Bounding-box -> Nx3 world-frame point cloud
    # ------------------------------------------------------------------
    @staticmethod
    def _deproject_bbox(bbox, depth_image, intrinsics, cam_xpos, cam_xmat,
                        depth_band: float = 0.05) -> np.ndarray:
        """
        De-project all pixels inside `bbox` (xyxy) whose depth is within
        `depth_band` meters of the bbox-center depth, into world coordinates.

        Filtering by a depth band around the central pixel is a cheap way to
        drop background pixels (table, floor, objects behind) so the centroid
        and PCA represent the actual detected object.

        Returns (N, 3) array; may be empty.
        """
        h, w = depth_image.shape[:2]
        f = intrinsics["f"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        x1, y1, x2, y2 = bbox
        u1 = int(max(0, np.floor(x1)))
        v1 = int(max(0, np.floor(y1)))
        u2 = int(min(w - 1, np.ceil(x2)))
        v2 = int(min(h - 1, np.ceil(y2)))
        if u2 <= u1 or v2 <= v1:
            return np.empty((0, 3))

        uc = int(np.clip((u1 + u2) // 2, 0, w - 1))
        vc = int(np.clip((v1 + v2) // 2, 0, h - 1))
        z_center = float(depth_image[vc, uc])
        if not np.isfinite(z_center) or z_center <= 0.0:
            # Center pixel hit the far plane; fall back to the bbox median.
            patch = depth_image[v1:v2 + 1, u1:u2 + 1]
            valid = patch[(patch > 0.0) & np.isfinite(patch)]
            if valid.size == 0:
                return np.empty((0, 3))
            z_center = float(np.median(valid))

        patch = depth_image[v1:v2 + 1, u1:u2 + 1]
        # Mask: finite, positive, and within +/- depth_band of the center depth.
        mask = (
            np.isfinite(patch)
            & (patch > 0.0)
            & (np.abs(patch - z_center) <= depth_band)
        )
        if not np.any(mask):
            return np.empty((0, 3))

        vs, us = np.nonzero(mask)
        zs = patch[vs, us].astype(np.float64)
        # Translate patch-local pixel coords back to full-image coords.
        us_full = us + u1
        vs_full = vs + v1

        x_c = (us_full - cx) * zs / f
        y_c = -(vs_full - cy) * zs / f
        z_c = -zs  # MuJoCo camera looks down -Z

        pts_cam = np.stack([x_c, y_c, z_c], axis=1)  # (N, 3)
        R = np.asarray(cam_xmat).reshape(3, 3)
        pts_world = pts_cam @ R.T + np.asarray(cam_xpos).reshape(1, 3)
        return pts_world

    # ------------------------------------------------------------------
    # Per-detection pose
    # ------------------------------------------------------------------
    def estimate_pose(self, class_name, bbox, score, depth_image,
                      intrinsics, cam_xpos, cam_xmat,
                      min_points: int = 30) -> Optional[ObjectPose]:
        """
        Returns an ObjectPose for a single YOLO detection, or None if there
        aren't enough valid depth samples to estimate a centroid reliably.
        """
        pts = self._deproject_bbox(bbox, depth_image, intrinsics, cam_xpos, cam_xmat)
        if pts.shape[0] < min_points:
            return None

        position = pts.mean(axis=0)

        if class_name in ("pen", "book"):
            # Project into XY plane, run PCA on (X, Y) only -> first principal
            # vector is the object's long axis.
            #   - pen:  yaw = long-axis angle, so jaws close ACROSS the long
            #           axis (jaws span the pen's short width).
            #   - book: yaw = long-axis angle + 90 deg, so jaws close ALONG
            #           the long axis -- fingers land on the two SHORT edges
            #           at the book's center (instead of the long edges).
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(xy, full_matrices=False)
                axis_xy = vh[0]  # principal direction in XY
            except np.linalg.LinAlgError:
                axis_xy = np.array([1.0, 0.0])
            long_yaw = float(np.arctan2(axis_xy[1], axis_xy[0]))
            yaw = long_yaw if class_name == "pen" else long_yaw + np.pi / 2.0
            quat = top_down_quat(yaw)
            principal_axis = np.array([axis_xy[0], axis_xy[1], 0.0])
        else:
            # cup, PuzzleBase, PuzzleCircle, PuzzleSquare, PuzzleTriangle
            quat = top_down_quat(0.0)
            principal_axis = None

        return ObjectPose(
            class_name=class_name,
            position=position,
            quaternion=quat,
            score=float(score),
            n_points=int(pts.shape[0]),
            principal_axis=principal_axis,
        )

    # ------------------------------------------------------------------
    # Whole-scene scan
    # ------------------------------------------------------------------
    def scan_scene(self, rgb_image, depth_image, cam_xpos, cam_xmat,
                   intrinsics, min_points: int = 30,
                   min_score: float = 0.25) -> List[ObjectPose]:
        """
        Runs YOLO on `rgb_image`, then estimates a 6-DoF grasp pose for every
        detection above `min_score` that has at least `min_points` valid
        masked depth samples. Returns the list (may be empty).
        """
        results, _ = self.detect(rgb_image)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(self.yolo, "names", {})
        poses: List[ObjectPose] = []

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

        for bbox, c, s in zip(xyxy, cls, conf):
            if s < min_score:
                continue
            cls_idx = int(c)
            class_name = names[cls_idx] if cls_idx in names else str(cls_idx)
            pose = self.estimate_pose(
                class_name, bbox, s, depth_image, intrinsics,
                cam_xpos, cam_xmat, min_points=min_points,
            )
            if pose is not None:
                poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Multi-camera scan + cross-view merge
    # ------------------------------------------------------------------
    def scan_multi(self, views: Sequence[CameraView],
                   min_points: int = 30, min_score: float = 0.25,
                   merge_radius: float = 0.04) -> List[ObjectPose]:
        """
        Run YOLO + pose estimation independently on every CameraView, then
        merge detections that share a class and lie within `merge_radius`
        meters of each other. The merged pose averages the two positions
        (weighted by point count) and keeps the higher-confidence quaternion.

        Multi-view helps with occlusion: e.g. a tall book hides a pen from the
        overhead cam, but a side cam still sees the pen.
        """
        all_poses: List[ObjectPose] = []
        for view in views:
            poses = self.scan_scene(
                view.rgb, view.depth, view.cam_xpos, view.cam_xmat,
                view.intrinsics, min_points=min_points, min_score=min_score,
            )
            for p in poses:
                p.source_cam = view.name
            all_poses.extend(poses)

        return self._merge_poses(all_poses, merge_radius)

    @staticmethod
    def _merge_poses(poses: List[ObjectPose], radius: float) -> List[ObjectPose]:
        """Greedy single-link clustering by (class_name, position)."""
        merged: List[ObjectPose] = []
        for p in poses:
            absorbed = False
            for i, q in enumerate(merged):
                if q.class_name != p.class_name:
                    continue
                if np.linalg.norm(q.position - p.position) > radius:
                    continue
                wq, wp = max(q.n_points, 1), max(p.n_points, 1)
                new_pos = (q.position * wq + p.position * wp) / (wq + wp)
                if p.score > q.score:
                    new_quat = p.quaternion
                    new_axis = p.principal_axis
                    new_score = p.score
                    new_src = p.source_cam
                else:
                    new_quat = q.quaternion
                    new_axis = q.principal_axis
                    new_score = q.score
                    new_src = q.source_cam
                merged[i] = ObjectPose(
                    class_name=q.class_name,
                    position=new_pos,
                    quaternion=new_quat,
                    score=new_score,
                    n_points=wq + wp,
                    principal_axis=new_axis,
                    source_cam=new_src + "+" + p.source_cam if new_src != p.source_cam else new_src,
                )
                absorbed = True
                break
            if not absorbed:
                merged.append(p)
        return merged

    @staticmethod
    def filter_by_robot_proximity(poses: List[ObjectPose],
                                  robot_points: np.ndarray,
                                  min_dist: float = 0.12) -> List[ObjectPose]:
        """
        Drop any detection whose 3D centroid is within `min_dist` meters of
        any of the supplied robot body positions. Last-line defense if the
        geom-group hiding leaks a robot pixel through.
        """
        if robot_points.size == 0:
            return list(poses)
        kept: List[ObjectPose] = []
        for p in poses:
            d = np.linalg.norm(robot_points - p.position.reshape(1, 3), axis=1)
            if float(d.min()) >= min_dist:
                kept.append(p)
        return kept
