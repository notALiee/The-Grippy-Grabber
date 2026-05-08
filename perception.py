import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionSystem:
    def __init__(self, yolo_model_path="best.pt"):
        self.yolo = YOLO(yolo_model_path)
        
    def detect(self, rgb_image):
        """Runs YOLO on an RGB image, returns results and annotated BGR frame."""
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        results = self.yolo(bgr_image, verbose=False)
        annotated_frame = results[0].plot()
        return results, annotated_frame

    def get_3d_point(self, u, v, depth_image, fovy_deg, width, height, cam_xpos, cam_xmat):
        """
        De-projects a 2D pixel to 3D given the depth map and camera parameters,
        and transforms it into the global coordinate frame.
        """
        # Ensure coordinates are within image bounds
        u = int(np.clip(u, 0, width - 1))
        v = int(np.clip(v, 0, height - 1))
        
        # Depth in meters
        z_c = depth_image[v, u]
        
        # Focal length
        f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        cx, cy = width / 2.0, height / 2.0
        
        # MuJoCo Camera Frame: +X is right, +Y is up, -Z is pointing forward
        # Image coordinates (u, v): origin is top-left, +u is right, +v is down
        x_c = (u - cx) * z_c / f
        y_c = -(v - cy) * z_c / f
        z_c_mujoco = -z_c 
        
        point_camera = np.array([x_c, y_c, z_c_mujoco])
        
        # Transform to global World Coordinates
        cam_xmat = cam_xmat.reshape(3, 3)
        point_world = cam_xpos + cam_xmat @ point_camera
        
        return point_world
