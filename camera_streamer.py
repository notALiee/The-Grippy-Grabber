import mujoco
import cv2
import numpy as np
import socket
import struct
import threading

class CameraStreamer:
    def __init__(self, model, camera_name="gripper_depth_camera", fps=30, port=9999):
        self.camera_name = camera_name
        self.fps = fps
        self.render_interval = 1.0 / self.fps
        self.last_render_time = 0.0

        # Initialize renderer
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        # Uncomment to stream depth instead of RGB:
        self.renderer.enable_depth_rendering()

        # Set up socket server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', port))
        self.server.listen(1)
        
        self.client_socket = None
        self.lock = threading.Lock()
        
        threading.Thread(target=self._accept_clients, daemon=True).start()

    def _accept_clients(self):
        print("\nCamera Streamer listening on 127.0.0.1:9999 (Run viewcamera.py to see feed)")
        while True:
            try:
                conn, addr = self.server.accept()
                print(f"Viewer connected from {addr}")
                with self.lock:
                    self.client_socket = conn
            except Exception as e:
                print(f"Streamer accept error: {e}")
                import time
                time.sleep(1)

    def update(self, data):
        # Only render at the specified FPS
        if data.time - self.last_render_time < self.render_interval:
            return

        self.last_render_time += self.render_interval

        with self.lock:
            # Skip if nobody is watching to save massive physics overhead
            if self.client_socket is None:
                return

            self.renderer.update_scene(data, camera=self.camera_name)
            pixels = self.renderer.render()

            if len(pixels.shape) == 2:  
                depth_normalized = np.clip(pixels, 0.0, 2.0) / 2.0
                frame = (depth_normalized * 255).astype(np.uint8)
            else: 
                frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            try:
                success, byte_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    msg = struct.pack("Q", len(byte_buffer)) + byte_buffer.tobytes()
                    self.client_socket.sendall(msg)
            except Exception:
                self.client_socket.close()
                self.client_socket = None
                print("Viewer disconnected.")