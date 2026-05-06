import cv2
import socket
import struct
import numpy as np

def start_viewer():
    print("Waiting for simulation on 127.0.0.1:9999...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('127.0.0.1', 9999))
        print("Connected to MuJoCo stream.")
    except ConnectionRefusedError:
        print("Could not connect. Is the simulation running on port 9999?")
        return

    data = b""
    payload_size = struct.calcsize("Q")
    
    cv2.namedWindow("Gripper Camera Live Feed", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Receive size descriptor
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
                
            if len(data) < payload_size: break
                
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Receive frame data based on size
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
                
            if len(data) < msg_size: break
                
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # Decode the JPEG buffer into an image array and display it
            frame_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                cv2.imshow("Gripper Camera Live Feed", frame)
                # Allow user to quit the viewer smoothly
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Stream interrupted: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("Viewer closed.")

if __name__ == "__main__":
    start_viewer()