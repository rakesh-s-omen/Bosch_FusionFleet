from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (serialCamera, SteerMotor, DrivingMode, kl, obcamera,ImuData,SpeedMotor)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import threading
import base64
import cv2
import time
import numpy as np
#from mpu6050 import mpu6050


class threadLane(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        super().__init__()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.initial_midpoint = None
        self.initialcomp = None
        self.delta_comp = 0
        self.lane_history = {'left': [], 'right': []}
        self.smooth_factor = 5
        self.max_deviation= 100
        self.initial_position_set = False  # Flag to indicate if initial position is set
        self.subscribe()
        self.mohan = False
        

    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)
        #self.imuDataSender = messageHandlerSender(self.queuesList, ImuData)
        self.motorspSender=messageHandlerSender(self.queuesList, SpeedMotor)
    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)
        #self.sensor = mpu6050(0x68)
    ''''def read_sensor_data(self):
        """Read accelerometer and gyroscope data from MPU6050"""
        accel_data = self.sensor.get_accel_data()
        gyro_data = self.sensor.get_gyro_data()
        data = {
            "roll": round(gyro_data['x'], 3),
            "pitch": round(gyro_data['y'], 3),
            "yaw": round(gyro_data['z'], 3),
            "accelx": round(accel_data['x'], 3),
            "accely": round(accel_data['y'], 3),
            "accelz": round(accel_data['z'], 3),
        }
        self.imuDataSender.send(str(data))
        return round(accel_data['x'], 3)'''

    
        
    def get_birds_eye_view(self,image):
        """Applies a perspective transform to get a bird's eye view of the track"""
        height, width = image.shape[:2]

        # Define 4 points on the original image (adjust based on your perspective)
        src_pts = np.float32([ 
            [width * 0.15, height * 0.9],  
            [width * 0.85, height * 0.9],  
            [width * 0.2, height * 0.35],  
            [width * 0.8, height * 0.35]   
        ])

        # Define where the points should be mapped to (straight rectangle)
        dst_pts = np.float32([
            [width * 0.2, height],
            [width * 0.8, height],
            [width * 0.2, 0],
            [width * 0.8, 0]
        ])

        # Compute the transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped, M
    

    def map_to_steering_angle(self, delta_comp):
        max_delta = self.frame_width / 2  # Assume max delta_comp is half the frame width
        normalized_delta = delta_comp / max_delta  # Normalize delta_comp to range [-1, 1]

        # Apply a non-linear mapping using a quadratic function
        # Steering angle will be proportional to the square of the deviation
        non_linear_factor = np.sign(normalized_delta) * (normalized_delta ** 2)
        steering_angle = np.clip(non_linear_factor * 250, -25, 25)  # Scale to [-25, 25]
        
        return steering_angle    
    def detect_lanes(self,cap):
        """Detect lanes in a given frame."""
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Region of Interest
        height, width = edges.shape
        roi_bottom_height = height // 2  
        vertices = np.array([[ 
            (0, height),                  
            (0, roi_bottom_height),       
            (width, roi_bottom_height),   
            (width, height)               
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Line Detection
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  
                if abs(slope) > 0.5:  
                    if slope < 0 and x1 < width / 2 and x2 < width / 2:
                        left_lines.append(line[0])
                    elif slope > 0 and x1 > width / 2 and x2 > width / 2:
                        right_lines.append(line[0])
        def get_average_line(lines):
            if len(lines) == 0:
                return None
            x1s = np.array([line[0] for line in lines])
            y1s = np.array([line[1] for line in lines])
            x2s = np.array([line[2] for line in lines])
            y2s = np.array([line[3] for line in lines])
            return [int(np.mean(x1s)), int(np.mean(y1s)), 
                    int(np.mean(x2s)), int(np.mean(y2s))]
        avg_left = get_average_line(left_lines)
        avg_right =get_average_line(right_lines)
        def get_midpoint(left, right):
            if left is None or right is None:
                return None
            x1, y1, x2, y2 = left
            x3, y3, x4, y4 = right
            midpoint_x = int((x1 + x3 + x2 + x4) / 4)
            midpoint_y = int((y1 + y3 + y2 + y4) / 4)
            return(midpoint_x)
        midpoint_x = get_midpoint(avg_left, avg_right)
        
        if midpoint_x is not None:
                    if not self.initial_position_set or abs(midpoint_x - self.initial_midpoint) > self.max_deviation:
                        self.initial_midpoint = midpoint_x
                        self.initialcomp = midpoint_x
                        self.initial_position_set = True
                        return None
                    delta_comp = midpoint_x - self.initialcomp
                    self.logging.info(delta_comp)
                    return delta_comp
        return 0
    def run(self):
        send = False
        while self._running:
            try:
                camera_data = self.serialCameraSubscriber.receive()
                if camera_data and send:
                    self.send()
                    self.obSender.send(camera_data)
                yk = self.klvalSubscriber.receive()

                if yk is not None:
                    if yk == "hiiii":
                        mode = self.DrivingModeSubscriber.receive()
                        if mode in ["stop", "manual", "legacy"]:
                            self.send()
                            self.steerSender.send("0")
                            self.mohan = False
                            self.initial_position_set = False
                        if mode == "auto":
                            self.send()
                            self.mohan = False
                            self.initial_position_set = False
                            self.mohan = True

                if self.mohan:
                    if camera_data:
                        image_data = base64.b64decode(camera_data)
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        send=not send
                        #imu=self.read_sensor_data()
                        if frame is not None:
                            self.cap = frame
                            self.frame_width = frame.shape[1]
                            self.frame_height = frame.shape[0]
                            warped, _ = self.get_birds_eye_view(frame)
                            # Detect lanes in the warped frame
                            delta_comp = self.detect_lanes(warped)
                            
                        '''if imu < -0.850: 
                            #self.motorspSender.send("150")
                            self.logging.info(imu)
                        if (imu > 1.300 and imu <2.5):
                            self.logging.info(imu)
                            #self.motorspSender.send("500")'''
                        if delta_comp is not None:
                                steering_angle = self.map_to_steering_angle(delta_comp)
                                steering_angle_int = int(round(steering_angle))
                                self.logging.info(f"Delta: {delta_comp}, Steering angle (rounded): {steering_angle_int}")
                                self.steerSender.send(str(steering_angle_int*10))
            except Exception as e:
                self.logging.error(f"Error in threadLane: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Detection thread stopped successfully.")

