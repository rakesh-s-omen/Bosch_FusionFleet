from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (serialCamera, SteerMotor, DrivingMode, kl, obcamera, ImuData)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import threading
import base64
import cv2
import time
import numpy as np
from mpu6050 import mpu6050



    
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
        self.smooth_factor_right = 5
        self.smooth_factor_left=3
        self.initial_position_set = False  # Flag to indicate if initial position is set
        self.subscribe()
        self.mohan = False
        self.max_allowed_deviation = 100 
        self.min_lane_gap=100
        self.max_deviation=100
        # Threshold for reinitializing midpoint (can be tuned)

        # Initialize LaneDetection



    def process_frame(self, frame):
        """Detect lanes and compute lane midpoint"""
        frame_height, frame_width = frame.shape[:2]
        frame_center = frame_width // 2  # Calculate center of the frame
        left_lane_boundary = int(frame_width * 0.1)  # Prevent left lane from reaching the edge

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define Region of Interest (ROI)
        roi_top = int(frame_height * 0.6)
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, frame_height), (frame_width, frame_height), 
                             (frame_width, roi_top), (0, roi_top)]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(edges, mask)

        # Separate Hough Transform for Left and Right
        left_lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=25)
        right_lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=25)

        left_lane_lines, right_lane_lines = [], []
        
        # Classify Left and Right Lines
        if left_lines is not None:
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
                if slope < -0.4 and left_lane_boundary < x2 < frame_center:  # Ensure left lane stays within bounds
                    left_lane_lines.append((x1, y1, x2, y2))

        if right_lines is not None:
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
                if slope > 0.5 and x2 > frame_center:  # Ensure right lane stays on the right
                    right_lane_lines.append((x1, y1, x2, y2))

        # Process left and right lanes separately
        left_lane = self.smooth_lanes(self.weighted_avg_lines(left_lane_lines, bias=1.5), self.lane_history['left'], self.smooth_factor_left)
        right_lane = self.smooth_lanes(self.weighted_avg_lines(right_lane_lines, bias=1.0), self.lane_history['right'], self.smooth_factor_right)

        # Prevent Left Lane from Reaching the Edge
        if left_lane:
            left_x2 = left_lane[2]  # Endpoint of the left lane
            if left_x2 < left_lane_boundary:
                print("Warning: Left lane too close to edge! Resetting left lane.")
                left_lane = None  # Ignore this lane if it's too close to the edge

        # Prevent Left Lane from Crossing Center
        if left_lane:
            left_x2 = left_lane[2]
            if left_x2 >= frame_center:
                print("Warning: Left lane crossed the center! Resetting left lane.")
                left_lane = None  # Ignore this lane if it crosses center

        # Prevent Right Lane from Crossing Center
        if right_lane:
            right_x2 = right_lane[2]  # Endpoint of the right lane
            if right_x2 <= frame_center:
                print("Warning: Right lane crossed the center! Resetting right lane.")
                right_lane = None  # Ignore this lane if it crosses center

        # Prevent Left and Right Lanes from Swapping or Overlapping
        if left_lane and right_lane:
            left_x2 = left_lane[2]
            right_x2 = right_lane[2]
            
            if left_x2 >= right_x2 - self.min_lane_gap:
                print("Warning: Lanes too close or swapped! Ignoring frame.")
                return None  # Ignore processing for this frame

        # Draw lanes on frame
        if left_lane:
            cv2.line(frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 5)
        if right_lane:
            cv2.line(frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 5)

        # Compute lane midpoint
        if left_lane and right_lane:
            midpoint_x = int(round((left_lane[2] + right_lane[2]) / 2))
        elif right_lane:
            midpoint_x = right_lane[2] - self.min_lane_gap
        elif left_lane:
            midpoint_x = left_lane[2] + self.min_lane_gap
        else:
            return None  

        midpoint_y = frame_height  

        # Draw midpoint on frame
        cv2.circle(frame, (midpoint_x, midpoint_y), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"Midpoint: {midpoint_x}", (midpoint_x - 50, midpoint_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Initialize or reinitialize midpoint
        if not self.initial_position_set or abs(midpoint_x - self.initial_midpoint) > self.max_allowed_deviation:
            self.initial_midpoint = midpoint_x
            self.initial_position_set = True
            return None

        # Compute lane deviation
        delta_comp = midpoint_x - self.initial_midpoint
        return delta_comp

    def smooth_lanes(self, current_lane, lane_history, smooth_factor):
        """Smooth detected lanes over multiple frames"""
        if current_lane is not None:
            lane_history.append(current_lane)
            if len(lane_history) > smooth_factor:
                lane_history.pop(0)
        else:
            if lane_history:
                lane_history.pop(0)

        if lane_history:
            avg_lane = np.mean(lane_history, axis=0)
            return tuple(map(int, avg_lane))
        return None

    def weighted_avg_lines(self, lines, bias=1.0):
        """Compute weighted average of detected lane lines with bias for left lanes"""
        if not lines:
            return None

        # Assign higher weight to longer lines
        weights = [bias * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in lines]
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]

        avg_x1 = int(sum([line[0] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_y1 = int(sum([line[1] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_x2 = int(sum([line[2] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_y2 = int(sum([line[3] * weight for line, weight in zip(lines, normalized_weights)]))

        return (avg_x1, avg_y1, avg_x2, avg_y2)



    def read_sensor_data(self):
        """Read accelerometer and gyroscope data from MPU6050"""
        sensor = mpu6050(0x68)
        accel_data = sensor.get_accel_data()
        gyro_data = sensor.get_gyro_data()

        data = {
            "roll": round(gyro_data['x'], 3),
            "pitch": round(gyro_data['y'], 3),
            "yaw": round(gyro_data['z'], 3),
            "accelx": round(accel_data['x'], 3),
            "accely": round(accel_data['y'], 3),
            "accelz": round(accel_data['z'], 3),
        }

        self.imuDataSender.send(str(data))

    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)
        self.imuDataSender = messageHandlerSender(self.queuesList, ImuData)

    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)

    def map_to_steering_angle(self, delta_comp):
            max_delta = self.frame_width / 2  # Assume max delta_comp is half the frame width
            normalized_delta = delta_comp / max_delta  # Normalize delta_comp to range [-1, 1]

            # Dead zone for small deviations
            dead_zone = 0.1  # Adjust this value as needed
            if abs(normalized_delta) < dead_zone:
                return 0  # No steering correction for small deviations

            # Non-linear mapping (quadratic or linear)
            non_linear_factor = np.sign(normalized_delta) * (normalized_delta ** 2)  # Quadratic
            # non_linear_factor = normalized_delta  # Linear

            # Scale to steering angle range
            scaling_factor = 200  # Adjust this value as needed
            steering_angle = np.clip(non_linear_factor * scaling_factor, -25, 25)  # Range [-25, 25]

            # Log the mapping for analysis
            self.logging.info(f"Delta: {delta_comp}, Normalized Delta: {normalized_delta}, Steering Angle: {steering_angle}")

            return steering_angle

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
                        send = not send
                        self.read_sensor_data()
                        if frame is not None:
                            self.cap = frame
                            self.frame_width = frame.shape[1]
                            self.frame_height = frame.shape[0]
                            delta_comp = self.process_frame(frame)  # Use LaneDetection
                            if delta_comp is not None:
                                steering_angle = self.map_to_steering_angle(delta_comp)
                                steering_angle_int = int(round(steering_angle))
                                self.logging.info(f"Delta: {delta_comp}, Steering angle (rounded): {steering_angle_int}")
                                self.steerSender.send(str(steering_angle_int * 10))
                time.sleep(0.05)
            except Exception as e:
                self.logging.error(f"Error in threadLane: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
