import threading
import cv2
import time
import numpy as np
import base64
import tensorflow.lite as tflite
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, DrivingMode, kl, SpeedMotor, WarningSignal, obcamera, SteerMotor
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

class threadObject(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        super().__init__()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.load_model()
        self.subscribe()
       
        self.motorspSender = None
        self.var = False
    
    def load_model(self):
        """Load the TFLite object detection model."""
        model_path = "/home/raspi/Bosch/Brain/src/Detection/Object/threads/best-int8.tflite"  # Update with correct model path
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape and dtype
        self.input_shape = self.input_details[0]['shape']
        self.input_height, self.input_width = self.input_shape[1], self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']

        # Define class labels
        self.class_labels = [
            "car", "closed-road-stand", "crosswalk-sign", "highway-entry-sign",
            "highway-exit-sign", "no-entry-road-sign", "one-way-road-sign",
            "parking-sign", "parking-spot", "pedestrian", "priority-sign",
            "round-about-sign", "stop-line", "stop-sign", "traffic-light"
        ]

    def subscribe(self):
        """Subscribe to required message channels."""
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, obcamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.HiSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)

    def send(self):
        """Initialize message senders."""
        self.motorspSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.wa = messageHandlerSender(self.queuesList, WarningSignal)
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)

    def preprocess_frame(self, frame):
        """Preprocess the frame for inference."""
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized_frame, axis=0).astype(self.input_dtype)
        return input_data

    def model_infer(self, frame):
        """Run inference on the preprocessed frame."""
        input_data = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def is_object_detected(self, results, frame):
        """Process inference results and check for detected objects."""
        grid_size = results.shape[1]
        num_classes = len(self.class_labels)
        num_outputs_per_cell = 20

        boxes = results[0, :, :4]
        objectness = results[0, :, 4]
        class_probs = results[0, :, 5:]

        objectness_threshold = 0.5
        detected_boxes = []
        detected_classes = []
        detected_scores = []

        for i in range(grid_size):
            if objectness[i] > objectness_threshold:
                class_id = np.argmax(class_probs[i])
                score = class_probs[i, class_id] * objectness[i]

                x_center, y_center, width, height = boxes[i]
                xmin = int((x_center - width / 2) * frame.shape[1])
                ymin = int((y_center - height / 2) * frame.shape[0])
                xmax = int((x_center + width / 2) * frame.shape[1])
                ymax = int((y_center + height / 2) * frame.shape[0])

                detected_boxes.append([xmin, ymin, xmax, ymax])
                detected_classes.append(self.class_labels[class_id])
                detected_scores.append(score)

        for i in range(len(detected_boxes)):
            xmin, ymin, xmax, ymax = detected_boxes[i]
            label = f"{detected_classes[i]}: {detected_scores[i]:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Log or print the detected class name
            self.logging.info(f"Detected class: {detected_classes[i]}, Confidence: {detected_scores[i]:.2f}")

        return len(detected_boxes) > 0

    def run(self):
        """Run the detection process after an initial delay."""
        while self._running:
            try:
                camera_data = self.serialCameraSubscriber.receive()
                yk = self.HiSubscriber.receive()

                if yk is not None:
                    if yk == "hiiii":
                        mode = self.DrivingModeSubscriber.receive()
                        if mode == "stop" or mode == "manual" or mode == "legacy":
                            self.send()
                            self.steerSender.send("0")
                            self.motorspSender.send("0")
                            self.var = False
                        if mode == "auto":
                            self.send()
                            self.motorspSender.send("250")
                            self.steerSender.send("0")
                            self.var = True

                if self.var:
                    if camera_data:
                        image_data = base64.b64decode(camera_data)
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

                    if frame is not None:
                        detection_results = self.model_infer(frame)
                        if self.is_object_detected(detection_results, frame):
                            self.logging.info("Object detected.")
                        else:
                            self.logging.info("No object detected.")
                            self.motorspSender.send("250")
                            self.steerSender.send("0")
                    else:
                        self.logging.error("Decoded frame is None.")

            except Exception as e:
                self.logging.error(f"Error in main thread loop: {e}")

    def stop(self):
        """Stop the detection thread."""
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
