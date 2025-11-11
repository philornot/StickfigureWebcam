"""
Detection worker module for threaded pose and face detection.

This module handles all MediaPipe detection operations in separate threads
for optimal performance, including pose detection, face mesh, mouth opening,
and eye blink detection.
"""

import queue
import threading
import time

import cv2
import mediapipe as mp

import config
from face_detection import calculate_mouth_openness, calculate_eye_aspect_ratio


class DetectionWorker:
    """
    Worker class for handling camera capture and detection in separate threads.

    This class manages two threads:
    - Capture thread: Continuously reads frames from camera
    - Detection thread: Processes frames through MediaPipe models
    """

    def __init__(self, camera_capture):
        """
        Initialize the detection worker.

        Args:
            camera_capture: OpenCV VideoCapture object for camera access.
        """
        self.cap = camera_capture

        # Initialize MediaPipe solutions (but not models yet)
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        # Models will be initialized lazily in the detection thread
        self.pose = None
        self.face_mesh = None

        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue(maxsize=2)
        self.running = threading.Event()
        self.state_lock = threading.Lock()

        # Eye detection state
        self.eyes_closed_frame_counter = 0
        self.eyes_closed = False

        # Thread references
        self.capture_thread = None
        self.detection_thread = None

        print("[DetectionWorker] Initialized")

    def start(self):
        """Start the worker threads."""
        self.running.set()

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CaptureThread"
        )
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="DetectionThread"
        )

        self.capture_thread.start()
        self.detection_thread.start()

        print("[DetectionWorker] Threads started")

    def stop(self):
        """Stop the worker threads."""
        print("[DetectionWorker] Stopping threads...")
        self.running.clear()

        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)

        print("[DetectionWorker] Threads stopped")

    def get_detection_result(self, timeout=0.1):
        """
        Get the next detection result from the queue.

        Args:
            timeout: Maximum time to wait for a result in seconds.

        Returns:
            dict or None: Detection result dictionary with keys:
                - frame: Original camera frame
                - pose_results: MediaPipe pose detection results
                - face_results: MediaPipe face mesh results
                - mouth_open: Boolean indicating if mouth is open
                - eyes_closed: Boolean indicating if eyes are closed
        """
        try:
            return self.detection_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self.pose:
            self.pose.close()
        if self.face_mesh:
            self.face_mesh.close()
        print("[DetectionWorker] Resources cleaned up")

    def _capture_loop(self):
        """
        Thread 1: Continuously capture frames from camera.

        This thread runs independently and puts frames into the queue
        for processing by the detection thread.
        """
        print("[CaptureThread] Started")

        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("[CaptureThread] Failed to read frame")
                time.sleep(0.01)
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Non-blocking put - drop frame if queue is full
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full

        print("[CaptureThread] Stopped")

    def _detection_loop(self):
        """
        Thread 2: Process frames for pose and face detection.

        This thread takes frames from the capture queue, runs MediaPipe
        detections, and puts results into the detection queue.
        """
        print("[DetectionThread] Started")

        # Lazy initialization: Initialize MediaPipe models only when needed
        # This speeds up startup time significantly
        models_initialized = False

        while self.running.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Initialize MediaPipe models on first frame
            if not models_initialized:
                print("[DetectionThread] Initializing MediaPipe models...")
                self._initialize_mediapipe_models()
                models_initialized = True
                print("[DetectionThread] MediaPipe models ready")

            # Process on smaller frame for performance
            processing_frame = cv2.resize(
                frame,
                (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

            # Run detections
            pose_results = self.pose.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)

            # Process face landmarks
            mouth_open, eyes_closed = self._process_face_landmarks(face_results)

            # Package results
            detection_data = {
                'frame': frame,
                'pose_results': pose_results,
                'face_results': face_results,
                'mouth_open': mouth_open,
                'eyes_closed': eyes_closed
            }

            # Non-blocking put - drop result if queue is full
            try:
                self.detection_queue.put(detection_data, block=False)
            except queue.Full:
                pass  # Skip result if queue is full

        print("[DetectionThread] Stopped")

    def _initialize_mediapipe_models(self):
        """
        Initialize MediaPipe models.

        This method is called lazily when the first frame arrives,
        avoiding blocking camera initialization at startup.
        """
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=config.POSE_MODEL_COMPLEXITY
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE
        )

    def _process_face_landmarks(self, face_results):
        """
        Process face landmarks to detect mouth opening and eye closure.

        Args:
            face_results: MediaPipe face mesh detection results.

        Returns:
            tuple: (mouth_open: bool, eyes_closed: bool)
        """
        mouth_open = False
        eyes_closed = False

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Detect mouth opening
            mouth_open = calculate_mouth_openness(
                landmarks,
                config.PROCESSING_WIDTH,
                config.PROCESSING_HEIGHT
            )

            # Detect closed eyes
            ear_ratio = calculate_eye_aspect_ratio(
                landmarks,
                config.PROCESSING_WIDTH,
                config.PROCESSING_HEIGHT
            )

            # Thread-safe state update for eye blink detection
            with self.state_lock:
                if ear_ratio < config.EYES_CLOSED_RATIO_THRESHOLD:
                    self.eyes_closed_frame_counter += 1
                else:
                    self.eyes_closed_frame_counter = 0

                eyes_closed = (
                        self.eyes_closed_frame_counter >=
                        config.EYES_CLOSED_CONSECUTIVE_FRAMES
                )
                self.eyes_closed = eyes_closed
        else:
            # No face detected, reset state
            with self.state_lock:
                self.eyes_closed_frame_counter = 0
                self.eyes_closed = False

        return mouth_open, eyes_closed
