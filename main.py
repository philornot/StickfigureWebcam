"""
Main application module for Stickfigure Webcam.

This module contains the main application loop that ties together
all components: camera, detection, drawing, UI, and virtual camera output.
"""

import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

import config
from face_detection import (
    calculate_mouth_openness,
    calculate_eye_aspect_ratio,
    draw_face_landmarks
)
from stickfigure import draw_stickfigure
from ui import (
    FPSCounter,
    draw_no_person_message,
    create_debug_overlay,
    print_startup_info,
    handle_key_press
)
from virtual_camera import VirtualCameraOutput
from camera_selector import show_camera_selector


class StickfigureWebcam:
    """
    Main application class for Stickfigure Webcam.
    """

    def __init__(self, camera_id=None):
        """Initialize the application.

        Args:
            camera_id: Optional camera ID. If None, will show selection dialog.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize MediaPipe models
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

        # Determine camera ID
        if camera_id is None:
            camera_id = show_camera_selector(use_gui=True)
            if camera_id is None:
                raise RuntimeError("No camera selected")

        # Initialize camera with selected ID
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Cannot open camera {self.camera_id}")

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize FPS counter and debug mode
        self.fps_counter = FPSCounter(config.FPS_UPDATE_INTERVAL)
        self.debug_mode = False

        # Eye detection state (shared between threads)
        self.eyes_closed_frame_counter = 0
        self.eyes_closed = False
        self.state_lock = threading.Lock()

        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)  # Raw frames from camera
        self.detection_queue = queue.Queue(maxsize=2)  # Detection results
        self.running = threading.Event()
        self.running.set()

        self.capture_thread = None
        self.detection_thread = None

        # Initialize Virtual Camera
        self.vcam = VirtualCameraOutput(
            self.width,
            self.height,
            fps=config.CAMERA_FPS,
            mirror=config.VCAM_MIRROR_OUTPUT
        )
        self.vcam.start()

        print_startup_info(self.width, self.height)
        print(f"Using camera: {self.camera_id}")

    def capture_frames(self):
        """Thread 1: Continuously capture frames from camera.

        This thread runs independently and puts frames into the queue
        for processing by the detection thread.
        """
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Cannot read frame from camera.")
                time.sleep(0.01)
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Non-blocking put - drop frame if queue is full
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full

    def process_detections(self):
        """Thread 2: Process frames for pose and face detection.

        This thread takes frames from the capture queue, runs MediaPipe
        detections, and puts results into the detection queue.
        """
        while self.running.is_set():
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Process on smaller frame for performance
            processing_frame = cv2.resize(
                frame,
                (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )

            # Single color conversion to RGB
            frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

            # Run detections
            pose_results = self.pose.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)

            mouth_open = False
            eyes_closed = False

            # Process face landmarks if detected
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

                # Thread-safe state update
                with self.state_lock:
                    if ear_ratio < config.EYES_CLOSED_RATIO_THRESHOLD:
                        self.eyes_closed_frame_counter += 1
                    else:
                        self.eyes_closed_frame_counter = 0

                    eyes_closed = (
                            self.eyes_closed_frame_counter >= config.EYES_CLOSED_CONSECUTIVE_FRAMES
                    )
                    self.eyes_closed = eyes_closed
            else:
                # No face detected, reset state
                with self.state_lock:
                    self.eyes_closed_frame_counter = 0
                    self.eyes_closed = False

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

    def render_stickfigure_view(self, pose_results, mouth_open, eyes_closed):
        """Render the main stickfigure view without any debug text.

        Args:
            pose_results: MediaPipe pose detection results.
            mouth_open: Whether mouth is detected as open.
            eyes_closed: Whether eyes are detected as closed.

        Returns:
            numpy.ndarray: Canvas with stickfigure.
        """
        # Create black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw stick figure or message
        if pose_results.pose_landmarks:
            draw_stickfigure(
                canvas,
                pose_results.pose_landmarks.landmark,
                self.width,
                self.height,
                mouth_open,
                eyes_closed,
                draw_debug_markers=False
            )
        else:
            draw_no_person_message(canvas, self.width, self.height)

        return canvas

    def render_debug_view(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """Render the debug camera view.

        Args:
            frame: Original camera frame.
            pose_results: MediaPipe pose detection results.
            face_results: MediaPipe face mesh results.
            mouth_open: Whether mouth is detected as open.
            eyes_closed: Whether eyes are detected as closed.

        Returns:
            numpy.ndarray: Frame with debug overlay.
        """
        current_fps = self.fps_counter.get_fps()
        landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

        return create_debug_overlay(
            frame,
            pose_results,
            face_results,
            landmarks,
            self.width,
            self.height,
            mouth_open,
            eyes_closed,
            current_fps,
            draw_stickfigure,
            draw_face_landmarks
        )

    def run(self):
        """Main application loop.

        Main thread handles rendering and display while capture and
        detection run in separate threads.
        """
        try:
            # Start worker threads
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.detection_thread = threading.Thread(target=self.process_detections, daemon=True)

            self.capture_thread.start()
            self.detection_thread.start()
        except Exception as e:
            print(f"Capture and Detection threads failed to start. Error: {e}")

        while self.running.is_set():
            try:
                # Get detection results with timeout
                detection_data = self.detection_queue.get(timeout=0.1)
            except queue.Empty:
                # No new data, continue waiting
                continue

            frame = detection_data['frame']
            pose_results = detection_data['pose_results']
            face_results = detection_data['face_results']
            mouth_open = detection_data['mouth_open']

            # Get thread-safe eyes_closed state
            with self.state_lock:
                eyes_closed = self.eyes_closed

            # Update FPS
            self.fps_counter.update()

            # Render main stickfigure view
            stickfigure_canvas = self.render_stickfigure_view(
                pose_results, mouth_open, eyes_closed
            )
            cv2.imshow(config.WINDOW_NAME_STICKFIGURE, stickfigure_canvas)

            # Send to virtual camera
            if self.vcam.is_active:
                self.vcam.send_frame(stickfigure_canvas)
                self.vcam.sleep_until_next_frame()

            # Render debug view if enabled
            if self.debug_mode:
                debug_canvas = self.render_debug_view(
                    frame, pose_results, face_results, mouth_open, eyes_closed
                )
                cv2.imshow(config.WINDOW_NAME_DEBUG, debug_canvas)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            should_quit, self.debug_mode = handle_key_press(key, self.debug_mode)

            # Mirror toggle
            if key == ord('m'):
                self.vcam.set_mirror(not self.vcam.mirror)

            if should_quit:
                self.running.clear()
                break

        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

    def cleanup(self):
        """Clean up resources and stop all threads."""
        self.running.clear()

        if self.vcam.is_active:
            self.vcam.stop()

        self.pose.close()
        self.face_mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application cleaned up and exited.")


def main():
    """Application entry point."""
    app = None
    try:
        app = StickfigureWebcam()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if app is not None:
            app.cleanup()


if __name__ == "__main__":
    main()
