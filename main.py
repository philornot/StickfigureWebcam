"""
Main application module for Stickfigure Webcam.

This module contains the main application loop that ties together
all components: camera, detection, drawing, and UI.
"""

import cv2
import mediapipe as mp
import numpy as np

import config
from face_detection import calculate_mouth_openness, draw_face_landmarks
from stickfigure import draw_stickfigure
from ui import (
    FPSCounter,
    draw_no_person_message,
    create_debug_overlay,
    print_startup_info,
    handle_key_press
)
from virtual_camera import VirtualCameraOutput


class StickfigureWebcam:
    """
    Main application class for Stickfigure Webcam.
    """

    def __init__(self):
        """Initialize the application with all required components."""
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

        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_ID)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Cannot open camera")

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize FPS counter and debug mode
        self.fps_counter = FPSCounter(config.FPS_UPDATE_INTERVAL)
        self.debug_mode = False

        # Initialize Virtual Camera
        self.vcam = VirtualCameraOutput(
            self.width,
            self.height,
            fps=config.CAMERA_FPS
        )
        self.vcam.start()

        print_startup_info(self.width, self.height)

    def process_frame(self, frame):
        """
        Process a single frame for pose and face detection.

        Args:
            frame: Input frame from camera

        Returns:
            tuple: (pose_results, face_results, mouth_open)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Process on smaller frame for performance
        processing_frame = cv2.resize(
            frame,
            (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
        )
        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Run detections
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        # Detect mouth opening
        mouth_open = False
        if face_results.multi_face_landmarks:
            mouth_open = calculate_mouth_openness(
                face_results.multi_face_landmarks[0].landmark,
                config.PROCESSING_WIDTH,
                config.PROCESSING_HEIGHT
            )

        return frame, pose_results, face_results, mouth_open

    def render_stickfigure_view(self, pose_results, mouth_open):
        """
        Render the main stickfigure view.

        Args:
            pose_results: MediaPipe pose detection results
            mouth_open: Whether mouth is detected as open

        Returns:
            numpy.ndarray: Canvas with stickfigure
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
                draw_debug_markers=False  # Always false for clean main view
            )
        else:
            draw_no_person_message(canvas, self.width, self.height)

        # Do NOT draw any debug overlays here to keep main window clean.
        return canvas

    def render_debug_view(self, frame, pose_results, face_results, mouth_open):
        """
        Render the debug camera view.

        Args:
            frame: Original camera frame
            pose_results: MediaPipe pose detection results
            face_results: MediaPipe face mesh results
            mouth_open: Whether mouth is detected as open

        Returns:
            numpy.ndarray: Frame with debug overlay
        """
        current_fps = self.fps_counter.get_fps()

        return create_debug_overlay(
            frame,
            pose_results,
            face_results,
            pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
            self.width,
            self.height,
            mouth_open,
            current_fps,
            draw_stickfigure,
            draw_face_landmarks
        )

    def run(self):
        """
        Main application loop.

        Continuously processes frames, updates displays, and handles user input
        until the user quits.
        """
        while True:
            # Read and process frame
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, pose_results, face_results, mouth_open = self.process_frame(frame)

            # Update FPS
            self.fps_counter.update()

            # Render main stickfigure view (always clean)
            stickfigure_canvas = self.render_stickfigure_view(pose_results, mouth_open)
            cv2.imshow(config.WINDOW_NAME_STICKFIGURE, stickfigure_canvas)

            # Send frame to virtual camera
            if self.vcam.is_active:
                self.vcam.send_frame(stickfigure_canvas)

            # Render debug view if enabled
            if self.debug_mode:
                debug_canvas = self.render_debug_view(frame, pose_results, face_results, mouth_open)
                cv2.imshow(config.WINDOW_NAME_DEBUG, debug_canvas)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            should_quit, self.debug_mode = handle_key_press(key, self.debug_mode)

            if should_quit:
                break

    def cleanup(self):
        """Clean up resources."""
        # Stop virtual camera
        if self.vcam.is_active:
            self.vcam.stop()

        self.pose.close()
        self.face_mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Application entry point.
    """
    try:
        app = StickfigureWebcam()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'app' in locals():
            app.cleanup()


if __name__ == "__main__":
    main()
