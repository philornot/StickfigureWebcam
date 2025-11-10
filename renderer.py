"""
Rendering module for stickfigure and debug views.

This module handles all rendering operations, including the main stickfigure
view and debug overlay with detection visualizations.
"""

import numpy as np

import config
from stickfigure import draw_stickfigure
from ui import draw_no_person_message, create_debug_overlay
from face_detection import draw_face_landmarks


class Renderer:
    """
    Renderer class for creating stickfigure and debug visualizations.

    This class handles all rendering operations for both the main stickfigure
    output and the optional debug view with detection overlays.
    """

    def __init__(self, width, height):
        """
        Initialize the renderer.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self.width = width
        self.height = height
        print(f"[Renderer] Initialized with resolution {width}x{height}")

    def render_stickfigure_view(self, pose_results, mouth_open, eyes_closed):
        """
        Render the main stickfigure view without any debug overlays.

        Creates a clean black canvas with only the stickfigure or a
        "no person detected" message.

        Args:
            pose_results: MediaPipe pose detection results.
            mouth_open: Whether mouth is detected as open.
            eyes_closed: Whether eyes are detected as closed.

        Returns:
            numpy.ndarray: Canvas with stickfigure rendered (BGR format).
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

    def render_debug_view(self, frame, pose_results, face_results,
                          mouth_open, eyes_closed, fps):
        """
        Render the debug camera view with detection overlays.

        Shows the original camera frame with stickfigure overlay,
        face landmarks, and detection statistics.

        Args:
            frame: Original camera frame (BGR format).
            pose_results: MediaPipe pose detection results.
            face_results: MediaPipe face mesh results.
            mouth_open: Whether mouth is detected as open.
            eyes_closed: Whether eyes are detected as closed.
            fps: Current frames per second.

        Returns:
            numpy.ndarray: Frame with debug overlay (BGR format).
        """
        landmarks = (pose_results.pose_landmarks.landmark
                     if pose_results.pose_landmarks else None)

        return create_debug_overlay(
            frame,
            pose_results,
            face_results,
            landmarks,
            self.width,
            self.height,
            mouth_open,
            eyes_closed,
            fps,
            draw_stickfigure,
            draw_face_landmarks
        )

    def render_both_views(self, detection_data, fps, debug_mode=False):
        """
        Convenience method to render both main and debug views.

        Args:
            detection_data: Dictionary with detection results containing:
                - frame: Original camera frame
                - pose_results: MediaPipe pose results
                - face_results: MediaPipe face results
                - mouth_open: Boolean for mouth state
                - eyes_closed: Boolean for eye state
            fps: Current frames per second.
            debug_mode: Whether to render debug view.

        Returns:
            tuple: (stickfigure_canvas, debug_canvas or None)
        """
        # Render main stickfigure view
        stickfigure_canvas = self.render_stickfigure_view(
            detection_data['pose_results'],
            detection_data['mouth_open'],
            detection_data['eyes_closed']
        )

        # Render debug view if enabled
        debug_canvas = None
        if debug_mode:
            debug_canvas = self.render_debug_view(
                detection_data['frame'],
                detection_data['pose_results'],
                detection_data['face_results'],
                detection_data['mouth_open'],
                detection_data['eyes_closed'],
                fps
            )

        return stickfigure_canvas, debug_canvas